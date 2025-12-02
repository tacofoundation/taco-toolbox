"""
Metadata generation for TACO containers.

This module generates the dual metadata system used by both ZIP and FOLDER containers:

1. CONSOLIDATED METADATA (METADATA/levelX files):
   - One file per hierarchy level (level0.parquet, level1.parquet, etc.)
   - Contains ALL samples at that level across the entire dataset
   - Includes internal:parent_id for relational queries (all levels)
   - Includes internal:relative_path for fast SQL queries (level 1+ only)
   - Used for: sql queries, navigation, statistics

2. LOCAL METADATA (DATA/folder/__meta__ files):
   - One file per FOLDER (only for level 1+)
   - Contains only the direct children of that specific folder
   - Does NOT include internal:parent_id (navigation via folder structure)
   - Does NOT include internal:relative_path (path is implicit from location)
   - Used for: efficient/lazy folder-level access

CRITICAL DESIGN PRINCIPLES:
- This module is SHARED between ZIP and FOLDER containers
- ZIP-specific operations (offset/size calculation) belong in zip_writer.py
- Relies on TACO's HOMOGENEITY guarantee:
  * All folders at same level have SAME number of children
  * All folders at same level have SAME child IDs (with padding)
  * Padding (__TACOPAD__*) ensures structural uniformity
"""

from typing import TYPE_CHECKING, Any, cast

import polars as pl

from tacotoolbox._column_utils import (
    align_dataframe_schemas,
    remove_empty_columns,
    reorder_internal_columns,
)
from tacotoolbox._constants import METADATA_PARENT_ID, METADATA_RELATIVE_PATH
from tacotoolbox._utils import is_padding_id
from tacotoolbox.tortilla.datamodel import Tortilla

if TYPE_CHECKING:
    from tacotoolbox.sample.datamodel import Sample
    from tacotoolbox.taco.datamodel import Taco


class MetadataPackage:
    """
    Complete metadata bundle for TACO containers.

    This package contains everything needed to create both ZIP and FOLDER containers:
    - Consolidated metadata for all levels (METADATA/)
    - Local metadata for individual folders (DATA/folder/)
    - Collection metadata (COLLECTION.json)
    - Schema information (PIT + field schemas)

    The internal:parent_id column enables hierarchical navigation via SQL JOINs.
    The internal:relative_path column enables fast queries without JOINs.
    Level 0 uses row index as parent_id, while level 1+ link to parent rows.

    SQL examples:
        # Level 0: Use 'id' directly (no relative_path)
        SELECT * FROM level0 WHERE id = 'A'

        # Level 1+: Use relative_path (fast, no JOIN needed)
        SELECT * FROM level1
        WHERE "internal:relative_path" LIKE 'A/%'

        # Traditional JOIN approach (works but slower):
        SELECT l1.id, l0.id as parent_folder
        FROM level0 l0
        JOIN level1 l1 ON l1."internal:parent_id" = l0."internal:parent_id"
        WHERE l0.id = 'A'
    """

    def __init__(
        self,
        levels: list[pl.DataFrame],
        local_metadata: dict[str, pl.DataFrame],
        collection: dict[str, Any],
        pit_schema: dict[str, Any],
        field_schema: dict[str, Any],
        max_depth: int,
    ):
        self.levels = levels
        self.local_metadata = local_metadata
        self.collection = collection
        self.pit_schema = pit_schema
        self.field_schema = field_schema
        self.max_depth = max_depth


class MetadataGenerator:
    """Generate complete metadata package for TACO containers."""

    def __init__(self, taco: "Taco", debug: bool = False) -> None:
        """Initialize metadata generator."""
        self.taco = taco
        self.debug = debug
        self.max_depth = min(taco.tortilla._current_depth, 5)

    def generate_all_levels(self) -> MetadataPackage:
        """Generate complete metadata package for both ZIP and FOLDER containers."""
        levels = []
        dataframes = []

        # Generate consolidated metadata for each level
        for depth in range(self.max_depth + 1):
            df = self.taco.tortilla.export_metadata(deep=depth)
            df = self._clean_dataframe(df)

            # Add internal:parent_id for level 0 as row index
            if depth == 0:
                df = df.with_columns(
                    pl.arange(0, len(df)).cast(pl.Int64).alias(METADATA_PARENT_ID)
                )
                df = reorder_internal_columns(df)

            dataframes.append(df)

        # Add internal:relative_path for fast SQL queries
        dataframes = self._add_relative_paths(dataframes)

        # Generate schemas
        pit_schema = generate_pit_schema(dataframes, debug=self.debug)
        field_schema = generate_field_schema(dataframes, self.taco)

        levels = dataframes

        # Generate local metadata for folders (level 1+ only)
        local_metadata = {}

        for sample in self.taco.tortilla.samples:
            if sample.type == "FOLDER":
                folder_path = f"DATA/{sample.id}/"
                folder_df = self._generate_folder_metadata(sample)
                local_metadata[folder_path] = folder_df

                # Recursively process nested folders
                nested = self._generate_nested_folders(sample, f"DATA/{sample.id}/")
                local_metadata.update(nested)

        # Generate collection metadata
        collection = generate_collection_json(self.taco)

        return MetadataPackage(
            levels=levels,
            local_metadata=local_metadata,
            collection=collection,
            pit_schema=pit_schema,
            field_schema=field_schema,
            max_depth=self.max_depth,
        )

    def _add_relative_paths(self, levels: list[pl.DataFrame]) -> list[pl.DataFrame]:
        """Add internal:relative_path to level 1+ (consolidated metadata only)."""
        result_levels = []

        for depth, df in enumerate(levels):
            if depth == 0:
                # Level 0: NO relative_path - just use id directly
                result_levels.append(df)
                continue

            if len(df) == 0:
                # Empty DataFrame - just add empty column
                df = df.with_columns(
                    pl.lit(None, dtype=pl.Utf8).alias(METADATA_RELATIVE_PATH)
                )
                result_levels.append(reorder_internal_columns(df))
                continue

            # Level 1+: Build relative path from parent's id (level 0) or relative_path (level 1+)
            parent_df = result_levels[depth - 1]

            # For level 1, parent is level 0, use 'id'
            # For level 2+, parent has relative_path, use that
            if depth == 1:
                # Parent is level 0 - use 'id' + '/' for FOLDERs
                parent_ids = parent_df["id"].to_list()
                parent_types = parent_df["type"].to_list()
                parent_paths = [
                    f"{pid}/" if ptype == "FOLDER" else pid
                    for pid, ptype in zip(parent_ids, parent_types, strict=False)
                ]
            else:
                # Parent is level 1+ - already has relative_path
                parent_paths = parent_df[METADATA_RELATIVE_PATH].to_list()

            # Build paths for current level
            relative_paths = []
            for row in df.iter_rows(named=True):
                parent_id = row[METADATA_PARENT_ID]
                parent_path = parent_paths[parent_id]

                if row["type"] == "FOLDER":
                    relative_paths.append(f"{parent_path}{row['id']}/")
                else:
                    relative_paths.append(f"{parent_path}{row['id']}")

            # Add column and reorder
            df = df.with_columns(pl.Series(METADATA_RELATIVE_PATH, relative_paths))
            df = reorder_internal_columns(df)
            result_levels.append(df)

        return result_levels

    def _generate_folder_metadata(self, folder_sample: "Sample") -> pl.DataFrame:
        """
        Generate local metadata for a single folder.

        Critical: Children may have different extensions (e.g., cloudmask vs s2data vs thumbnail),
        resulting in different schema widths. Must align schemas before concatenation.
        """
        # Cast path to Tortilla to access .samples
        tortilla = cast(Tortilla, folder_sample.path)
        samples = tortilla.samples
        metadata_dfs = [s.export_metadata() for s in samples]

        # Align schemas before concatenating (samples may have different extensions)
        # Without this, pl.concat fails with: ShapeError: unable to append DataFrame of width X with width Y
        aligned_dfs = align_dataframe_schemas(metadata_dfs)
        df = pl.concat(aligned_dfs, how="vertical")

        return self._clean_dataframe(df)

    def _generate_nested_folders(
        self, parent_sample: "Sample", parent_path: str
    ) -> dict[str, pl.DataFrame]:
        """Recursively generate local metadata for nested folders."""
        result = {}

        # Cast path to Tortilla to access .samples
        tortilla = cast(Tortilla, parent_sample.path)
        for child in tortilla.samples:
            if child.type == "FOLDER":
                folder_path = f"{parent_path}{child.id}/"
                folder_df = self._generate_folder_metadata(child)
                result[folder_path] = folder_df

                # Recurse into nested folders
                nested = self._generate_nested_folders(child, folder_path)
                result.update(nested)

        return result

    def _clean_dataframe(self, df: pl.DataFrame) -> pl.DataFrame:
        """Clean DataFrame by removing unnecessary columns."""
        # Remove path column (filesystem-specific, not container-relevant)
        container_irrelevant_cols = ["path"]
        df = df.drop([col for col in container_irrelevant_cols if col in df.columns])

        # Remove empty columns (preserves core + internal automatically)
        df = remove_empty_columns(df, preserve_core=True, preserve_internal=True)

        if len(df.columns) == 0:
            raise ValueError(
                "DataFrame cleaning resulted in no columns. "
                "This should never happen as core fields are preserved."
            )

        return df


def generate_field_schema(levels: list[pl.DataFrame], taco: "Taco") -> dict[str, Any]:
    """
    Generate field schema with descriptions from extensions.

    Collects field descriptions from all Sample and Tortilla extensions,
    then generates arrays of [name, type, description] for each field.
    If no description exists for a field, uses empty string.
    """
    from tacotoolbox._constants import (
        CORE_FIELD_DESCRIPTIONS,
        INTERNAL_FIELD_DESCRIPTIONS,
    )

    # Collect all field descriptions
    all_descriptions: dict[str, str] = {}

    # 1. Add core and internal field descriptions first
    all_descriptions.update(CORE_FIELD_DESCRIPTIONS)
    all_descriptions.update(INTERNAL_FIELD_DESCRIPTIONS)

    # 2. From tortilla (overrides core/internal if redefined)
    if hasattr(taco.tortilla, "_field_descriptions"):
        all_descriptions.update(taco.tortilla._field_descriptions)

    # 3. From samples recursively (overrides previous)
    def collect_from_samples(samples: list["Sample"]) -> None:
        for sample in samples:
            if hasattr(sample, "_field_descriptions"):
                all_descriptions.update(sample._field_descriptions)

            # Recurse into FOLDER samples
            if sample.type == "FOLDER":
                tortilla_path = cast(Tortilla, sample.path)
                if hasattr(tortilla_path, "samples"):
                    collect_from_samples(tortilla_path.samples)

    collect_from_samples(taco.tortilla.samples)

    # Generate field schema with descriptions
    field_schema = {}

    for i, level_df in enumerate(levels):
        fields = []
        for col_name, col_type in level_df.schema.items():
            type_name = str(col_type).lower()
            description = all_descriptions.get(col_name, "")
            fields.append([col_name, type_name, description])

        field_schema[f"level{i}"] = fields

    return field_schema


def generate_pit_schema(  # noqa: C901
    dataframes: list[pl.DataFrame], debug: bool = False
) -> dict[str, Any]:
    """
    Generate PIT schema.

    NOTE: This function has high cyclomatic complexity (C901) but is intentionally
    kept as a single function for algorithmic clarity. The PIT schema generation
    is inherently complex and splitting it would reduce readability.
    """
    if not dataframes:
        raise ValueError("Need at least one DataFrame to generate schema")

    df0 = dataframes[0]
    if "type" not in df0.columns:
        raise ValueError("Level 0 missing 'type' column")

    # Root level
    root_type = df0["type"][0]
    root = {"n": len(df0), "type": root_type}

    hierarchy: dict[str, list[dict]] = {}

    # Process each hierarchical level
    for depth in range(1, len(dataframes)):
        df = dataframes[depth]
        parent_df = dataframes[depth - 1]

        if len(df) == 0:
            continue

        if METADATA_PARENT_ID not in df.columns:
            raise ValueError(f"Depth {depth} missing '{METADATA_PARENT_ID}' column")

        if depth == 1:
            # LEVEL 1: Find folder with MOST real (non-padding) samples
            children_per_parent = len(df) // len(parent_df)
            target_real_count = children_per_parent

            max_real_count = 0
            best_group_ids = []
            best_group_types = []

            for parent_idx in range(len(parent_df)):
                start_idx = parent_idx * children_per_parent
                end_idx = start_idx + children_per_parent
                group = df[start_idx:end_idx]

                ids = group["id"].to_list()
                types = group["type"].to_list()

                real_count = sum(1 for id_val in ids if not is_padding_id(id_val))

                if real_count > max_real_count:
                    max_real_count = real_count
                    best_group_ids = ids
                    best_group_types = types

                if real_count == target_real_count:
                    if debug:
                        parent_id = parent_df["id"][parent_idx]
                        print(
                            f"Found canonical folder '{parent_id}' at depth {depth} "
                            f"with {real_count}/{children_per_parent} real samples"
                        )
                    break

            pattern = {"n": len(df), "type": best_group_types, "id": best_group_ids}
            hierarchy[str(depth)] = [pattern]

        else:
            # LEVEL 2+: Multiple folder positions possible
            parent_schema = hierarchy[str(depth - 1)]
            parent_pattern = parent_schema[0]["type"]
            pattern_size = len(parent_pattern)
            num_groups = len(parent_df) // pattern_size

            folder_positions = [
                i for i, t in enumerate(parent_pattern) if t == "FOLDER"
            ]

            if not folder_positions:
                continue

            all_patterns: list[dict] = []

            for position_idx in folder_positions:
                parent_ids_for_position = [
                    group_idx * pattern_size + position_idx
                    for group_idx in range(num_groups)
                ]

                position_children = df.filter(
                    pl.col(METADATA_PARENT_ID).is_in(parent_ids_for_position)
                )

                if len(position_children) == 0:
                    continue

                samples_per_group = len(position_children) // num_groups

                if samples_per_group == 0:
                    continue

                target_real_count = samples_per_group
                max_real_count = 0
                best_group_ids = []
                best_group_types = []

                for group_idx in range(num_groups):
                    start_idx = group_idx * samples_per_group
                    end_idx = start_idx + samples_per_group
                    group = position_children[start_idx:end_idx]

                    ids = group["id"].to_list()
                    types = group["type"].to_list()

                    real_count = sum(1 for id_val in ids if not is_padding_id(id_val))

                    if real_count > max_real_count:
                        max_real_count = real_count
                        best_group_ids = ids
                        best_group_types = types

                    if real_count == target_real_count:
                        if debug:
                            parent_row_idx = parent_ids_for_position[group_idx]
                            parent_id = parent_df["id"][parent_row_idx]
                            print(
                                f"Found canonical folder '{parent_id}' at depth {depth} "
                                f"position {position_idx} with {real_count}/{samples_per_group} real samples"
                            )
                        break

                total_nodes = num_groups * len(best_group_types)

                pattern_dict = {
                    "n": total_nodes,
                    "type": best_group_types,
                    "id": best_group_ids,
                }
                all_patterns.append(pattern_dict)

            hierarchy[str(depth)] = all_patterns

    return {"root": root, "hierarchy": hierarchy}


def generate_collection_json(taco: "Taco") -> dict[str, Any]:
    """Generate COLLECTION.json content from TACO object."""
    collection = taco.model_dump()
    collection.pop("tortilla", None)
    return collection
