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

import pyarrow as pa
import pyarrow.compute as pc

from tacotoolbox._column_utils import (
    align_arrow_schemas,
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
        levels: list[pa.Table],
        local_metadata: dict[str, pa.Table],
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
        tables = []

        # Generate consolidated metadata for each level
        for depth in range(self.max_depth + 1):
            table = self.taco.tortilla.export_metadata(deep=depth)
            table = self._clean_table(table)

            # Add internal:parent_id for level 0 as row index
            if depth == 0:
                parent_id_array = pa.array(range(table.num_rows), type=pa.int64())
                parent_id_field = pa.field(METADATA_PARENT_ID, pa.int64())
                table = table.append_column(parent_id_field, parent_id_array)
                table = reorder_internal_columns(table)

            tables.append(table)

        # Add internal:relative_path for fast SQL queries
        tables = self._add_relative_paths(tables)

        # Generate schemas
        pit_schema = generate_pit_schema(tables, debug=self.debug)
        field_schema = generate_field_schema(tables, self.taco)

        levels = tables

        # Generate local metadata for folders (level 1+ only)
        local_metadata = {}

        for sample in self.taco.tortilla.samples:
            if sample.type == "FOLDER":
                folder_path = f"DATA/{sample.id}/"
                folder_table = self._generate_folder_metadata(sample)
                local_metadata[folder_path] = folder_table

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

    def _add_relative_paths(self, levels: list[pa.Table]) -> list[pa.Table]:
        """Add internal:relative_path to level 1+ (consolidated metadata only)."""
        result_levels = []

        for depth, table in enumerate(levels):
            if depth == 0:
                # Level 0: NO relative_path - just use id directly
                result_levels.append(table)
                continue

            if table.num_rows == 0:
                # Empty Table - just add empty column
                null_array = pa.nulls(table.num_rows, type=pa.string())
                relative_path_field = pa.field(METADATA_RELATIVE_PATH, pa.string())
                table = table.append_column(relative_path_field, null_array)
                result_levels.append(reorder_internal_columns(table))
                continue

            # Level 1+: Build relative path from parent's id (level 0) or relative_path (level 1+)
            parent_table = result_levels[depth - 1]

            # For level 1, parent is level 0, use 'id'
            # For level 2+, parent has relative_path, use that
            if depth == 1:
                # Parent is level 0 - use 'id' + '/' for FOLDERs
                parent_ids = parent_table.column("id").to_pylist()
                parent_types = parent_table.column("type").to_pylist()
                parent_paths = [
                    f"{pid}/" if ptype == "FOLDER" else pid
                    for pid, ptype in zip(parent_ids, parent_types, strict=False)
                ]
            else:
                # Parent is level 1+ - already has relative_path
                parent_paths = parent_table.column(METADATA_RELATIVE_PATH).to_pylist()

            # Build paths for current level
            relative_paths = []
            parent_id_column = table.column(METADATA_PARENT_ID)
            id_column = table.column("id")
            type_column = table.column("type")

            for i in range(table.num_rows):
                parent_id = parent_id_column[i].as_py()
                sample_id = id_column[i].as_py()
                sample_type = type_column[i].as_py()

                parent_path = parent_paths[parent_id]

                if sample_type == "FOLDER":
                    relative_paths.append(f"{parent_path}{sample_id}/")
                else:
                    relative_paths.append(f"{parent_path}{sample_id}")

            # Add column and reorder
            relative_path_array = pa.array(relative_paths, type=pa.string())
            relative_path_field = pa.field(METADATA_RELATIVE_PATH, pa.string())
            table = table.append_column(relative_path_field, relative_path_array)
            table = reorder_internal_columns(table)
            result_levels.append(table)

        return result_levels

    def _generate_folder_metadata(self, folder_sample: "Sample") -> pa.Table:
        """
        Generate local metadata for a single folder.

        Critical: Children may have different extensions (e.g., cloudmask vs s2data vs thumbnail),
        resulting in different schema widths. Must align schemas before concatenation.
        """
        tortilla = cast(Tortilla, folder_sample.path)
        samples = tortilla.samples
        metadata_tables = []

        for sample in samples:
            metadata_table = sample.export_metadata()
            metadata_tables.append(metadata_table)

        # Align schemas before concatenating
        aligned_tables = align_arrow_schemas(metadata_tables)
        table = pa.concat_tables(aligned_tables)

        return self._clean_table(table)

    def _generate_nested_folders(
        self, parent_sample: "Sample", parent_path: str
    ) -> dict[str, pa.Table]:
        """Recursively generate local metadata for nested folders."""
        result = {}

        tortilla = cast(Tortilla, parent_sample.path)
        for child in tortilla.samples:
            if child.type == "FOLDER":
                folder_path = f"{parent_path}{child.id}/"
                folder_table = self._generate_folder_metadata(child)
                result[folder_path] = folder_table

                # Recurse into nested folders
                nested = self._generate_nested_folders(child, folder_path)
                result.update(nested)

        return result

    def _clean_table(self, table: pa.Table) -> pa.Table:
        """Clean Table by removing unnecessary columns."""
        # Remove path column (filesystem-specific, not container-relevant)
        container_irrelevant_cols = ["path"]
        cols_to_drop = [
            col for col in container_irrelevant_cols if col in table.schema.names
        ]

        if cols_to_drop:
            table = table.drop(cols_to_drop)

        # Remove empty columns (preserves core + internal automatically)
        table = remove_empty_columns(table, preserve_core=True, preserve_internal=True)

        if table.num_columns == 0:
            raise ValueError(
                "Table cleaning resulted in no columns. "
                "This should never happen as core fields are preserved."
            )

        return table


def generate_field_schema(levels: list[pa.Table], taco: "Taco") -> dict[str, Any]:
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

    for i, level_table in enumerate(levels):
        fields = []
        for field in level_table.schema:
            col_name = field.name
            type_name = str(field.type).lower()
            description = all_descriptions.get(col_name, "")
            fields.append([col_name, type_name, description])

        field_schema[f"level{i}"] = fields

    return field_schema


def generate_pit_schema(  # noqa: C901
    tables: list[pa.Table], debug: bool = False
) -> dict[str, Any]:
    """
    Generate PIT schema.

    NOTE: This function has high cyclomatic complexity (C901) but is intentionally
    kept as a single function for algorithmic clarity. The PIT schema generation
    is inherently complex and splitting it would reduce readability.
    """
    if not tables:
        raise ValueError("Need at least one Table to generate schema")

    table0 = tables[0]
    if "type" not in table0.schema.names:
        raise ValueError("Level 0 missing 'type' column")

    # Root level
    root_type = table0.column("type")[0].as_py()
    root = {"n": table0.num_rows, "type": root_type}

    hierarchy: dict[str, list[dict]] = {}

    # Process each hierarchical level
    for depth in range(1, len(tables)):
        table = tables[depth]
        parent_table = tables[depth - 1]

        if table.num_rows == 0:
            continue

        if METADATA_PARENT_ID not in table.schema.names:
            raise ValueError(f"Depth {depth} missing '{METADATA_PARENT_ID}' column")

        if depth == 1:
            # LEVEL 1: Find folder with MOST real (non-padding) samples
            children_per_parent = table.num_rows // parent_table.num_rows
            target_real_count = children_per_parent

            max_real_count = 0
            best_group_ids = []
            best_group_types = []

            for parent_idx in range(parent_table.num_rows):
                start_idx = parent_idx * children_per_parent
                end_idx = start_idx + children_per_parent
                group = table.slice(start_idx, children_per_parent)

                ids = group.column("id").to_pylist()
                types = group.column("type").to_pylist()

                real_count = sum(1 for id_val in ids if not is_padding_id(id_val))

                if real_count > max_real_count:
                    max_real_count = real_count
                    best_group_ids = ids
                    best_group_types = types

                if real_count == target_real_count:
                    if debug:
                        parent_id = parent_table.column("id")[parent_idx].as_py()
                        print(
                            f"Found canonical folder '{parent_id}' at depth {depth} "
                            f"with {real_count}/{children_per_parent} real samples"
                        )
                    break

            pattern = {
                "n": table.num_rows,
                "type": best_group_types,
                "id": best_group_ids,
            }
            hierarchy[str(depth)] = [pattern]

        else:
            # LEVEL 2+: Multiple folder positions possible
            parent_schema = hierarchy[str(depth - 1)]
            parent_pattern = parent_schema[0]["type"]
            pattern_size = len(parent_pattern)
            num_groups = parent_table.num_rows // pattern_size

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

                # Filter table for matching parent_ids
                parent_id_column = table.column(METADATA_PARENT_ID)
                mask = pc.is_in(parent_id_column, pa.array(parent_ids_for_position))
                position_children = table.filter(mask)

                if position_children.num_rows == 0:
                    continue

                samples_per_group = position_children.num_rows // num_groups

                if samples_per_group == 0:
                    continue

                target_real_count = samples_per_group
                max_real_count = 0
                best_group_ids = []
                best_group_types = []

                for group_idx in range(num_groups):
                    start_idx = group_idx * samples_per_group
                    group = position_children.slice(start_idx, samples_per_group)

                    ids = group.column("id").to_pylist()
                    types = group.column("type").to_pylist()

                    real_count = sum(1 for id_val in ids if not is_padding_id(id_val))

                    if real_count > max_real_count:
                        max_real_count = real_count
                        best_group_ids = ids
                        best_group_types = types

                    if real_count == target_real_count:
                        if debug:
                            parent_row_idx = parent_ids_for_position[group_idx]
                            parent_id = parent_table.column("id")[
                                parent_row_idx
                            ].as_py()
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
