"""
Metadata generation for TACO containers.

This module generates the dual metadata system used by both ZIP and FOLDER containers:

1. CONSOLIDATED METADATA (METADATA/levelX files):
   - One file per hierarchy level (level0.parquet, level1.parquet, etc.)
   - Contains ALL samples at that level across the entire dataset
   - Includes internal:parent_id for relational queries (all levels)
   - Includes internal:relative_path for fast SQL queries (level 1+ only)
   - Used for: sql, queries, navigation, statistics
   
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

Example structure:
    Level 0: [folder_A, folder_B, folder_C]  (3 folders)
    Level 1: Each folder has exactly 4 children (homogeneous):
        folder_A: [file_1, file_2, __TACOPAD__0, __TACOPAD__1]
        folder_B: [file_1, file_2, file_3, file_4]
        folder_C: [file_1, file_2, file_3, __TACOPAD__2]
    
    Even with different real files, padding ensures ALL have 4 children.
"""

from typing import TYPE_CHECKING, Any

import polars as pl

# Import from new modules
from tacotoolbox._column_utils import remove_empty_columns, reorder_internal_columns
from tacotoolbox._constants import (
    METADATA_PARENT_ID,
    METADATA_RELATIVE_PATH,
    is_padding_id,
)

if TYPE_CHECKING:
    from tacotoolbox.sample.datamodel import Sample
    from tacotoolbox.taco.datamodel import Taco


class PITValidationError(Exception):
    """Raised when Position-Isomorphic Tree (PIT) constraint is violated."""


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

    Example hierarchy with parent_id and relative_path:

        level0 (parent_id = row index, NO relative_path):
        ┌─────┬──────────┬──────────────────────┐
        │ id  │ type     │ internal:parent_id   │
        ├─────┼──────────┼──────────────────────┤
        │ A   │ FOLDER   │ 0                    │
        │ B   │ FOLDER   │ 1                    │
        └─────┴──────────┴──────────────────────┘

        level1 (parent_id links to level0 row, HAS relative_path):
        ┌─────┬──────────┬──────────────────────┬─────────────────────────────────┐
        │ id  │ type     │ internal:parent_id   │ internal:relative_path          │
        ├─────┼──────────┼──────────────────────┼─────────────────────────────────┤
        │ f1  │ FILE     │ 0                    │ A/f1                            │
        │ f2  │ FILE     │ 0                    │ A/f2                            │
        │ f3  │ FILE     │ 1                    │ B/f3                            │
        └─────┴──────────┴──────────────────────┴─────────────────────────────────┘

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

    Attributes:
        levels: List of DataFrames, one per level (0-5)
                Each DataFrame contains ALL samples at that level
                Level 0: Has parent_id, NO relative_path (use 'id' directly)
                Level 1+: Has parent_id AND relative_path for fast queries
                NO offset/size (added by zip_writer if needed)

        local_metadata: Dict mapping folder_path -> DataFrame
                       Contains metadata for direct children of each folder
                       WITHOUT parent_id or relative_path (path implicit, navigation via folder)
                       Example: {"DATA/folder_A/": df_with_4_children}

        collection: Dictionary with COLLECTION.json content
                   Contains all TACO metadata except tortilla

        pit_schema: Position-Isomorphic Tree schema
                   Describes the hierarchical structure with types and IDs
                   Format: {"root": {...}, "hierarchy": {"1": [...], "2": [...]}}

        field_schema: Field schema per level
                     Lists column names and types for each level
                     Format: {"level0": [["id", "string"], ...], "level1": [...]}

        max_depth: Maximum hierarchy depth (0-5)
                  Example: depth=2 means 3 levels (0, 1, 2)

    Example usage:
        >>> generator = MetadataGenerator(taco, quiet=True)
        >>> package = generator.generate_all_levels()
        >>>
        >>> # Access consolidated metadata
        >>> level0_df = package.levels[0]  # All root samples
        >>> level1_df = package.levels[1]  # All level 1 samples
        >>>
        >>> # Access local metadata
        >>> folder_meta = package.local_metadata["DATA/folder_A/"]
        >>>
        >>> # Access schemas
        >>> pit = package.pit_schema
        >>> fields = package.field_schema

        >>> # SQL navigation using relative_path (FAST - no JOINs, level 1+ only)
        >>> import duckdb
        >>> duckdb.sql('''
        ...     SELECT * FROM level1
        ...     WHERE "internal:relative_path" LIKE 'A/%'
        ... ''')
        >>>
        >>> # Level 0: Use 'id' directly (no relative_path column)
        >>> duckdb.sql('''
        ...     SELECT * FROM level0 WHERE id = 'A'
        ... ''')
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
    """
    Generate complete metadata package for TACO containers.

    This generator creates all metadata needed for both ZIP and FOLDER containers:
    - Consolidated metadata (METADATA/levelX) for all levels
    - Local metadata (DATA/folder/__meta__) for each folder
    - Collection metadata (COLLECTION.json)
    - Schema information (PIT + field schemas)

    The generator validates TACO's Position-Isomorphic Tree (PIT) constraints:
    - All samples at same level have same type
    - All folders at same level have same number of children
    - All folders at same level have same child IDs (with padding)

    Usage:
        >>> taco = Taco(...)
        >>> generator = MetadataGenerator(taco, quiet=False)
        >>> package = generator.generate_all_levels()
        >>>
        >>> # Now use package.levels for consolidated metadata
        >>> # Use package.local_metadata for folder-specific metadata
    """

    def __init__(self, taco: "Taco", quiet: bool = False) -> None:
        """
        Initialize metadata generator.

        Args:
            taco: TACO object containing tortilla with samples
            quiet: If True, suppress progress messages
        """
        self.taco = taco
        self.quiet = quiet
        self.max_depth = min(taco.tortilla._current_depth, 5)

    def generate_all_levels(self) -> MetadataPackage:
        """
        Generate complete metadata package for both ZIP and FOLDER containers.

        Process:
        1. Export metadata from tortilla for each level (0 to max_depth)
        2. Clean DataFrames (remove empty columns, container-irrelevant fields)
        3. Add internal:parent_id for relational queries
        4. Validate PIT constraints (homogeneity, structure)
        5. Add internal:relative_path for fast queries (level 1+ only, consolidated metadata)
        6. Generate PIT schema and field schema
        7. Generate local metadata for each folder
        8. Generate collection metadata

        Returns:
            MetadataPackage with all metadata components

        Raises:
            PITValidationError: If TACO homogeneity constraints are violated
        """
        levels = []
        dataframes = []

        # Generate consolidated metadata for each level
        for depth in range(self.max_depth + 1):
            df = self.taco.tortilla.export_metadata(deep=depth)
            df = self._clean_dataframe(df)

            # Add internal:parent_id for level 0 as row index
            # This enables uniform JOIN queries across all levels
            # Level 1+ already have parent_id from tortilla.export_metadata(deep=N)
            if depth == 0:
                df = df.with_columns(
                    pl.arange(0, len(df)).cast(pl.Int64).alias(METADATA_PARENT_ID)
                )
                # Move internal:parent_id to end for consistency
                df = reorder_internal_columns(df)

            dataframes.append(df)

            # Validate PIT constraints
            if depth == 0:
                self._validate_pit_level0(df)
            else:
                self._validate_pit_depth(df, dataframes[depth - 1], depth)

        # Add internal:relative_path for fast SQL queries
        dataframes = self._add_relative_paths(dataframes)

        # Generate schemas
        pit_schema = generate_pit_schema(dataframes, quiet=self.quiet)
        field_schema = generate_field_schema(dataframes)

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
        """
        Add internal:relative_path to level 1+ (consolidated metadata only).

        Level 0 does NOT get relative_path - just use 'id' directly.
        Level 1+ gets full relative paths built from parent paths.

        Path format:
        - Relative to DATA/ directory (no DATA/ prefix)
        - FOLDERs end with "/"
        - FILEs have no trailing slash

        Uses internal:parent_id to lookup parent paths for level 1+.

        Args:
            levels: List of DataFrames, one per level

        Returns:
            List of DataFrames with internal:relative_path added (level 1+ only)

        Example:
            Level 0: NO relative_path column (use id directly)
            Level 1: "Landslide_001/imagery/" (FOLDER), "Landslide_001/label.json" (FILE)
            Level 2: "Landslide_001/imagery/before.tif" (FILE)
        """
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

        This creates the DATA/folder/__meta__ file content containing
        metadata for the direct children of this folder only.

        Args:
            folder_sample: Sample object of type FOLDER

        Returns:
            DataFrame with metadata for direct children
        """
        samples = folder_sample.path.samples
        metadata_dfs = [s.export_metadata() for s in samples]
        df = pl.concat(metadata_dfs, how="vertical")
        return self._clean_dataframe(df)

    def _generate_nested_folders(
        self, parent_sample: "Sample", parent_path: str
    ) -> dict[str, pl.DataFrame]:
        """
        Recursively generate local metadata for nested folders.

        Args:
            parent_sample: Parent folder sample
            parent_path: Path to parent folder (e.g., "DATA/folder_A/")

        Returns:
            Dictionary mapping nested folder paths to their metadata DataFrames

        Example:
            >>> nested = self._generate_nested_folders(folder_A, "DATA/folder_A/")
            >>> nested
            {
                "DATA/folder_A/subfolder_B/": df_subfolder_B,
                "DATA/folder_A/subfolder_B/deepfolder/": df_deepfolder,
            }
        """
        result = {}

        for child in parent_sample.path.samples:
            if child.type == "FOLDER":
                folder_path = f"{parent_path}{child.id}/"
                folder_df = self._generate_folder_metadata(child)
                result[folder_path] = folder_df

                # Recurse into nested folders
                nested = self._generate_nested_folders(child, folder_path)
                result.update(nested)

        return result

    def _validate_pit_level0(self, df: pl.DataFrame) -> None:
        """
        Validate PIT constraint for level 0 (root).

        Rule: All samples at root level must have the same type.
        Either all FILE or all FOLDER (never mixed).

        Args:
            df: DataFrame with level 0 metadata

        Raises:
            PITValidationError: If multiple types found at root
        """
        if "type" not in df.columns:
            raise PITValidationError("Level 0 missing 'type' column")

        types = df["type"].to_list()
        unique_types = list(set(types))

        if len(unique_types) != 1:
            raise PITValidationError(
                f"PIT constraint violated at level 0:\n"
                f"All nodes must have the same type (all FILE or all FOLDER).\n"
                f"Found types: {unique_types}\n"
                f"This violates TACO's homogeneity requirement."
            )

    def _validate_pit_depth(
        self, df: pl.DataFrame, parent_df: pl.DataFrame, depth: int
    ) -> None:
        """
        Validate PIT constraints for hierarchical levels (depth 1+).

        Rules:
        1. Parent level must contain at least one FOLDER
        2. All folders at parent level must have same structure
        3. Each folder position must have consistent pattern

        Args:
            df: DataFrame with current level metadata
            parent_df: DataFrame with parent level metadata
            depth: Current depth (1+)

        Raises:
            PITValidationError: If PIT constraints violated
        """
        if "type" not in df.columns:
            raise PITValidationError(f"Depth {depth} missing 'type' column")

        parent_types = parent_df["type"].to_list()
        parent_pattern = self._infer_unique_pattern(parent_types, depth - 1)
        folder_positions = [i for i, t in enumerate(parent_pattern) if t == "FOLDER"]

        if not folder_positions:
            raise PITValidationError(
                f"Depth {depth} exists but no FOLDERs at depth {depth - 1}.\n"
                f"This is impossible in valid TACO structure."
            )

        num_parents = len(parent_df)
        child_types = df["type"].to_list()

        # Validate each folder position has consistent pattern
        for folder_idx, position in enumerate(folder_positions):
            chunk_pattern = self._extract_chunk_pattern(
                child_types, num_parents, len(folder_positions), folder_idx
            )

            if chunk_pattern is None:
                raise PITValidationError(
                    f"PIT constraint violated at depth {depth}:\n"
                    f"Cannot extract consistent pattern for FOLDER at position {position}.\n"
                    f"All folders must have same structure (use pad_to to ensure homogeneity)."
                )

            # Validate pattern consistency across all parents
            for parent_idx in range(num_parents):
                chunk_start = (parent_idx * len(folder_positions) + folder_idx) * len(
                    chunk_pattern
                )
                chunk_end = chunk_start + len(chunk_pattern)
                actual_chunk = child_types[chunk_start:chunk_end]

                if actual_chunk != chunk_pattern:
                    raise PITValidationError(
                        f"PIT constraint violated at depth {depth}:\n"
                        f"FOLDER at position {position}, parent {parent_idx} has different pattern.\n"
                        f"Expected: {chunk_pattern}\n"
                        f"Actual: {actual_chunk}\n"
                        f"All folders must have identical child structure."
                    )

    def _infer_unique_pattern(self, types: list[str], depth: int) -> list[str]:
        """
        Infer the repeating pattern of types at a level.

        TACO homogeneity means samples repeat in patterns.
        Example: [FILE, FILE, FOLDER, FILE, FILE, FOLDER] has pattern [FILE, FILE, FOLDER]

        Args:
            types: List of types at a level
            depth: Depth level (for error context)

        Returns:
            Shortest repeating pattern

        Example:
            >>> types = ["FILE", "FILE", "FOLDER"] * 3
            >>> self._infer_unique_pattern(types, 0)
            ["FILE", "FILE", "FOLDER"]
        """
        total = len(types)

        # Try to find shortest repeating pattern
        for pattern_len in range(1, total // 2 + 1):
            if total % pattern_len != 0:
                continue

            pattern = types[:pattern_len]
            num_repeats = total // pattern_len

            # Check if pattern repeats exactly
            if all(
                types[i * pattern_len : (i + 1) * pattern_len] == pattern
                for i in range(num_repeats)
            ):
                return pattern

        # No repeating pattern found - entire list is the pattern
        return types

    def _extract_chunk_pattern(
        self,
        types: list[str],
        num_parents: int,
        num_folders_per_parent: int,
        folder_idx: int,
    ) -> list[str] | None:
        """
        Extract the pattern for a specific folder position.

        When parent level has multiple folder positions, each position
        can have different child patterns. This extracts the pattern
        for one specific folder position.

        Args:
            types: All child types at current level
            num_parents: Number of parent samples
            num_folders_per_parent: Number of folders per parent pattern
            folder_idx: Which folder position to extract (0-indexed)

        Returns:
            Pattern for this folder position, or None if inconsistent
        """
        total_types = len(types)
        expected_total = num_parents * num_folders_per_parent

        if total_types % expected_total != 0:
            return None

        chunk_size = total_types // expected_total

        # Extract pattern from first parent's folder at this position
        first_chunk_start = folder_idx * chunk_size
        first_chunk_end = first_chunk_start + chunk_size
        pattern = types[first_chunk_start:first_chunk_end]

        # Validate pattern is consistent across all parents
        for parent_idx in range(num_parents):
            for fld_idx in range(num_folders_per_parent):
                if fld_idx == folder_idx:
                    chunk_start = (
                        parent_idx * num_folders_per_parent + fld_idx
                    ) * chunk_size
                    chunk_end = chunk_start + chunk_size
                    if types[chunk_start:chunk_end] != pattern:
                        return None

        return pattern

    def _clean_dataframe(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Clean DataFrame by removing unnecessary columns.

        Removes:
        - Container-irrelevant columns (path - filesystem specific)
        - Empty columns (all null or empty string)

        Preserves:
        - Core fields (id, type)
        - All internal:* columns (even if empty)
        - Any non-empty user columns

        Args:
            df: Raw metadata DataFrame

        Returns:
            Cleaned DataFrame

        Raises:
            ValueError: If cleaning results in no columns (should never happen)
        """
        # Remove path column (filesystem-specific, not container-relevant)
        container_irrelevant_cols = ["path"]
        df = df.drop([col for col in container_irrelevant_cols if col in df.columns])

        # Remove empty columns (preserves core + internal automatically)
        df = remove_empty_columns(df, preserve_core=True, preserve_internal=True)

        # Safety check (should never happen due to preserve_core)
        if len(df.columns) == 0:
            raise ValueError(
                "DataFrame cleaning resulted in no columns. "
                "This should never happen as core fields are preserved."
            )

        return df


def generate_field_schema(levels: list[pl.DataFrame]) -> dict[str, Any]:
    """
    Generate field schema describing columns at each level.

    Creates a schema that lists all column names and their types
    for each hierarchical level. Used for validation and documentation.

    Args:
        levels: List of DataFrames, one per level

    Returns:
        Dictionary mapping level -> list of [column_name, type] pairs

    Example:
        >>> field_schema = generate_field_schema([level0_df, level1_df])
        >>> field_schema
        {
            "level0": [
                ["id", "string"],
                ["type", "string"],
                ["internal:parent_id", "int64"]
            ],
            "level1": [
                ["id", "string"],
                ["type", "string"],
                ["internal:parent_id", "int64"],
                ["internal:relative_path", "string"],
                ["custom_field", "float64"]
            ]
        }
    """
    field_schema = {}

    for i, level_df in enumerate(levels):
        fields = []
        for col_name, col_type in level_df.schema.items():
            type_name = str(col_type).lower()
            fields.append([col_name, type_name])

        field_schema[f"level{i}"] = fields

    return field_schema


def generate_pit_schema(
    dataframes: list[pl.DataFrame], quiet: bool = True
) -> dict[str, Any]:
    """
    Generate Position-Isomorphic Tree (PIT) schema.

    PIT schema describes the hierarchical structure of the dataset:
    - Root level: type and count
    - Each subsequent level: patterns of types and IDs

    Relies on TACO's HOMOGENEITY guarantee:
    - All folders at same level have SAME number of children
    - All folders at same level have SAME child IDs (with padding)
    - Padding (__TACOPAD__*) ensures structural uniformity

    This allows us to describe the ENTIRE structure by examining
    just ONE representative folder per level (the one with most
    real samples to avoid describing padding-heavy folders).

    Args:
        dataframes: List of DataFrames, one per level (0 to max_depth)
        quiet: If True, suppress debug messages

    Returns:
        Dictionary with PIT schema:
        {
            "root": {"n": count, "type": "FILE"|"FOLDER"},
            "hierarchy": {
                "1": [{"n": total_count, "type": [...], "id": [...]}],
                "2": [{"n": total_count, "type": [...], "id": [...]}, ...],
                ...
            }
        }

    Raises:
        PITValidationError: If DataFrames missing required columns

    Example:
        >>> pit_schema = generate_pit_schema([level0_df, level1_df])
        >>> pit_schema
        {
            "root": {"n": 3, "type": "FOLDER"},
            "hierarchy": {
                "1": [{
                    "n": 12,
                    "type": ["FILE", "FILE", "FILE", "FILE"],
                    "id": ["file_1", "file_2", "file_3", "file_4"]
                }]
            }
        }
    """
    if not dataframes:
        raise PITValidationError("Need at least one DataFrame to generate schema")

    df0 = dataframes[0]
    if "type" not in df0.columns:
        raise PITValidationError("Level 0 missing 'type' column")

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
            raise PITValidationError(
                f"Depth {depth} missing '{METADATA_PARENT_ID}' column"
            )

        if depth == 1:
            # LEVEL 1: Find folder with MOST real (non-padding) samples
            # This is the canonical representation of the structure

            # TACO HOMOGENEITY: All folders have same number of children
            children_per_parent = len(df) // len(parent_df)

            # Target: folder with ALL real samples (no padding)
            # If we find one with children_per_parent real samples, that's perfect!
            target_real_count = children_per_parent

            # Find the folder with maximum real samples
            max_real_count = 0
            best_group_ids = []
            best_group_types = []

            for parent_idx in range(len(parent_df)):
                start_idx = parent_idx * children_per_parent
                end_idx = start_idx + children_per_parent
                group = df[start_idx:end_idx]

                ids = group["id"].to_list()
                types = group["type"].to_list()

                # Count real (non-padding) samples
                real_count = sum(1 for id_val in ids if not is_padding_id(id_val))

                if real_count > max_real_count:
                    max_real_count = real_count
                    best_group_ids = ids
                    best_group_types = types

                # OPTIMIZATION: If we found a folder with ALL real samples, stop!
                # This is the canonical representation - no need to continue
                if real_count == target_real_count:
                    if not quiet:
                        parent_id = parent_df["id"][parent_idx]
                        print(
                            f"  ✓ Found canonical folder '{parent_id}' at depth {depth} "
                            f"with {real_count}/{children_per_parent} real samples (early stop)"
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
                # Get parent_ids for this folder position across all groups
                parent_ids_for_position = [
                    group_idx * pattern_size + position_idx
                    for group_idx in range(num_groups)
                ]

                # Filter children by these parent_ids
                position_children = df.filter(
                    pl.col(METADATA_PARENT_ID).is_in(parent_ids_for_position)
                )

                if len(position_children) == 0:
                    continue

                # TACO HOMOGENEITY: All parents have same number of children
                samples_per_group = len(position_children) // num_groups

                if samples_per_group == 0:
                    continue

                # Target: group with ALL real samples (no padding)
                target_real_count = samples_per_group

                # Find the group with maximum REAL (non-padding) samples
                max_real_count = 0
                best_group_ids = []
                best_group_types = []

                for group_idx in range(num_groups):
                    start_idx = group_idx * samples_per_group
                    end_idx = start_idx + samples_per_group
                    group = position_children[start_idx:end_idx]

                    ids = group["id"].to_list()
                    types = group["type"].to_list()

                    # Count real (non-padding) samples
                    real_count = sum(1 for id_val in ids if not is_padding_id(id_val))

                    if real_count > max_real_count:
                        max_real_count = real_count
                        best_group_ids = ids
                        best_group_types = types

                    # OPTIMIZATION: If we found a group with ALL real samples, stop!
                    # This is the canonical representation - no need to continue
                    if real_count == target_real_count:
                        if not quiet:
                            # Get parent ID for logging
                            parent_row_idx = parent_ids_for_position[group_idx]
                            parent_id = parent_df["id"][parent_row_idx]
                            print(
                                f"  ✓ Found canonical folder '{parent_id}' at depth {depth} position {position_idx} "
                                f"with {real_count}/{samples_per_group} real samples (early stop)"
                            )
                        break

                # Calculate total nodes for this position
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
    """
    Generate COLLECTION.json content from TACO object.

    Extracts all metadata from the TACO object except the tortilla
    (which contains the hierarchical sample structure - not needed
    in COLLECTION.json).

    Args:
        taco: TACO object with metadata

    Returns:
        Dictionary with collection metadata (without tortilla)
    """
    collection = taco.model_dump()
    collection.pop("tortilla", None)
    return collection
