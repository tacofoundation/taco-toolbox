"""
Metadata generation for TACO containers.

This module generates the dual metadata system used by both ZIP and FOLDER containers:

1. CONSOLIDATED METADATA (METADATA/levelX files):
   - One file per hierarchy level (level0.parquet, level1.parquet, etc.)
   - Contains ALL samples at that level across the entire dataset
   - Includes internal:current_id for O(1) position lookups (all levels)
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
    reorder_internal_columns,
)
from tacotoolbox._constants import (
    CORE_FIELD_DESCRIPTIONS,
    INTERNAL_FIELD_DESCRIPTIONS,
    METADATA_CURRENT_ID,
    METADATA_PARENT_ID,
    METADATA_RELATIVE_PATH,
    PADDING_PREFIX,
)
from tacotoolbox._logging import get_logger
from tacotoolbox.tortilla.datamodel import Tortilla

if TYPE_CHECKING:
    from tacotoolbox.sample.datamodel import Sample
    from tacotoolbox.taco.datamodel import Taco

logger = get_logger(__name__)


class MetadataPackage:
    """
    Complete metadata bundle for TACO containers.

    This package contains everything needed to create both ZIP and FOLDER containers:
    - Consolidated metadata for all levels (METADATA/)
    - Local metadata for individual folders (DATA/folder/)
    - Collection metadata (COLLECTION.json)
    - Schema information (PIT + field schemas)

    The internal:current_id column stores each sample's position (0, 1, 2...) at its level.
    The internal:parent_id column enables hierarchical navigation via SQL JOINs.
    The internal:relative_path column enables fast queries without JOINs.

    SQL examples:
        # Simple JOIN using current_id and parent_id
        SELECT l1.id, l0.id as parent_folder
        FROM level0 l0
        JOIN level1 l1 ON l1."internal:parent_id" = l0."internal:current_id"
        WHERE l0.id = 'A'

        # Level 1+: Use relative_path (fast, no JOIN needed)
        SELECT * FROM level1
        WHERE "internal:relative_path" LIKE 'A/%'
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

    def __init__(self, taco: "Taco") -> None:
        """Initialize metadata generator."""
        self.taco = taco
        self.max_depth = min(taco.tortilla._current_depth, 5)

    def generate_all_levels(self) -> MetadataPackage:
        """Generate complete metadata package for both ZIP and FOLDER containers."""
        levels = []
        tables = []

        # Generate consolidated metadata for each level
        for depth in range(self.max_depth + 1):
            table = self.taco.tortilla.export_metadata(deep=depth)
            table = self._clean_table(table)

            # Add internal:current_id if not already present (deep>0 already has it)
            if METADATA_CURRENT_ID not in table.schema.names:
                current_id_array = pa.array(range(table.num_rows), type=pa.int64())
                current_id_field = pa.field(METADATA_CURRENT_ID, pa.int64())
                table = table.append_column(current_id_field, current_id_array)

            # Add internal:parent_id for level 0 (uses current_id values)
            if depth == 0:
                # For level0, parent_id equals current_id (self-referential for consistency)
                parent_id_array = pa.array(range(table.num_rows), type=pa.int64())
                parent_id_field = pa.field(METADATA_PARENT_ID, pa.int64())
                table = table.append_column(parent_id_field, parent_id_array)

            table = reorder_internal_columns(table)
            tables.append(table)

        # Add internal:relative_path for fast SQL queries
        tables = self._add_relative_paths(tables)

        # Generate schemas
        pit_schema = generate_pit_schema(tables)
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

        # Align schemas before concatenating (required for heterogeneous extensions)
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

        # Do not remove empty columns when using group_by.
        # When creating multiple ZIPs via grouping, some groups may have all-None
        # for certain columns while others have values. Removing empty columns
        # breaks schema consistency across groups, causing TacoCat consolidation
        # to fail with schema mismatch errors.
        # table = remove_empty_columns(table, preserve_core=True, preserve_internal=True)

        if table.num_columns == 0:
            raise ValueError(
                "Table cleaning resulted in no columns. "
                "This should never happen as core fields are preserved."
            )

        return table


# PIT schema generation
#
# PIT (Position-Invariant Tree) schema describes the hierarchical structure
# of a TACO container. It enables deterministic navigation without reading
# all metadata files.
#
# The algorithm finds "canonical" folders - those with the most real (non-padding)
# samples - to represent each level's structure. Padding samples (__TACOPAD__*)
# are excluded from the canonical pattern.
#
# Uses vectorized PyArrow operations instead of Python loops. Key optimizations:
# - pc.starts_with for padding detection.
# - pc.sum on boolean masks for counting.
# - Single to_pylist() call at the end, not per-iteration.


def _compute_real_mask(ids_column: pa.ChunkedArray) -> pa.ChunkedArray:
    """
    Compute boolean mask: True for real samples, False for padding.

    Uses pc.starts_with which executes in C++, avoiding Python iteration.
    PADDING_PREFIX is "__TACOPAD__" defined in _constants.py.
    """
    is_padding = pc.starts_with(ids_column, PADDING_PREFIX)
    return pc.invert(is_padding)


def _count_real_in_slice(is_real_mask: pa.ChunkedArray, start: int, length: int) -> int:
    """
    Count real samples in a slice of the boolean mask.

    Uses pc.sum on the slice - True counts as 1, False as 0.
    Avoids converting to Python list for counting.
    """
    return pc.sum(is_real_mask.slice(start, length)).as_py()


def _find_best_group(
    table: pa.Table,
    is_real_mask: pa.ChunkedArray,
    group_size: int,
    num_groups: int,
) -> tuple[list[str], list[str], int]:
    """
    Find the group with the most real (non-padding) samples.

    Iterates through groups tracking max real count. Early-exits if a
    "perfect" group (all real, no padding) is found.

    Critical: Only converts to Python (to_pylist) once at the end for the
    best group, not during iteration. This is the key performance optimization.

    Returns:
        Tuple of (best_ids, best_types, best_group_idx)
    """
    max_real = 0
    best_idx = 0

    for idx in range(num_groups):
        real_count = _count_real_in_slice(is_real_mask, idx * group_size, group_size)

        if real_count > max_real:
            max_real = real_count
            best_idx = idx

        # Early exit: perfect group found (all samples are real, no padding)
        if real_count == group_size:
            break

    # Single conversion to Python at the end
    best_group = table.slice(best_idx * group_size, group_size)
    return (
        best_group.column("id").to_pylist(),
        best_group.column("type").to_pylist(),
        best_idx,
    )


def _process_level1(
    table: pa.Table,
    parent_table: pa.Table,
) -> dict[str, Any]:
    """
    Process level 1: direct children of root folders.

    Level 1 is simpler than level 2+ because all children share the same
    parent structure. Each parent folder has exactly children_per_parent
    children due to TACO's homogeneity guarantee.
    """
    children_per_parent = table.num_rows // parent_table.num_rows

    # Compute mask once for entire level (vectorized)
    is_real_mask = _compute_real_mask(table.column("id"))

    best_ids, best_types, best_idx = _find_best_group(
        table, is_real_mask, children_per_parent, parent_table.num_rows
    )

    parent_id = parent_table.column("id")[best_idx].as_py()
    real_count = _count_real_in_slice(
        is_real_mask, best_idx * children_per_parent, children_per_parent
    )
    logger.debug(
        f"Level 1 canonical folder '{parent_id}' "
        f"with {real_count}/{children_per_parent} samples"
    )

    return {"n": table.num_rows, "type": best_types, "id": best_ids}


def _process_level_n(
    table: pa.Table,
    parent_table: pa.Table,
    parent_pattern: list[str],
    depth: int,
) -> list[dict[str, Any]]:
    """
    Process level 2+: children of nested folders.

    Level 2+ is more complex because folders can appear at multiple positions
    within the parent pattern. For example, if parent_pattern is
    ["FILE", "FOLDER", "FILE", "FOLDER"], there are FOLDERs at positions 1 and 3.

    Each folder position may have different children structures, so we generate
    a separate pattern for each position.

    Uses pc.is_in for vectorized filtering instead of Python loops.
    """
    pattern_size = len(parent_pattern)
    num_groups = parent_table.num_rows // pattern_size

    # Find positions in parent pattern that are FOLDERs
    folder_positions = [i for i, t in enumerate(parent_pattern) if t == "FOLDER"]
    if not folder_positions:
        return []

    parent_id_column = table.column(METADATA_PARENT_ID)
    all_patterns = []

    for pos_idx in folder_positions:
        # Calculate parent_ids for this folder position across all groups
        # E.g., if pattern_size=4 and pos_idx=1: parent_ids = [1, 5, 9, 13, ...]
        parent_ids = pa.array([g * pattern_size + pos_idx for g in range(num_groups)])

        # Vectorized filter using pc.is_in (much faster than Python loop)
        mask = pc.is_in(parent_id_column, parent_ids)
        children = table.filter(mask)

        if children.num_rows == 0:
            continue

        samples_per_group = children.num_rows // num_groups
        if samples_per_group == 0:
            continue

        # Compute mask for filtered children
        is_real_mask = _compute_real_mask(children.column("id"))

        best_ids, best_types, best_idx = _find_best_group(
            children, is_real_mask, samples_per_group, num_groups
        )

        logger.debug(
            f"Level {depth} position {pos_idx}: "
            f"canonical at group {best_idx} with {samples_per_group} samples"
        )

        all_patterns.append(
            {
                "n": num_groups * len(best_types),
                "type": best_types,
                "id": best_ids,
            }
        )

    return all_patterns


def generate_pit_schema(
    tables: list[pa.Table],
) -> dict[str, Any]:
    """
    Generate PIT schema from metadata tables.

    The PIT schema describes the hierarchical structure of a TACO container,
    enabling deterministic navigation without reading all metadata.

    Uses vectorized PyArrow operations for performance on large datasets.

    Args:
        tables: List of PyArrow tables, one per hierarchy level (level0, level1, ...)

    Returns:
        PIT schema dict with structure:
        {
            "root": {"n": 100, "type": "FOLDER"},
            "hierarchy": {
                "1": [{"n": 500, "type": ["FILE", "FOLDER"], "id": ["a", "b"]}],
                "2": [{"n": 200, "type": ["FILE"], "id": ["x"]}]
            }
        }

    Raises:
        ValueError: If tables is empty or missing required columns
    """
    if not tables:
        raise ValueError("Need at least one Table to generate schema")

    table0 = tables[0]
    if "type" not in table0.schema.names:
        raise ValueError("Level 0 missing 'type' column")

    # Root level: just count and type
    root = {"n": table0.num_rows, "type": table0.column("type")[0].as_py()}
    hierarchy: dict[str, list[dict[str, Any]]] = {}

    for depth in range(1, len(tables)):
        table = tables[depth]
        parent_table = tables[depth - 1]

        if table.num_rows == 0:
            continue

        if METADATA_PARENT_ID not in table.schema.names:
            raise ValueError(f"Depth {depth} missing '{METADATA_PARENT_ID}'")

        if depth == 1:
            # Level 1: direct children of root
            hierarchy["1"] = [_process_level1(table, parent_table)]
        else:
            # Level 2+: nested folders, may have multiple folder positions
            parent_pattern = hierarchy[str(depth - 1)][0]["type"]
            patterns = _process_level_n(table, parent_table, parent_pattern, depth)
            if patterns:
                hierarchy[str(depth)] = patterns

    return {"root": root, "hierarchy": hierarchy}


# Field schema & collection
def generate_field_schema(levels: list[pa.Table], taco: "Taco") -> dict[str, Any]:
    """
    Generate field schema with descriptions from extensions.

    Collects field descriptions from all Sample and Tortilla extensions,
    then generates arrays of [name, type, description] for each field.
    If no description exists for a field, uses empty string.
    """
    # Collect all field descriptions (later sources override earlier)
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


def generate_collection_json(taco: "Taco") -> dict[str, Any]:
    """Generate COLLECTION.json content from TACO object."""
    collection = taco.model_dump()
    collection.pop("tortilla", None)
    return collection
