"""
TacoCollection - Merge multiple TACO datasets into global collection metadata.

Internal module for consolidating COLLECTION.json metadata from multiple TACO
datasets. Can be used standalone or called by tacocat to generate metadata.

Key features:
- Read COLLECTION.json from multiple TACO files
- Validate schema consistency (pit_schema and field_schema)
- Sum 'n' values across datasets
- Merge spatial/temporal extents into global coverage
- Store individual partition extents for query routing

Main function:
    create_tacollection(): Merge multiple TACO datasets into single collection
"""

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tacozip

from tacotoolbox._exceptions import TacoConsolidationError, TacoSchemaError
from tacotoolbox._validation import validate_common_directory


def create_tacollection(
    inputs: Sequence[str | Path],
    output: str | Path | None = None,
    validate_schema: bool = True,
) -> None:
    """
    Create global collection metadata from multiple TACO datasets.

    Generates a consolidated metadata file.

    Validates that all datasets have:
    - Identical taco:pit_schema structure (types)
    - Identical taco:field_schema (columns and types)

    Sums 'n' values across all datasets.
    Stores individual partition extents for query routing.
    Uses first dataset as base for other metadata.

    Args:
        inputs: Sequence of .tacozip file paths to consolidate
        output: Output file path (e.g., 'TACOLLECTION.json' or '.tacocat/COLLECTION.json')
                If None, creates COLLECTION.json in same dir as inputs
        validate_schema: If True, validate schemas match across datasets

    Raises:
        TacoValidationError: If inputs are invalid or in different directories
        TacoConsolidationError: If consolidation fails
        TacoSchemaError: If schema validation fails
    """
    if not inputs:
        raise TacoConsolidationError("No datasets provided")

    # Auto-detect output path if not specified
    if output is None:
        common_dir = validate_common_directory(inputs)
        output_path = common_dir / "COLLECTION.json"
    else:
        output_path = Path(output)

    # Validate output path
    if output_path.exists():
        raise TacoConsolidationError(f"Output file already exists: {output_path}")

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read all collections
    collections = _read_collections(inputs)

    if not collections:
        raise TacoConsolidationError("No valid collections could be read")

    # Validate schemas
    if validate_schema:
        _validate_pit_structure(collections)
        _validate_field_schema(collections)

    # Sum pit_schema n values
    global_pit = _sum_pit_schemas(collections)

    # Merge extents from all partitions into global coverage
    global_spatial = _merge_spatial_extents(collections)
    global_temporal = _merge_temporal_extents(collections)

    # Collect individual partition extents
    partition_extents = _collect_partition_extents(collections, inputs)

    # Create global collection
    global_collection = collections[0].copy()
    global_collection["taco:pit_schema"] = global_pit

    # Add global extent (merged from all partitions)
    global_collection["extent"] = {
        "spatial": global_spatial,
        "temporal": global_temporal,
    }

    global_collection["taco:sources"] = {
        "count": len(inputs),
        "ids": [c.get("id", "unknown") for c in collections],
        "files": [Path(p).name for p in inputs],
        "extents": partition_extents,
    }

    # Write to file
    with open(output_path, "w") as f:
        json.dump(global_collection, f, indent=4)


def _read_collections(tacozips: Sequence[str | Path]) -> list[dict[str, Any]]:
    """
    Read COLLECTION.json from all tacozips.

    COLLECTION.json is always the last entry in TACO_HEADER.
    """
    collections = []

    for tacozip_path in tacozips:
        collection = _read_single_collection(tacozip_path)
        collections.append(collection)

    return collections


def _read_single_collection(tacozip_path: str | Path) -> dict[str, Any]:
    """Read COLLECTION.json from a single tacozip file."""
    path = Path(tacozip_path)

    # Validate file
    _validate_tacozip_file(path)

    # Read header and collection
    try:
        entries = tacozip.read_header(str(path))

        if len(entries) == 0:
            _raise_empty_header_error(path)

        # COLLECTION.json is always the LAST entry
        collection_offset, collection_size = entries[-1]

        if collection_size == 0:
            _raise_empty_collection_error(path)

        # Read COLLECTION.json directly
        with open(path, "rb") as f:
            f.seek(collection_offset)
            collection_bytes = f.read(collection_size)
            collection = json.loads(collection_bytes)

            # Validate required fields
            _validate_collection_fields(collection, path)

            return collection

    except json.JSONDecodeError as e:
        raise TacoConsolidationError(f"Invalid JSON in {path}: {e}") from e
    except TacoConsolidationError:
        raise
    except Exception as e:
        raise TacoConsolidationError(
            f"Failed to read collection from {path}: {e}"
        ) from e


def _raise_empty_header_error(path: Path) -> None:
    """Helper to raise empty TACO_HEADER error."""
    raise TacoConsolidationError(f"Empty TACO_HEADER in {path}")


def _raise_empty_collection_error(path: Path) -> None:
    """Helper to raise empty COLLECTION.json error."""
    raise TacoConsolidationError(f"Empty COLLECTION.json in {path}")


def _validate_tacozip_file(path: Path) -> None:
    """Validate that the tacozip file exists and is readable."""
    if not path.exists():
        raise TacoConsolidationError(f"File not found: {path}")

    if not path.is_file():
        raise TacoConsolidationError(f"Path is not a file: {path}")

    if path.stat().st_size == 0:
        raise TacoConsolidationError(f"File is empty: {path}")


def _validate_collection_fields(collection: dict[str, Any], path: Path) -> None:
    """Validate that collection has required fields."""
    if "taco:pit_schema" not in collection:
        raise TacoConsolidationError(f"Missing 'taco:pit_schema' in {path}")

    if "taco:field_schema" not in collection:
        raise TacoConsolidationError(f"Missing 'taco:field_schema' in {path}")


def _validate_pit_structure(collections: list[dict[str, Any]]) -> None:
    """
    Validate that all collections have identical taco:pit_schema structure.

    Checks:
    - Root type must be identical
    - Hierarchy depth must be identical
    - Type arrays must be identical per level
    - ID arrays must be identical per level
    """
    if not collections:
        return

    reference_pit = collections[0].get("taco:pit_schema")
    if not reference_pit:
        raise TacoSchemaError("First collection missing taco:pit_schema")

    reference_root_type = reference_pit.get("root", {}).get("type")
    reference_hierarchy = reference_pit.get("hierarchy", {})

    for idx, collection in enumerate(collections[1:], start=1):
        _validate_single_pit_structure(
            collection, idx, reference_root_type, reference_hierarchy
        )


def _validate_single_pit_structure(
    collection: dict[str, Any],
    idx: int,
    reference_root_type: Any,
    reference_hierarchy: dict[str, Any],
) -> None:
    """Validate PIT structure for a single collection against reference."""
    current_pit = collection.get("taco:pit_schema")
    if not current_pit:
        raise TacoSchemaError(f"Collection {idx} missing taco:pit_schema")

    # Validate root type
    current_root_type = current_pit.get("root", {}).get("type")
    if current_root_type != reference_root_type:
        raise TacoSchemaError(
            f"Collection {idx} has different root type:\n"
            f"  Expected: {reference_root_type}\n"
            f"  Got: {current_root_type}"
        )

    # Validate hierarchy structure
    current_hierarchy = current_pit.get("hierarchy", {})

    if set(current_hierarchy.keys()) != set(reference_hierarchy.keys()):
        raise TacoSchemaError(
            f"Collection {idx} has different hierarchy levels:\n"
            f"  Expected: {sorted(reference_hierarchy.keys())}\n"
            f"  Got: {sorted(current_hierarchy.keys())}"
        )

    # Validate type/id arrays per level
    _validate_hierarchy_patterns(idx, reference_hierarchy, current_hierarchy)


def _validate_hierarchy_patterns(
    idx: int,
    reference_hierarchy: dict[str, Any],
    current_hierarchy: dict[str, Any],
) -> None:
    """Validate hierarchy patterns for a single collection."""
    for level, ref_patterns in reference_hierarchy.items():
        curr_patterns = current_hierarchy[level]

        if len(ref_patterns) != len(curr_patterns):
            raise TacoSchemaError(
                f"Collection {idx} level {level} has different pattern count:\n"
                f"  Expected: {len(ref_patterns)}\n"
                f"  Got: {len(curr_patterns)}"
            )

        for pattern_idx, (ref_pattern, curr_pattern) in enumerate(
            zip(ref_patterns, curr_patterns, strict=False)
        ):
            _validate_single_pattern(idx, level, pattern_idx, ref_pattern, curr_pattern)


def _validate_single_pattern(
    idx: int,
    level: str,
    pattern_idx: int,
    ref_pattern: dict[str, Any],
    curr_pattern: dict[str, Any],
) -> None:
    """Validate a single pattern against reference."""
    # Check type arrays
    ref_types = ref_pattern.get("type", [])
    curr_types = curr_pattern.get("type", [])

    if ref_types != curr_types:
        raise TacoSchemaError(
            f"Collection {idx} level {level} pattern {pattern_idx} "
            f"has different types:\n"
            f"  Expected: {ref_types}\n"
            f"  Got: {curr_types}"
        )

    # Check id arrays
    ref_ids = ref_pattern.get("id", [])
    curr_ids = curr_pattern.get("id", [])

    if ref_ids != curr_ids:
        raise TacoSchemaError(
            f"Collection {idx} level {level} pattern {pattern_idx} "
            f"has different ids:\n"
            f"  Expected: {ref_ids}\n"
            f"  Got: {curr_ids}"
        )


def _validate_field_schema(collections: list[dict[str, Any]]) -> None:
    """Validate that all collections have identical taco:field_schema."""
    if not collections:
        return

    reference_fields = collections[0].get("taco:field_schema")
    if not reference_fields:
        raise TacoSchemaError("First collection missing taco:field_schema")

    for idx, collection in enumerate(collections[1:], start=1):
        current_fields = collection.get("taco:field_schema")
        if not current_fields:
            raise TacoSchemaError(f"Collection {idx} missing taco:field_schema")

        # Check same levels
        if set(current_fields.keys()) != set(reference_fields.keys()):
            raise TacoSchemaError(
                f"Collection {idx} has different field schema levels:\n"
                f"  Expected: {sorted(reference_fields.keys())}\n"
                f"  Got: {sorted(current_fields.keys())}"
            )

        # Check fields per level
        for level, ref_schema in reference_fields.items():
            curr_schema = current_fields[level]

            if ref_schema != curr_schema:
                raise TacoSchemaError(
                    f"Collection {idx} has different field schema at {level}:\n"
                    f"  Expected: {ref_schema}\n"
                    f"  Got: {curr_schema}"
                )


def _sum_pit_schemas(collections: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum 'n' values across all taco:pit_schemas."""
    if not collections:
        raise TacoConsolidationError("Cannot sum schemas from empty collections list")

    # Validate first collection has pit_schema
    if "taco:pit_schema" not in collections[0]:
        raise TacoConsolidationError("First collection missing taco:pit_schema")

    # Start with first schema as base (deep copy)
    result = collections[0]["taco:pit_schema"].copy()

    if "root" not in result:
        raise TacoConsolidationError("First collection missing 'root' in pit_schema")

    # Sum root n
    root_n_sum = sum(
        c.get("taco:pit_schema", {}).get("root", {}).get("n", 0) for c in collections
    )

    if root_n_sum == 0:
        raise TacoConsolidationError(
            "Sum of root 'n' values is zero across all collections"
        )

    result["root"]["n"] = root_n_sum

    # Sum hierarchy n values
    if "hierarchy" in result:
        for level, patterns in result["hierarchy"].items():
            for pattern_idx in range(len(patterns)):
                n_sum = sum(
                    c.get("taco:pit_schema", {})
                    .get("hierarchy", {})
                    .get(level, [{}])[pattern_idx]
                    .get("n", 0)
                    for c in collections
                )
                result["hierarchy"][level][pattern_idx]["n"] = n_sum

    return result


def _merge_spatial_extents(collections: list[dict[str, Any]]) -> list[float]:
    """
    Merge spatial extents from all collections into global bounding box.

    Computes the union of all spatial extents to create a global bbox that
    covers all partitions. If no extents found, defaults to global coverage.

    Examples:
        Partition 1: [-10, 30, 0, 40]  (Europe)
        Partition 2: [100, -10, 110, 0]  (Indonesia)
        Result: [-10, -10, 110, 40]  (covers both regions)
    """
    all_extents = []

    for collection in collections:
        extent = collection.get("extent")
        if extent and "spatial" in extent and extent["spatial"]:
            all_extents.append(extent["spatial"])

    if not all_extents:
        # Default to global extent if none found
        return [-180.0, -90.0, 180.0, 90.0]

    # Extract min/max from all bboxes: [min_lon, min_lat, max_lon, max_lat]
    min_lons = [e[0] for e in all_extents]
    min_lats = [e[1] for e in all_extents]
    max_lons = [e[2] for e in all_extents]
    max_lats = [e[3] for e in all_extents]

    return [min(min_lons), min(min_lats), max(max_lons), max(max_lats)]


def _merge_temporal_extents(collections: list[dict[str, Any]]) -> list[str] | None:
    """
    Merge temporal extents from all collections into global time range.

    Finds the earliest start datetime and latest end datetime across all
    partitions to create a global temporal coverage.

    Examples:
        Partition 1: ["2023-01-01T00:00:00Z", "2023-06-30T23:59:59Z"]
        Partition 2: ["2023-07-01T00:00:00Z", "2023-12-31T23:59:59Z"]
        Result: ["2023-01-01T00:00:00Z", "2023-12-31T23:59:59Z"]
    """
    all_temporals = []

    for collection in collections:
        extent = collection.get("extent")
        if extent and "temporal" in extent and extent["temporal"]:
            all_temporals.append(extent["temporal"])

    if not all_temporals:
        return None

    # Parse all datetime strings to find global range
    starts = []
    ends = []

    for temporal in all_temporals:
        start_str, end_str = temporal

        # Parse ISO 8601 strings, handle 'Z' suffix
        if start_str.endswith("Z"):
            start_dt = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
        else:
            start_dt = datetime.fromisoformat(start_str)

        if end_str.endswith("Z"):
            end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
        else:
            end_dt = datetime.fromisoformat(end_str)

        # Ensure UTC timezone for comparison
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc)
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=timezone.utc)

        starts.append(start_dt)
        ends.append(end_dt)

    # Get global temporal range
    earliest = min(starts)
    latest = max(ends)

    # Convert back to ISO 8601 strings with 'Z' suffix
    return [
        earliest.isoformat().replace("+00:00", "Z"),
        latest.isoformat().replace("+00:00", "Z"),
    ]


def _collect_partition_extents(
    collections: list[dict[str, Any]], tacozips: Sequence[str | Path]
) -> list[dict[str, Any]]:
    """
    Collect individual spatial/temporal extents from each partition.

    Stores each partition's coverage for query routing and visualization.

    Examples:
        [
            {
                "file": "europe.taco",
                "id": "europe",
                "spatial": [-10, 30, 0, 40],
                "temporal": ["2020-01-01Z", "2022-12-31Z"]
            },
            {
                "file": "asia.taco",
                "id": "asia",
                "spatial": [100, -10, 110, 0],
                "temporal": ["2021-01-01Z", "2023-12-31Z"]
            }
        ]
    """
    partition_extents = []

    for collection, tacozip_path in zip(collections, tacozips, strict=True):
        extent = collection.get("extent", {})

        partition_info = {
            "file": Path(tacozip_path).name,
            "id": collection.get("id", "unknown"),
        }

        # Add spatial if exists
        if extent.get("spatial"):
            partition_info["spatial"] = extent["spatial"]

        # Add temporal if exists
        if extent.get("temporal"):
            partition_info["temporal"] = extent["temporal"]

        partition_extents.append(partition_info)

    return partition_extents
