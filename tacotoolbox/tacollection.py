"""
TacoCollection - Merge multiple TACO datasets into global collection metadata.

This module provides functionality to consolidate COLLECTION.json metadata
from multiple TACO datasets. It validates schema consistency and sums sample
counts across datasets to create a unified global collection.

Key features:
- Read COLLECTION.json from multiple TACO files
- Validate schema consistency (pit_schema and field_schema)
- Sum 'n' values across datasets
- Generate global COLLECTION.json with consolidated metadata

Main function:
    create_tacollection(): Merge multiple TACO datasets into single collection

Example:
    >>> from tacotoolbox import create_tacollection
    >>> 
    >>> # Merge all TACO files in directory
    >>> tacozips = list(Path("datasets/").glob("*.tacozip"))
    >>> create_tacollection(tacozips, output_path="global_collection.json")
"""

import json
from pathlib import Path
from typing import Any

import tacozip

# ============================================================================
# EXCEPTIONS
# ============================================================================


class CollectionError(Exception):
    """Base exception for collection operations."""


class SchemaValidationError(CollectionError):
    """Raised when schema validation fails."""


# ============================================================================
# MAIN API
# ============================================================================


def create_tacollection(
    tacozips: list[str | Path],
    output_path: str | Path,
    validate_schema: bool = True,
) -> None:
    """
    Create global COLLECTION.json from multiple TACO datasets.

    Validates that all datasets have:
    - Identical taco:pit_schema structure (types)
    - Identical taco:field_schema (columns and types)

    Sums 'n' values across all datasets.
    Uses first dataset as base for other metadata.

    Args:
        tacozips: List of TACO file paths
        output_path: Output path for COLLECTION.json
        validate_schema: Validate schema consistency (default: True)

    Raises:
        CollectionError: If no datasets provided or read fails
        SchemaValidationError: If validation fails

    Example:
        >>> from tacotoolbox import create_tacollection
        >>>
        >>> create_tacollection(
        ...     tacozips=list(Path("data/").glob("*.tacozip")),
        ...     output_path="COLLECTION.json"
        ... )
    """
    if not tacozips:
        raise CollectionError("No datasets provided")

    if len(tacozips) < 2:
        raise CollectionError(
            f"Need at least 2 datasets to create collection, got {len(tacozips)}"
        )

    # Validate output path
    output_path = Path(output_path)
    if output_path.exists():
        raise CollectionError(f"Output file already exists: {output_path}")

    if not output_path.parent.exists():
        raise CollectionError(f"Parent directory does not exist: {output_path.parent}")

    # Read all collections
    collections = _read_collections(tacozips)

    if not collections:
        raise CollectionError("No valid collections could be read")

    # Validate schemas
    if validate_schema:
        _validate_pit_structure(collections)
        _validate_field_schema(collections)

    # Sum pit_schema n values
    global_pit = _sum_pit_schemas(collections)

    # Create global collection
    global_collection = collections[0].copy()
    global_collection["taco:pit_schema"] = global_pit
    global_collection["taco:sources"] = {
        "count": len(tacozips),
        "ids": [c.get("id", "unknown") for c in collections],
        "files": [Path(p).name for p in tacozips],
    }

    # Write to file
    with open(output_path, "w") as f:
        json.dump(global_collection, f, indent=4)


# ============================================================================
# INTERNAL FUNCTIONS
# ============================================================================


def _read_collections(tacozips: list[str | Path]) -> list[dict[str, Any]]:
    """
    Read COLLECTION.json from all tacozips.

    COLLECTION.json is always the last entry in TACO_HEADER.

    Args:
        tacozips: List of TACO file paths

    Returns:
        List of collection dictionaries

    Raises:
        CollectionError: If reading fails
    """
    collections = []

    for tacozip_path in tacozips:
        path = Path(tacozip_path)

        if not path.exists():
            raise CollectionError(f"File not found: {path}")

        if not path.is_file():
            raise CollectionError(f"Path is not a file: {path}")

        if path.stat().st_size == 0:
            raise CollectionError(f"File is empty: {path}")

        try:
            # Read TACO_HEADER
            entries = tacozip.read_header(str(path))

            if len(entries) == 0:
                raise CollectionError(f"Empty TACO_HEADER in {path}")

            # COLLECTION.json is always the LAST entry
            collection_offset, collection_size = entries[-1]

            if collection_size == 0:
                raise CollectionError(f"Empty COLLECTION.json in {path}")

            # Read COLLECTION.json directly
            with open(path, "rb") as f:
                f.seek(collection_offset)
                collection_bytes = f.read(collection_size)
                collection = json.loads(collection_bytes)

                # Validate required fields
                if "taco:pit_schema" not in collection:
                    raise CollectionError(f"Missing 'taco:pit_schema' in {path}")

                if "taco:field_schema" not in collection:
                    raise CollectionError(f"Missing 'taco:field_schema' in {path}")

                collections.append(collection)

        except json.JSONDecodeError as e:
            raise CollectionError(f"Invalid JSON in {path}: {e}")
        except CollectionError:
            raise
        except Exception as e:
            raise CollectionError(f"Failed to read collection from {path}: {e}")

    return collections


def _validate_pit_structure(collections: list[dict[str, Any]]) -> None:
    """
    Validate that all collections have identical taco:pit_schema structure.

    Checks:
    - Root type must be identical
    - Hierarchy depth must be identical
    - Type arrays must be identical per level
    - ID arrays must be identical per level

    Args:
        collections: List of collection dictionaries

    Raises:
        SchemaValidationError: If structure differs
    """
    if not collections:
        return

    reference_pit = collections[0].get("taco:pit_schema")
    if not reference_pit:
        raise SchemaValidationError("First collection missing taco:pit_schema")

    reference_root_type = reference_pit.get("root", {}).get("type")
    reference_hierarchy = reference_pit.get("hierarchy", {})

    for idx, collection in enumerate(collections[1:], start=1):
        current_pit = collection.get("taco:pit_schema")
        if not current_pit:
            raise SchemaValidationError(f"Collection {idx} missing taco:pit_schema")

        # Validate root type
        current_root_type = current_pit.get("root", {}).get("type")
        if current_root_type != reference_root_type:
            raise SchemaValidationError(
                f"Collection {idx} has different root type:\n"
                f"  Expected: {reference_root_type}\n"
                f"  Got: {current_root_type}"
            )

        # Validate hierarchy structure
        current_hierarchy = current_pit.get("hierarchy", {})

        if set(current_hierarchy.keys()) != set(reference_hierarchy.keys()):
            raise SchemaValidationError(
                f"Collection {idx} has different hierarchy levels:\n"
                f"  Expected: {sorted(reference_hierarchy.keys())}\n"
                f"  Got: {sorted(current_hierarchy.keys())}"
            )

        # Validate type/id arrays per level
        for level, ref_patterns in reference_hierarchy.items():
            curr_patterns = current_hierarchy[level]

            if len(ref_patterns) != len(curr_patterns):
                raise SchemaValidationError(
                    f"Collection {idx} level {level} has different pattern count:\n"
                    f"  Expected: {len(ref_patterns)}\n"
                    f"  Got: {len(curr_patterns)}"
                )

            for pattern_idx, (ref_pattern, curr_pattern) in enumerate(
                zip(ref_patterns, curr_patterns, strict=False)
            ):
                # Check type arrays
                ref_types = ref_pattern.get("type", [])
                curr_types = curr_pattern.get("type", [])

                if ref_types != curr_types:
                    raise SchemaValidationError(
                        f"Collection {idx} level {level} pattern {pattern_idx} "
                        f"has different types:\n"
                        f"  Expected: {ref_types}\n"
                        f"  Got: {curr_types}"
                    )

                # Check id arrays
                ref_ids = ref_pattern.get("id", [])
                curr_ids = curr_pattern.get("id", [])

                if ref_ids != curr_ids:
                    raise SchemaValidationError(
                        f"Collection {idx} level {level} pattern {pattern_idx} "
                        f"has different ids:\n"
                        f"  Expected: {ref_ids}\n"
                        f"  Got: {curr_ids}"
                    )


def _validate_field_schema(collections: list[dict[str, Any]]) -> None:
    """
    Validate that all collections have identical taco:field_schema.

    Args:
        collections: List of collection dictionaries

    Raises:
        SchemaValidationError: If schemas differ
    """
    if not collections:
        return

    reference_fields = collections[0].get("taco:field_schema")
    if not reference_fields:
        raise SchemaValidationError("First collection missing taco:field_schema")

    for idx, collection in enumerate(collections[1:], start=1):
        current_fields = collection.get("taco:field_schema")
        if not current_fields:
            raise SchemaValidationError(f"Collection {idx} missing taco:field_schema")

        # Check same levels
        if set(current_fields.keys()) != set(reference_fields.keys()):
            raise SchemaValidationError(
                f"Collection {idx} has different field schema levels:\n"
                f"  Expected: {sorted(reference_fields.keys())}\n"
                f"  Got: {sorted(current_fields.keys())}"
            )

        # Check fields per level
        for level, ref_schema in reference_fields.items():
            curr_schema = current_fields[level]

            if ref_schema != curr_schema:
                raise SchemaValidationError(
                    f"Collection {idx} has different field schema at {level}:\n"
                    f"  Expected: {ref_schema}\n"
                    f"  Got: {curr_schema}"
                )


def _sum_pit_schemas(collections: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Sum 'n' values across all taco:pit_schemas.

    Args:
        collections: List of collection dictionaries

    Returns:
        Merged pit_schema with summed 'n' values

    Raises:
        CollectionError: If collections list is empty or schemas are invalid
    """
    if not collections:
        raise CollectionError("Cannot sum schemas from empty collections list")

    # Validate first collection has pit_schema
    if "taco:pit_schema" not in collections[0]:
        raise CollectionError("First collection missing taco:pit_schema")

    # Start with first schema as base (deep copy)
    result = collections[0]["taco:pit_schema"].copy()

    if "root" not in result:
        raise CollectionError("First collection missing 'root' in pit_schema")

    # Sum root n
    root_n_sum = sum(
        c.get("taco:pit_schema", {}).get("root", {}).get("n", 0) for c in collections
    )

    if root_n_sum == 0:
        raise CollectionError("Sum of root 'n' values is zero across all collections")

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
