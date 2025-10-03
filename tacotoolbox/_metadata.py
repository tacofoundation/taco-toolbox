import pathlib
from typing import TYPE_CHECKING, Any

import polars as pl

from tacotoolbox._types import (
    DataLookup,
    LevelMetadata,
    MetadataPackage,
    PITHierarchyLevel,
    PITRootLevel,
    PITSchema,
)

if TYPE_CHECKING:
    from tacotoolbox.taco.datamodel import Taco


class PITValidationError(Exception):
    """Raised when Position-Isomorphic Tree constraint is violated."""


class MetadataGenerator:
    """Generate consolidated metadata tables for all hierarchy levels."""

    def __init__(self, taco: "Taco", quiet: bool = False) -> None:
        """
        Initialize metadata generator.

        Args:
            taco: TACO object with tortilla samples
            quiet: Suppress warning messages
        """
        self.taco = taco
        self.quiet = quiet
        self.max_depth = taco.tortilla._current_depth

    def generate_all_levels(self) -> MetadataPackage:
        """
        Generate metadata for all hierarchy levels.

        Validates PIT constraint and generates PIT schema for deterministic navigation.

        Returns:
            MetadataPackage with levels, collection, max_depth, and pit_schema

        Raises:
            PITValidationError: If PIT constraint violated at any level
        """
        levels: list[LevelMetadata] = []
        dataframes: list[pl.DataFrame] = []

        for depth in range(self.max_depth + 1):
            df = self.taco.tortilla.export_metadata(deep=depth)
            df = self._clean_dataframe(df)
            dataframes.append(df)

            # Validate PIT for all levels
            if depth == 0:
                self._validate_pit_level0(df)
            else:
                # For level 1+, validate without parent_id (assumes ordered data)
                self._validate_pit_ordered(df, depth)

            levels.append({"dataframe": df})

        collection = generate_collection_json(self.taco)
        pit_schema = generate_pit_schema(dataframes)

        return {
            "levels": levels,
            "collection": collection,
            "max_depth": self.max_depth,
            "pit_schema": pit_schema,
        }

    def _validate_pit_level0(self, df: pl.DataFrame) -> None:
        """
        Validate PIT constraint for level 0 (root collection).

        Level 0 should have uniform node types (all TORTILLA or all SAMPLE).

        Args:
            df: Level 0 DataFrame

        Raises:
            PITValidationError: If multiple node types at level 0
        """
        if "type" not in df.columns:
            raise PITValidationError("Level 0 missing 'type' column")

        unique_types = df["type"].unique().to_list()

        if len(unique_types) != 1:
            raise PITValidationError(
                f"PIT constraint violated at level 0:\n"
                f"All nodes must have the same type.\n"
                f"Found types: {unique_types}\n\n"
                f"TACO requires Position-Isomorphic Tree (PIT) structure.\n"
                f"Level 0 must be homogeneous (all TORTILLA or all SAMPLE)."
            )

    def _validate_pit_ordered(self, df: pl.DataFrame, level: int) -> None:
        """
        Validate PIT constraint for levels 1+ assuming ordered data.

        Without parent_id column, we assume rows are ordered by parent and
        validate that the pattern repeats uniformly.

        Args:
            df: DataFrame for current level
            level: Current hierarchy level

        Raises:
            PITValidationError: If pattern doesn't repeat uniformly
        """
        if "type" not in df.columns:
            raise PITValidationError(f"Level {level} missing 'type' column")

        total_rows = len(df)
        if total_rows == 0:
            raise PITValidationError(f"Level {level} has no rows")

        # Infer pattern length by finding first repeat of type sequence
        types = df["type"].to_list()
        pattern_length = self._infer_pattern_length(types)

        if pattern_length is None:
            raise PITValidationError(
                f"PIT constraint violated at level {level}:\n"
                f"Cannot infer uniform pattern from types.\n"
                f"Data may not be properly ordered by parent."
            )

        # Validate total rows is divisible by pattern length
        if total_rows % pattern_length != 0:
            raise PITValidationError(
                f"PIT constraint violated at level {level}:\n"
                f"Total rows ({total_rows}) not divisible by pattern length ({pattern_length}).\n"
                f"Expected uniform repetition of pattern."
            )

        # Validate that pattern repeats uniformly
        expected_pattern = types[:pattern_length]
        num_parents = total_rows // pattern_length

        for parent_idx in range(num_parents):
            start = parent_idx * pattern_length
            end = start + pattern_length
            actual_pattern = types[start:end]

            if actual_pattern != expected_pattern:
                raise PITValidationError(
                    f"PIT constraint violated at level {level}:\n"
                    f"Parent {parent_idx} has different pattern.\n"
                    f"Expected: {expected_pattern}\n"
                    f"Actual: {actual_pattern}\n\n"
                    f"TACO requires Position-Isomorphic Tree (PIT) structure.\n"
                    f"All parents must have identical child patterns."
                )

    def _infer_pattern_length(self, types: list[str]) -> int | None:
        """
        Infer pattern length from list of types.

        Tries pattern lengths from 1 to len(types)//2 and checks if
        the pattern repeats uniformly.

        Args:
            types: List of node types

        Returns:
            Pattern length if found, None otherwise
        """
        total = len(types)

        for pattern_len in range(1, total // 2 + 1):
            if total % pattern_len != 0:
                continue

            # Check if this pattern repeats
            pattern = types[:pattern_len]
            num_repeats = total // pattern_len
            is_valid = True

            for i in range(num_repeats):
                start = i * pattern_len
                end = start + pattern_len
                if types[start:end] != pattern:
                    is_valid = False
                    break

            if is_valid:
                return pattern_len

        # If no pattern found, assume entire list is the pattern (single parent)
        return total

    def _clean_dataframe(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Remove completely null or empty columns.

        Args:
            df: Input DataFrame

        Returns:
            Cleaned DataFrame
        """
        cols_to_keep = []

        for col in df.columns:
            # Skip if all nulls
            if df[col].is_null().all():
                continue

            # For string columns, check for empty strings
            if df[col].dtype == pl.Utf8:
                non_empty = df.filter(
                    (pl.col(col).is_not_null())
                    & (pl.col(col) != "")
                    & (pl.col(col) != "None")
                ).height
                if non_empty > 0:
                    cols_to_keep.append(col)
            else:
                cols_to_keep.append(col)

        return df.select(cols_to_keep) if cols_to_keep else df.select([df.columns[0]])


def generate_pit_schema(dataframes: list[pl.DataFrame]) -> PITSchema:
    """
    Generate PIT schema from DataFrames.

    Args:
        dataframes: List of DataFrames (one per level)

    Returns:
        PIT schema for deterministic navigation

    Raises:
        PITValidationError: If schema cannot be generated
    """
    if not dataframes:
        raise PITValidationError("Need at least one DataFrame to generate schema")

    # Level 0 (root collection)
    df0 = dataframes[0]
    if "type" not in df0.columns:
        raise PITValidationError("Level 0 missing 'type' column")

    root_type = df0["type"][0]  # Already validated to be uniform
    root: PITRootLevel = {"n": len(df0), "type": root_type}

    # Levels 1+ (hierarchy)
    hierarchy: list[PITHierarchyLevel] = []

    for depth in range(1, len(dataframes)):
        df = dataframes[depth]
        total_rows = len(df)

        if total_rows == 0:
            continue

        # Get pattern from first repetition
        types = df["type"].to_list()
        pattern_length = _infer_pattern_length_for_schema(types)
        pattern = types[:pattern_length]

        hierarchy_level: PITHierarchyLevel = {
            "depth": depth,
            "n": total_rows,
            "children": pattern,
        }
        hierarchy.append(hierarchy_level)

    return {"root": root, "hierarchy": hierarchy}


def _infer_pattern_length_for_schema(types: list[str]) -> int:
    """
    Infer pattern length for schema generation.

    Args:
        types: List of node types

    Returns:
        Pattern length
    """
    total = len(types)

    for pattern_len in range(1, total // 2 + 1):
        if total % pattern_len != 0:
            continue

        pattern = types[:pattern_len]
        num_repeats = total // pattern_len
        is_valid = True

        for i in range(num_repeats):
            start = i * pattern_len
            end = start + pattern_len
            if types[start:end] != pattern:
                is_valid = False
                break

        if is_valid:
            return pattern_len

    # Default: entire list is the pattern
    return total


class OffsetEnricher:
    """Enrich metadata with internal ZIP offsets and headers."""

    def __init__(self, zip_path: pathlib.Path, quiet: bool = False) -> None:
        """
        Initialize offset enricher.

        Args:
            zip_path: Path to ZIP file with DATA/
            quiet: Suppress warning messages
        """
        self.zip_path = zip_path
        self.quiet = quiet

    def enrich_metadata(
        self, df: pl.DataFrame, data_lookup: DataLookup
    ) -> pl.DataFrame:
        """
        Add internal:offset, internal:size, internal:header columns.

        Args:
            df: Input metadata DataFrame
            data_lookup: Mapping of sample_id to (offset, size)

        Returns:
            Enriched DataFrame with internal columns
        """
        # Add empty internal columns
        df = df.with_columns(
            [
                pl.lit(None, dtype=pl.Int64).alias("internal:offset"),
                pl.lit(None, dtype=pl.Int64).alias("internal:size"),
                pl.lit(None, dtype=pl.Binary).alias("internal:header"),
            ]
        )

        # Process each row
        rows_data = []
        for row in df.iter_rows(named=True):
            row_dict = dict(row)
            sample_id = row_dict["id"]

            # Lookup offset/size in DATA/
            if sample_id in data_lookup:
                offset, size = data_lookup[sample_id]
                row_dict["internal:offset"] = offset
                row_dict["internal:size"] = size

            # Extract TACOTIFF header if applicable
            if (
                row_dict["type"] == "TACOTIFF"
                and "path" in row_dict
                and row_dict["path"]
            ):
                row_dict["internal:header"] = self._get_tacotiff_header(
                    row_dict["path"]
                )

            rows_data.append(row_dict)

        return pl.DataFrame(rows_data, schema=df.schema)

    def _get_tacotiff_header(self, path: str) -> bytes | None:
        """
        Extract TACOTIFF header as binary data.

        Args:
            path: Path to TACOTIFF file

        Returns:
            Header bytes or None if extraction fails
        """
        try:
            import tacotiff

            header_data = tacotiff.metadata_from_tiff(path)
            if header_data is None:
                return None

            return (
                header_data
                if isinstance(header_data, bytes)
                else header_data.encode("utf-8")
            )

        except Exception as e:
            if not self.quiet:
                print(f"Warning: Could not extract TACOTIFF header for {path}: {e}")
            return None


def generate_collection_json(taco: "Taco") -> dict[str, Any]:
    """
    Generate COLLECTION.json dictionary from TACO object.

    Args:
        taco: TACO object with metadata

    Returns:
        Dictionary suitable for JSON serialization
    """
    collection = taco.model_dump()
    collection.pop("tortilla", None)  # Remove tortilla, only keep metadata
    return collection


class RelativePathEnricher:
    """Enrich metadata with internal:relative_path for folder containers."""

    def __init__(self, samples: list[Any], quiet: bool = False) -> None:
        """
        Initialize relative path enricher.

        Args:
            samples: List of Sample objects from tortilla
            quiet: Suppress warning messages
        """
        self.samples = samples
        self.quiet = quiet
        # Build mapping: sample_id -> relative_path
        self.path_mapping = self._build_path_mapping()

    def _build_path_mapping(self) -> dict[str, str]:
        """
        Build mapping of sample_id to relative path.

        Returns:
            Dictionary mapping sample_id to relative path from container root
        """
        mapping: dict[str, str] = {}
        self._traverse_samples(self.samples, mapping, path_prefix="")
        return mapping

    def _traverse_samples(
        self, samples: list[Any], mapping: dict[str, str], path_prefix: str
    ) -> None:
        """
        Recursively traverse samples to build path mapping.

        Args:
            samples: List of Sample objects
            mapping: Dictionary to populate
            path_prefix: Current path prefix
        """
        for sample in samples:
            if sample.type == "TORTILLA":
                # TORTILLA gets directory path
                new_prefix = (
                    f"{path_prefix}{sample.id}/" if path_prefix else f"{sample.id}/"
                )
                mapping[sample.id] = f"DATA/{new_prefix}"

                # Recurse into children
                self._traverse_samples(sample.path.samples, mapping, new_prefix)
            else:
                # File gets full path with extension
                file_suffix = sample.path.suffix
                mapping[sample.id] = f"DATA/{path_prefix}{sample.id}{file_suffix}"

    def enrich_metadata(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add internal:relative_path column.

        Args:
            df: Input metadata DataFrame

        Returns:
            Enriched DataFrame with internal:relative_path column
        """
        # Add empty relative_path column
        df = df.with_columns(
            [
                pl.lit(None, dtype=pl.Utf8).alias("internal:relative_path"),
            ]
        )

        # Process each row
        rows_data = []
        for row in df.iter_rows(named=True):
            row_dict = dict(row)
            sample_id = row_dict["id"]

            # Lookup relative path
            if sample_id in self.path_mapping:
                row_dict["internal:relative_path"] = self.path_mapping[sample_id]

            rows_data.append(row_dict)

        return pl.DataFrame(rows_data, schema=df.schema)
