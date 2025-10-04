import pathlib
from typing import TYPE_CHECKING, Any

import polars as pl

from tacotoolbox._types import (
    DataLookup,
    LevelMetadata,
    MetadataPackage,
    PITPattern,
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

            if depth == 0:
                self._validate_pit_level0(df)
            else:
                self._validate_pit_depth(df, dataframes[depth - 1], depth)

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
            df: DataFrame for level 0

        Raises:
            PITValidationError: If multiple node types at level 0
        """
        if "type" not in df.columns:
            raise PITValidationError("Level 0 missing 'type' column")

        normalized_types = [_normalize_type(t) for t in df["type"].to_list()]
        unique_types = list(set(normalized_types))

        if len(unique_types) != 1:
            raise PITValidationError(
                f"PIT constraint violated at level 0:\n"
                f"All nodes must have the same type.\n"
                f"Found normalized types: {unique_types}\n\n"
                f"TACO requires Position-Isomorphic Tree (PIT) structure.\n"
                f"Level 0 must be homogeneous (all TORTILLA or all SAMPLE)."
            )

    def _validate_pit_depth(
        self, df: pl.DataFrame, parent_df: pl.DataFrame, depth: int
    ) -> None:
        """
        Validate PIT constraint for depth >= 1.

        Args:
            df: DataFrame for current depth
            parent_df: DataFrame for parent depth
            depth: Current hierarchy depth

        Raises:
            PITValidationError: If PIT constraint violated
        """
        if "type" not in df.columns:
            raise PITValidationError(f"Depth {depth} missing 'type' column")

        parent_types = [_normalize_type(t) for t in parent_df["type"].to_list()]
        parent_pattern = self._infer_unique_pattern(parent_types, depth - 1)
        tortilla_positions = [
            i for i, t in enumerate(parent_pattern) if t == "TORTILLA"
        ]

        if not tortilla_positions:
            raise PITValidationError(
                f"Depth {depth} exists but no TORTILLAs at depth {depth - 1}"
            )

        num_parents = len(parent_df)
        child_types = [_normalize_type(t) for t in df["type"].to_list()]

        for tortilla_idx, position in enumerate(tortilla_positions):
            chunk_pattern = self._extract_chunk_pattern(
                child_types, num_parents, len(tortilla_positions), tortilla_idx
            )

            if chunk_pattern is None:
                raise PITValidationError(
                    f"PIT constraint violated at depth {depth}:\n"
                    f"Cannot extract consistent pattern for TORTILLA at position {position}"
                )

            for parent_idx in range(num_parents):
                chunk_start = (
                    parent_idx * len(tortilla_positions) + tortilla_idx
                ) * len(chunk_pattern)
                chunk_end = chunk_start + len(chunk_pattern)
                actual_chunk = child_types[chunk_start:chunk_end]

                if actual_chunk != chunk_pattern:
                    raise PITValidationError(
                        f"PIT constraint violated at depth {depth}:\n"
                        f"TORTILLA at position {position}, parent {parent_idx} has different pattern.\n"
                        f"Expected: {chunk_pattern}\n"
                        f"Actual: {actual_chunk}"
                    )

    def _infer_unique_pattern(self, types: list[str], depth: int) -> list[str]:
        """
        Infer unique pattern from types list.

        Args:
            types: List of node types
            depth: Current depth for error reporting

        Returns:
            Inferred pattern

        Raises:
            PITValidationError: If no unique pattern found
        """
        total = len(types)

        for pattern_len in range(1, total // 2 + 1):
            if total % pattern_len != 0:
                continue

            pattern = types[:pattern_len]
            num_repeats = total // pattern_len

            if all(
                types[i * pattern_len : (i + 1) * pattern_len] == pattern
                for i in range(num_repeats)
            ):
                return pattern

        return types

    def _extract_chunk_pattern(
        self,
        types: list[str],
        num_parents: int,
        num_tortillas_per_parent: int,
        tortilla_idx: int,
    ) -> list[str] | None:
        """
        Extract pattern for specific TORTILLA position.

        Args:
            types: All child types at this depth
            num_parents: Number of parents from previous depth
            num_tortillas_per_parent: Number of TORTILLAs per parent
            tortilla_idx: Index of TORTILLA position we're extracting

        Returns:
            Pattern for this TORTILLA position or None if inconsistent
        """
        total_types = len(types)
        expected_total = num_parents * num_tortillas_per_parent

        if total_types % expected_total != 0:
            return None

        chunk_size = total_types // expected_total

        first_chunk_start = tortilla_idx * chunk_size
        first_chunk_end = first_chunk_start + chunk_size
        pattern = types[first_chunk_start:first_chunk_end]

        for parent_idx in range(num_parents):
            for tort_idx in range(num_tortillas_per_parent):
                if tort_idx == tortilla_idx:
                    chunk_start = (
                        parent_idx * num_tortillas_per_parent + tort_idx
                    ) * chunk_size
                    chunk_end = chunk_start + chunk_size
                    if types[chunk_start:chunk_end] != pattern:
                        return None

        return pattern

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
            if df[col].is_null().all():
                continue

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
        dataframes: List of DataFrames (one per depth)

    Returns:
        PIT schema for deterministic navigation

    Raises:
        PITValidationError: If schema cannot be generated
    """
    if not dataframes:
        raise PITValidationError("Need at least one DataFrame to generate schema")

    df0 = dataframes[0]
    if "type" not in df0.columns:
        raise PITValidationError("Level 0 missing 'type' column")

    root_type = _normalize_type(df0["type"][0])
    root: PITRootLevel = {"n": len(df0), "type": root_type}

    hierarchy: dict[str, list[PITPattern]] = {}

    for depth in range(1, len(dataframes)):
        df = dataframes[depth]
        parent_df = dataframes[depth - 1]

        if len(df) == 0:
            continue

        parent_types = [_normalize_type(t) for t in parent_df["type"].to_list()]
        parent_pattern = _infer_pattern(parent_types)
        tortilla_positions = [
            i for i, t in enumerate(parent_pattern) if t == "TORTILLA"
        ]

        if not tortilla_positions:
            continue

        patterns: list[PITPattern] = []
        num_parents = len(parent_df)
        child_types = [_normalize_type(t) for t in df["type"].to_list()]

        for tortilla_idx in range(len(tortilla_positions)):
            chunk_pattern = _extract_tortilla_pattern(
                child_types, num_parents, len(tortilla_positions), tortilla_idx
            )
            chunk_total = num_parents * len(chunk_pattern)
            patterns.append({"n": chunk_total, "children": chunk_pattern})

        hierarchy[str(depth)] = patterns

    return {"root": root, "hierarchy": hierarchy}


def _normalize_type(type_str: str) -> str:
    """
    Normalize asset type to PIT conceptual type.

    Args:
        type_str: Original asset type

    Returns:
        Normalized type: "SAMPLE" or "TORTILLA"
    """
    return "SAMPLE" if type_str != "TORTILLA" else "TORTILLA"


def _infer_pattern(types: list[str]) -> list[str]:
    """Infer repeating pattern from types list."""
    total = len(types)

    for pattern_len in range(1, total // 2 + 1):
        if total % pattern_len != 0:
            continue

        pattern = types[:pattern_len]
        num_repeats = total // pattern_len

        if all(
            types[i * pattern_len : (i + 1) * pattern_len] == pattern
            for i in range(num_repeats)
        ):
            return pattern

    return types


def _extract_tortilla_pattern(
    types: list[str],
    num_parents: int,
    num_tortillas_per_parent: int,
    tortilla_idx: int,
) -> list[str]:
    """Extract pattern for specific TORTILLA position."""
    total_types = len(types)
    expected_total = num_parents * num_tortillas_per_parent
    chunk_size = total_types // expected_total

    first_chunk_start = tortilla_idx * chunk_size
    first_chunk_end = first_chunk_start + chunk_size
    return types[first_chunk_start:first_chunk_end]


class OffsetEnricher:
    """Enrich metadata with internal ZIP offsets and headers."""

    def __init__(self, zip_path: pathlib.Path, quiet: bool = False) -> None:
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
        df = df.with_columns(
            [
                pl.lit(None, dtype=pl.Int64).alias("internal:offset"),
                pl.lit(None, dtype=pl.Int64).alias("internal:size"),
                pl.lit(None, dtype=pl.Binary).alias("internal:header"),
            ]
        )

        rows_data = []
        for row in df.iter_rows(named=True):
            row_dict = dict(row)
            sample_id = row_dict["id"]

            if sample_id in data_lookup:
                offset, size = data_lookup[sample_id]
                row_dict["internal:offset"] = offset
                row_dict["internal:size"] = size

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
    collection.pop("tortilla", None)
    return collection


class RelativePathEnricher:
    """Enrich metadata with internal:relative_path for folder containers."""

    def __init__(self, samples: list[Any], quiet: bool = False) -> None:
        self.samples = samples
        self.quiet = quiet
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
                new_prefix = (
                    f"{path_prefix}{sample.id}/" if path_prefix else f"{sample.id}/"
                )
                mapping[sample.id] = f"DATA/{new_prefix}"
                self._traverse_samples(sample.path.samples, mapping, new_prefix)
            else:
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
        df = df.with_columns(
            [
                pl.lit(None, dtype=pl.Utf8).alias("internal:relative_path"),
            ]
        )

        rows_data = []
        for row in df.iter_rows(named=True):
            row_dict = dict(row)
            sample_id = row_dict["id"]

            if sample_id in self.path_mapping:
                row_dict["internal:relative_path"] = self.path_mapping[sample_id]

            rows_data.append(row_dict)

        return pl.DataFrame(rows_data, schema=df.schema)
