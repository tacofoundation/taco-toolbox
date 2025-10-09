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


def _remove_empty_internal_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Remove optional internal: columns that are completely None.

    Structural columns (internal:offset, internal:size, internal:relative_path)
    are never removed even if None, as they are required by readers.

    Only removes optional content columns like internal:header.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame without empty optional internal: columns
    """
    protected_columns = {
        "internal:offset",
        "internal:size",
        "internal:relative_path",
    }

    cols_to_drop = []

    for col in df.columns:
        if (
            col.startswith("internal:")
            and col not in protected_columns
            and df[col].is_null().all()
        ):
            cols_to_drop.append(col)

    return df.drop(cols_to_drop) if cols_to_drop else df


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

        for _depth in range(self.max_depth + 1):
            df = self.taco.tortilla.export_metadata(deep=_depth)
            df = self._clean_dataframe(df)
            dataframes.append(df)

            if _depth == 0:
                self._validate_pit_level0(df)
            else:
                self._validate_pit_depth(df, dataframes[_depth - 1], _depth)

        collection = generate_collection_json(self.taco)

        # Generate schema BEFORE cleaning (needs internal:temporal_parent_id)
        pit_schema = generate_pit_schema(dataframes)

        # NOW clean the dataframes (remove internal:temporal_parent_id)
        cleaned_dataframes = []
        for df in dataframes:
            if "internal:temporal_parent_id" in df.columns:
                df = df.drop("internal:temporal_parent_id")
            cleaned_dataframes.append(df)

        # Update levels with cleaned dataframes
        for _depth, df in enumerate(cleaned_dataframes):
            levels.append({"dataframe": df})

        return {
            "levels": levels,
            "collection": collection,
            "max_depth": self.max_depth,
            "pit_schema": pit_schema,
        }

    def _validate_pit_level0(self, df: pl.DataFrame) -> None:
        """
        Validate PIT constraint for level 0 (root collection).

        Level 0 should have uniform node types (all FOLDER or all FILE).

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
                f"Found types: {unique_types}\n\n"
                f"TACO requires Position-Isomorphic Tree (PIT) structure.\n"
                f"Level 0 must be homogeneous (all FOLDER or all FILE)."
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
        folder_positions = [i for i, t in enumerate(parent_pattern) if t == "FOLDER"]

        if not folder_positions:
            raise PITValidationError(
                f"Depth {depth} exists but no FOLDERs at depth {depth - 1}"
            )

        num_parents = len(parent_df)
        child_types = [_normalize_type(t) for t in df["type"].to_list()]

        for folder_idx, position in enumerate(folder_positions):
            chunk_pattern = self._extract_chunk_pattern(
                child_types, num_parents, len(folder_positions), folder_idx
            )

            if chunk_pattern is None:
                raise PITValidationError(
                    f"PIT constraint violated at depth {depth}:\n"
                    f"Cannot extract consistent pattern for FOLDER at position {position}"
                )

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
        num_folders_per_parent: int,
        folder_idx: int,
    ) -> list[str] | None:
        """
        Extract pattern for specific FOLDER position.

        Args:
            types: All child types at this depth
            num_parents: Number of parents from previous depth
            num_folders_per_parent: Number of FOLDERs per parent
            folder_idx: Index of FOLDER position we're extracting

        Returns:
            Pattern for this FOLDER position or None if inconsistent
        """
        total_types = len(types)
        expected_total = num_parents * num_folders_per_parent

        if total_types % expected_total != 0:
            return None

        chunk_size = total_types // expected_total

        first_chunk_start = folder_idx * chunk_size
        first_chunk_end = first_chunk_start + chunk_size
        pattern = types[first_chunk_start:first_chunk_end]

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
        Remove completely null or empty columns, and columns not relevant for containers.

        Args:
            df: Input DataFrame

        Returns:
            Cleaned DataFrame
        """
        container_irrelevant_cols = ["path"]
        df = df.drop([col for col in container_irrelevant_cols if col in df.columns])

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
    Generate PIT schema from DataFrames using internal:temporal_parent_id.

    Args:
        dataframes: List of DataFrames with internal:temporal_parent_id

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

        if "internal:temporal_parent_id" not in df.columns:
            raise PITValidationError(
                f"Depth {depth} missing 'internal:temporal_parent_id' column"
            )

        if depth == 1:
            children_per_parent = len(df) // len(parent_df)
            first_parent_children = df.head(children_per_parent)
            child_types = [
                _normalize_type(t) for t in first_parent_children["type"].to_list()
            ]

            pattern: PITPattern = {"n": len(df), "children": child_types}
            hierarchy[str(depth)] = [pattern]

        else:
            parent_schema = hierarchy[str(depth - 1)]
            parent_pattern = parent_schema[0]["children"]
            pattern_size = len(parent_pattern)
            num_groups = len(parent_df) // pattern_size

            folder_positions = [
                i for i, t in enumerate(parent_pattern) if t == "FOLDER"
            ]

            if not folder_positions:
                continue

            all_patterns: list[PITPattern] = []

            for position_idx in folder_positions:
                position_children = df.filter(
                    pl.col("internal:temporal_parent_id") == position_idx
                )

                if len(position_children) == 0:
                    continue

                types = position_children["type"].to_list()
                types_normalized = [_normalize_type(t) for t in types]
                total_nodes = num_groups * len(types_normalized)

                pattern_dict: PITPattern = {
                    "n": total_nodes,
                    "children": types_normalized,
                }
                all_patterns.append(pattern_dict)

            hierarchy[str(depth)] = all_patterns

    return {"root": root, "hierarchy": hierarchy}


def generate_pit_schema_clean(dataframes: list[pl.DataFrame]) -> PITSchema:
    """
    Generate PIT schema and remove internal:temporal_parent_id from dataframes.

    DEPRECATED: Use generate_pit_schema() directly and handle cleanup separately.
    This function is kept for backwards compatibility.

    Args:
        dataframes: List of DataFrames with internal:temporal_parent_id

    Returns:
        PIT schema (dataframes modified in-place to remove parent_id)
    """
    schema = generate_pit_schema(dataframes)

    # Remove internal:temporal_parent_id from dataframes in-place
    for i, df in enumerate(dataframes):
        if "internal:temporal_parent_id" in df.columns:
            dataframes[i] = df.drop("internal:temporal_parent_id")

    return schema


def _normalize_type(type_str: str) -> str:
    """
    Return asset type without normalization.

    Previously this function normalized FILE→SAMPLE and FOLDER→TORTILLA,
    but now returns the original type directly for use in PIT schema.

    Args:
        type_str: Original asset type ("FILE" or "FOLDER")

    Returns:
        Original type: "FILE" or "FOLDER"

    Example:
        >>> _normalize_type("FILE")
        'FILE'
        >>> _normalize_type("FOLDER")
        'FOLDER'
    """
    return type_str


class OffsetEnricher:
    """Enrich metadata with internal ZIP offsets and headers."""

    def __init__(
        self, zip_path: pathlib.Path, arc_files: list[str], quiet: bool = False
    ) -> None:
        self.zip_path = zip_path
        self.arc_files = arc_files
        self.quiet = quiet

    def enrich_metadata(
        self, df: pl.DataFrame, data_lookup: DataLookup
    ) -> pl.DataFrame:
        """
        Add internal:offset, internal:size, internal:header columns using positional matching.

        Args:
            df: Input metadata DataFrame
            data_lookup: Mapping of archive path to (offset, size)

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
        for row_idx, row in enumerate(df.iter_rows(named=True)):
            row_dict = dict(row)

            # Use row position to get corresponding archive file
            if row_idx < len(self.arc_files):
                arc_file = self.arc_files[row_idx]

                # Lookup offset/size using archive path
                if arc_file in data_lookup:
                    offset, size = data_lookup[arc_file]
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

        result_df = pl.DataFrame(rows_data, schema=df.schema)
        return _remove_empty_internal_columns(result_df)

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
        self.path_list = self._build_path_list()

    def _build_path_list(self) -> list[str]:
        """
        Build ordered list of relative paths matching DataFrame row order.

        Returns:
            List of relative paths in same order as samples appear in flattened hierarchy
        """
        paths: list[str] = []
        self._traverse_samples(self.samples, paths, path_prefix="")
        return paths

    def _traverse_samples(
        self, samples: list[Any], paths: list[str], path_prefix: str
    ) -> None:
        """
        Recursively traverse samples to build ordered path list.

        Args:
            samples: List of Sample objects
            paths: List to populate (in order)
            path_prefix: Current path prefix
        """
        for sample in samples:
            if sample.type == "FOLDER":
                new_prefix = (
                    f"{path_prefix}{sample.id}/" if path_prefix else f"{sample.id}/"
                )
                paths.append(f"DATA/{new_prefix}")
                self._traverse_samples(sample.path.samples, paths, new_prefix)
            else:
                file_suffix = sample.path.suffix
                paths.append(f"DATA/{path_prefix}{sample.id}{file_suffix}")

    def enrich_metadata(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add internal:relative_path column using positional matching.

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
        for row_idx, row in enumerate(df.iter_rows(named=True)):
            row_dict = dict(row)

            # Use row position to get corresponding path
            if row_idx < len(self.path_list):
                row_dict["internal:relative_path"] = self.path_list[row_idx]

            rows_data.append(row_dict)

        result_df = pl.DataFrame(rows_data, schema=df.schema)
        return _remove_empty_internal_columns(result_df)
