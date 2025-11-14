"""
Column manipulation utilities for tacotoolbox.

Functions:
    - reorder_internal_columns: Place internal:* columns at end of DataFrame
    - remove_empty_columns: Remove columns with all null/empty values
    - validate_schema_consistency: Check schema compatibility across DataFrames
    - read_metadata_file: Read Parquet metadata files
    - write_parquet_file: Write Parquet for local __meta__ files
    - write_parquet_file_with_cdc: Write Parquet with CDC for consolidated metadata
    - cast_dataframe_to_schema: Cast columns to match schema spec
"""

from pathlib import Path

import polars as pl

from tacotoolbox._constants import (
    PARQUET_CDC_DEFAULT_CONFIG,
    METADATA_COLUMNS_ORDER,
    SHARED_CORE_FIELDS,
    is_internal_column,
)


def reorder_internal_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Reorder DataFrame columns to place internal:* columns at the end.

    Column order:
        1. Regular columns (non-internal)
        2. internal:parent_id (if present)
        3. internal:offset (if present)
        4. internal:size (if present)
        5. Other internal:* columns (if any)

    Args:
        df: DataFrame with mixed column order

    Returns:
        DataFrame with internal:* columns at the end

    Example:
        >>> df = pl.DataFrame({
        ...     "id": ["a", "b"],
        ...     "internal:offset": [100, 200],
        ...     "type": ["FILE", "FILE"],
        ...     "internal:parent_id": [0, 0],
        ... })
        >>> df = reorder_internal_columns(df)
        >>> df.columns
        ['id', 'type', 'internal:parent_id', 'internal:offset']
    """
    regular_cols = [col for col in df.columns if not is_internal_column(col)]
    ordered_internal = [col for col in METADATA_COLUMNS_ORDER if col in df.columns]
    other_internal = [
        col
        for col in df.columns
        if is_internal_column(col) and col not in METADATA_COLUMNS_ORDER
    ]

    new_order = regular_cols + ordered_internal + other_internal
    return df.select(new_order)


def remove_empty_columns(
    df: pl.DataFrame, preserve_core: bool = True, preserve_internal: bool = True
) -> pl.DataFrame:
    """
    Remove columns that are completely empty (all null or empty strings).

    Args:
        df: DataFrame to clean
        preserve_core: If True, never remove core fields (id, type, path)
        preserve_internal: If True, never remove internal:* columns

    Returns:
        DataFrame with empty columns removed

    Example:
        >>> df = pl.DataFrame({
        ...     "id": ["a", "b"],
        ...     "empty_col": [None, None],
        ...     "internal:offset": [100, 200],
        ...     "empty_string": ["", ""],
        ... })
        >>> df = remove_empty_columns(df)
        >>> df.columns
        ['id', 'internal:offset']
    """
    cols_to_keep = []

    for col in df.columns:
        if preserve_core and col in SHARED_CORE_FIELDS:
            cols_to_keep.append(col)
            continue

        if preserve_internal and is_internal_column(col):
            cols_to_keep.append(col)
            continue

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

    if not cols_to_keep:
        cols_to_keep = [df.columns[0]]

    return df.select(cols_to_keep)


def validate_schema_consistency(
    dataframes: list[pl.DataFrame], context: str = "DataFrame list"
) -> None:
    """
    Validate that all DataFrames have consistent schemas.

    Ensures that DataFrames can be safely concatenated vertically by
    checking that they all have the same columns with the same types.

    Args:
        dataframes: List of DataFrames to validate
        context: Description of where these DataFrames come from

    Raises:
        ValueError: If schemas are inconsistent

    Example:
        >>> df1 = pl.DataFrame({"id": ["a"], "type": ["FILE"]})
        >>> df2 = pl.DataFrame({"id": ["b"], "type": ["FILE"]})
        >>> validate_schema_consistency([df1, df2])  # OK
        >>>
        >>> df3 = pl.DataFrame({"id": ["c"], "name": ["x"]})
        >>> validate_schema_consistency([df1, df3])  # Raises ValueError
    """
    if not dataframes:
        raise ValueError(f"{context}: Cannot validate empty DataFrame list")

    if len(dataframes) == 1:
        return

    reference_schema = dict(dataframes[0].schema)
    reference_columns = set(reference_schema.keys())

    for i, df in enumerate(dataframes[1:], start=1):
        current_schema = dict(df.schema)
        current_columns = set(current_schema.keys())

        missing_columns = reference_columns - current_columns
        extra_columns = current_columns - reference_columns

        if missing_columns or extra_columns:
            error_msg = f"{context}: Schema inconsistency at index {i}:\n"

            if missing_columns:
                error_msg += f"  Missing columns: {sorted(missing_columns)}\n"

            if extra_columns:
                error_msg += f"  Extra columns: {sorted(extra_columns)}\n"

            error_msg += f"  Expected schema: {sorted(reference_columns)}\n"
            error_msg += f"  Actual schema: {sorted(current_columns)}"

            raise ValueError(error_msg)

        type_mismatches = []
        for col in reference_columns:
            ref_type = reference_schema[col]
            curr_type = current_schema[col]

            if ref_type != curr_type:
                type_mismatches.append(
                    f"  Column '{col}': expected {ref_type}, got {curr_type}"
                )

        if type_mismatches:
            error_msg = f"{context}: Type mismatches at index {i}:\n"
            error_msg += "\n".join(type_mismatches)
            raise ValueError(error_msg)


def ensure_columns_exist(
    df: pl.DataFrame, required_columns: list[str], context: str = "DataFrame"
) -> None:
    """
    Validate that DataFrame contains all required columns.

    Args:
        df: DataFrame to check
        required_columns: List of column names that must be present
        context: Description of the DataFrame

    Raises:
        ValueError: If any required columns are missing
    """
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"{context}: Missing required columns: {missing}\n"
            f"Available columns: {df.columns}"
        )


def filter_columns_by_prefix(
    df: pl.DataFrame, prefix: str, exclude: bool = False
) -> pl.DataFrame:
    """
    Select or exclude columns based on name prefix.

    Args:
        df: DataFrame to filter
        prefix: Column name prefix to match
        exclude: If True, exclude matching columns. If False, keep only matching.

    Returns:
        DataFrame with filtered columns
    """
    if exclude:
        cols = [col for col in df.columns if not col.startswith(prefix)]
    else:
        cols = [col for col in df.columns if col.startswith(prefix)]

    if not cols:
        return df.select([])

    return df.select(cols)


def add_columns_if_missing(
    df: pl.DataFrame, columns: dict[str, pl.DataType], default_value: any = None
) -> pl.DataFrame:
    """
    Add columns to DataFrame if they don't exist, with default values.

    Args:
        df: DataFrame to modify
        columns: Dictionary mapping column_name -> polars DataType
        default_value: Value to fill new columns with (default: None)

    Returns:
        DataFrame with all specified columns present
    """
    for col_name, col_type in columns.items():
        if col_name not in df.columns:
            df = df.with_columns(pl.lit(default_value).cast(col_type).alias(col_name))

    return df


def get_column_statistics(df: pl.DataFrame) -> dict[str, dict]:
    """
    Get basic statistics about DataFrame columns.

    Args:
        df: DataFrame to analyze

    Returns:
        Dictionary mapping column_name -> statistics dict
    """
    stats = {}

    for col in df.columns:
        col_stats = {
            "dtype": str(df[col].dtype),
            "null_count": df[col].is_null().sum(),
            "null_percentage": (df[col].is_null().sum() / len(df)) * 100,
            "is_empty": df[col].is_null().all(),
        }

        if df[col].dtype in [pl.Int64, pl.Int32, pl.Float64, pl.Float32]:
            if not df[col].is_null().all():
                col_stats["min"] = df[col].min()
                col_stats["max"] = df[col].max()
                col_stats["mean"] = df[col].mean()

        stats[col] = col_stats

    return stats


def read_metadata_file(file_path: Path | str) -> pl.DataFrame:
    """
    Read metadata file in Parquet format.

    Args:
        file_path: Path to metadata file (.parquet)

    Returns:
        DataFrame with proper column names

    Raises:
        ValueError: If file format is not .parquet

    Example:
        >>> # Reading Parquet
        >>> df = read_metadata_file("METADATA/level0.parquet")
        >>> "internal:parent_id" in df.columns
        True
    """
    file_path = Path(file_path)

    if file_path.suffix == ".parquet":
        return pl.read_parquet(file_path)
    else:
        raise ValueError(
            f"Unsupported metadata format: {file_path.suffix}\n"
            f"Expected .parquet, got {file_path}"
        )


def write_parquet_file(df: pl.DataFrame, output_path: Path | str) -> None:
    """
    Write DataFrame to Parquet file (for local __meta__ files).

    Parquet natively supports colons in column names, no sanitization needed.
    Used for local metadata (__meta__) in FOLDER containers.

    Args:
        df: DataFrame to write
        output_path: Target path for .parquet file

    Example:
        >>> df = pl.DataFrame({
        ...     "id": ["a", "b"],
        ...     "internal:parent_id": [0, 1],
        ...     "stac:crs": ["EPSG:4326", "EPSG:32633"]
        ... })
        >>> write_parquet_file(df, "metadata.parquet")
    """
    df.write_parquet(output_path, compression="zstd")


def write_parquet_file_with_cdc(df: pl.DataFrame, output_path: Path | str) -> None:
    """
    Write DataFrame to Parquet with CDC (for consolidated metadata).

    Content-Defined Chunking ensures consistent data page boundaries for
    efficient deduplication on content-addressable storage systems.

    Args:
        df: DataFrame to write
        output_path: Target path for .parquet file

    Example:
        >>> df = pl.DataFrame({
        ...     "id": ["a", "b"],
        ...     "internal:parent_id": [0, 1]
        ... })
        >>> write_parquet_file_with_cdc(df, "level0.parquet")
        # Parquet with CDC enabled for incremental updates
    """
    df.write_parquet(output_path, **PARQUET_CDC_DEFAULT_CONFIG)


def cast_dataframe_to_schema(df: pl.DataFrame, schema_spec: list) -> pl.DataFrame:
    """
    Cast DataFrame columns to match schema specification from taco:field_schema.

    Converts DataFrame types to match the expected schema defined in
    collection["taco:field_schema"]["levelX"]. Handles Null columns gracefully
    by adding missing columns and coercing type mismatches.

    Args:
        df: Polars DataFrame with potentially inconsistent types
        schema_spec: List of [column_name, type_string] from taco:field_schema

    Returns:
        DataFrame with all columns cast to correct types

    Example:
        >>> schema = [
        ...     ["id", "string"],
        ...     ["type", "string"],
        ...     ["internal:parent_id", "int64"],
        ... ]
        >>> df = cast_dataframe_to_schema(df, schema)
    """
    type_mapping = {
        "string": pl.Utf8,
        "int64": pl.Int64,
        "int32": pl.Int32,
        "float64": pl.Float64,
        "float32": pl.Float32,
        "binary": pl.Binary,
        "bool": pl.Boolean,
        "list(int64)": pl.List(pl.Int64),
        "list(float32)": pl.List(pl.Float32),
        "list(float64)": pl.List(pl.Float64),
    }

    for col_name, type_str in schema_spec:
        target_type = type_mapping.get(type_str)
        if target_type is None:
            continue

        if col_name not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=target_type).alias(col_name))
            continue

        current_type = df[col_name].dtype

        if current_type == target_type:
            continue

        try:
            df = df.with_columns(pl.col(col_name).cast(target_type))
        except Exception:
            df = df.with_columns(
                pl.col(col_name).fill_null(pl.lit(None)).cast(target_type)
            )

    return df