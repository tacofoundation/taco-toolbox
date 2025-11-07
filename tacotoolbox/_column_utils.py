"""
Column manipulation utilities for tacotoolbox.

This module provides shared utilities for working with DataFrame columns,
particularly for handling internal:* metadata columns consistently across
both ZIP and FOLDER containers.

Functions:
    - reorder_internal_columns: Place internal:* columns at end of DataFrame
    - remove_empty_columns: Remove columns with all null/empty values
    - validate_schema_consistency: Check schema compatibility across DataFrames
    - sanitize_avro_columns: Replace colons for Avro serialization
    - desanitize_avro_columns: Restore colons after Avro deserialization
    - read_metadata_file: Read Parquet or Avro with auto-format detection
    - cast_dataframe_to_schema: Cast columns to match schema spec
"""

from pathlib import Path

import polars as pl

from tacotoolbox._constants import (
    AVRO_COLON_REPLACEMENT,
    METADATA_COLUMNS_ORDER,
    SHARED_CORE_FIELDS,
    is_internal_column,
)


def reorder_internal_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Reorder DataFrame columns to place internal:* columns at the end.

    This function ensures consistent column ordering across all metadata
    DataFrames in TACO containers. Regular columns come first, followed
    by internal:* columns in a preferred order.

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
    # Separate regular and internal columns
    regular_cols = [col for col in df.columns if not is_internal_column(col)]

    # Get internal columns in preferred order
    ordered_internal = [col for col in METADATA_COLUMNS_ORDER if col in df.columns]

    # Get any other internal:* columns not in the preferred list
    other_internal = [
        col
        for col in df.columns
        if is_internal_column(col) and col not in METADATA_COLUMNS_ORDER
    ]

    # Combine: regular + ordered internal + other internal
    new_order = regular_cols + ordered_internal + other_internal

    return df.select(new_order)


def remove_empty_columns(
    df: pl.DataFrame, preserve_core: bool = True, preserve_internal: bool = True
) -> pl.DataFrame:
    """
    Remove columns that are completely empty (all null or empty strings).

    This function cleans up metadata DataFrames by removing columns that
    contain no useful information. Core fields and internal:* columns are
    protected by default.

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
        # Always preserve core fields if requested
        if preserve_core and col in SHARED_CORE_FIELDS:
            cols_to_keep.append(col)
            continue

        # Always preserve internal:* columns if requested
        if preserve_internal and is_internal_column(col):
            cols_to_keep.append(col)
            continue

        # Check if column has any non-null, non-empty values
        if df[col].is_null().all():
            # All nulls - skip this column
            continue

        # For string columns, also check for empty strings
        if df[col].dtype == pl.Utf8:
            non_empty = df.filter(
                (pl.col(col).is_not_null())
                & (pl.col(col) != "")
                & (pl.col(col) != "None")
            ).height

            if non_empty > 0:
                cols_to_keep.append(col)
        else:
            # Non-string column with some non-null values
            cols_to_keep.append(col)

    # Ensure we keep at least one column
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
        context: Description of where these DataFrames come from (for error messages)

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
        return  # Single DataFrame is always consistent with itself

    # Use first DataFrame as reference
    reference_schema = dict(dataframes[0].schema)
    reference_columns = set(reference_schema.keys())

    # Check each subsequent DataFrame
    for i, df in enumerate(dataframes[1:], start=1):
        current_schema = dict(df.schema)
        current_columns = set(current_schema.keys())

        # Check for column differences
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

        # Check for type mismatches in common columns
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
        context: Description of the DataFrame (for error messages)

    Raises:
        ValueError: If any required columns are missing

    Example:
        >>> df = pl.DataFrame({"id": ["a"], "type": ["FILE"]})
        >>> ensure_columns_exist(df, ["id", "type"])  # OK
        >>> ensure_columns_exist(df, ["id", "missing"])  # Raises ValueError
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

    Example:
        >>> df = pl.DataFrame({
        ...     "id": ["a"],
        ...     "internal:offset": [100],
        ...     "internal:size": [50],
        ...     "custom": ["x"]
        ... })
        >>> # Keep only internal:* columns
        >>> internal_df = filter_columns_by_prefix(df, "internal:")
        >>> internal_df.columns
        ['internal:offset', 'internal:size']
        >>>
        >>> # Exclude internal:* columns
        >>> regular_df = filter_columns_by_prefix(df, "internal:", exclude=True)
        >>> regular_df.columns
        ['id', 'custom']
    """
    if exclude:
        cols = [col for col in df.columns if not col.startswith(prefix)]
    else:
        cols = [col for col in df.columns if col.startswith(prefix)]

    if not cols:
        # Return empty DataFrame with same height but no columns
        return df.select([])

    return df.select(cols)


def add_columns_if_missing(
    df: pl.DataFrame, columns: dict[str, pl.DataType], default_value: any = None
) -> pl.DataFrame:
    """
    Add columns to DataFrame if they don't exist, with default values.

    Useful for ensuring schema compatibility when some DataFrames might
    be missing certain columns.

    Args:
        df: DataFrame to modify
        columns: Dictionary mapping column_name -> polars DataType
        default_value: Value to fill new columns with (default: None)

    Returns:
        DataFrame with all specified columns present

    Example:
        >>> df = pl.DataFrame({"id": ["a", "b"]})
        >>> df = add_columns_if_missing(
        ...     df,
        ...     {"internal:offset": pl.Int64(), "internal:size": pl.Int64()},
        ...     default_value=0
        ... )
        >>> df.columns
        ['id', 'internal:offset', 'internal:size']
    """
    for col_name, col_type in columns.items():
        if col_name not in df.columns:
            # Create column with default values
            df = df.with_columns(pl.lit(default_value).cast(col_type).alias(col_name))

    return df


def get_column_statistics(df: pl.DataFrame) -> dict[str, dict]:
    """
    Get basic statistics about DataFrame columns.

    Useful for debugging and understanding DataFrame contents.

    Args:
        df: DataFrame to analyze

    Returns:
        Dictionary mapping column_name -> statistics dict

    Example:
        >>> df = pl.DataFrame({
        ...     "id": ["a", "b", "c"],
        ...     "value": [1, None, 3]
        ... })
        >>> stats = get_column_statistics(df)
        >>> stats["value"]["null_count"]
        1
    """
    stats = {}

    for col in df.columns:
        col_stats = {
            "dtype": str(df[col].dtype),
            "null_count": df[col].is_null().sum(),
            "null_percentage": (df[col].is_null().sum() / len(df)) * 100,
            "is_empty": df[col].is_null().all(),
        }

        # Add type-specific stats
        if df[col].dtype in [pl.Int64, pl.Int32, pl.Float64, pl.Float32]:
            if not df[col].is_null().all():
                col_stats["min"] = df[col].min()
                col_stats["max"] = df[col].max()
                col_stats["mean"] = df[col].mean()

        stats[col] = col_stats

    return stats


def sanitize_avro_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Sanitize DataFrame column names for Avro serialization.

    Replaces colons with AVRO_COLON_REPLACEMENT since Avro specification
    does not allow colons in field names. This affects all columns with
    colons including internal:* columns and STAC/custom metadata fields.

    Args:
        df: DataFrame with potentially colons in column names

    Returns:
        DataFrame with sanitized column names safe for Avro

    Example:
        >>> df = pl.DataFrame({
        ...     "id": ["a"],
        ...     "internal:parent_id": [0],
        ...     "stac:crs": ["EPSG:4326"]
        ... })
        >>> sanitized = sanitize_avro_columns(df)
        >>> sanitized.columns
        ['id', 'internal_COLON_parent_id', 'stac_COLON_crs']
    """
    return df.rename(
        {col: col.replace(":", AVRO_COLON_REPLACEMENT) for col in df.columns}
    )


def desanitize_avro_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Restore original column names after Avro deserialization.

    Replaces AVRO_COLON_REPLACEMENT back with colons to restore
    original column names after reading from Avro files. This is the
    inverse operation of sanitize_avro_columns().

    Args:
        df: DataFrame with sanitized column names from Avro

    Returns:
        DataFrame with restored original column names

    Example:
        >>> df = pl.DataFrame({
        ...     "id": ["a"],
        ...     "internal_COLON_parent_id": [0],
        ...     "stac_COLON_crs": ["EPSG:4326"]
        ... })
        >>> restored = desanitize_avro_columns(df)
        >>> restored.columns
        ['id', 'internal:parent_id', 'stac:crs']
    """
    return df.rename(
        {col: col.replace(AVRO_COLON_REPLACEMENT, ":") for col in df.columns}
    )


def read_metadata_file(file_path: Path | str) -> pl.DataFrame:
    """
    Read metadata file in Parquet or Avro format with automatic format detection.

    Handles Avro column name desanitization automatically. Parquet files
    preserve colons in column names, but Avro requires sanitization/desanitization.

    Supported formats:
        - .parquet: Read directly (colons preserved)
        - .avro: Read and desanitize column names (restore colons)

    Args:
        file_path: Path to metadata file (.parquet or .avro)

    Returns:
        DataFrame with proper column names (colons restored if from Avro)

    Raises:
        ValueError: If file format is not .parquet or .avro

    Example:
        >>> # Reading Avro (auto-desanitizes column names)
        >>> df = read_metadata_file("METADATA/level0.avro")
        >>> "internal:parent_id" in df.columns
        True
        >>>
        >>> # Reading Parquet (no changes needed)
        >>> df = read_metadata_file("METADATA/level0.parquet")
        >>> "internal:parent_id" in df.columns
        True
    """
    file_path = Path(file_path)

    if file_path.suffix == ".parquet":
        return pl.read_parquet(file_path)
    elif file_path.suffix == ".avro":
        df = pl.read_avro(file_path)
        return desanitize_avro_columns(df)
    else:
        raise ValueError(
            f"Unsupported metadata format: {file_path.suffix}\n"
            f"Expected .parquet or .avro, got {file_path}"
        )


def write_avro_file(df: pl.DataFrame, output_path: Path | str) -> None:
    """
    Write DataFrame to Avro file with automatic column name sanitization.

    Sanitizes column names (replaces : with _COLON_) before writing since
    Avro specification does not allow colons in field names. The output
    file can later be read with read_metadata_file() which will automatically
    restore the original column names.

    Args:
        df: DataFrame to write
        output_path: Target path for .avro file

    Example:
        >>> df = pl.DataFrame({
        ...     "id": ["a", "b"],
        ...     "internal:parent_id": [0, 1],
        ...     "stac:crs": ["EPSG:4326", "EPSG:32633"]
        ... })
        >>> write_avro_file(df, "metadata.avro")
        # File contains: id, internal_COLON_parent_id, stac_COLON_crs
        >>>
        >>> # Read it back with automatic desanitization
        >>> df_read = read_metadata_file("metadata.avro")
        >>> df_read.columns
        ['id', 'internal:parent_id', 'stac:crs']
    """
    sanitized_df = sanitize_avro_columns(df)
    sanitized_df.write_avro(output_path, name="TacoMetadata")


def write_parquet_file(df: pl.DataFrame, output_path: Path | str) -> None:
    """
    Write DataFrame to Parquet file.

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
        # Colons preserved directly in column names
    """
    df.write_parquet(output_path, compression="zstd")


def cast_dataframe_to_schema(df: pl.DataFrame, schema_spec: list) -> pl.DataFrame:
    """
    Cast DataFrame columns to match schema specification from taco:field_schema.

    Converts DataFrame types to match the expected schema defined in
    collection["taco:field_schema"]["levelX"]. Handles Null columns gracefully
    by adding missing columns and coercing type mismatches.

    This function ensures all DataFrames have identical schemas before concatenation,
    preventing "type X is incompatible with expected type Y" errors.

    Args:
        df: Polars DataFrame with potentially inconsistent types
        schema_spec: List of [column_name, type_string] from taco:field_schema
                     Example: [["id", "string"], ["internal:parent_id", "int64"], ...]

    Returns:
        DataFrame with all columns cast to correct types and all schema columns present

    Example:
        >>> schema = [
        ...     ["id", "string"],
        ...     ["type", "string"],
        ...     ["internal:parent_id", "int64"],
        ...     ["stac:time_start", "int64"],
        ...     ["stac:centroid", "binary"]
        ... ]
        >>> df = cast_dataframe_to_schema(df, schema)
        >>> # All columns present with correct types
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

        # Add missing column as Null with target type
        if col_name not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=target_type).alias(col_name))
            continue

        current_type = df[col_name].dtype

        # Skip if already correct type
        if current_type == target_type:
            continue

        # ALWAYS cast to target type (handles Null -> Int64, Int64 -> Null, etc.)
        try:
            df = df.with_columns(pl.col(col_name).cast(target_type))
        except Exception:
            # If cast fails, fill nulls first then cast
            df = df.with_columns(
                pl.col(col_name).fill_null(pl.lit(None)).cast(target_type)
            )

    return df
