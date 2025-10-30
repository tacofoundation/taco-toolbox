"""
Column manipulation utilities for tacotoolbox.

This module provides shared utilities for working with DataFrame columns,
particularly for handling internal:* metadata columns consistently across
both ZIP and FOLDER containers.

Functions:
    - reorder_internal_columns: Place internal:* columns at end of DataFrame
    - remove_empty_columns: Remove columns with all null/empty values
    - validate_schema_consistency: Check schema compatibility across DataFrames
"""

import polars as pl

from tacotoolbox._constants import (
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
