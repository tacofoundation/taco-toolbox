"""
Column manipulation utilities for tacotoolbox.

Key functions:
- align_dataframe_schemas: Align schemas before vertical concatenation
- reorder_internal_columns: Place internal:* columns at end
- remove_empty_columns: Remove all-null columns
- validate_schema_consistency: Check schema compatibility
- write_parquet_file_with_cdc: Write Parquet with Content-Defined Chunking
- cast_dataframe_to_schema: Cast columns to match schema spec
"""

from pathlib import Path

import polars as pl

from tacotoolbox._constants import (
    PARQUET_CDC_DEFAULT_CONFIG,
    METADATA_COLUMNS_ORDER,
    SHARED_CORE_FIELDS,
)
from tacotoolbox._utils import is_internal_column


def align_dataframe_schemas(
    dfs: list[pl.DataFrame], core_fields: list[str] | None = None
) -> list[pl.DataFrame]:
    """
    Align schemas before vertical concatenation.

    CRITICAL for concatenating DataFrames from samples with different extensions.
    Without alignment, pl.concat(dfs, how="vertical") fails with ShapeError.

    Algorithm:
    1. Collect all unique columns from all DataFrames with their types
    2. Order columns: core fields first, then extension fields alphabetically
    3. Add missing columns to each DataFrame with None values (proper type)
    4. Reorder all DataFrames to identical column order

    Default core_fields: ["id", "type", "path"]
    """
    if len(dfs) <= 1:
        return dfs

    if core_fields is None:
        core_fields = ["id", "type", "path"]

    # Collect complete schema with types from all DataFrames
    complete_schema: dict[str, pl.DataType] = {}
    for df in dfs:
        for col_name, dtype in df.schema.items():
            if col_name not in complete_schema:
                complete_schema[col_name] = dtype

    # Define column ordering: core first, then extensions alphabetically
    extension_fields = sorted(
        [col for col in complete_schema.keys() if col not in core_fields]
    )

    ordered_columns = [
        col for col in core_fields if col in complete_schema
    ] + extension_fields

    # Align all DataFrames
    aligned_dfs = []
    for df in dfs:
        # Add missing columns with proper types
        for col_name, dtype in complete_schema.items():
            if col_name not in df.columns:
                df = df.with_columns(pl.lit(None, dtype=dtype).alias(col_name))

        df = df.select(ordered_columns)
        aligned_dfs.append(df)

    return aligned_dfs


def reorder_internal_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Place internal:* columns at end.

    Order: regular columns → internal:parent_id → internal:offset →
           internal:size → other internal:* columns
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
    """Remove columns that are all null or empty strings."""
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
    Validate all DataFrames have consistent schemas for safe vertical concatenation.

    Checks same columns with same types.
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
    """Validate DataFrame contains all required columns."""
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"{context}: Missing required columns: {missing}\n"
            f"Available columns: {df.columns}"
        )


def read_metadata_file(file_path: Path | str) -> pl.DataFrame:
    """Read metadata file in Parquet format."""
    file_path = Path(file_path)

    if file_path.suffix == ".parquet":
        return pl.read_parquet(file_path)
    else:
        raise ValueError(
            f"Unsupported metadata format: {file_path.suffix}\n"
            f"Expected .parquet, got {file_path}"
        )


def write_parquet_file(df: pl.DataFrame, output_path: Path | str, **kwargs) -> None:
    """
    Write DataFrame to Parquet (for local __meta__ files).

    Used for local metadata in FOLDER containers.
    Parquet natively supports colons in column names.
    """
    default_config = {"compression": "zstd"}
    parquet_config = {**default_config, **kwargs}
    df.write_parquet(output_path, **parquet_config)


def write_parquet_file_with_cdc(
    df: pl.DataFrame, output_path: Path | str, **kwargs
) -> None:
    """
    Write Parquet with Content-Defined Chunking (for consolidated metadata).

    CDC ensures consistent data page boundaries for efficient deduplication
    on content-addressable storage systems.

    CRITICAL: Uses PyArrow's pq.write_table() because CDC is only available
    in PyArrow, not in Polars write_parquet().
    """
    import pyarrow.parquet as pq

    parquet_config = {**PARQUET_CDC_DEFAULT_CONFIG, **kwargs}
    arrow_table = df.to_arrow()
    pq.write_table(arrow_table, output_path, **parquet_config)


def cast_dataframe_to_schema(df: pl.DataFrame, schema_spec: list) -> pl.DataFrame:
    """
    Cast DataFrame columns to match schema spec from taco:field_schema.

    Handles Null columns gracefully by adding missing columns and
    coercing type mismatches.

    schema_spec: List of [column_name, type_string] from taco:field_schema
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
