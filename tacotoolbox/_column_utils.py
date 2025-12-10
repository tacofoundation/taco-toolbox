"""
Column manipulation utilities for tacotoolbox.

Key functions:
- align_arrow_schemas: Align schemas before vertical concatenation
- reorder_internal_columns: Place internal:* columns at end
- remove_empty_columns: Remove all-null columns
- validate_schema_consistency: Check schema compatibility
- write_parquet_file_with_cdc: Write Parquet with Content-Defined Chunking
"""

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from tacotoolbox._constants import (
    METADATA_COLUMNS_ORDER,
    PARQUET_CDC_DEFAULT_CONFIG,
    SHARED_CORE_FIELDS,
)


def is_internal_column(column_name: str) -> bool:
    """Check if column is internal metadata (starts with 'internal:')."""
    return column_name.startswith("internal:")


def align_arrow_schemas(
    tables: list[pa.Table],
    core_fields: list[str] | None = None,
) -> list[pa.Table]:
    """
    Align schemas before vertical concatenation.

    CRITICAL for concatenating Tables from samples with different extensions.
    Without alignment, pa.concat_tables(tables) fails with schema mismatch.

    Algorithm:
    1. Collect all unique columns from all Tables with their types
    2. Order columns: core fields first, then extension fields alphabetically
    3. Add missing columns to each Table with None values (proper type)
    4. Reorder all Tables to identical column order

    Default core_fields: ["id", "type", "path"]
    """
    if not tables:
        return tables

    if len(tables) == 1:
        return tables

    if core_fields is None:
        core_fields = ["id", "type", "path"]

    # Collect all unique field names and types across all tables
    # Use dict to preserve order and track types
    all_fields: dict[str, pa.DataType] = {}

    for table in tables:
        for field in table.schema:
            if field.name not in all_fields:
                all_fields[field.name] = field.type
            # If same column appears with different types, keep first occurrence
            # This ensures consistent schema when tables have conflicting types

    # Define column ordering: core first, then extensions alphabetically
    extension_fields = sorted([col for col in all_fields if col not in core_fields])
    ordered_columns = [
        col for col in core_fields if col in all_fields
    ] + extension_fields

    # Build target schema with ordered columns
    target_schema = pa.schema(
        [pa.field(name, all_fields[name]) for name in ordered_columns]
    )

    # Align each table to target schema
    aligned_tables = []

    for table in tables:
        current_columns = set(table.schema.names)
        missing_columns = set(target_schema.names) - current_columns

        if not missing_columns:
            # Reorder columns to match target
            reordered_arrays = [table.column(name) for name in target_schema.names]
            aligned_table = pa.Table.from_arrays(reordered_arrays, schema=target_schema)
            aligned_tables.append(aligned_table)
            continue

        # Add missing columns with None values
        arrays = [table.column(name) for name in table.schema.names]
        schema_fields = list(table.schema)

        for col_name in missing_columns:
            field_idx = target_schema.get_field_index(col_name)
            field = target_schema.field(field_idx)

            null_array = pa.nulls(table.num_rows, type=field.type)

            arrays.append(null_array)
            schema_fields.append(field)

        # Reorder columns to match target schema order
        new_schema = pa.schema(schema_fields)
        temp_table = pa.Table.from_arrays(arrays, schema=new_schema)

        reordered_arrays = [temp_table.column(name) for name in target_schema.names]
        aligned_table = pa.Table.from_arrays(reordered_arrays, schema=target_schema)

        aligned_tables.append(aligned_table)

    return aligned_tables


def reorder_internal_columns(table: pa.Table) -> pa.Table:
    """
    Place internal:* columns at end.

    Order: regular columns → internal:parent_id → internal:offset →
           internal:size → other internal:* columns
    """
    regular_cols = [col for col in table.schema.names if not is_internal_column(col)]
    ordered_internal = [
        col for col in METADATA_COLUMNS_ORDER if col in table.schema.names
    ]
    other_internal = [
        col
        for col in table.schema.names
        if is_internal_column(col) and col not in METADATA_COLUMNS_ORDER
    ]

    new_order = regular_cols + ordered_internal + other_internal
    return table.select(new_order)


def remove_empty_columns(
    table: pa.Table,
    preserve_core: bool = True,
    preserve_internal: bool = True,
) -> pa.Table:
    """Remove columns that are all null or empty strings."""
    cols_to_keep = []

    for col_name in table.schema.names:
        # Preserve core fields if requested
        if preserve_core and col_name in SHARED_CORE_FIELDS:
            cols_to_keep.append(col_name)
            continue

        # Preserve internal columns if requested
        if preserve_internal and is_internal_column(col_name):
            cols_to_keep.append(col_name)
            continue

        column = table.column(col_name)

        # Check if all null
        if pc.all(pc.is_null(column)).as_py():
            continue

        # For string columns, check for empty/None strings
        if pa.types.is_string(column.type) or pa.types.is_large_string(column.type):
            # Check if any non-null, non-empty values exist
            is_valid = pc.and_(
                pc.is_valid(column),
                pc.and_(pc.not_equal(column, ""), pc.not_equal(column, "None")),
            )
            if pc.any(is_valid).as_py():
                cols_to_keep.append(col_name)
        else:
            cols_to_keep.append(col_name)

    # Keep at least one column
    if not cols_to_keep:
        cols_to_keep = [table.schema.names[0]]

    return table.select(cols_to_keep)


def validate_schema_consistency(
    tables: list[pa.Table], context: str = "Table list"
) -> None:
    """
    Validate all Tables have consistent schemas for safe vertical concatenation.

    Checks same columns with same types.
    """
    if not tables:
        raise ValueError(f"{context}: Cannot validate empty Table list")

    if len(tables) == 1:
        return

    reference_schema = tables[0].schema
    reference_columns = set(reference_schema.names)

    for i, table in enumerate(tables[1:], start=1):
        current_schema = table.schema
        current_columns = set(current_schema.names)

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

        # Check type consistency
        type_mismatches = []
        for col in reference_columns:
            ref_type = reference_schema.field(col).type
            curr_type = current_schema.field(col).type

            if ref_type != curr_type:
                type_mismatches.append(
                    f"  Column '{col}': expected {ref_type}, got {curr_type}"
                )

        if type_mismatches:
            error_msg = f"{context}: Type mismatches at index {i}:\n"
            error_msg += "\n".join(type_mismatches)
            raise ValueError(error_msg)


def ensure_columns_exist(
    table: pa.Table, required_columns: list[str], context: str = "Table"
) -> None:
    """Validate Table contains all required columns."""
    missing = [col for col in required_columns if col not in table.schema.names]

    if missing:
        raise ValueError(
            f"{context}: Missing required columns: {missing}\n"
            f"Available columns: {table.schema.names}"
        )


def read_metadata_file(file_path: Path | str) -> pa.Table:
    """Read metadata file in Parquet format."""
    file_path = Path(file_path)

    if file_path.suffix == ".parquet":
        return pq.read_table(file_path)
    else:
        raise ValueError(
            f"Unsupported metadata format: {file_path.suffix}\n"
            f"Expected .parquet, got {file_path}"
        )


def write_parquet_file(table: pa.Table, output_path: Path | str, **kwargs: Any) -> None:
    """
    Write Table to Parquet (for local __meta__ files).

    Used for local metadata in FOLDER containers.
    Parquet natively supports colons in column names.
    """
    default_config = {"compression": "zstd"}
    parquet_config = {**default_config, **kwargs}
    pq.write_table(table, output_path, **parquet_config)


def write_parquet_file_with_cdc(
    table: pa.Table, output_path: Path | str, **kwargs: Any
) -> None:
    """
    Write Parquet with Content-Defined Chunking (for consolidated metadata).

    CDC ensures consistent data page boundaries for efficient deduplication
    on content-addressable storage systems.
    """
    parquet_config = {**PARQUET_CDC_DEFAULT_CONFIG, **kwargs}
    pq.write_table(table, output_path, **parquet_config)
