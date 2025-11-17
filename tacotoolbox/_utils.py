"""Utility functions for tacotoolbox."""

from tacotoolbox._constants import PADDING_PREFIX


def is_padding_id(sample_id: str) -> bool:
    """Check if sample ID is auto-generated padding."""
    return sample_id.startswith(PADDING_PREFIX)


def is_internal_column(column_name: str) -> bool:
    """Check if column is internal metadata (starts with 'internal:')."""
    return column_name.startswith("internal:")
