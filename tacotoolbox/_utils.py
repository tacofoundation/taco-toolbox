"""Utility functions for tacotoolbox."""

from tacotoolbox._constants import PADDING_PREFIX, SHARED_MAX_DEPTH


def is_padding_id(sample_id: str) -> bool:
    """Check if sample ID is auto-generated padding."""
    return sample_id.startswith(PADDING_PREFIX)


def is_internal_column(column_name: str) -> bool:
    """Check if column is internal metadata (starts with 'internal:')."""
    return column_name.startswith("internal:")


def validate_depth(depth: int, context: str = "operation") -> None:
    """
    Validate that depth is within allowed range.

    Used by tortilla datamodel for depth validation.

    Args:
        depth: Depth value to validate
        context: Context string for error message

    Raises:
        ValueError: If depth is invalid

    Example:
        >>> validate_depth(3, "export")
        >>> validate_depth(6, "export")  # Raises ValueError
    """
    if depth < 0:
        raise ValueError(f"{context}: depth must be non-negative, got {depth}")

    if depth > SHARED_MAX_DEPTH:
        raise ValueError(
            f"{context}: depth {depth} exceeds maximum of {SHARED_MAX_DEPTH} "
            f"(levels 0-{SHARED_MAX_DEPTH})"
        )
