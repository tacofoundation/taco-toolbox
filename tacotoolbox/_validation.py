"""
Validation utilities for TACO container creation.

This module provides validation functions used during TACO creation to ensure:
- Output paths are valid and available
- Size parameters are correctly formatted
- Format options are compatible with other parameters

All validation functions raise TacoValidationError with clear, actionable
error messages when validation fails.

Functions:
    - validate_output_path: Check output path availability
    - validate_split_size: Parse and validate split size parameter
    - validate_format_value: Validate format parameter value
    - validate_format_and_split: Check format/split compatibility
    - parse_size: Convert human-readable sizes to bytes

Exception:
    - TacoValidationError: Base exception for all validation errors
"""

import pathlib
import re
from typing import Literal


class TacoValidationError(Exception):
    """
    Raised when TACO creation validation fails.

    This exception is raised by validation functions when input parameters
    or paths don't meet requirements for container creation.

    Error messages are designed to be actionable, telling users exactly
    what's wrong and how to fix it.

    Example:
        >>> try:
        ...     validate_output_path(Path("existing.tacozip"), "zip")
        ... except TacoValidationError as e:
        ...     print(e)
        Output file already exists: existing.tacozip
        Remove it or choose a different output path.
    """

    pass


def validate_output_path(
    path: pathlib.Path, output_format: Literal["zip", "folder"]
) -> None:
    """
    Validate that output path is available for creation.

    Checks that:
    - Output path doesn't already exist (prevents accidental overwrite)
    - Parent directory exists (can't create in non-existent directory)

    Args:
        path: Target output path for container
        output_format: Creation format ("zip" or "folder")

    Raises:
        TacoValidationError: If path exists or parent directory missing

    Example:
        >>> # Valid case
        >>> validate_output_path(Path("new_dataset.tacozip"), "zip")

        >>> # Invalid - path exists
        >>> validate_output_path(Path("existing.tacozip"), "zip")
        TacoValidationError: Output file already exists: existing.tacozip

        >>> # Invalid - parent missing
        >>> validate_output_path(Path("/nonexistent/data.tacozip"), "zip")
        TacoValidationError: Parent directory does not exist: /nonexistent
    """
    if path.exists():
        if output_format == "zip":
            raise TacoValidationError(
                f"Output file already exists: {path}\n"
                f"Remove it or choose a different output path."
            )
        else:
            raise TacoValidationError(
                f"Output directory already exists: {path}\n"
                f"Remove it or choose a different output path."
            )

    if not path.parent.exists():
        raise TacoValidationError(
            f"Parent directory does not exist: {path.parent}\n"
            f"Create the parent directory first."
        )


def validate_split_size(size_str: str) -> int:
    """
    Validate and parse split_size parameter to bytes.

    Parses human-readable size strings (e.g., "4GB") and validates that:
    - Format is valid (parseable)
    - Value is positive (greater than zero)

    Uses parse_size() internally for actual parsing.

    Args:
        size_str: Size string like "4GB", "100GB", "1TB"

    Returns:
        Size in bytes

    Raises:
        TacoValidationError: If size format invalid or value non-positive

    Example:
        >>> validate_split_size("4GB")
        4294967296

        >>> validate_split_size("invalid")
        TacoValidationError: Invalid split_size format: ...

        >>> validate_split_size("0GB")
        TacoValidationError: split_size must be positive. Requested: 0GB
    """
    try:
        size_bytes = parse_size(size_str)
    except ValueError as e:
        raise TacoValidationError(f"Invalid split_size format: {e}") from e

    if size_bytes <= 0:
        raise TacoValidationError(f"split_size must be positive. Requested: {size_str}")

    return size_bytes


def validate_format_and_split(
    output_format: Literal["zip", "folder"], split_size: str | None
) -> None:
    """
    Validate compatibility between format and split_size parameters.

    Ensures that split_size is only used with ZIP format, as FOLDER
    format doesn't support splitting.

    Args:
        output_format: Creation format ("zip" or "folder")
        split_size: Optional split size parameter

    Raises:
        TacoValidationError: If folder format used with split_size

    Example:
        >>> # Valid combinations
        >>> validate_format_and_split("zip", "4GB")
        >>> validate_format_and_split("zip", None)
        >>> validate_format_and_split("folder", None)

        >>> # Invalid combination
        >>> validate_format_and_split("folder", "4GB")
        TacoValidationError: split_size is not supported with format='folder'
    """
    if output_format == "folder" and split_size is not None:
        raise TacoValidationError(
            "split_size is not supported with format='folder'.\n"
            "Splitting is only available for format='zip'."
        )


def validate_format_value(output_format: str) -> None:
    """
    Validate that format parameter has allowed value.

    Only "zip" and "folder" are valid TACO container formats.

    Args:
        output_format: Format string to validate

    Raises:
        TacoValidationError: If format is not "zip" or "folder"

    Example:
        >>> validate_format_value("zip")     # OK
        >>> validate_format_value("folder")  # OK
        >>> validate_format_value("tar")     # FAIL
        TacoValidationError: Invalid format: 'tar'. Must be 'zip' or 'folder'.
    """
    if output_format not in ("zip", "folder"):
        raise TacoValidationError(
            f"Invalid format: '{output_format}'. Must be 'zip' or 'folder'."
        )


def parse_size(size_str: str) -> int:
    """
    Parse human-readable size string to bytes.

    Supports common size formats with optional suffixes:
    - GB/G: Gigabytes (1024^3 bytes)
    - MB/M: Megabytes (1024^2 bytes)
    - KB/K: Kilobytes (1024 bytes)
    - B or no suffix: Bytes

    Decimal values are supported (e.g., "4.5GB").
    Case-insensitive (e.g., "4gb" == "4GB").
    Whitespace between number and unit is allowed.

    Args:
        size_str: Size string to parse (e.g., "4GB", "512M", "1024KB")

    Returns:
        Size in bytes

    Raises:
        ValueError: If format is invalid or cannot be parsed

    Example:
        >>> parse_size("4GB")
        4294967296
        >>> parse_size("512MB")
        536870912
        >>> parse_size("2.5GB")
        2684354560
        >>> parse_size("1024")
        1024
        >>> parse_size("4 GB")  # Whitespace allowed
        4294967296
        >>> parse_size("4gb")   # Case insensitive
        4294967296
    """
    size_str = size_str.strip().upper()

    # Match pattern: number + optional unit
    match = re.match(r"^(\d+(?:\.\d+)?)\s*(GB?|MB?|KB?|B?)$", size_str)

    if not match:
        raise ValueError(
            f"Invalid size format: '{size_str}'. "
            f"Use format like '4GB', '512MB', '1024KB', or '2048B'"
        )

    value = float(match.group(1))
    unit = match.group(2)

    # Convert to bytes using binary prefixes (1024-based)
    multipliers = {
        "B": 1,
        "KB": 1024,
        "K": 1024,
        "MB": 1024**2,
        "M": 1024**2,
        "GB": 1024**3,
        "G": 1024**3,
    }

    if not unit or unit == "B":
        return int(value)

    return int(value * multipliers.get(unit, 1))
