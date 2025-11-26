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

    Error messages are designed to be actionable, telling users exactly
    what's wrong and how to fix it.
    """

    pass


def validate_output_path(
    path: pathlib.Path, output_format: Literal["zip", "folder"]
) -> None:
    """
    Validate that output path is available for creation.

    Checks that output path doesn't already exist to prevent accidental overwrite.
    Parent directories are created automatically if they don't exist.
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


def validate_split_size(size_str: str) -> int:
    """
    Validate and parse split_size parameter to bytes.

    Parses human-readable size strings (e.g., "4GB") and validates that:
    - Format is valid (parseable)
    - Value is positive (greater than zero)

    Uses parse_size() internally for actual parsing.
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
    """
    size_str = size_str.strip().upper()

    match = re.match(r"^(\d+(?:\.\d+)?)\s*(GB?|MB?|KB?|B?)$", size_str)

    if not match:
        raise ValueError(
            f"Invalid size format: '{size_str}'. "
            f"Use format like '4GB', '512MB', '1024KB', or '2048B'"
        )

    value = float(match.group(1))
    unit = match.group(2)

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