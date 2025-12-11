"""
Validation utilities for TACO container creation.

This module provides validation functions used during TACO creation to ensure:
- Output paths are valid and available
- Size parameters are correctly formatted
- Format options are compatible with other parameters
- Input files are in valid locations

All validation functions raise TacoValidationError with clear, actionable
error messages when validation fails.

Functions:
    - is_padding_id: Check if sample ID is auto-generated padding
    - validate_output_path: Check output path availability
    - validate_common_directory: Validate all inputs are in same directory
    - validate_split_size: Parse and validate split size parameter
    - validate_format_value: Validate format parameter value
    - validate_format_and_split: Check format/split compatibility
    - parse_size: Convert human-readable sizes to bytes
"""

import pathlib
import re
from collections.abc import Sequence
from typing import Literal

from tacotoolbox._constants import PADDING_PREFIX
from tacotoolbox._exceptions import TacoValidationError


def is_padding_id(sample_id: str) -> bool:
    """Check if sample ID is auto-generated padding."""
    return sample_id.startswith(PADDING_PREFIX)


def validate_output_path(
    path: pathlib.Path, output_format: Literal["zip", "folder"]
) -> None:
    """
    Validate that output path is available for creation.

    Checks that output path doesn't already exist to prevent accidental overwrite.
    Parent directories are created automatically if they don't exist.

    Raises:
        TacoValidationError: If output path already exists
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


def validate_common_directory(
    inputs: Sequence[str | pathlib.Path],
) -> pathlib.Path:
    """
    Validate that all input files are in the same directory.

    Returns the common parent directory for output auto-detection.
    Used by tacocat and tacollection when output directory is not specified.

    Args:
        inputs: Sequence of input file paths

    Returns:
        pathlib.Path: Common parent directory

    Raises:
        TacoValidationError: If files are in different directories or no inputs
    """
    if not inputs:
        raise TacoValidationError("No input files provided")

    input_paths = [pathlib.Path(p).resolve() for p in inputs]
    parent_dirs = [p.parent for p in input_paths]
    first_parent = parent_dirs[0]

    if not all(parent == first_parent for parent in parent_dirs):
        raise TacoValidationError(
            "Input files are in different directories. "
            "Please specify output directory explicitly."
        )

    return first_parent


def validate_split_size(size_str: str) -> int:
    """
    Validate and parse split_size parameter to bytes.

    Parses human-readable size strings (e.g., "4GB") and validates that:
    - Format is valid (parseable)
    - Value is positive (greater than zero)

    Uses parse_size() internally for actual parsing.

    Returns:
        int: Size in bytes

    Raises:
        TacoValidationError: If size format is invalid or value is non-positive
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

    Raises:
        TacoValidationError: If split_size is used with folder format
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

    Raises:
        TacoValidationError: If format is not 'zip' or 'folder'
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

    Returns:
        int: Size in bytes

    Raises:
        ValueError: If size format is invalid
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
