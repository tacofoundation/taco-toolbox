import pathlib
from typing import Literal

from tacotoolbox._helpers import parse_size

# Maximum split size (4GB limit - 100MB safety margin for metadata)
MAX_SPLIT_SIZE = int(3.9 * 1024**3)  # 3.9GB in bytes


class TacoValidationError(Exception):
    """Raised when TACO creation validation fails."""

    pass


def validate_output_path(
    path: pathlib.Path, output_format: Literal["zip", "folder"]
) -> None:
    """
    Validate that output path is available.

    Args:
        path: Target output path
        output_format: Creation format ("zip" or "folder")

    Raises:
        TacoValidationError: If path exists or parent doesn't exist
    """
    # Check if output already exists
    if path.exists():
        if output_format == "zip":
            raise TacoValidationError(
                f"Output file already exists: {path}\n"
                f"Remove it or choose a different output path."
            )
        else:  # folder
            raise TacoValidationError(
                f"Output directory already exists: {path}\n"
                f"Remove it or choose a different output path."
            )

    # Check if parent directory exists
    if not path.parent.exists():
        raise TacoValidationError(
            f"Parent directory does not exist: {path.parent}\n"
            f"Create the parent directory first."
        )


def validate_split_size(size_str: str) -> int:
    """
    Validate and parse split_size parameter.

    Args:
        size_str: Size string like "4GB", "2GB"

    Returns:
        Size in bytes

    Raises:
        TacoValidationError: If size is invalid or exceeds maximum
    """
    try:
        size_bytes = parse_size(size_str)
    except ValueError as e:
        raise TacoValidationError(f"Invalid split_size format: {e}") from e

    if size_bytes > MAX_SPLIT_SIZE:
        max_gb = MAX_SPLIT_SIZE / (1024**3)
        raise TacoValidationError(
            f"split_size cannot exceed {max_gb:.1f}GB (regular ZIP limit minus metadata space).\n"
            f"Requested: {size_str}"
        )

    return size_bytes


def validate_format_and_split(
    output_format: Literal["zip", "folder"], split_size: str | None
) -> None:
    """
    Validate format and split_size compatibility.

    Args:
        output_format: Creation format
        split_size: Optional split size parameter

    Raises:
        TacoValidationError: If folder format used with split_size
    """
    if output_format == "folder" and split_size is not None:
        raise TacoValidationError(
            "split_size is not supported with format='folder'.\n"
            "Splitting is only available for format='zip'."
        )


def validate_format_value(output_format: str) -> None:
    """
    Validate format parameter value.

    Args:
        output_format: Format string to validate

    Raises:
        TacoValidationError: If format is not "zip" or "folder"
    """
    if output_format not in ("zip", "folder"):
        raise TacoValidationError(
            f"Invalid format: '{output_format}'. Must be 'zip' or 'folder'."
        )
