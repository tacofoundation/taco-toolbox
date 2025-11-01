"""
Global constants for tacotoolbox.

Organization:
- SHARED_*    : Used by both ZIP and FOLDER containers
- ZIP_*       : ZIP container specific
- FOLDER_*    : FOLDER container specific  
- TACOCAT_*   : TacoCat format specific
- METADATA_*  : Metadata column names
- VALIDATION_*: Validation rules and limits
- PADDING_*   : Padding-related constants
"""

import re

# =============================================================================
# METADATA COLUMNS (SHARED - used by both containers)
# =============================================================================

METADATA_PARENT_ID = "internal:parent_id"
"""Parent sample index in previous level DataFrame (enables relational queries)."""

METADATA_OFFSET = "internal:offset"
"""Byte offset in container file where data starts. Only for zip containers."""

METADATA_SIZE = "internal:size"
"""Size in bytes of the data. Only for zip containers."""

METADATA_RELATIVE_PATH = "internal:relative_path"
"""Relative path from DATA/ directory (for consolidated metadata only)."""


METADATA_COLUMNS_ORDER = [
    METADATA_PARENT_ID,
    METADATA_OFFSET,
    METADATA_SIZE,
    METADATA_RELATIVE_PATH,
]
"""Preferred order for internal:* columns at end of DataFrames."""

METADATA_PROTECTED_COLUMNS = {
    METADATA_PARENT_ID,
    METADATA_OFFSET,
    METADATA_SIZE,
    METADATA_RELATIVE_PATH,
}
"""
Protected internal:* columns that should not be dropped during cleaning.
These are the core internal columns created by tacotoolbox.
"""

# =============================================================================
# CORE FIELDS (SHARED - used in Sample datamodel)
# =============================================================================

SHARED_CORE_FIELDS = {"id", "type", "path"}
"""Core Sample fields that cannot be overwritten by extensions."""

SHARED_PROTECTED_FIELDS = SHARED_CORE_FIELDS
"""Alias for protected fields (same as core fields)."""

# =============================================================================
# HIERARCHY LIMITS (SHARED)
# =============================================================================

SHARED_MAX_DEPTH = 5
"""Maximum hierarchy depth (0-5 means 6 levels total)."""

SHARED_MAX_LEVELS = 6
"""
Total number of possible levels (0 through 5) plus COLLECTION.json location.
In TACO containers, we have 5 metadata levels (level0-level5) plus the 
COLLECTION.json entry, making 6 entries total in the TACO_HEADER.
"""

# =============================================================================
# PADDING (SHARED)
# =============================================================================

PADDING_PREFIX = "__TACOPAD__"
"""Prefix for auto-generated padding sample IDs."""

# =============================================================================
# VALIDATION (SHARED)
# =============================================================================

VALIDATION_KEY_PATTERN = re.compile(r"^[a-zA-Z0-9_]+(?:[:][\w]+)?$")
"""Regex pattern for valid metadata key names (e.g., 'key', 'stac:title')."""

VALIDATION_MAX_TITLE_LENGTH = 250
"""Maximum length for dataset title field."""

VALIDATION_MIN_SPLIT_SIZE = 1024
"""Minimum size in bytes for dataset splitting (1 KB)."""

# =============================================================================
# ZIP CONTAINER SPECIFIC
# =============================================================================

ZIP_LFH_BASE_SIZE = 30
"""ZIP Local File Header base size in bytes (before filename)."""

ZIP_ZIP64_EXTRA_FIELD_SIZE = 20
"""ZIP64 extra field size in bytes."""

ZIP_ZIP64_THRESHOLD = 4_294_967_295
"""File size threshold for ZIP64 format (4GB - 1 byte)."""

ZIP_TACO_HEADER_FILENAME = "TACO_HEADER"
"""Filename for TACO header entry in ZIP."""

ZIP_TACO_HEADER_FILENAME_LEN = 11
"""Length of TACO_HEADER filename."""

ZIP_TACO_HEADER_LFH_SIZE = ZIP_LFH_BASE_SIZE + ZIP_TACO_HEADER_FILENAME_LEN  # 41
"""Total Local File Header size for TACO_HEADER entry."""

ZIP_TACO_HEADER_DATA_SIZE = 116
"""Data payload size of TACO_HEADER (stores entries table)."""

ZIP_TACO_HEADER_TOTAL_SIZE = ZIP_TACO_HEADER_LFH_SIZE + ZIP_TACO_HEADER_DATA_SIZE  # 157
"""Total size of TACO_HEADER in ZIP (LFH + data)."""

# =============================================================================
# FOLDER CONTAINER SPECIFIC
# =============================================================================

FOLDER_DATA_DIR = "DATA"
"""Directory name for data files in FOLDER container."""

FOLDER_METADATA_DIR = "METADATA"
"""Directory name for consolidated metadata in FOLDER container."""

FOLDER_META_FILENAME = "__meta__"
"""Filename for local metadata files in FOLDER container."""

FOLDER_COLLECTION_FILENAME = "COLLECTION.json"
"""Filename for collection metadata."""

# =============================================================================
# TACOCAT FORMAT SPECIFIC
# =============================================================================

TACOCAT_MAGIC = b"TACOCAT\x00"
"""Magic number identifying TacoCat files (8 bytes)."""

TACOCAT_VERSION = 1
"""TacoCat format version (uint32)."""

TACOCAT_MAX_LEVELS = 6
"""
Fixed number of levels in TacoCat format (always 6 entries).
Structure: 5 metadata levels (level0-level5) + COLLECTION.json.
When a level doesn't exist in the dataset, its entry contains zeros (offset=0, size=0).
This static structure allows for deterministic offset calculations.
"""

TACOCAT_HEADER_SIZE = 16
"""TacoCat file header size: Magic(8) + Version(4) + MaxDepth(4)."""

TACOCAT_INDEX_ENTRY_SIZE = 16
"""Size of each index entry: Offset(8) + Size(8)."""

TACOCAT_INDEX_SIZE = (
    TACOCAT_MAX_LEVELS * TACOCAT_INDEX_ENTRY_SIZE + TACOCAT_INDEX_ENTRY_SIZE
)  # 112
"""Total index block size: 7 entries x 16 bytes."""

TACOCAT_TOTAL_HEADER_SIZE = TACOCAT_HEADER_SIZE + TACOCAT_INDEX_SIZE  # 128
"""Total header + index size (data starts at byte 128)."""

TACOCAT_DEFAULT_PARQUET_CONFIG = {
    "compression": "zstd",
    "compression_level": 13,  # Balanced compression (fast + good ratio)
    "row_group_size": 122_880,  # DuckDB default for parallelization
    "statistics": True,  # CRITICAL for predicate pushdown
}
"""
Default Parquet configuration optimized for DuckDB queries.s
"""

TACOCAT_FILENAME = "__TACOCAT__"
"""Fixed filename for TacoCat consolidated files."""

# =============================================================================
# FILE PATHS (SHARED)
# =============================================================================

SHARED_DATA_PREFIX = "DATA/"
"""Prefix for data files in archive paths."""

SHARED_METADATA_PREFIX = "METADATA/"
"""Prefix for metadata files in archive paths."""

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def is_padding_id(sample_id: str) -> bool:
    """
    Check if a sample ID represents padding.

    Args:
        sample_id: Sample ID to check

    Returns:
        True if ID starts with padding prefix

    Example:
        >>> is_padding_id("__TACOPAD__0")
        True
        >>> is_padding_id("real_sample")
        False
    """
    return sample_id.startswith(PADDING_PREFIX)


def is_internal_column(column_name: str) -> bool:
    """
    Check if a column name is an internal metadata column.

    Args:
        column_name: Column name to check

    Returns:
        True if column starts with "internal:"

    Example:
        >>> is_internal_column("internal:parent_id")
        True
        >>> is_internal_column("custom_field")
        False
    """
    return column_name.startswith("internal:")


def is_protected_column(column_name: str) -> bool:
    """
    Check if a column is protected and should not be dropped.

    ALL internal:* columns are now protected. This ensures that any
    internal metadata added by extensions or processing is preserved.

    Args:
        column_name: Column name to check

    Returns:
        True if column starts with "internal:"

    Example:
        >>> is_protected_column("internal:offset")
        True
        >>> is_protected_column("internal:custom_metadata")
        True
        >>> is_protected_column("user_field")
        False
    """
    return is_internal_column(column_name)


def validate_depth(depth: int, context: str = "operation") -> None:
    """
    Validate that depth is within allowed range.

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


# =============================================================================
# VERSION INFO
# =============================================================================

TACO_SPECIFICATION_VERSION = "2.0.0"
"""Current TACO specification version."""

TACOTOOLBOX_MIN_PYTHON = "3.10"
"""Minimum required Python version."""
