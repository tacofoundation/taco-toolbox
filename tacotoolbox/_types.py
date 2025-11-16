"""
Type definitions and aliases for tacotoolbox.

Provides type hints and type aliases used throughout the TACO creation
and manipulation codebase. Includes types for:
- Offset pairs and file mappings
- PIT (Position-Isomorphic Tree) schema structures
- Metadata packages and field schemas
- Virtual ZIP structures
- Sample grouping for dataset splitting

These types improve code clarity and enable better static type checking.
"""

from typing import TypeAlias

import polars as pl

# ============================================================================
# CORE OFFSET TYPES
# ============================================================================

OffsetPair: TypeAlias = tuple[int, int]
"""Offset and size pair: (offset, size)"""

FilePair: TypeAlias = tuple[str, str]
"""Source and archive path pair: (src_path, arc_path)"""


# ============================================================================
# LOOKUP DICTIONARIES
# ============================================================================

DataLookup: TypeAlias = dict[str, OffsetPair]
"""Lookup for data files: arc_path -> (offset, size)"""

OffsetMap: TypeAlias = dict[str, OffsetPair]
"""Generic offset mapping: path -> (offset, size)"""


# ============================================================================
# PIT SCHEMA TYPES
# ============================================================================


class PITRootLevel:
    """Root level of PIT schema (level 0 - the collection)."""

    n: int
    type: str


class PITPattern:
    """Pattern descriptor for a specific position in the hierarchy."""

    n: int
    children: list[str]


PITSchema: TypeAlias = dict[str, object]
"""
Position-Isomorphic Tree schema for deterministic navigation.

Structure:
    {
        "root": {"n": int, "type": str},
        "hierarchy": {
            "1": [{"n": int, "children": [str, ...]}, ...],
            "2": [{"n": int, "children": [str, ...]}, ...],
        }
    }
"""


# ============================================================================
# METADATA PACKAGE TYPES (NEW DUAL SYSTEM)
# ============================================================================


class LocalMetadata:
    """Metadata for a single folder (local __metadata__). Used for DATA/folder/__metadata__ files (level 1+ only)."""

    folder_path: str
    samples: list  # list[Sample] - can't import to avoid circular dependency
    metadata_df: pl.DataFrame


class MetadataPackage:
    """
    Complete metadata bundle with dual system.

    Contains both:
    - Consolidated metadata (METADATA/levelX.parquet) for ALL levels
    - Local metadata (DATA/folder/__metadata__) for FOLDERs only (level 1+)

    The internal:parent_id column is permanent in all metadata files (level 1+).
    Enables relational queries in DuckDB via JOINs.
    """

    levels: list[pl.DataFrame]
    local_metadata: dict[str, pl.DataFrame]
    collection: dict[str, object]
    pit_schema: PITSchema
    max_depth: int


# ============================================================================
# EXTRACTED FILES TYPES
# ============================================================================


class ExtractedFiles:
    """Files extracted from samples for DATA/."""

    src_files: list[str]
    arc_files: list[str]


ExtractedFilesDict: TypeAlias = dict[str, list[str]]
"""Dictionary format for ExtractedFiles: {"src_files": [...], "arc_files": [...]}"""


# ============================================================================
# ZIP CREATION TYPES
# ============================================================================


class ZipCreationResult:
    """Result of ZIP creation process (currently unused, kept for compatibility)."""

    path: str
    data_offsets: list[OffsetPair]
    metadata_offsets: list[OffsetPair]
    collection_offset: OffsetPair


# ============================================================================
# VIRTUAL ZIP TYPES
# ============================================================================


class VirtualFileInfo:
    """Information about a virtual file in VirtualTACOZIP."""

    src_path: str | None
    arc_path: str
    file_size: int
    lfh_offset: int
    lfh_size: int
    data_offset: int
    needs_zip64: bool


VirtualZipSummary: TypeAlias = dict[str, int | bool]
"""
Summary statistics from VirtualTACOZIP.

Keys: header_size, num_files, zip64_files, total_data_size, total_lfh_size, total_zip_size, needs_zip64
"""


# ============================================================================
# SAMPLE GROUPING TYPES
# ============================================================================

SampleChunk: TypeAlias = list  # list[Sample]
"""List of samples grouped together for splitting"""

SampleChunks: TypeAlias = list[list]  # list[list[Sample]]
"""Multiple chunks of samples for multi-part ZIPs"""


# ============================================================================
# VALIDATION TYPES
# ============================================================================

ValidationErrors: TypeAlias = list[str]
"""List of validation error messages"""