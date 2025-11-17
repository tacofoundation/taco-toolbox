"""
Type definitions and aliases for tacotoolbox.

Provides type hints and type aliases used throughout the TACO creation
and manipulation codebase. Includes types for:
- Offset pairs and file mappings
- PIT (Position-Isomorphic Tree) schema structures
- Metadata packages and field schemas
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
# METADATA PACKAGE TYPES
# ============================================================================


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
