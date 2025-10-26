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
    """
    Root level of PIT schema (level 0 - the collection).
    
    Attributes:
        n: Number of items in collection
        type: Node type ("FILE" or "FOLDER")
    
    Example:
        >>> root = {"n": 100, "type": "FOLDER"}
    """
    n: int
    type: str


class PITPattern:
    """
    Pattern descriptor for a specific position in the hierarchy.
    
    Attributes:
        n: Total nodes at this depth for this pattern
        children: Ordered array of child types
    
    Example:
        >>> pattern = {"n": 200, "children": ["FILE", "FILE"]}
    """
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
            ...
        }
    }

Example:
    >>> pit_schema = {
    ...     "root": {"n": 3, "type": "FOLDER"},
    ...     "hierarchy": {
    ...         "1": [{"n": 6, "children": ["FILE", "FILE"]}]
    ...     }
    ... }
"""


# ============================================================================
# METADATA PACKAGE TYPES (NEW DUAL SYSTEM)
# ============================================================================

class LocalMetadata:
    """
    Metadata for a single folder (local __metadata__).
    
    Used for DATA/folder/__metadata__ files (level 1+ only).
    
    Attributes:
        folder_path: Path in ZIP (e.g., "DATA/folder_A/")
        samples: List of Sample objects in this folder
        metadata_df: DataFrame with metadata including internal:offset/size
    
    Example:
        >>> local = LocalMetadata(
        ...     folder_path="DATA/folder_A/",
        ...     samples=[sample1, sample2],
        ...     metadata_df=df_with_offsets
        ... )
    """
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
    It contains the index of the parent sample in the previous level's DataFrame,
    enabling relational queries in DuckDB and other databases.
    
    Attributes:
        levels: List of DataFrames for consolidated metadata (one per level 0-5)
        local_metadata: Dict mapping folder_path -> DataFrame for local metadata
        collection: COLLECTION.json content (dict)
        pit_schema: PIT schema for navigation
        max_depth: Maximum hierarchy depth (0-5, meaning 6 levels)
    
    Example:
        >>> pkg = MetadataPackage(
        ...     levels=[level0_df, level1_df],
        ...     local_metadata={
        ...         "DATA/folder_A/": folder_a_df,
        ...         "DATA/folder_B/": folder_b_df
        ...     },
        ...     collection={"id": "dataset", ...},
        ...     pit_schema={"root": {...}, "hierarchy": {...}},
        ...     max_depth=1
        ... )
        
        >>> # Query using internal:parent_id
        >>> import duckdb
        >>> duckdb.sql("
        ...     SELECT l2.id, l2.type 
        ...     FROM level1 l1
        ...     JOIN level2 l2 ON l2."internal:parent_id" = l1.rowid
        ...     WHERE l1."internal:parent_id" = 0
        ... ")
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
    """
    Files extracted from samples for DATA/.
    
    Attributes:
        src_files: Original filesystem paths (absolute)
        arc_files: Archive paths inside ZIP (e.g., "DATA/sample_001.tif")
    
    Example:
        >>> extracted = {
        ...     "src_files": ["/data/image1.tif", "/data/image2.tif"],
        ...     "arc_files": ["DATA/image1.tif", "DATA/image2.tif"]
        ... }
    """
    src_files: list[str]
    arc_files: list[str]


ExtractedFilesDict: TypeAlias = dict[str, list[str]]
"""Dictionary format for ExtractedFiles: {"src_files": [...], "arc_files": [...]}"""


# ============================================================================
# ZIP CREATION TYPES
# ============================================================================

class ZipCreationResult:
    """
    Result of ZIP creation process (currently unused, kept for compatibility).
    
    Attributes:
        path: Path to created ZIP file
        data_offsets: List of (offset, size) for data files
        metadata_offsets: List of (offset, size) for metadata files
        collection_offset: (offset, size) for COLLECTION.json
    """
    path: str
    data_offsets: list[OffsetPair]
    metadata_offsets: list[OffsetPair]
    collection_offset: OffsetPair


# ============================================================================
# VIRTUAL ZIP TYPES
# ============================================================================

class VirtualFileInfo:
    """
    Information about a virtual file in VirtualTACOZIP.
    
    Attributes:
        src_path: Source file path (or None for in-memory)
        arc_path: Archive path in ZIP
        file_size: Size in bytes
        lfh_offset: Local File Header offset
        lfh_size: Local File Header size
        data_offset: Actual data offset
        needs_zip64: Whether ZIP64 format is needed
    
    Example:
        >>> vfile = VirtualFileInfo(
        ...     src_path="/data/image.tif",
        ...     arc_path="DATA/image.tif",
        ...     file_size=100_000_000,
        ...     lfh_offset=157,
        ...     lfh_size=42,
        ...     data_offset=199,
        ...     needs_zip64=False
        ... )
    """
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

Keys:
    - header_size: Size of TACO_HEADER
    - num_files: Total number of files
    - zip64_files: Number of files using ZIP64
    - total_data_size: Sum of all file sizes
    - total_lfh_size: Sum of all LFH sizes
    - total_zip_size: Total ZIP size
    - needs_zip64: Whether ZIP64 format is needed
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