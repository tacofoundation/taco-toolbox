"""
Global constants for tacotoolbox.

Organized by: Metadata Columns, Core Fields, Hierarchy Limits, Padding,
Parquet Configuration, ZIP Format, FOLDER Format, TacoCat Format,
Reserved Names, Cloud Storage, Documentation Templates.
"""

# Metadata Columns (shared with tacoreader)
METADATA_CURRENT_ID = "internal:current_id"
"""Current sample index at this level (enables O(1) position lookups and relational queries)."""

METADATA_PARENT_ID = "internal:parent_id"
"""Parent sample index in previous level DataFrame (enables relational queries)."""

METADATA_OFFSET = "internal:offset"
"""Byte offset in container file where data starts. Only for ZIP/TACOCAT containers."""

METADATA_SIZE = "internal:size"
"""Size in bytes of the sample data. Derived from sample._size_bytes at construction."""

METADATA_RELATIVE_PATH = "internal:relative_path"
"""Relative path from DATA/ directory (for consolidated metadata only)."""

METADATA_COLUMNS_ORDER = [
    METADATA_CURRENT_ID,
    METADATA_PARENT_ID,
    METADATA_OFFSET,
    METADATA_SIZE,
    METADATA_RELATIVE_PATH,
]
"""Preferred order for internal:* columns at end of DataFrames."""


# Core Fields (shared with tacoreader)
SHARED_CORE_FIELDS = {"id", "type", "path"}
"""Core Sample fields that cannot be overwritten by extensions."""


# Field Descriptions
CORE_FIELD_DESCRIPTIONS: dict[str, str] = {
    "id": "Unique sample identifier within parent scope. Must be unique among siblings.",
    "type": "Sample type discriminator (FILE or FOLDER).",
}
"""
Core field descriptions for field_schema generation.
Note: 'path' is excluded as it's removed during container materialization.
"""

INTERNAL_FIELD_DESCRIPTIONS: dict[str, str] = {
    "internal:current_id": "Current sample position at this level (0-indexed). Enables O(1) random access and relational JOINs (ZIP, FOLDER, TACOCAT).",
    "internal:parent_id": "Foreign key referencing parent sample position in previous level (ZIP, FOLDER, TACOCAT).",
    "internal:offset": "Byte offset in container file where sample data begins. Used for GDAL /vsisubfile/ paths (ZIP, TACOCAT).",
    "internal:size": "Size in bytes of sample data. Derived from sample._size_bytes. Combined with offset for /vsisubfile/ in ZIP/TACOCAT (ZIP, FOLDER, TACOCAT).",
    "internal:gdal_vsi": "Complete GDAL Virtual File System path for direct data access (ZIP, FOLDER, TACOCAT).",
    "internal:source_file": "Original source tacozip filename in TACOCAT consolidated datasets. Disambiguates samples from multiple sources (only TACOCAT).",
    "internal:relative_path": "Relative path from DATA/ directory. Format: {parent_path}/{id} or {id} for level0 (ZIP, FOLDER, TACOCAT).",
}
"""Internal field descriptions for field_schema generation."""


# Hierarchy Limits (shared with tacoreader)
SHARED_MAX_DEPTH = 5
"""Maximum hierarchy depth (0-5 means 6 levels total)."""

SHARED_MAX_LEVELS = 6
"""
Total number of possible levels (0 through 5) plus COLLECTION.json location.
In TACO containers, we have 5 metadata levels (level0-level5) plus the
COLLECTION.json entry, making 6 entries total in the TACO_HEADER.
"""


# Padding (shared with tacoreader)
PADDING_PREFIX = "__TACOPAD__"
"""Prefix for auto-generated padding sample IDs."""


# Parquet Configuration
PARQUET_CDC_DEFAULT_CONFIG = {
    "compression": "zstd",
    "compression_level": 13,
    "use_content_defined_chunking": True,
    "row_group_size": 65_536,
    "write_statistics": True,
}
"""
Default Parquet config with CDC enabled for consolidated metadata.

Content-Defined Chunking (CDC) ensures consistent data page boundaries for
efficient deduplication on content-addressable storage systems.

CRITICAL: This config is for PyArrow's pq.write_table(), NOT Polars df.write_parquet().
Both ZIP and FOLDER writers must use PyArrow directly:
    arrow_table = df.to_arrow()
    pq.write_table(arrow_table, path, **PARQUET_CDC_DEFAULT_CONFIG)

Parameters:
    compression: zstd for good compression ratio
    compression_level: 13 for balanced compression/speed
    use_content_defined_chunking: Enable CDC for deduplication
    row_group_size: 65_536 rows per row group
    write_statistics: Enable min/max/null_count for query pushdown

User-provided kwargs will override these defaults via merge:
    config = {**PARQUET_CDC_DEFAULT_CONFIG, **user_kwargs}
"""


# ZIP Container Format
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


# FOLDER Container Format
FOLDER_DATA_DIR = "DATA"
"""Directory name for data files in FOLDER container."""

FOLDER_METADATA_DIR = "METADATA"
"""Directory name for consolidated metadata in FOLDER container."""

FOLDER_META_FILENAME = "__meta__"
"""Filename for local metadata files in FOLDER container."""

FOLDER_COLLECTION_FILENAME = "COLLECTION.json"
"""Filename for collection metadata."""


# TacoCat Format (shared with tacoreader)
TACOCAT_FOLDER_NAME = ".tacocat"
"""Fixed folder name for TACOCAT consolidated datasets."""

TACOCAT_MAX_LEVELS = 6
"""
Maximum number of levels in TACOCAT datasets.
Structure: level0.parquet through level5.parquet (if they exist).
"""

TACOCAT_DEFAULT_PARQUET_CONFIG = {
    "compression": "zstd",
    "compression_level": 19,
    "row_group_size": 65_536,
    "write_statistics": True,
    "use_content_defined_chunking": True,
}
"""
Default Parquet config for TACOCAT with CDC and optimized compression.

Higher compression_level (19) than standard config since TACOCAT files
are written once and read many times.
"""


# Reserved Folder Names (shared with tacoreader)
RESERVED_FOLDER_NAMES = frozenset(
    {TACOCAT_FOLDER_NAME, FOLDER_DATA_DIR, FOLDER_METADATA_DIR}
)
"""
Folder names reserved by TACO specification.

These names cannot be used as output folder names in FOLDER format:
- .tacocat: Reserved for TacoCat consolidated format
- DATA: Reserved for FOLDER format data directory
- METADATA: Reserved for FOLDER format metadata directory

Using any of these names would conflict with TACO container structure.
"""


# File Paths (shared with tacoreader)
SHARED_DATA_PREFIX = "DATA/"
"""Prefix for data files in archive paths."""

SHARED_METADATA_PREFIX = "METADATA/"
"""Prefix for metadata files in archive paths."""


# Cloud Storage & Protocols (shared with tacoreader)
PROTOCOL_MAPPINGS = {
    "s3": {"standard": "s3://", "vsi": "/vsis3/"},
    "gcs": {"standard": "gs://", "vsi": "/vsigs/"},
    "azure": {"standard": "az://", "vsi": "/vsiaz/", "alt": "azure://"},
    "oss": {"standard": "oss://", "vsi": "/vsioss/"},
    "swift": {"standard": "swift://", "vsi": "/vsiswift/"},
    "http": {"standard": "http://", "vsi": "/vsicurl/"},
    "https": {"standard": "https://", "vsi": "/vsicurl/"},
}
"""
Unified protocol mappings for cloud storage and GDAL VSI.

Maps storage protocols to their standard URL scheme and GDAL VSI prefix.
'alt' key provides alternative protocol names (e.g., azure:// vs az://).
"""


# Documentation Templates & Assets
DOCS_TEMPLATE_HTML = "collection.html"
"""Jinja2 template for interactive HTML documentation."""

DOCS_TEMPLATE_MD = "collection.md"
"""Jinja2 template for Markdown documentation."""

DOCS_CSS_FILE = "styles.css"
"""CSS stylesheet for HTML documentation."""

DOCS_JS_PIT = "pit_graph.js"
"""JavaScript for PIT graph visualization (D3.js)."""

DOCS_JS_MAP = "map.js"
"""JavaScript for spatial coverage map (Leaflet)."""

DOCS_JS_UI = "ui-interactions.js"
"""JavaScript for UI interactions (tabs, copy buttons, syntax highlighting)."""
