from typing import TypeAlias, TypedDict

import polars as pl

# Core offset types
OffsetPair: TypeAlias = tuple[int, int]  # (offset, size)
FilePair: TypeAlias = tuple[str, str]  # (src_path, arc_path)

# Lookup dictionaries
DataLookup: TypeAlias = dict[str, OffsetPair]  # sample_id -> (offset, size)
HierarchyLookup: TypeAlias = dict[str, OffsetPair]  # tortilla_id -> (offset, size)


# PIT Schema types
class PITRootLevel(TypedDict):
    """Root level of PIT schema (level 0 - the collection)."""

    n: int  # Number of items in collection
    type: str  # Node type: "TORTILLA" or "SAMPLE"


class PITHierarchyLevel(TypedDict):
    """Hierarchy level of PIT schema (levels 1+)."""

    depth: int  # Level depth (1, 2, 3, ...)
    n: int  # Total number of nodes at this level
    children: list[str]  # Pattern of child types for each parent


class PITSchema(TypedDict):
    """Position-Isomorphic Tree schema for deterministic navigation."""

    root: PITRootLevel
    hierarchy: list[PITHierarchyLevel]


# Metadata bundle per level
class LevelMetadata(TypedDict):
    """Metadata for a single hierarchy level."""

    dataframe: pl.DataFrame


# Complete metadata package
class MetadataPackage(TypedDict):
    """Complete metadata bundle for TACO creation."""

    levels: list[LevelMetadata]
    collection: dict[str, object]  # COLLECTION.json content
    max_depth: int
    pit_schema: PITSchema  # PIT schema for deterministic navigation


# File extraction results
class ExtractedFiles(TypedDict):
    """Files extracted from samples for DATA/."""

    src_files: list[str]  # Original filesystem paths
    arc_files: list[str]  # Archive paths (DATA/...)


# ZIP creation results
class ZipCreationResult(TypedDict):
    """Result of ZIP creation process."""

    path: str
    data_offsets: list[OffsetPair]
    metadata_offsets: list[OffsetPair]
    collection_offset: OffsetPair