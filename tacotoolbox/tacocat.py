"""
TacoCat Writer - Consolidate multiple TACO datasets into single high-performance file.

TacoCat is a custom binary format designed to solve the "many small files" problem
in large-scale Earth Observation datasets. Instead of having hundreds of individual
TACO datasets (each with their own level0.parquet, level1.parquet, etc.), TacoCat
consolidates all datasets into a single file with optimal query performance.

┌─────────────────────────────────────────────────────────────────────────────┐
│                        TACOCAT FILE FORMAT SPECIFICATION                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  HEADER (16 bytes) - FIXED                                                  │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │ MAGIC_NUMBER     : 8 bytes  │  b"TACOCAT\x00"                      │     │
│  │ VERSION          : 4 bytes  │  uint32 = 1                          │     │
│  │ MAX_DEPTH        : 4 bytes  │  uint32 = 0-5 (maximum level depth)  │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  INDEX BLOCK (112 bytes) - FIXED (7 entries x 16 bytes each)                │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │ LEVEL0_OFFSET    : 8 bytes  │  uint64 (byte offset in file)        │     │
│  │ LEVEL0_SIZE      : 8 bytes  │  uint64 (size in bytes)              │     │
│  ├────────────────────────────────────────────────────────────────────┤     │
│  │ LEVEL1_OFFSET    : 8 bytes  │  uint64 (FREE if MAX_DEPTH < 1)      │     │
│  │ LEVEL1_SIZE      : 8 bytes  │  uint64 (FREE if MAX_DEPTH < 1)      │     │
│  ├────────────────────────────────────────────────────────────────────┤     │
│  │ LEVEL2_OFFSET    : 8 bytes  │  uint64 (FREE if MAX_DEPTH < 2)      │     │
│  │ LEVEL2_SIZE      : 8 bytes  │  uint64 (FREE if MAX_DEPTH < 2)      │     │
│  ├────────────────────────────────────────────────────────────────────┤     │
│  │ LEVEL3_OFFSET    : 8 bytes  │  uint64 (FREE if MAX_DEPTH < 3)      │     │
│  │ LEVEL3_SIZE      : 8 bytes  │  uint64 (FREE if MAX_DEPTH < 3)      │     │
│  ├────────────────────────────────────────────────────────────────────┤     │
│  │ LEVEL4_OFFSET    : 8 bytes  │  uint64 (FREE if MAX_DEPTH < 4)      │     │
│  │ LEVEL4_SIZE      : 8 bytes  │  uint64 (FREE if MAX_DEPTH < 4)      │     │
│  ├────────────────────────────────────────────────────────────────────┤     │
│  │ LEVEL5_OFFSET    : 8 bytes  │  uint64 (FREE if MAX_DEPTH < 5)      │     │
│  │ LEVEL5_SIZE      : 8 bytes  │  uint64 (FREE if MAX_DEPTH < 5)      │     │
│  ├────────────────────────────────────────────────────────────────────┤     │
│  │ COLLECTION_OFFSET: 8 bytes  │  uint64 (always present)             │     │
│  │ COLLECTION_SIZE  : 8 bytes  │  uint64 (always present)             │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  DATA SECTION (variable size) - STARTS AT BYTE 128                          │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │ CONSOLIDATED_LEVEL0.parquet  │  All level0 DataFrames concatenated │     │
│  ├────────────────────────────────────────────────────────────────────┤     │
│  │ CONSOLIDATED_LEVEL1.parquet  │  All level1 DataFrames concatenated │     │
│  ├────────────────────────────────────────────────────────────────────┤     │
│  │ CONSOLIDATED_LEVEL2.parquet  │  All level2 DataFrames concatenated │     │
│  ├────────────────────────────────────────────────────────────────────┤     │
│  │ ...                          │  (only if levels exist)             │     │
│  ├────────────────────────────────────────────────────────────────────┤     │
│  │ COLLECTION.json              │  Merged collection metadata         │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""

import json
import struct
from io import BytesIO
from pathlib import Path
from typing import Any

import polars as pl
import tacozip

from tacotoolbox._constants import (
    TACOCAT_DEFAULT_PARQUET_CONFIG,
    TACOCAT_MAGIC,
    TACOCAT_MAX_LEVELS,
    TACOCAT_TOTAL_HEADER_SIZE,
    TACOCAT_VERSION,
)

# ============================================================================
# EXCEPTIONS
# ============================================================================


class TacoCatError(Exception):
    """Base exception for TacoCat operations."""


class SchemaValidationError(TacoCatError):
    """Raised when dataset schemas don't match."""


# ============================================================================
# MAIN API
# ============================================================================


def create_tacocat(
    tacozips: list[str | Path],
    output_path: str | Path,
    parquet_kwargs: dict[str, Any] | None = None,
    validate_schema: bool = True,
    quiet: bool = True,
) -> None:
    """
    Create TacoCat file from multiple TACO datasets.

    Consolidates multiple .tacozip files into a single high-performance
    TacoCat file optimized for DuckDB queries.

    Args:
        tacozips: List of .tacozip file paths to consolidate
        output_path: Output path for .tacocat file
        parquet_kwargs: Custom Parquet writer parameters (overrides defaults)
        validate_schema: If True, validate all datasets have same schema per level
        quiet: If True, suppress progress output

    Raises:
        TacoCatError: If no datasets provided or file operations fail
        SchemaValidationError: If validate_schema=True and schemas don't match

    Default parquet_kwargs (optimized for DuckDB):
        compression: "zstd"
        compression_level: 19  # High compression for archival
        row_group_size: 122_880  # DuckDB default
        statistics: True  # Enable for pushdown optimization

    Example:
        >>> # Basic usage with defaults
        >>> create_tacocat(
        ...     tacozips=["dataset1.tacozip", "dataset2.tacozip"],
        ...     output_path="consolidated.tacocat"
        ... )

        >>> # Maximum compression for cold storage
        >>> create_tacocat(
        ...     tacozips=list(Path("data/").glob("*.tacozip")),
        ...     output_path="archive.tacocat",
        ...     parquet_kwargs={"compression_level": 22}
        ... )

        >>> # Faster writes, less compression
        >>> create_tacocat(
        ...     tacozips=my_datasets,
        ...     output_path="quick.tacocat",
        ...     parquet_kwargs={"compression_level": 3}
        ... )
    """
    writer = TacoCatWriter(
        output_path=output_path,
        parquet_kwargs=parquet_kwargs,
        quiet=quiet,
    )

    for tacozip in tacozips:
        writer.add_dataset(tacozip)

    writer.write(validate_schema=validate_schema)


# ============================================================================
# WRITER CLASS
# ============================================================================


class TacoCatWriter:
    """Writer for TacoCat format - consolidates multiple TACO datasets."""

    def __init__(
        self,
        output_path: str | Path,
        parquet_kwargs: dict[str, Any] | None = None,
        quiet: bool = False,
    ):
        """
        Initialize TacoCat writer.

        Args:
            output_path: Output path for .tacocat file
            parquet_kwargs: Custom Parquet writer parameters
            quiet: Suppress progress output
        """
        self.output_path = Path(output_path)
        self.datasets: list[Path] = []
        self.max_depth = 0
        self.quiet = quiet

        # Merge user kwargs with defaults
        self.parquet_config = TACOCAT_DEFAULT_PARQUET_CONFIG.copy()
        if parquet_kwargs:
            self.parquet_config.update(parquet_kwargs)

    def add_dataset(self, tacozip_path: str | Path) -> None:
        """
        Add a TACO dataset to be consolidated.

        Args:
            tacozip_path: Path to TACO file

        Raises:
            FileNotFoundError: If dataset file doesn't exist
            TacoCatError: If file is not readable or invalid format
        """
        path = Path(tacozip_path)

        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        if not path.is_file():
            raise TacoCatError(f"Path is not a file: {path}")

        if path.stat().st_size == 0:
            raise TacoCatError(f"Dataset file is empty: {path}")

        # Try to read header to validate it's a proper TACO file
        try:
            tacozip.read_header(str(path))
        except Exception as e:
            raise TacoCatError(
                f"Invalid TACO file (cannot read header): {path}\n  Error: {e}"
            )

        self.datasets.append(path)

    def write(self, validate_schema: bool = True) -> None:
        """
        Write consolidated TacoCat file.

        Args:
            validate_schema: If True, validate all datasets have same schema

        Raises:
            TacoCatError: If no datasets added
            SchemaValidationError: If schema validation fails
        """
        if not self.datasets:
            raise TacoCatError("No datasets added. Use add_dataset() first.")

        if not self.quiet:
            print(f"Consolidating {len(self.datasets)} TACO datasets into TacoCat...")

        # Step 1: Consolidate Parquet files per level
        levels_bytes = self._consolidate_parquet_files(validate_schema)

        # Step 2: Merge collection metadata
        collection_bytes = self._merge_collections()

        # Step 3: Calculate offsets
        offsets = self._calculate_offsets(levels_bytes, collection_bytes)

        # Step 4: Write TacoCat file
        self._write_file(offsets, levels_bytes, collection_bytes)

        if not self.quiet:
            file_size_gb = self.output_path.stat().st_size / (1024**3)
            print(f"TacoCat file created: {self.output_path}")
            print(f"   Datasets: {len(self.datasets)}")
            print(f"   Max depth: {self.max_depth}")
            print(f"   File size: {file_size_gb:.2f} GB")

    def _consolidate_parquet_files(self, validate_schema: bool) -> dict[int, bytes]:
        """
        Consolidate Parquet files from all datasets by level using direct reads.

        Uses tacozip.read_header() to get offsets, then reads directly from file.

        Returns:
            Dictionary mapping level -> consolidated parquet bytes
        """
        levels_data: dict[int, list[pl.DataFrame]] = {}
        reference_schemas: dict[int, dict] = {}

        for idx, dataset_path in enumerate(self.datasets):
            if not self.quiet:
                print(
                    f"  [{idx+1}/{len(self.datasets)}] Processing {dataset_path.name}"
                )

            # Read TACO_HEADER to get offsets
            try:
                entries = tacozip.read_header(str(dataset_path))
            except Exception as e:
                raise TacoCatError(f"Failed to read header from {dataset_path}: {e}")

            # Read parquet files directly using offsets
            # NOTE: Last entry is COLLECTION.json, so iterate entries[:-1]
            with open(dataset_path, "rb") as f:
                for level_idx, (offset, size) in enumerate(entries[:-1]):
                    level = level_idx

                    # Skip if size is 0 (level doesn't exist)
                    if size == 0:
                        continue

                    # Update max depth
                    self.max_depth = max(self.max_depth, level)

                    # Read parquet bytes directly
                    f.seek(offset)
                    parquet_bytes = f.read(size)
                    df = pl.read_parquet(BytesIO(parquet_bytes))

                    # Schema validation
                    if validate_schema:
                        current_schema = dict(df.schema)

                        if level in reference_schemas:
                            if current_schema != reference_schemas[level]:
                                raise SchemaValidationError(
                                    f"Schema mismatch at level {level} in {dataset_path.name}\n"
                                    f"Expected columns: {list(reference_schemas[level].keys())}\n"
                                    f"Got columns: {list(current_schema.keys())}"
                                )
                        else:
                            reference_schemas[level] = current_schema

                    # Store DataFrame
                    if level not in levels_data:
                        levels_data[level] = []

                    levels_data[level].append(df)

        # Consolidate DataFrames per level and write to bytes
        levels_bytes = {}

        for level in sorted(levels_data.keys()):
            if not self.quiet:
                print(
                    f"  Consolidating level {level} ({len(levels_data[level])} DataFrames)..."
                )

            # Concatenate all DataFrames for this level
            consolidated_df = pl.concat(levels_data[level], how="vertical")

            # Write to bytes with optimized Parquet config
            buffer = BytesIO()
            consolidated_df.write_parquet(buffer, **self.parquet_config)
            levels_bytes[level] = buffer.getvalue()

            if not self.quiet:
                size_mb = len(levels_bytes[level]) / (1024**2)
                print(
                    f"     Level {level}: {len(consolidated_df)} rows, {size_mb:.2f} MB"
                )

        return levels_bytes

    def _merge_collections(self) -> bytes:
        """
        Merge COLLECTION.json from all datasets using direct reads.

        COLLECTION.json is always the last entry in TACO_HEADER.

        Returns:
            JSON bytes for merged collection

        Raises:
            TacoCatError: If no collections could be read
        """
        collections = []

        for dataset_path in self.datasets:
            try:
                entries = tacozip.read_header(str(dataset_path))

                if len(entries) == 0:
                    raise TacoCatError(f"Empty header in {dataset_path.name}")

                # COLLECTION.json is always the LAST entry
                collection_offset, collection_size = entries[-1]

                if collection_size == 0:
                    raise TacoCatError(f"Empty collection in {dataset_path.name}")

                # Read COLLECTION.json directly
                with open(dataset_path, "rb") as f:
                    f.seek(collection_offset)
                    collection_bytes = f.read(collection_size)
                    collection = json.loads(collection_bytes)
                    collections.append(collection)
            except Exception as e:
                raise TacoCatError(
                    f"Failed to read collection from {dataset_path.name}: {e}"
                )

        if not collections:
            raise TacoCatError("No valid collections found in any dataset")

        # Take first collection as base
        merged_collection = collections[0]

        # Add consolidation metadata
        merged_collection["_tacocat"] = {
            "version": TACOCAT_VERSION,
            "num_datasets": len(self.datasets),
            "dataset_ids": [c.get("id", "unknown") for c in collections],
        }

        return json.dumps(merged_collection, indent=2).encode("utf-8")

    def _calculate_offsets(
        self, levels_bytes: dict[int, bytes], collection_bytes: bytes
    ) -> dict[int | str, tuple[int, int]]:
        """
        Calculate byte offsets for all data sections.

        Args:
            levels_bytes: Consolidated Parquet bytes per level
            collection_bytes: Collection JSON bytes

        Returns:
            Dictionary mapping level/collection -> (offset, size)
        """
        offsets = {}
        current_offset = TACOCAT_TOTAL_HEADER_SIZE  # Start at byte 128

        # Calculate offsets for each level (0-5)
        for level in range(TACOCAT_MAX_LEVELS):
            if level in levels_bytes:
                size = len(levels_bytes[level])
                offsets[level] = (current_offset, size)
                current_offset += size
            else:
                # Level doesn't exist - mark as FREE
                offsets[level] = (0, 0)

        # Collection offset (always present)
        offsets["collection"] = (current_offset, len(collection_bytes))

        return offsets

    def _write_file(
        self,
        offsets: dict[int | str, tuple[int, int]],
        levels_bytes: dict[int, bytes],
        collection_bytes: bytes,
    ) -> None:
        """
        Write final TacoCat file with header, index, and data sections.

        Args:
            offsets: Byte offsets for all sections
            levels_bytes: Consolidated Parquet bytes per level
            collection_bytes: Collection JSON bytes
        """
        with open(self.output_path, "wb") as f:
            # ===== HEADER (16 bytes) =====
            f.write(TACOCAT_MAGIC)  # 8 bytes
            f.write(struct.pack("<I", TACOCAT_VERSION))  # 4 bytes
            f.write(struct.pack("<I", self.max_depth))  # 4 bytes

            # ===== INDEX BLOCK (112 bytes) =====
            # Write 7 entries: level0-5 + collection (each 16 bytes)
            for level in range(TACOCAT_MAX_LEVELS):
                offset, size = offsets[level]
                f.write(struct.pack("<Q", offset))  # 8 bytes - offset
                f.write(struct.pack("<Q", size))  # 8 bytes - size

            # Collection entry
            offset, size = offsets["collection"]
            f.write(struct.pack("<Q", offset))  # 8 bytes
            f.write(struct.pack("<Q", size))  # 8 bytes

            # ===== DATA SECTION (variable) =====
            # Write consolidated Parquet files in order
            for level in sorted(levels_bytes.keys()):
                f.write(levels_bytes[level])

            # Write collection JSON
            f.write(collection_bytes)
