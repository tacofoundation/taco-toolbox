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
│  METADATA COLUMNS (all levels):                                             │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │ Standard columns from original tacozip files, plus:                │     │
│  │ internal:source_file : str  │  Source tacozip filename             │     │
│  │                                                                     │     │
│  │ Original internal:offset and internal:size are preserved.          │     │
│  │ GDAL VSI construction:                                              │     │
│  │ /vsisubfile/{offset}_{size},{base_path}/{internal:source_file}     │     │
│  │                                                                     │     │
│  │ Example:                                                            │     │
│  │ /vsisubfile/1024_5000,/vsis3/bucket/base_dir/part0001.tacozip      │     │
│  └────────────────────────────────────────────────────────────────────┘     │
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
    TACOCAT_FILENAME,
    TACOCAT_MAGIC,
    TACOCAT_MAX_LEVELS,
    TACOCAT_TOTAL_HEADER_SIZE,
    TACOCAT_VERSION,
)


class TacoCatError(Exception):
    """Base exception for TacoCat operations."""


class SchemaValidationError(TacoCatError):
    """Raised when dataset schemas don't match."""


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
        output_path: Output directory where __TACOCAT__ will be created
        parquet_kwargs: Custom Parquet writer parameters (overrides defaults)
        validate_schema: If True, validate all datasets have same schema per level
        quiet: If True, suppress progress output

    Raises:
        TacoCatError: If no datasets provided or file operations fail
        ValueError: If output_path is a file or has a file extension
        SchemaValidationError: If validate_schema=True and schemas don't match

    Default parquet_kwargs (optimized for DuckDB):
        compression: "zstd"
        compression_level: 13
        row_group_size: 122_880
        statistics: True

    Example:
        >>> create_tacocat(
        ...     tacozips=["dataset1.tacozip", "dataset2.tacozip"],
        ...     output_path="/data/archive/"
        ... )

        >>> create_tacocat(
        ...     tacozips=list(Path("data/").glob("*.tacozip")),
        ...     output_path="/output/consolidated/",
        ...     parquet_kwargs={"compression_level": 22}
        ... )

        >>> create_tacocat(
        ...     tacozips=my_datasets,
        ...     output_path="/quick/dir/",
        ...     parquet_kwargs={"compression_level": 3}
        ... )
    """
    output_dir = Path(output_path)
    
    if output_dir.exists() and output_dir.is_file():
        raise ValueError(
            f"output_path must be a directory, not a file: {output_path}\n"
            f"Example: create_tacocat(datasets, '/data/output_dir/')"
        )
    
    if output_dir.suffix:
        raise ValueError(
            f"output_path should not have a file extension: {output_path}\n"
            f"The file will be automatically named '{TACOCAT_FILENAME}'\n"
            f"Example: create_tacocat(datasets, '/data/output_dir/')"
        )
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    final_output = output_dir / TACOCAT_FILENAME
    
    writer = TacoCatWriter(
        output_path=final_output,
        parquet_kwargs=parquet_kwargs,
        quiet=quiet,
    )

    for tacozip_path in tacozips:
        writer.add_dataset(tacozip_path)

    writer.write(validate_schema=validate_schema)


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
            output_path: Full path to __TACOCAT__ file (typically set by create_tacocat)
            parquet_kwargs: Custom Parquet writer parameters
            quiet: Suppress progress output
        """
        self.output_path = Path(output_path)
        self.datasets: list[Path] = []
        self.max_depth = 0
        self.quiet = quiet

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

        levels_bytes = self._consolidate_parquet_files(validate_schema)
        collection_bytes = self._merge_collections()
        offsets = self._calculate_offsets(levels_bytes, collection_bytes)
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
        Adds internal:source_file column to track original tacozip.

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

            try:
                entries = tacozip.read_header(str(dataset_path))
            except Exception as e:
                raise TacoCatError(f"Failed to read header from {dataset_path}: {e}")

            with open(dataset_path, "rb") as f:
                for level_idx, (offset, size) in enumerate(entries[:-1]):
                    level = level_idx

                    if size == 0:
                        continue

                    self.max_depth = max(self.max_depth, level)

                    f.seek(offset)
                    parquet_bytes = f.read(size)
                    df = pl.read_parquet(BytesIO(parquet_bytes))

                    df = df.with_columns(
                        pl.lit(dataset_path.name).alias("internal:source_file")
                    )

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

                    if level not in levels_data:
                        levels_data[level] = []

                    levels_data[level].append(df)

        levels_bytes = {}

        for level in sorted(levels_data.keys()):
            if not self.quiet:
                print(
                    f"  Consolidating level {level} ({len(levels_data[level])} DataFrames)..."
                )

            consolidated_df = pl.concat(levels_data[level], how="vertical")

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

                collection_offset, collection_size = entries[-1]

                if collection_size == 0:
                    raise TacoCatError(f"Empty collection in {dataset_path.name}")

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

        merged_collection = collections[0]

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
        current_offset = TACOCAT_TOTAL_HEADER_SIZE

        for level in range(TACOCAT_MAX_LEVELS):
            if level in levels_bytes:
                size = len(levels_bytes[level])
                offsets[level] = (current_offset, size)
                current_offset += size
            else:
                offsets[level] = (0, 0)

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
            f.write(TACOCAT_MAGIC)
            f.write(struct.pack("<I", TACOCAT_VERSION))
            f.write(struct.pack("<I", self.max_depth))

            for level in range(TACOCAT_MAX_LEVELS):
                offset, size = offsets[level]
                f.write(struct.pack("<Q", offset))
                f.write(struct.pack("<Q", size))

            offset, size = offsets["collection"]
            f.write(struct.pack("<Q", offset))
            f.write(struct.pack("<Q", size))

            for level in sorted(levels_bytes.keys()):
                f.write(levels_bytes[level])

            f.write(collection_bytes)