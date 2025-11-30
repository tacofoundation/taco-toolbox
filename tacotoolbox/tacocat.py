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
│  │                                                                    │     │
│  │ Original internal:offset and internal:size are preserved.          │     │
│  │ GDAL VSI construction:                                             │     │
│  │ /vsisubfile/{offset}_{size},{base_path}/{internal:source_file}     │     │
│  │                                                                    │     │
│  │ Example:                                                           │     │
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
import pyarrow.parquet as pq
import tacozip

from tacotoolbox._column_utils import align_dataframe_schemas
from tacotoolbox._constants import (
    TACOCAT_FILENAME,
    TACOCAT_MAGIC,
    TACOCAT_MAX_LEVELS,
    TACOCAT_TOTAL_HEADER_SIZE,
    TACOCAT_VERSION,
)
from tacotoolbox._logging import get_logger

logger = get_logger(__name__)


class TacoCatError(Exception):
    """Base exception for TacoCat operations."""


class SchemaValidationError(TacoCatError):
    """Raised when dataset schemas don't match."""


def _estimate_row_group_size(num_rows: int, num_cols: int) -> int:
    """
    Estimate optimal row_group_size for Parquet files.

    Target: ~128MB uncompressed row groups for optimal DuckDB performance.
    Assumes ~1KB average row size (conservative estimate).

    Args:
        num_rows: Total number of rows in DataFrame
        num_cols: Number of columns in DataFrame

    Returns:
        Optimal row_group_size (capped between 10K and 1M rows)
    """
    estimated_row_bytes = max(100 * num_cols, 500)

    target_row_group_bytes = 128 * 1024 * 1024
    estimated_row_group_size = target_row_group_bytes // estimated_row_bytes

    row_group_size = max(10_000, min(estimated_row_group_size, 1_000_000))

    if num_rows < 100_000:
        row_group_size = min(row_group_size, num_rows)

    return int(row_group_size)


def _get_default_parquet_config() -> dict[str, Any]:
    """
    Get default Parquet configuration optimized for DuckDB queries.

    Returns:
        PyArrow Parquet writer configuration
    """
    return {
        "compression": "zstd",
        "compression_level": 19,
        "use_dictionary": True,
        "write_statistics": True,
        "use_content_defined_chunking": True,
    }


def _detect_common_directory(inputs: list[str | Path]) -> Path:
    """
    Detect common parent directory for all input files.

    All input files must be in the same directory for auto-detection.
    Otherwise raises TacoCatError.
    """
    input_paths = [Path(p).resolve() for p in inputs]
    parent_dirs = [p.parent for p in input_paths]

    first_parent = parent_dirs[0]

    if not all(parent == first_parent for parent in parent_dirs):
        raise TacoCatError(
            "Input files are in different directories. "
            "Please specify output directory explicitly."
        )

    return first_parent


def create_tacocat(
    inputs: list[str | Path],
    output: str | Path | None = None,
    parquet_kwargs: dict[str, Any] | None = None,
    validate_schema: bool = True,
) -> None:
    """
    Create TacoCat file from multiple TACO datasets.

    Consolidates multiple .tacozip files into a single high-performance
    TacoCat file optimized for DuckDB queries.

    If output is not specified, all input files must be in the same directory
    and the TacoCat file will be created there.

    Default parquet_kwargs (optimized for DuckDB, uses PyArrow):
        compression: "zstd"
        compression_level: 19
        use_dictionary: True
        write_statistics: True
        use_content_defined_chunking: True
        row_group_size: auto-estimated (10K-1M rows, ~128MB target)

    Args:
        inputs: List of .tacozip file paths to consolidate
        output: Output directory path (file will be named 'tacocat')
        parquet_kwargs: Optional PyArrow Parquet writer parameters to override defaults
        validate_schema: If True, validate schemas match across datasets
    """
    if output is None:
        output_dir = _detect_common_directory(inputs)
    else:
        output_dir = Path(output)

    if output_dir.exists() and output_dir.is_file():
        raise ValueError(
            f"output must be a directory, not a file: {output}\n"
            f"Example: create_tacocat(datasets, '/data/output_dir/')"
        )

    if output_dir.suffix:
        raise ValueError(
            f"output should not have a file extension: {output}\n"
            f"The file will be automatically named '{TACOCAT_FILENAME}'\n"
            f"Example: create_tacocat(datasets, '/data/output_dir/')"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    final_output = output_dir / TACOCAT_FILENAME

    writer = TacoCatWriter(
        output_path=final_output,
        parquet_kwargs=parquet_kwargs,
    )

    for tacozip_path in inputs:
        writer.add_dataset(tacozip_path)

    writer.write(validate_schema=validate_schema)


class TacoCatWriter:
    """Writer for TacoCat format - consolidates multiple TACO datasets."""

    def __init__(
        self,
        output_path: str | Path,
        parquet_kwargs: dict[str, Any] | None = None,
    ):
        """
        Initialize TacoCat writer.

        Args:
            output_path: Path to output tacocat file
            parquet_kwargs: Optional PyArrow Parquet writer parameters
        """
        self.output_path = Path(output_path)
        self.datasets: list[Path] = []
        self.max_depth = 0

        self.parquet_config = _get_default_parquet_config()
        if parquet_kwargs:
            self.parquet_config.update(parquet_kwargs)

    def add_dataset(self, tacozip_path: str | Path) -> None:
        """Add a TACO dataset to be consolidated."""
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
            ) from e

        self.datasets.append(path)

    def write(self, validate_schema: bool = True) -> None:
        """Write consolidated TacoCat file."""
        if not self.datasets:
            raise TacoCatError("No datasets added. Use add_dataset() first.")

        logger.info(f"Consolidating {len(self.datasets)} TACO datasets into TacoCat...")

        levels_bytes, consolidated_counts = self._consolidate_parquet_files(
            validate_schema
        )
        collection_bytes = self._merge_collections(consolidated_counts)
        offsets = self._calculate_offsets(levels_bytes, collection_bytes)
        self._write_file(offsets, levels_bytes, collection_bytes)

        file_size_gb = self.output_path.stat().st_size / (1024**3)
        logger.info(f"TacoCat file created: {self.output_path}")
        logger.info(f"   Datasets: {len(self.datasets)}")
        logger.info(f"   Max depth: {self.max_depth}")
        logger.info(f"   File size: {file_size_gb:.2f} GB")

    def _consolidate_parquet_files(
        self, validate_schema: bool
    ) -> tuple[dict[int, bytes], dict[int, int]]:
        """
        Consolidate Parquet files from all datasets by level using direct reads.

        Uses tacozip.read_header() to get offsets, then reads directly from file.
        Adds internal:source_file column to track original tacozip.

        CRITICAL: Different tacozips may have samples with different extensions
        (e.g., dataset1 has geotiff:stats, dataset2 has scaling:scale_factor).
        Must align schemas before concatenation to avoid ShapeError.
        """
        levels_data: dict[int, list[pl.DataFrame]] = {}
        reference_schemas: dict[int, dict] = {}

        for idx, dataset_path in enumerate(self.datasets):
            logger.info(
                f"  [{idx+1}/{len(self.datasets)}] Processing {dataset_path.name}"
            )

            self._process_single_dataset(
                dataset_path, levels_data, reference_schemas, validate_schema
            )

        return self._consolidate_levels(levels_data)

    def _process_single_dataset(
        self,
        dataset_path: Path,
        levels_data: dict[int, list[pl.DataFrame]],
        reference_schemas: dict[int, dict],
        validate_schema: bool,
    ) -> None:
        """Process a single dataset and add its DataFrames to levels_data."""
        try:
            entries = tacozip.read_header(str(dataset_path))
        except Exception as e:
            raise TacoCatError(f"Failed to read header from {dataset_path}: {e}") from e

        with open(dataset_path, "rb") as f:
            for level_idx, (offset, size) in enumerate(entries[:-1]):
                if size == 0:
                    continue

                level = level_idx
                self.max_depth = max(self.max_depth, level)

                df = self._read_level_dataframe(f, offset, size, dataset_path)

                if validate_schema:
                    self._validate_schema_for_level(
                        df, level, dataset_path, reference_schemas
                    )

                if level not in levels_data:
                    levels_data[level] = []

                levels_data[level].append(df)

    def _read_level_dataframe(
        self, f: Any, offset: int, size: int, dataset_path: Path
    ) -> pl.DataFrame:
        """Read a single level DataFrame from file."""
        f.seek(offset)
        parquet_bytes = f.read(size)
        df = pl.read_parquet(BytesIO(parquet_bytes))
        return df.with_columns(pl.lit(dataset_path.name).alias("internal:source_file"))

    def _validate_schema_for_level(
        self,
        df: pl.DataFrame,
        level: int,
        dataset_path: Path,
        reference_schemas: dict[int, dict],
    ) -> None:
        """Validate DataFrame schema against reference schema for level."""
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

    def _consolidate_levels(
        self, levels_data: dict[int, list[pl.DataFrame]]
    ) -> tuple[dict[int, bytes], dict[int, int]]:
        """
        Consolidate DataFrames by level and serialize to Parquet bytes using PyArrow.
        """
        levels_bytes = {}
        consolidated_counts = {}

        for level in sorted(levels_data.keys()):
            logger.info(
                f"  Consolidating level {level} ({len(levels_data[level])} DataFrames)..."
            )

            aligned_dfs = align_dataframe_schemas(levels_data[level])
            consolidated_df = pl.concat(aligned_dfs, how="vertical")
            consolidated_counts[level] = len(consolidated_df)

            num_rows = len(consolidated_df)
            num_cols = len(consolidated_df.columns)
            row_group_size = _estimate_row_group_size(num_rows, num_cols)

            arrow_table = consolidated_df.to_arrow()

            pq_config = self.parquet_config.copy()

            if "row_group_size" not in self.parquet_config:
                pq_config["row_group_size"] = row_group_size

            buffer = BytesIO()
            pq.write_table(arrow_table, buffer, **pq_config)
            levels_bytes[level] = buffer.getvalue()

            size_mb = len(levels_bytes[level]) / (1024**2)
            actual_row_group_size = pq_config.get("row_group_size", row_group_size)
            logger.info(
                f"     Level {level}: {num_rows:,} rows x {num_cols} cols = {size_mb:.2f} MB "
                f"(row_group_size={actual_row_group_size:,})"
            )

        return levels_bytes, consolidated_counts

    def _raise_empty_header_error(self, dataset_name: str) -> None:
        """Helper to raise empty header error."""
        raise TacoCatError(f"Empty header in {dataset_name}")

    def _raise_empty_collection_error(self, dataset_name: str) -> None:
        """Helper to raise empty collection error."""
        raise TacoCatError(f"Empty collection in {dataset_name}")

    def _merge_collections(self, consolidated_counts: dict[int, int]) -> bytes:
        """
        Merge COLLECTION.json from all datasets using direct reads.

        Updates the PIT schema counts to reflect the consolidated data.
        Stores dataset_sources (original .tacozip filenames) for provenance tracking.
        """
        collections = []

        for dataset_path in self.datasets:
            collection = self._read_single_collection(dataset_path)
            collections.append(collection)

        if not collections:
            raise TacoCatError("No valid collections found in any dataset")

        merged_collection = collections[0].copy()
        self._update_pit_schema_counts(merged_collection, consolidated_counts)
        self._add_tacocat_metadata(merged_collection)

        return json.dumps(merged_collection, indent=2).encode("utf-8")

    def _read_single_collection(self, dataset_path: Path) -> dict[str, Any]:
        """Read COLLECTION.json from a single dataset."""
        try:
            entries = tacozip.read_header(str(dataset_path))

            if len(entries) == 0:
                self._raise_empty_header_error(dataset_path.name)

            collection_offset, collection_size = entries[-1]

            if collection_size == 0:
                self._raise_empty_collection_error(dataset_path.name)

            with open(dataset_path, "rb") as f:
                f.seek(collection_offset)
                collection_bytes = f.read(collection_size)
                return json.loads(collection_bytes)
        except Exception as e:
            raise TacoCatError(
                f"Failed to read collection from {dataset_path.name}: {e}"
            ) from e

    def _update_pit_schema_counts(
        self, collection: dict[str, Any], consolidated_counts: dict[int, int]
    ) -> None:
        """Update PIT schema counts with consolidated values."""
        if "taco:pit_schema" not in collection:
            return

        pit_schema = collection["taco:pit_schema"]

        if 0 in consolidated_counts:
            pit_schema["root"]["n"] = consolidated_counts[0]

        if "hierarchy" in pit_schema:
            for depth_str, patterns in pit_schema["hierarchy"].items():
                depth = int(depth_str)
                if depth in consolidated_counts:
                    for pattern in patterns:
                        pattern["n"] = consolidated_counts[depth]

    def _add_tacocat_metadata(self, collection: dict[str, Any]) -> None:
        """Add TacoCat-specific metadata to collection."""
        collection["_tacocat"] = {
            "version": TACOCAT_VERSION,
            "num_datasets": len(self.datasets),
            "dataset_sources": sorted(
                [dataset_path.name for dataset_path in self.datasets]
            ),
        }

    def _calculate_offsets(
        self, levels_bytes: dict[int, bytes], collection_bytes: bytes
    ) -> dict[int | str, tuple[int, int]]:
        """Calculate byte offsets for all data sections."""
        offsets: dict[int | str, tuple[int, int]] = {}
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
        """Write final TacoCat file with header, index, and data sections."""
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