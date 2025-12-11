"""
TacoCat - Consolidate multiple TACO datasets into .tacocat/ folder.

Consolidates multiple .tacozip files into a single .tacocat/ folder with
unified parquet files optimized for DuckDB queries plus COLLECTION.json metadata.
"""

from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import tacozip

from tacotoolbox._column_utils import align_arrow_schemas
from tacotoolbox._constants import (
    TACOCAT_DEFAULT_PARQUET_CONFIG,
    TACOCAT_FOLDER_NAME,
)
from tacotoolbox._exceptions import (
    TacoConsolidationError,
    TacoSchemaError,
    TacoValidationError,
)
from tacotoolbox._logging import get_logger
from tacotoolbox._tacollection import create_tacollection
from tacotoolbox._validation import validate_common_directory

logger = get_logger(__name__)


def _estimate_row_group_size(num_rows: int, num_cols: int) -> int:
    """
    Estimate optimal row_group_size for Parquet files.

    Target: ~128MB uncompressed row groups for optimal DuckDB performance.
    Assumes ~1KB average row size (conservative estimate).
    """
    estimated_row_bytes = max(100 * num_cols, 500)
    target_row_group_bytes = 128 * 1024 * 1024
    estimated_row_group_size = target_row_group_bytes // estimated_row_bytes
    row_group_size = max(10_000, min(estimated_row_group_size, 1_000_000))

    if num_rows < 100_000:
        row_group_size = min(row_group_size, num_rows)

    return int(row_group_size)


def create_tacocat(
    inputs: Sequence[str | Path],
    output: str | Path | None = None,
    parquet_kwargs: dict[str, Any] | None = None,
    validate_schema: bool = True,
) -> None:
    """
    Create .tacocat/ folder from multiple TACO datasets.

    Consolidates multiple .tacozip files into a single .tacocat/ folder
    with unified parquet files optimized for DuckDB queries plus consolidated
    COLLECTION.json metadata.

    If output is not specified, all input files must be in the same directory
    and .tacocat/ will be created there.

    Default parquet_kwargs (optimized for DuckDB):
        compression: "zstd"
        compression_level: 19
        use_dictionary: True
        write_statistics: True
        use_content_defined_chunking: True
        row_group_size: auto-estimated (10K-1M rows, ~128MB target)

    Output structure:
        .tacocat/
        ├── level0.parquet
        ├── level1.parquet
        ├── level2.parquet
        └── COLLECTION.json

    Args:
        inputs: Sequence of .tacozip file paths to consolidate
        output: Output directory path (folder .tacocat/ will be created inside)
        parquet_kwargs: Optional PyArrow Parquet writer parameters
        validate_schema: If True, validate schemas match across datasets

    Raises:
        TacoValidationError: If inputs are invalid or in different directories
        TacoConsolidationError: If consolidation fails
        TacoSchemaError: If schema validation fails
    """
    output_dir = validate_common_directory(inputs) if output is None else Path(output)

    if output_dir.exists() and output_dir.is_file():
        raise TacoConsolidationError(
            f"output must be a directory, not a file: {output}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    tacocat_folder = output_dir / TACOCAT_FOLDER_NAME

    if tacocat_folder.exists():
        raise TacoConsolidationError(
            f"{TACOCAT_FOLDER_NAME} already exists in {output_dir}"
        )

    tacocat_folder.mkdir(parents=True, exist_ok=True)

    writer = TacoCatWriter(
        output_path=tacocat_folder,
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
            output_path: Path to .tacocat/ folder
            parquet_kwargs: Optional PyArrow Parquet writer parameters
        """
        self.output_path = Path(output_path)
        self.datasets: list[Path] = []
        self.max_depth = 0

        self.parquet_config = TACOCAT_DEFAULT_PARQUET_CONFIG.copy()
        if parquet_kwargs:
            self.parquet_config.update(parquet_kwargs)

    def add_dataset(self, tacozip_path: str | Path) -> None:
        """
        Add a TACO dataset to be consolidated.

        Raises:
            TacoConsolidationError: If dataset is invalid or cannot be read
        """
        path = Path(tacozip_path)

        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        if not path.is_file():
            raise TacoConsolidationError(f"Path is not a file: {path}")

        if path.stat().st_size == 0:
            raise TacoConsolidationError(f"Dataset file is empty: {path}")

        try:
            tacozip.read_header(str(path))
        except Exception as e:
            raise TacoConsolidationError(
                f"Invalid TACO file (cannot read header): {path}\n  Error: {e}"
            ) from e

        self.datasets.append(path)

    def write(self, validate_schema: bool = True) -> None:
        """
        Write consolidated .tacocat/ folder with parquets + COLLECTION.json.

        Raises:
            TacoConsolidationError: If no datasets added or writing fails
            TacoSchemaError: If schema validation fails
        """
        if not self.datasets:
            raise TacoConsolidationError("No datasets added. Use add_dataset() first.")

        logger.info(
            f"Consolidating {len(self.datasets)} TACO datasets into .tacocat/..."
        )

        # 1. Consolidate and write parquet files
        levels_data = self._consolidate_parquet_files(validate_schema)
        self._write_parquet_files(levels_data)

        # 2. Generate COLLECTION.json metadata
        self._write_collection_metadata(validate_schema)

        # Calculate total folder size
        folder_size_gb = sum(f.stat().st_size for f in self.output_path.glob("*")) / (
            1024**3
        )

        logger.info(f"TacoCat created: {self.output_path}")
        logger.info(f"   Datasets: {len(self.datasets)}")
        logger.info(f"   Max depth: {self.max_depth}")
        logger.info(f"   Total size: {folder_size_gb:.2f} GB")

    def _consolidate_parquet_files(self, validate_schema: bool) -> dict[int, pa.Table]:
        """
        Consolidate Parquet files from all datasets by level.
        Adds internal:source_file column to track original tacozip.
        """
        levels_data: dict[int, list[pa.Table]] = {}
        reference_schemas: dict[int, pa.Schema] = {}

        for idx, dataset_path in enumerate(self.datasets):
            logger.info(
                f"  [{idx+1}/{len(self.datasets)}] Processing {dataset_path.name}"
            )

            self._process_single_dataset(
                dataset_path, levels_data, reference_schemas, validate_schema
            )

        return self._merge_levels(levels_data)

    def _process_single_dataset(
        self,
        dataset_path: Path,
        levels_data: dict[int, list[pa.Table]],
        reference_schemas: dict[int, pa.Schema],
        validate_schema: bool,
    ) -> None:
        """
        Process a single dataset and add its Tables to levels_data.

        Raises:
            TacoConsolidationError: If dataset cannot be read
            TacoSchemaError: If schema validation fails
        """
        try:
            entries = tacozip.read_header(str(dataset_path))
        except Exception as e:
            raise TacoConsolidationError(
                f"Failed to read header from {dataset_path}: {e}"
            ) from e

        with open(dataset_path, "rb") as f:
            for level_idx, (offset, size) in enumerate(entries[:-1]):
                if size == 0:
                    continue

                level = level_idx
                self.max_depth = max(self.max_depth, level)

                table = self._read_level_table(f, offset, size, dataset_path)

                if validate_schema:
                    self._validate_schema_for_level(
                        table, level, dataset_path, reference_schemas
                    )

                if level not in levels_data:
                    levels_data[level] = []

                levels_data[level].append(table)

    def _read_level_table(
        self, f: Any, offset: int, size: int, dataset_path: Path
    ) -> pa.Table:
        """Read a single level Table from file."""
        f.seek(offset)
        parquet_bytes = f.read(size)
        table = pq.read_table(BytesIO(parquet_bytes))

        # Add internal:source_file column
        source_file_array = pa.array(
            [dataset_path.name] * table.num_rows, type=pa.string()
        )
        source_file_field = pa.field("internal:source_file", pa.string())

        return table.append_column(source_file_field, source_file_array)

    def _validate_schema_for_level(
        self,
        table: pa.Table,
        level: int,
        dataset_path: Path,
        reference_schemas: dict[int, pa.Schema],
    ) -> None:
        """
        Validate Table schema against reference schema for level.

        Raises:
            TacoSchemaError: If schemas don't match
        """
        current_schema = table.schema

        if level in reference_schemas:
            if not current_schema.equals(reference_schemas[level]):
                raise TacoSchemaError(
                    f"Schema mismatch at level {level} in {dataset_path.name}\n"
                    f"Expected columns: {reference_schemas[level].names}\n"
                    f"Got columns: {current_schema.names}"
                )
        else:
            reference_schemas[level] = current_schema

    def _merge_levels(
        self, levels_data: dict[int, list[pa.Table]]
    ) -> dict[int, pa.Table]:
        """Merge Tables by level using schema alignment."""
        merged = {}

        for level in sorted(levels_data.keys()):
            logger.info(
                f"  Consolidating level {level} ({len(levels_data[level])} Tables)..."
            )

            aligned_tables = align_arrow_schemas(levels_data[level])
            merged[level] = pa.concat_tables(aligned_tables)

            num_rows = merged[level].num_rows
            num_cols = merged[level].num_columns
            logger.info(f"     Level {level}: {num_rows:,} rows x {num_cols} cols")

        return merged

    def _write_parquet_files(self, levels_data: dict[int, pa.Table]) -> None:
        """Write consolidated parquet files to .tacocat/ folder."""
        for level, table in levels_data.items():
            output_file = self.output_path / f"level{level}.parquet"

            num_rows = table.num_rows
            num_cols = table.num_columns
            row_group_size = _estimate_row_group_size(num_rows, num_cols)

            pq_config = self.parquet_config.copy()
            if "row_group_size" not in self.parquet_config:
                pq_config["row_group_size"] = row_group_size

            pq.write_table(table, output_file, **pq_config)

            size_mb = output_file.stat().st_size / (1024**2)
            actual_row_group_size = pq_config.get("row_group_size", row_group_size)
            logger.info(
                f"     Wrote {output_file.name}: {size_mb:.2f} MB "
                f"(row_group_size={actual_row_group_size:,})"
            )

    def _write_collection_metadata(self, validate_schema: bool) -> None:
        """Generate and write COLLECTION.json to .tacocat/ folder."""
        logger.info("  Generating COLLECTION.json metadata...")

        collection_path = self.output_path / "COLLECTION.json"

        # Use create_tacollection with full output path
        create_tacollection(
            inputs=self.datasets,
            output=collection_path,
            validate_schema=validate_schema,
        )

        size_kb = collection_path.stat().st_size / 1024
        logger.info(f"     Wrote COLLECTION.json: {size_kb:.2f} KB")
