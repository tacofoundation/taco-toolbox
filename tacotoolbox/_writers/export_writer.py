"""
Export Writer - Create FOLDER containers from filtered TacoDataset.

This module handles the creation of FOLDER-format TACO containers from filtered
TacoDataset instances (e.g., from .sql() queries) using concurrent downloads
for maximum throughput with remote data sources.

Key features:
- Concurrent file downloading from TacoDataset (S3/HTTP/local)
- Automatic parent_id reindexing for consistency
- Recursive FOLDER handling with local metadata
- Zero-copy data transfer with obstore
- High-performance concurrent downloads with progress bars

The ExportWriter uses async/await for optimal performance when downloading
data from remote sources like S3 or HTTP endpoints. Local file I/O uses
sync operations as they're fast and not the bottleneck.

FOLDER Structure example (level 0 = FILEs only):
    dataset/
    ├── DATA/
    │   ├── sample_001.tif
    │   ├── sample_002.tif
    │   └── sample_003.tif
    ├── METADATA/
    │   └── level0.parquet
    └── COLLECTION.json

FOLDER Structure example (level 0 = FOLDERs, level 1 = FILEs):
    dataset/
    ├── DATA/
    │   ├── folder_A/
    │   │   ├── __meta__
    │   │   ├── nested_001.tif
    │   │   └── nested_002.tif
    │   └── folder_B/
    │       ├── __meta__
    │       ├── nested_001.tif
    │       └── nested_002.tif
    ├── METADATA/
    │   ├── level0.parquet
    │   └── level1.parquet
    └── COLLECTION.json
"""

import asyncio
import json
import pathlib

import obstore as obs
import polars as pl
from tacoreader import TacoDataset
from tacoreader.utils.vsi import create_obstore_from_url, strip_vsi_prefix

from tacotoolbox._column_utils import (
    read_metadata_file,
    reorder_internal_columns,
    write_parquet_file,
    write_parquet_file_with_cdc,
)
from tacotoolbox._constants import FOLDER_DATA_DIR, FOLDER_METADATA_DIR
from tacotoolbox._logging import get_logger
from tacotoolbox._progress import progress_gather

logger = get_logger(__name__)


class ExportWriterError(Exception):
    """Raised when export writing operations fail."""


class ExportWriter:
    """
    Handle creation of FOLDER containers from filtered TacoDataset.

    This writer is specialized for exporting datasets that have been loaded
    and potentially filtered via TacoDataset operations. It uses concurrent
    downloads for file copying, metadata reindexing, and collection updates.
    """

    def __init__(
        self,
        dataset: TacoDataset,
        output: pathlib.Path,
        concurrency: int = 100,
        quiet: bool = False,
        debug: bool = False,
    ) -> None:
        """
        Initialize export writer.

        Args:
            dataset: TacoDataset with applied filters (e.g., from .sql())
            output: Path to output folder
            concurrency: Maximum concurrent async operations (default: 100)
            quiet: If True, hide progress bars (default: False - shows progress)
            debug: If True, enable debug logging (default: False)
        """
        self.dataset = dataset
        self.output = pathlib.Path(output)
        self.concurrency = concurrency
        self.quiet = quiet

        self.data_dir = self.output / FOLDER_DATA_DIR
        self.metadata_dir = self.output / FOLDER_METADATA_DIR

        logger.debug(
            f"ExportWriter initialized: output={output}, concurrency={concurrency}"
        )

    async def create_folder(self) -> pathlib.Path:
        """
        Create complete FOLDER TACO from filtered dataset.

        Orchestrates the entire export process:
        1. Validates dataset (no level1+ joins, not empty)
        2. Creates FOLDER structure (DATA/, METADATA/)
        3. Copies files concurrently (FILEs and FOLDERs) with progress bars
        4. Generates consolidated metadata with reindexed parent_id
        5. Creates COLLECTION.json with subset metadata

        Returns:
            pathlib.Path: Path to created FOLDER TACO

        Raises:
            ExportWriterError: If export fails
            ValueError: If dataset has level1+ joins or is empty
            FileExistsError: If output already exists
        """
        try:
            logger.info(f"Starting FOLDER export: {self.output}")

            self._validate_dataset()
            self._create_folder_structure()
            await self._copy_all_bytes()
            self._generate_consolidated_metadata()
            self._generate_collection_json()

            logger.info(f"FOLDER export complete: {self.output}")
            return self.output

        except Exception as e:
            logger.error(f"Failed to export FOLDER: {e}")
            raise ExportWriterError(f"Failed to export FOLDER: {e}") from e

    def _validate_dataset(self) -> None:
        """
        Validate dataset before export.

        Checks that:
        - Dataset has no level1+ joins (only level0 filters supported)
        - Dataset is not empty (at least one sample)
        - Output path doesn't exist

        Raises:
            ValueError: If dataset validation fails
            FileExistsError: If output exists
        """
        if self.dataset._has_level1_joins:
            raise ValueError(
                "Cannot export dataset with level1+ joins. "
                "Only level0 filters are supported."
            )

        count = len(self.dataset.data._data)
        if count == 0:
            raise ValueError("Cannot export empty dataset")

        if self.output.exists():
            raise FileExistsError(f"Output already exists: {self.output}")

        logger.debug(f"Dataset validated: {count} samples")

    def _create_folder_structure(self) -> None:
        """Create base DATA/ and METADATA/ folders."""
        self.data_dir.mkdir(parents=True)
        self.metadata_dir.mkdir(parents=True)

        logger.debug(f"Created {FOLDER_DATA_DIR}/ and {FOLDER_METADATA_DIR}/")

    async def _copy_all_bytes(self) -> None:
        """
        Copy all bytes from dataset to output/DATA/.

        Uses concurrent downloads for FILEs with progress bar.
        FOLDER copying is recursive and processes children concurrently.
        """
        df = self.dataset.data._data

        files_df = df.filter(pl.col("type") == "FILE")
        folders_df = df.filter(pl.col("type") == "FOLDER")

        semaphore = asyncio.Semaphore(self.concurrency)

        # Copy FILEs concurrently with progress bar
        if len(files_df) > 0:
            logger.info(f"Copying {len(files_df)} FILEs")

            tasks = []
            for row in files_df.iter_rows(named=True):
                vsi_path = row["internal:gdal_vsi"]

                if "internal:relative_path" in row and row["internal:relative_path"]:
                    relative_path = row["internal:relative_path"]
                else:
                    relative_path = row["id"]

                dest = self.data_dir / relative_path

                task = self._copy_single_file(vsi_path, dest, semaphore)
                tasks.append(task)

            # Progress bar for FILE copying
            await progress_gather(
                *tasks, desc="Copying FILEs", unit="file", colour="green"
            )

        # Copy FOLDERs recursively with progress bar
        if len(folders_df) > 0:
            logger.info(f"Copying {len(folders_df)} FOLDERs")

            tasks = []
            for row in folders_df.iter_rows(named=True):
                task = self._copy_folder_recursive(row, level=0, semaphore=semaphore)
                tasks.append(task)

            # Progress bar for FOLDER copying
            await progress_gather(
                *tasks, desc="Copying FOLDERs", unit="folder", colour="blue"
            )

    async def _copy_single_file(
        self, vsi_path: str, dest_path: pathlib.Path, semaphore: asyncio.Semaphore
    ) -> None:
        """
        Copy bytes from vsi_path to dest_path.

        Handles two types of source paths:
        - /vsisubfile/... paths (ZIP entries): Downloads bytes from offset/size using obstore
        - Regular filesystem paths: Direct file copy

        Remote sources use obstore for concurrent downloads.
        Local file I/O uses sync operations as they're fast and not the bottleneck.

        Args:
            vsi_path: Source path (may be /vsisubfile/... or regular path)
            dest_path: Destination path for copied file
            semaphore: Asyncio semaphore to limit concurrency
        """
        async with semaphore:
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            if vsi_path.startswith("/vsisubfile/"):
                # Parse vsisubfile format: /vsisubfile/offset_size,source_path
                parts = vsi_path.replace("/vsisubfile/", "").split(",", 1)
                offset, size = map(int, parts[0].split("_"))
                zip_path = parts[1]

                clean_url = strip_vsi_prefix(zip_path)

                if pathlib.Path(clean_url).exists():
                    # Local file: sync read (fast, not a bottleneck)
                    with open(clean_url, "rb") as f:
                        f.seek(offset)
                        data = f.read(size)
                else:
                    # Remote file: use obstore async for concurrent downloads
                    store = create_obstore_from_url(clean_url)
                    # obstore returns Bytes (zero-copy), convert to bytes
                    data = bytes(
                        await obs.get_range_async(
                            store, "", start=offset, end=offset + size
                        )
                    )

                # Write to dest (sync is fine - local disk I/O is fast)
                with open(dest_path, "wb") as f:
                    f.write(data)
            else:
                # Regular file copy (sync - both read and write are local and fast)
                with open(vsi_path, "rb") as src:
                    data = src.read()
                with open(dest_path, "wb") as dest:
                    dest.write(data)

    async def _copy_folder_recursive(
        self,
        folder_row: dict,
        level: int,
        semaphore: asyncio.Semaphore,
    ) -> None:
        """
        Recursively copy a FOLDER and its children.

        For each FOLDER:
        1. Create the folder directory in DATA/
        2. Query children from next level using parent_id
        3. Process all children concurrently (both FOLDERs and FILEs)
        4. Write local __meta__ file with children metadata (PARQUET format)

        Args:
            folder_row: Row from dataset.df for the FOLDER
            level: Current level of this folder (0, 1, 2, ...)
            semaphore: Asyncio semaphore to limit concurrency
        """
        if level == 0:
            relative_path = folder_row["id"]
        else:
            relative_path = folder_row["internal:relative_path"]

        folder_path = self.data_dir / relative_path
        folder_path.mkdir(parents=True, exist_ok=True)

        parent_id = folder_row["internal:parent_id"]
        next_level = level + 1

        view_name = f"level{next_level}"
        if view_name not in self.dataset.consolidated_files:
            return

        # Query children from next level
        if "internal:source_file" in folder_row and folder_row["internal:source_file"]:
            # TacoCat case: need to filter by both parent_id AND source_file
            children_df = pl.from_arrow(
                self.dataset._duckdb.execute(
                    f'SELECT * FROM {view_name} WHERE "internal:parent_id" = ? AND "internal:source_file" = ?',
                    [parent_id, folder_row["internal:source_file"]],
                ).fetch_arrow_table()
            )
        else:
            # Single TACO case: only filter by parent_id
            children_df = pl.from_arrow(
                self.dataset._duckdb.execute(
                    f'SELECT * FROM {view_name} WHERE "internal:parent_id" = ?',
                    [parent_id],
                ).fetch_arrow_table()
            )

        # Process each child concurrently
        tasks = []
        for child_row in children_df.iter_rows(named=True):
            child_type = child_row["type"]
            child_vsi = child_row["internal:gdal_vsi"]

            # Level 1+ has internal:relative_path, level 0 uses id
            if (
                "internal:relative_path" in child_row
                and child_row["internal:relative_path"]
            ):
                child_rel = child_row["internal:relative_path"]
            else:
                child_rel = child_row["id"]

            child_dest = self.data_dir / child_rel

            if child_type == "FILE":
                task = self._copy_single_file(child_vsi, child_dest, semaphore)
                tasks.append(task)
            elif child_type == "FOLDER":
                task = self._copy_folder_recursive(child_row, next_level, semaphore)
                tasks.append(task)

        # Wait for all children to complete
        await asyncio.gather(*tasks)

        # Write local __meta__ for this folder in PARQUET format
        meta_path = folder_path / "__meta__"
        write_parquet_file(children_df, meta_path)

    def _generate_consolidated_metadata(self) -> None:
        """
        Generate consolidated metadata files (METADATA/levelX.parquet).

        Filters existing consolidated metadata to selected samples and reindexes
        internal:parent_id to maintain relational consistency in the new dataset.

        Also removes ZIP-specific columns (internal:offset, internal:size) since
        FOLDER format doesn't use them.
        """
        logger.info("Generating consolidated metadata")

        selected_ids = set(self.dataset.data._data["id"].to_list())

        level_names = sorted(
            self.dataset.consolidated_files.keys(),
            key=lambda x: int(x.replace("level", "")),
        )

        parent_id_mapping = None

        for level_name in level_names:
            file_path = self.dataset.consolidated_files[level_name]

            # Read metadata file (auto-detects parquet)
            df = read_metadata_file(file_path)

            if level_name == "level0":
                # Filter to selected samples
                filtered_df = df.filter(pl.col("id").is_in(selected_ids))

                # Build mapping: old_parent_id -> new_parent_id
                old_parent_ids = filtered_df["internal:parent_id"].to_list()
                parent_id_mapping = {old: new for new, old in enumerate(old_parent_ids)}

                # Assign new sequential parent_id values
                filtered_df = filtered_df.with_columns(
                    pl.arange(0, len(filtered_df))
                    .cast(pl.Int64)
                    .alias("internal:parent_id")
                )
                filtered_df = filtered_df.rechunk()
                filtered_df = reorder_internal_columns(filtered_df)
            else:
                # Filter to children of selected parents
                old_parent_ids_to_keep = set(parent_id_mapping.keys())
                filtered_df = df.filter(
                    pl.col("internal:parent_id").is_in(old_parent_ids_to_keep)
                )

                # Remap parent_id values to new indices
                new_parent_ids = [
                    parent_id_mapping[old_pid]
                    for old_pid in filtered_df["internal:parent_id"].to_list()
                ]
                filtered_df = filtered_df.with_columns(
                    pl.Series("internal:parent_id", new_parent_ids)
                )
                filtered_df = filtered_df.rechunk()
                filtered_df = reorder_internal_columns(filtered_df)

            # Remove ZIP-specific columns (not used in FOLDER format)
            zip_specific_cols = ["internal:offset", "internal:size"]
            filtered_df = filtered_df.drop(
                [col for col in zip_specific_cols if col in filtered_df.columns]
            )

            # Write consolidated metadata as PARQUET with CDC
            output_path = self.metadata_dir / f"{level_name}.parquet"
            write_parquet_file_with_cdc(filtered_df, output_path)

            logger.debug(f"{level_name}.parquet: {len(filtered_df)} samples")

    def _generate_collection_json(self) -> None:
        """
        Generate COLLECTION.json with updated counts and subset info.

        Updates collection metadata to reflect the exported subset:
        - Updates sample count (taco:pit_schema.root.n)
        - Adds taco:subset_of field with original dataset ID
        - Adds taco:subset_date with export timestamp
        """
        logger.debug("Generating COLLECTION.json")

        collection = self.dataset.collection.copy()

        # Update sample count
        new_count = len(self.dataset.data._data)
        collection["taco:pit_schema"]["root"]["n"] = new_count

        # Add subset provenance
        collection["taco:subset_of"] = collection.get("id", "unknown")

        from datetime import datetime, timezone

        collection["taco:subset_date"] = datetime.now(timezone.utc).isoformat()

        # Write to file
        output_path = self.output / "COLLECTION.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(collection, f, indent=4, ensure_ascii=False)

        logger.debug("COLLECTION.json created")
