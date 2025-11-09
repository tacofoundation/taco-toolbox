"""
Export Writer - Create FOLDER containers from filtered TacoDataset.

This module handles the creation of FOLDER-format TACO containers from filtered
TacoDataset instances (e.g., from .sql() queries). It differs from FolderWriter
which creates containers from Tortilla/Sample objects during initial dataset
creation.

Key features:
- Parallel file copying from TacoDataset (local/remote sources)
- Automatic parent_id reindexing for consistency
- Recursive FOLDER handling with local metadata
- ZIP-specific column removal (offset, size)
- Collection metadata updates with subset provenance

The ExportWriter handles three main scenarios:
1. Direct dataset export (full or filtered)
2. ZIP to FOLDER conversion (via translate module)
3. Subset extraction with metadata reindexing

FOLDER Structure example (level 0 = FILEs only):
    dataset/
    ├── DATA/
    │   ├── sample_001.tif
    │   ├── sample_002.tif
    │   └── sample_003.tif
    ├── METADATA/
    │   └── level0.avro
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
    │   ├── level0.avro
    │   └── level1.avro
    └── COLLECTION.json
"""

import json
import pathlib
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

import obstore as obs
import polars as pl
from tacoreader import TacoDataset
from tacoreader.utils.vsi import create_obstore_from_url, strip_vsi_prefix

from tacotoolbox._column_utils import (
    read_metadata_file,
    reorder_internal_columns,
    write_avro_file,
    write_parquet_file,
)
from tacotoolbox._constants import FOLDER_DATA_DIR, FOLDER_METADATA_DIR


class ExportWriterError(Exception):
    """Raised when export writing operations fail."""


class ExportWriter:
    """
    Handle creation of FOLDER containers from filtered TacoDataset.

    This writer is specialized for exporting datasets that have been loaded
    and potentially filtered via TacoDataset operations. It handles parallel
    file copying, metadata reindexing, and collection updates.
    """

    def __init__(
        self,
        dataset: TacoDataset,
        output: pathlib.Path,
        nworkers: int = 4,
        quiet: bool = False,
    ) -> None:
        """
        Initialize export writer.

        Args:
            dataset: TacoDataset with applied filters (e.g., from .sql())
            output: Path to output folder
            nworkers: Number of parallel workers for file copying
            quiet: If True, suppress progress messages
        """
        self.dataset = dataset
        self.output = pathlib.Path(output)
        self.nworkers = nworkers
        self.quiet = quiet

        self.data_dir = self.output / FOLDER_DATA_DIR
        self.metadata_dir = self.output / FOLDER_METADATA_DIR

    def create_folder(self) -> pathlib.Path:
        """
        Create complete FOLDER TACO from filtered dataset.

        Orchestrates the entire export process:
        1. Validates dataset (no level1+ joins, not empty)
        2. Creates FOLDER structure (DATA/, METADATA/)
        3. Copies files in parallel (FILEs and FOLDERs)
        4. Generates consolidated metadata with reindexed parent_id
        5. Creates COLLECTION.json with subset metadata

        Returns:
            Path to created FOLDER TACO

        Raises:
            ExportWriterError: If export fails
            ValueError: If dataset has level1+ joins or is empty
            FileExistsError: If output already exists
        """
        try:
            if not self.quiet:
                print("Starting FOLDER export...")

            self._validate_dataset()
            self._create_folder_structure()
            self._copy_all_bytes()
            self._generate_consolidated_metadata()
            self._generate_collection_json()

            if not self.quiet:
                print(f"FOLDER export complete: {self.output}")

            return self.output

        except Exception as e:
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

    def _create_folder_structure(self) -> None:
        """Create base DATA/ and METADATA/ folders."""
        self.data_dir.mkdir(parents=True)
        self.metadata_dir.mkdir(parents=True)

        if not self.quiet:
            print(f"  Created {FOLDER_DATA_DIR}/ and {FOLDER_METADATA_DIR}/")

    def _copy_all_bytes(self) -> None:
        """
        Copy all bytes from dataset to output/DATA/.

        Uses parallel workers for FILE copying. FOLDER copying is recursive
        and single-threaded per folder tree.
        """
        df = self.dataset.data._data

        files_df = df.filter(pl.col("type") == "FILE")
        folders_df = df.filter(pl.col("type") == "FOLDER")

        # Copy FILEs in parallel
        if len(files_df) > 0:
            with ThreadPoolExecutor(max_workers=self.nworkers) as executor:
                futures = []
                for row in files_df.iter_rows(named=True):
                    vsi_path = row["internal:gdal_vsi"]

                    if (
                        "internal:relative_path" in row
                        and row["internal:relative_path"]
                    ):
                        relative_path = row["internal:relative_path"]
                    else:
                        relative_path = row["id"]

                    dest = self.data_dir / relative_path

                    future = executor.submit(self._copy_single_file, vsi_path, dest)
                    futures.append(future)

                # Wait for all FILEs to complete
                for future in as_completed(futures):
                    future.result()  # Raises exception if copy failed

            if not self.quiet:
                print(f"  Copied {len(files_df)} FILEs")

        # Copy FOLDERs recursively (single-threaded per tree)
        if len(folders_df) > 0:
            for row in folders_df.iter_rows(named=True):
                self._copy_folder_recursive(row, level=0)

            if not self.quiet:
                print(f"  Copied {len(folders_df)} FOLDERs")

    def _copy_single_file(self, vsi_path: str, dest_path: pathlib.Path) -> None:
        """
        Copy bytes from vsi_path to dest_path.

        Handles two types of source paths:
        - /vsisubfile/... paths (ZIP entries): Extracts bytes from offset/size
        - Regular filesystem paths: Direct file copy

        For vsisubfile paths, supports both local and remote (S3, HTTP) sources
        using obstore for efficient range reads.

        Args:
            vsi_path: Source path (may be /vsisubfile/... or regular path)
            dest_path: Destination path for copied file
        """
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        if vsi_path.startswith("/vsisubfile/"):
            # Parse vsisubfile format: /vsisubfile/offset_size,source_path
            parts = vsi_path.replace("/vsisubfile/", "").split(",", 1)
            offset, size = map(int, parts[0].split("_"))
            zip_path = parts[1]

            clean_url = strip_vsi_prefix(zip_path)

            if pathlib.Path(clean_url).exists():
                # Local file: direct read
                with open(clean_url, "rb") as f:
                    f.seek(offset)
                    data = f.read(size)
            else:
                # Remote file: use obstore for range read
                store = create_obstore_from_url(clean_url)
                data = bytes(obs.get_range(store, "", start=offset, end=offset + size))

            with open(dest_path, "wb") as f:
                f.write(data)
        else:
            # Regular file copy
            shutil.copy2(vsi_path, dest_path)

    def _copy_folder_recursive(
        self,
        folder_row: dict,
        level: int,
    ) -> None:
        """
        Recursively copy a FOLDER and its children.

        For each FOLDER:
        1. Create the folder directory in DATA/
        2. Query children from next level using parent_id
        3. Recursively copy child FOLDERs
        4. Copy child FILEs directly
        5. Write local __meta__ file with children metadata (PARQUET format)

        Args:
            folder_row: Row from dataset.df for the FOLDER
            level: Current level of this folder (0, 1, 2, ...)
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
        if (
            "internal:source_file" in folder_row
            and folder_row["internal:source_file"]
        ):
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

        # Process each child
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
                self._copy_single_file(child_vsi, child_dest)
            elif child_type == "FOLDER":
                self._copy_folder_recursive(child_row, next_level)

        # Write local __meta__ for this folder in PARQUET format
        meta_path = folder_path / "__meta__"
        write_parquet_file(children_df, meta_path)

    def _generate_consolidated_metadata(self) -> None:
        """
        Generate consolidated metadata files (METADATA/levelX.avro).

        Filters existing consolidated metadata to selected samples and reindexes
        internal:parent_id to maintain relational consistency in the new dataset.

        The reindexing process:
        1. Level0: Filter to selected samples, assign new parent_id = 0, 1, 2, ...
        2. Level1+: Filter to children of selected parents, remap parent_id values

        Also removes ZIP-specific columns (internal:offset, internal:size) since
        FOLDER format doesn't use them.
        """
        if not self.quiet:
            print("  Generating consolidated metadata...")

        selected_ids = set(self.dataset.data._data["id"].to_list())

        level_names = sorted(
            self.dataset.consolidated_files.keys(),
            key=lambda x: int(x.replace("level", "")),
        )

        parent_id_mapping = None

        for level_name in level_names:
            file_path = self.dataset.consolidated_files[level_name]

            # Read metadata file (auto-detects parquet/avro)
            df = read_metadata_file(file_path)

            if level_name == "level0":
                # Filter to selected samples
                filtered_df = df.filter(pl.col("id").is_in(selected_ids))

                # Build mapping: old_parent_id -> new_parent_id
                old_parent_ids = filtered_df["internal:parent_id"].to_list()
                parent_id_mapping = {
                    old: new for new, old in enumerate(old_parent_ids)
                }

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

            # Write consolidated metadata as AVRO
            output_path = self.metadata_dir / f"{level_name}.avro"
            write_avro_file(filtered_df, output_path)

            if not self.quiet:
                print(f"    {level_name}.avro: {len(filtered_df)} samples")

    def _generate_collection_json(self) -> None:
        """
        Generate COLLECTION.json with updated counts and subset info.

        Updates collection metadata to reflect the exported subset:
        - Updates sample count (taco:pit_schema.root.n)
        - Adds taco:subset_of field with original dataset ID
        - Adds taco:subset_date with export timestamp
        """
        if not self.quiet:
            print("  Generating COLLECTION.json...")

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