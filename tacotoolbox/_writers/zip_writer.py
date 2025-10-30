"""
ZIP container writer for TACO format.

This module handles the creation of .tacozip files with optimized structure:
- TACO_HEADER: Fixed 157-byte entry with offset table
- DATA files: Actual sample data
- Local __meta__ files: Folder-specific metadata (Parquet)
- Consolidated METADATA/levelX.parquet: Full level metadata
- COLLECTION.json: Dataset metadata

This is ZIP-SPECIFIC. Offset/size columns only apply here.

The writer uses a bottom-up approach:
1. Add all data files to VirtualZIP
2. Calculate initial offsets
3. Generate __meta__ files bottom-up (deepest first)
4. Recalculate offsets after each level
5. Rebuild consolidated metadata with offset/size columns
6. Write final ZIP with correct offsets in TACO_HEADER

ZIP Structure example (level 0 = FILEs only):
    dataset.tacozip
    ├── TACO_HEADER
    ├── DATA/
    │   ├── sample_001.tif
    │   ├── sample_002.tif
    │   └── sample_003.tif
    ├── METADATA/
    │   └── level0.parquet
    └── COLLECTION.json

ZIP Structure example (level 0 = FOLDERs, level 1 = FILEs):
    dataset.tacozip
    ├── TACO_HEADER
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

ZIP Structure example (Change Detection - 3 levels deep):
    dataset.tacozip
    ├── TACO_HEADER
    ├── DATA/
    │   ├── Landslide_001/
    │   │   ├── __meta__
    │   │   ├── label.json
    │   │   └── imagery/
    │   │       ├── __meta__
    │   │       ├── before.tif
    │   │       └── after.tif
    │   ├── Landslide_002/
    │   │   ├── __meta__
    │   │   ├── label.json
    │   │   └── imagery/
    │   │       ├── __meta__
    │   │       ├── before.tif
    │   │       └── after.tif
    │   └── ...
    ├── METADATA/
    │   ├── level0.parquet
    │   ├── level1.parquet
    │   └── level2.parquet
    └── COLLECTION.json

PIT Schema for Change Detection example:
    {
        "root": {"n": 500, "type": ["FOLDER"]},
        "hierarchy": {
            "1": [{"n": 1000, "type": ["FILE", "FOLDER"], "id": ["label", "imagery"]}],
            "2": [{"n": 1000, "type": ["FILE", "FILE"], "id": ["before", "after"]}]
        }
    }
"""

import json
import pathlib
import tempfile
import uuid
import zipfile
from typing import Any

import polars as pl
import pyarrow.parquet as pq
import tacozip

from tacotoolbox._column_utils import reorder_internal_columns
from tacotoolbox._constants import (
    METADATA_OFFSET,
    METADATA_PARENT_ID,
    METADATA_SIZE,
    is_padding_id,
)
from tacotoolbox._metadata import MetadataPackage
from tacotoolbox._virtual_zip import VirtualTACOZIP


class ZipWriterError(Exception):
    """Raised when ZIP writing operations fail."""


class ZipWriter:
    """
    Handle creation of .tacozip container files with precalculated offsets.

    The ZipWriter uses a sophisticated bottom-up approach to ensure all
    metadata files (__meta__) have correct offsets pointing to data files:

    1. Virtual ZIP stage: Calculate all file offsets without writing
    2. Bottom-up metadata: Generate __meta__ starting from deepest folders
    3. Offset propagation: Each level's offsets affect parent level metadata
    4. Final assembly: Write actual ZIP with all offsets correct

    This ensures that FOLDER samples in metadata point to their __meta__ files,
    and FILE samples point to their actual data, all with byte-perfect offsets.
    """

    def __init__(
        self,
        output_path: pathlib.Path,
        quiet: bool = True,
        temp_dir: pathlib.Path | None = None,
    ) -> None:
        """
        Initialize ZIP writer.

        Args:
            output_path: Path for output .tacozip file
            quiet: If True, suppress progress messages (default: True)
            temp_dir: Directory for temporary files (default: system temp)
        """
        self.output_path = output_path
        self.quiet = quiet

        if temp_dir is None:
            self.temp_dir = pathlib.Path(tempfile.gettempdir())
        else:
            self.temp_dir = pathlib.Path(temp_dir)

        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self._temp_files: list[pathlib.Path] = []

    def create_complete_zip(
        self,
        src_files: list[str],
        arc_files: list[str],
        metadata_package: MetadataPackage,
        **parquet_kwargs: Any,
    ) -> pathlib.Path:
        """
        Create complete .tacozip file with all metadata and data.

        This is the main entry point that orchestrates the entire ZIP creation
        process using the bottom-up approach.

        Args:
            src_files: Source file paths (filesystem)
            arc_files: Archive paths in ZIP (e.g., "DATA/sample.tif")
            metadata_package: Complete metadata from MetadataGenerator
            **parquet_kwargs: Additional arguments for Parquet writer

        Returns:
            Path to created .tacozip file

        Raises:
            ZipWriterError: If ZIP creation fails
        """
        try:
            if not self.quiet:
                print("=" * 60)
                print("STARTING BOTTOM-UP __meta__ GENERATION")
                print("=" * 60)

            if not self.quiet:
                print("\n[STEP 1] Adding data files to VirtualZIP...")

            virtual_zip = VirtualTACOZIP()
            num_entries = len(metadata_package.levels) + 1
            virtual_zip.add_header()

            for src_path, arc_path in zip(src_files, arc_files, strict=False):
                if pathlib.Path(src_path).exists():
                    virtual_zip.add_file(src_path, arc_path)

            if not self.quiet:
                print(f"  Added {len(src_files)} data files")

            if not self.quiet:
                print("\n[STEP 2] Calculating initial offsets for data files...")

            virtual_zip.calculate_offsets()
            offsets_map = virtual_zip.get_all_offsets()

            if not self.quiet:
                print(f"  Initial offsets calculated: {len(offsets_map)} entries")

            if not self.quiet:
                print("\n[STEP 3] Analyzing folder hierarchy...")

            folder_order = self._extract_folder_order(arc_files)
            folders_by_depth = self._group_folders_by_depth(folder_order)

            if not self.quiet:
                for depth in sorted(folders_by_depth.keys(), reverse=True):
                    print(f"  Depth {depth}: {len(folders_by_depth[depth])} folders")

            if not self.quiet:
                print("\n[STEP 4] Generating __meta__ files (bottom-up)...")

            temp_parquet_meta_files = {}
            enriched_metadata_by_depth = {}
            max_depth = max(folders_by_depth.keys()) if folders_by_depth else 0

            # Bottom-up: Start from deepest folders
            for depth in range(max_depth, 0, -1):
                if not self.quiet:
                    print(f"\n  Processing depth {depth}...")

                folders_at_depth = folders_by_depth[depth]
                enriched_metadata_by_depth[depth] = []

                for folder_path in folders_at_depth:
                    if folder_path not in metadata_package.local_metadata:
                        if not self.quiet:
                            print(
                                f"    WARNING: {folder_path} not in local_metadata, skipping"
                            )
                        continue

                    local_df = metadata_package.local_metadata[folder_path]

                    # Add ZIP-specific offset/size columns
                    enriched_df = _add_zip_offsets(local_df, offsets_map, folder_path)
                    enriched_metadata_by_depth[depth].append(enriched_df)

                    # Write __meta__ as Parquet
                    meta_arc_path = f"{folder_path}__meta__"
                    temp_parquet = self._write_single_parquet(enriched_df, folder_path)
                    temp_parquet_meta_files[meta_arc_path] = temp_parquet

                    # Add to virtual ZIP and recalculate offsets
                    real_size = temp_parquet.stat().st_size
                    virtual_zip.add_file(
                        str(temp_parquet), meta_arc_path, file_size=real_size
                    )

                    if not self.quiet:
                        print(f"    Created {meta_arc_path} ({real_size} bytes)")

                # Recalculate offsets after adding __meta__ files
                virtual_zip.calculate_offsets()
                offsets_map = virtual_zip.get_all_offsets()

                if not self.quiet:
                    print(f"  Recalculated offsets: {len(offsets_map)} entries")

            if not self.quiet:
                print(
                    "\n[STEP 5] Rebuilding consolidated metadata (METADATA/levelX.parquet)..."
                )

            # Level 0: Add offset/size columns
            original_level0 = metadata_package.levels[0]
            level0_with_offsets = _add_zip_offsets(
                original_level0, offsets_map, folder_path=""
            )

            # Combine: keep all original columns + add offset/size
            offset_size_cols = [METADATA_OFFSET, METADATA_SIZE]
            columns_to_add = [
                col for col in offset_size_cols if col in level0_with_offsets.columns
            ]

            if columns_to_add:
                offset_size_df = level0_with_offsets.select(columns_to_add)
                metadata_package.levels[0] = pl.concat(
                    [original_level0, offset_size_df], how="horizontal"
                )
                # Reorder to place internal:* columns at the end
                metadata_package.levels[0] = reorder_internal_columns(
                    metadata_package.levels[0]
                )
            else:
                metadata_package.levels[0] = original_level0

            if not self.quiet:
                cols_info = metadata_package.levels[0].columns
                print(
                    f"  Level 0: {len(metadata_package.levels[0])} samples (columns: {len(cols_info)})"
                )

            # Level 1+: Preserve parent_id from original, add offsets from concatenated
            for depth in range(1, len(metadata_package.levels)):
                if enriched_metadata_by_depth.get(depth):
                    # Original DataFrame (HAS parent_id, NO offset/size)
                    original_level = metadata_package.levels[depth]

                    # Concatenated DataFrame (HAS offset/size, NO parent_id)
                    concatenated = pl.concat(
                        enriched_metadata_by_depth[depth], how="vertical"
                    )

                    # Extract only offset/size columns
                    offset_size_cols = [METADATA_OFFSET, METADATA_SIZE]
                    columns_to_add = [
                        col for col in offset_size_cols if col in concatenated.columns
                    ]

                    if columns_to_add:
                        offset_size_df = concatenated.select(columns_to_add)

                        # Combine: original (with parent_id) + offset/size
                        enriched_level = pl.concat(
                            [original_level, offset_size_df], how="horizontal"
                        )

                        # Reorder to place internal:* columns at the end
                        enriched_level = reorder_internal_columns(enriched_level)

                        # Sort by parent_id to maintain hierarchical order
                        if METADATA_PARENT_ID in enriched_level.columns:
                            enriched_level = enriched_level.sort(
                                METADATA_PARENT_ID, maintain_order=True
                            )

                        metadata_package.levels[depth] = enriched_level
                    else:
                        metadata_package.levels[depth] = original_level

                    if not self.quiet:
                        cols_info = metadata_package.levels[depth].columns
                        has_parent_id = METADATA_PARENT_ID in cols_info
                        print(
                            f"  Level {depth}: {len(metadata_package.levels[depth])} samples "
                            f"(rebuilt from {len(enriched_metadata_by_depth[depth])} folders, "
                            f"parent_id={has_parent_id})"
                        )
                else:
                    if not self.quiet:
                        print(f"  Level {depth}: No data (skipped)")

            # Write consolidated METADATA/levelX.parquet files
            temp_parquet_level_files = {}
            for i, level_df in enumerate(metadata_package.levels):
                arc_path = f"METADATA/level{i}.parquet"
                temp_path = self.temp_dir / f"{uuid.uuid4().hex}_level{i}.parquet"
                arrow_table = level_df.to_arrow()
                pq.write_table(arrow_table, temp_path, **parquet_kwargs)
                real_size = temp_path.stat().st_size
                virtual_zip.add_file(str(temp_path), arc_path, file_size=real_size)
                temp_parquet_level_files[arc_path] = temp_path
                self._temp_files.append(temp_path)

                if not self.quiet:
                    print(f"  Added {arc_path} ({real_size} bytes)")

            if not self.quiet:
                print("\n[STEP 6] Adding COLLECTION.json...")

            collection = metadata_package.collection.copy()
            collection["taco:pit_schema"] = metadata_package.pit_schema
            collection["taco:field_schema"] = metadata_package.field_schema
            temp_json = self.temp_dir / f"{uuid.uuid4().hex}.json"
            with open(temp_json, "w", encoding="utf-8") as f:
                json.dump(collection, f, indent=4, ensure_ascii=False)
            collection_size = temp_json.stat().st_size
            virtual_zip.add_file(
                str(temp_json), "COLLECTION.json", file_size=collection_size
            )
            self._temp_files.append(temp_json)

            if not self.quiet:
                print(f"  Added COLLECTION.json ({collection_size} bytes)")

            if not self.quiet:
                print("\n[STEP 7] Final offset calculation...")

            virtual_zip.calculate_offsets()

            if not self.quiet:
                print("\n[STEP 8] Preparing final file lists for ZIP creation...")

            all_src_files = list(src_files)
            all_arc_files = list(arc_files)

            # Add __meta__ files (bottom-up order)
            for depth in range(max_depth, 0, -1):
                for folder_path in folders_by_depth[depth]:
                    meta_arc_path = f"{folder_path}__meta__"
                    if meta_arc_path in temp_parquet_meta_files:
                        temp_path = temp_parquet_meta_files[meta_arc_path]
                        all_src_files.append(str(temp_path))
                        all_arc_files.append(meta_arc_path)

            # Add METADATA/levelX.parquet files
            for i in range(len(metadata_package.levels)):
                arc_path = f"METADATA/level{i}.parquet"
                temp_path = temp_parquet_level_files[arc_path]
                all_src_files.append(str(temp_path))
                all_arc_files.append(arc_path)

            # Add COLLECTION.json
            all_src_files.append(str(temp_json))
            all_arc_files.append("COLLECTION.json")

            if not self.quiet:
                print(f"  Total files in ZIP: {len(all_src_files)}")

            if not self.quiet:
                print("\n[STEP 9] Writing final ZIP file...")

            header_entries = [(0, 0) for _ in range(num_entries)]

            tacozip.create(
                zip_path=str(self.output_path),
                src_files=all_src_files,
                arc_files=all_arc_files,
                entries=header_entries,
            )

            if not self.quiet:
                print("\n[STEP 10] Updating TACO_HEADER with real offsets...")

            metadata_offsets, metadata_sizes = self._get_metadata_offsets()
            collection_offset, collection_size = self._get_collection_offset()

            real_entries = [
                *zip(metadata_offsets, metadata_sizes, strict=False),
                (collection_offset, collection_size),
            ]

            tacozip.update_header(zip_path=str(self.output_path), entries=real_entries)

            if not self.quiet:
                print(f"\n{'='*60}")
                print(f"ZIP CREATED SUCCESSFULLY: {self.output_path}")
                print(f"{'='*60}\n")

        except Exception as e:
            raise ZipWriterError(f"Failed to create ZIP: {e}") from e
        else:
            return self.output_path
        finally:
            # CRITICAL: Always cleanup temporary files
            if not self.quiet:
                print("\n[CLEANUP] Removing temporary files...")
            self._cleanup()

    def _extract_folder_order(self, arc_files: list[str]) -> list[str]:
        """
        Extract folder paths that need __meta__ files.

        Only includes Level 1+ folders (not DATA/ root).

        Args:
            arc_files: Archive paths of data files

        Returns:
            List of folder paths (e.g., ["DATA/folder_A/", "DATA/folder_A/sub/"])
        """
        folder_set = set()

        for arc_path in arc_files:
            if "/" in arc_path:
                parts = arc_path.split("/")
                # Start from 2 to skip DATA/ root
                for i in range(2, len(parts)):
                    folder_path = "/".join(parts[:i]) + "/"
                    folder_set.add(folder_path)

        return sorted(folder_set)

    def _group_folders_by_depth(self, folder_order: list[str]) -> dict[int, list[str]]:
        """
        Group folders by depth for bottom-up processing.

        Args:
            folder_order: List of folder paths

        Returns:
            Dictionary mapping depth -> list of folder paths

        Example:
            >>> folders = ["DATA/a/", "DATA/a/b/", "DATA/a/b/c/"]
            >>> self._group_folders_by_depth(folders)
            {1: ["DATA/a/"], 2: ["DATA/a/b/"], 3: ["DATA/a/b/c/"]}
        """
        by_depth: dict[int, list[str]] = {}

        for folder in folder_order:
            depth = folder.count("/") - 1

            if depth not in by_depth:
                by_depth[depth] = []
            by_depth[depth].append(folder)

        return by_depth

    def _write_single_parquet(self, df: pl.DataFrame, folder_path: str) -> pathlib.Path:
        """
        Write a single __meta__ Parquet file to temp directory.

        Args:
            df: DataFrame with metadata
            folder_path: Folder path for identifier

        Returns:
            Path to temporary Parquet file
        """
        identifier = folder_path.replace("/", "_").strip("_")
        temp_path = self.temp_dir / f"{uuid.uuid4().hex}_{identifier}.parquet"

        filtered_df = self._filter_metadata_columns(df)
        arrow_table = filtered_df.to_arrow()
        pq.write_table(arrow_table, temp_path, compression="zstd")

        self._temp_files.append(temp_path)
        return temp_path

    def _filter_metadata_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Filter columns for __meta__ files.

        Removes container-irrelevant columns like 'path'.

        Args:
            df: DataFrame with all columns

        Returns:
            DataFrame with filtered columns
        """
        exclude_columns = {"path"}
        cols_to_keep = [col for col in df.columns if col not in exclude_columns]
        return df.select(cols_to_keep) if cols_to_keep else df

    def _get_metadata_offsets(self) -> tuple[list[int], list[int]]:
        """
        Get offsets and sizes for METADATA/levelX.parquet files.

        Returns:
            Tuple of (offsets, sizes) lists
        """
        offsets = []
        sizes = []

        with zipfile.ZipFile(self.output_path, "r") as zf:
            with open(self.output_path, "rb") as f:
                parquet_files = [
                    info
                    for info in zf.infolist()
                    if info.filename.startswith("METADATA/")
                    and info.filename.endswith(".parquet")
                ]

                parquet_files.sort(key=lambda x: x.filename)

                for info in parquet_files:
                    f.seek(info.header_offset)
                    lfh = f.read(30)

                    filename_len = int.from_bytes(lfh[26:28], "little")
                    extra_len = int.from_bytes(lfh[28:30], "little")

                    data_offset = info.header_offset + 30 + filename_len + extra_len
                    data_size = info.compress_size

                    offsets.append(data_offset)
                    sizes.append(data_size)

        return offsets, sizes

    def _get_collection_offset(self) -> tuple[int, int]:
        """
        Get offset and size for COLLECTION.json.

        Returns:
            Tuple of (offset, size)
        """
        with zipfile.ZipFile(self.output_path, "r") as zf:
            with open(self.output_path, "rb") as f:
                info = zf.getinfo("COLLECTION.json")

                f.seek(info.header_offset)
                lfh = f.read(30)

                filename_len = int.from_bytes(lfh[26:28], "little")
                extra_len = int.from_bytes(lfh[28:30], "little")

                data_offset = info.header_offset + 30 + filename_len + extra_len
                data_size = info.compress_size

                return data_offset, data_size

    def _cleanup(self) -> None:
        """
        Clean up all temporary files created during ZIP creation.

        This ensures no temporary files are left behind after the ZIP
        is successfully created or if an error occurs during creation.
        """
        if not self._temp_files:
            return

        cleaned = 0
        failed = 0

        for temp_file in self._temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
                    cleaned += 1
            except Exception as e:
                failed += 1
                if not self.quiet:
                    print(f"  Warning: Failed to cleanup {temp_file}: {e}")

        self._temp_files.clear()

        if not self.quiet and (cleaned > 0 or failed > 0):
            print(f"  Cleanup: {cleaned} files removed, {failed} failed")


def _add_zip_offsets(
    metadata_df: pl.DataFrame,
    offsets_map: dict[str, tuple[int, int]],
    folder_path: str = "",
) -> pl.DataFrame:
    """
    Add internal:offset and internal:size columns for ZIP containers.

    This is a ZIP-SPECIFIC function. FOLDER containers don't use offsets.

    Critical behavior:
    - FILE samples: offset points to actual data file in ZIP
    - FOLDER samples: offset points to child's __meta__ file in ZIP

    This function is ALWAYS strict: missing offsets for non-padding samples
    indicate a bug in offset calculation and will raise an error.

    Args:
        metadata_df: DataFrame with sample metadata
        offsets_map: Dictionary mapping arc_path -> (offset, size)
        folder_path: Current folder path (e.g., "DATA/folder_A/")

    Returns:
        DataFrame with added internal:offset and internal:size columns

    Raises:
        ValueError: If offsets missing for non-padding samples

    Example:
        >>> offsets_map = {
        ...     "DATA/file1.tif": (1000, 500),
        ...     "DATA/folder_A/__meta__": (1500, 200)
        ... }
        >>> df = _add_zip_offsets(metadata_df, offsets_map, folder_path="DATA/")
        >>> df.columns
        ['id', 'type', 'internal:parent_id', 'internal:offset', 'internal:size']
    """
    offsets = []
    sizes = []
    missing_offsets = []

    for row in metadata_df.iter_rows(named=True):
        sample_id = row["id"]
        sample_type = row["type"]

        # Determine archive path based on type
        if sample_type == "FOLDER":
            # FOLDER: points to __meta__ file
            if folder_path:
                arc_path = f"{folder_path}{sample_id}/__meta__"
            else:
                arc_path = f"DATA/{sample_id}/__meta__"
        else:
            # FILE: points to actual data file
            if folder_path:
                arc_path = f"{folder_path}{sample_id}"
            else:
                arc_path = f"DATA/{sample_id}"

        # Look up offset and size
        if arc_path in offsets_map:
            offset, size = offsets_map[arc_path]
            offsets.append(offset)
            sizes.append(size)
        else:
            # Missing offset - track for error
            if not is_padding_id(sample_id):
                missing_offsets.append(arc_path)

            offsets.append(None)
            sizes.append(None)

    # ALWAYS strict: missing offsets indicate a bug
    if missing_offsets:
        raise ValueError(
            f"Offsets not found for {len(missing_offsets)} non-padding samples:\n"
            f"  First 5: {missing_offsets[:5]}\n"
            f"  This indicates a bug in offset calculation or file mapping."
        )

    # Add offset and size columns
    result_df = metadata_df.with_columns(
        [pl.Series(METADATA_OFFSET, offsets), pl.Series(METADATA_SIZE, sizes)]
    )

    # Reorder to place internal:* columns at the end
    return reorder_internal_columns(result_df)
