"""
ZIP container writer for TACO format.

This module handles the creation of .tacozip files with optimized structure:
- TACO_HEADER: Fixed 157-byte entry with offset table
- DATA files: Actual sample data
- Local __meta__ files: Folder-specific metadata (Parquet)
- Consolidated METADATA/levelX.parquet: Full level metadata
- COLLECTION.json: Dataset metadata

This is ZIP-SPECIFIC. internal:offset and internal:size are added here.
Both values come from VirtualZIP, not from sample._size_bytes.

The writer uses a bottom-up approach:
1. Add all data files to VirtualZIP
2. Calculate initial offsets
3. Generate __meta__ files bottom-up (deepest first)
4. Recalculate offsets after each level
5. Rebuild consolidated metadata with offset and size columns
6. Write final ZIP with correct offsets in TACO_HEADER
"""

import json
import pathlib
import tempfile
import uuid
import zipfile
from contextlib import ExitStack
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import tacozip

from tacotoolbox._column_utils import (
    align_arrow_schemas,
    reorder_internal_columns,
)
from tacotoolbox._constants import (
    METADATA_OFFSET,
    METADATA_PARENT_ID,
    METADATA_SIZE,
)
from tacotoolbox._exceptions import TacoCreationError
from tacotoolbox._logging import get_logger
from tacotoolbox._metadata import MetadataPackage
from tacotoolbox._validation import is_padding_id
from tacotoolbox._virtual_zip import VirtualTACOZIP

logger = get_logger(__name__)


class ZipWriter:
    """
    Handle creation of .tacozip container files with precalculated offsets.

    The ZipWriter uses a sophisticated bottom-up approach to ensure all
    metadata files (__meta__) have correct offsets pointing to data files.

    All temporary files are automatically cleaned up via ExitStack, even
    if errors occur during ZIP creation.
    """

    def __init__(
        self,
        output_path: pathlib.Path,
        temp_dir: pathlib.Path | None = None,
    ) -> None:
        """
        Initialize ZIP writer.

        Args:
            output_path: Path for output .tacozip file
            temp_dir: Directory for temporary files (default: system temp)
        """
        self.output_path = output_path

        if temp_dir is None:
            self.temp_dir = pathlib.Path(tempfile.gettempdir())
        else:
            self.temp_dir = pathlib.Path(temp_dir)

        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Use ExitStack for robust cleanup
        self._cleanup_stack = ExitStack()

        logger.debug(
            f"ZipWriter initialized: output={output_path}, temp_dir={self.temp_dir}"
        )

    def create_complete_zip(  # noqa: C901
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
            **parquet_kwargs: Additional Parquet writer parameters

        Returns:
            pathlib.Path: Path to created .tacozip file

        Raises:
            TacoCreationError: If ZIP creation fails
        """
        try:
            logger.info("Starting bottom-up __meta__ generation")

            # Step 1: Add data files to VirtualZIP
            virtual_zip = VirtualTACOZIP()
            num_entries = len(metadata_package.levels) + 1
            virtual_zip.add_header()

            logger.debug(f"Adding {len(src_files)} data files to virtual ZIP")
            for src_path, arc_path in zip(src_files, arc_files, strict=False):
                if pathlib.Path(src_path).exists():
                    virtual_zip.add_file(src_path, arc_path)

            # Step 2: Calculate initial offsets
            virtual_zip.calculate_offsets()
            offsets_map = virtual_zip.get_all_offsets()
            logger.debug(f"Initial offsets calculated: {len(offsets_map)} entries")

            # Step 3: Analyze folder hierarchy
            folder_order = self._extract_folder_order(arc_files)
            folders_by_depth = self._group_folders_by_depth(folder_order)

            for depth in sorted(folders_by_depth.keys(), reverse=True):
                logger.debug(f"Depth {depth}: {len(folders_by_depth[depth])} folders")

            # Step 4: Generate __meta__ files (bottom-up)
            temp_meta_files = {}
            enriched_metadata_by_depth: dict[int, list[pa.Table]] = {}
            max_depth = max(folders_by_depth.keys()) if folders_by_depth else 0

            # Bottom-up: Start from deepest folders
            for depth in range(max_depth, 0, -1):
                logger.debug(f"Processing depth {depth}...")

                folders_at_depth = folders_by_depth[depth]
                enriched_metadata_by_depth[depth] = []

                for folder_path in folders_at_depth:
                    if folder_path not in metadata_package.local_metadata:
                        logger.warning(f"{folder_path} not in local_metadata")
                        continue

                    local_table = metadata_package.local_metadata[folder_path]

                    # Add ZIP-specific offset and size columns from VirtualZIP
                    enriched_table = _add_zip_offsets(
                        local_table, offsets_map, folder_path
                    )
                    enriched_metadata_by_depth[depth].append(enriched_table)

                    # Write __meta__ as Parquet
                    meta_arc_path = f"{folder_path}__meta__"
                    temp_parquet = self._write_single_parquet(
                        enriched_table, folder_path, **parquet_kwargs
                    )
                    temp_meta_files[meta_arc_path] = temp_parquet

                    # Add to virtual ZIP and recalculate offsets
                    real_size = temp_parquet.stat().st_size
                    virtual_zip.add_file(
                        str(temp_parquet), meta_arc_path, file_size=real_size
                    )

                # Recalculate offsets after adding __meta__ files
                virtual_zip.calculate_offsets()
                offsets_map = virtual_zip.get_all_offsets()

            logger.debug("Rebuilding consolidated metadata...")

            # Step 5: Rebuild consolidated metadata (METADATA/levelX.parquet)

            # Level 0: Add offset and size from VirtualZIP
            original_level0 = metadata_package.levels[0]
            metadata_package.levels[0] = _add_zip_offsets(
                original_level0, offsets_map, folder_path=""
            )

            logger.debug(f"Level 0: {metadata_package.levels[0].num_rows} samples")

            # Level 1+: Preserve parent_id from original, add offset and size from concatenated
            for depth in range(1, len(metadata_package.levels)):
                if enriched_metadata_by_depth.get(depth):
                    original_level = metadata_package.levels[depth]

                    # Concatenate enriched tables (already have correct types from Pydantic validation)
                    enriched_tables = enriched_metadata_by_depth[depth]

                    # Align schemas and concatenate
                    aligned_tables = align_arrow_schemas(enriched_tables)
                    concatenated = pa.concat_tables(aligned_tables)

                    # Extract offset and size columns from concatenated
                    if (
                        METADATA_OFFSET in concatenated.schema.names
                        and METADATA_SIZE in concatenated.schema.names
                    ):
                        offset_array = concatenated.column(METADATA_OFFSET)
                        offset_field = concatenated.schema.field(METADATA_OFFSET)

                        size_array = concatenated.column(METADATA_SIZE)
                        size_field = concatenated.schema.field(METADATA_SIZE)

                        # Combine: original (with parent_id) + size + offset from VirtualZIP
                        all_arrays = [
                            original_level.column(i)
                            for i in range(original_level.num_columns)
                        ]
                        all_arrays.extend([size_array, offset_array])

                        all_fields = list(original_level.schema)
                        all_fields.extend([size_field, offset_field])

                        combined_schema = pa.schema(all_fields)
                        enriched_level = pa.Table.from_arrays(
                            all_arrays, schema=combined_schema
                        )

                        # Reorder to place internal:* columns at the end
                        enriched_level = reorder_internal_columns(enriched_level)

                        # Sort by parent_id to maintain hierarchical order
                        if METADATA_PARENT_ID in enriched_level.schema.names:
                            import pyarrow.compute as pc

                            sort_indices = pc.sort_indices(
                                enriched_level.column(METADATA_PARENT_ID)
                            )
                            enriched_level = enriched_level.take(sort_indices)

                        metadata_package.levels[depth] = enriched_level
                    else:
                        metadata_package.levels[depth] = original_level

                    logger.debug(
                        f"Level {depth}: {metadata_package.levels[depth].num_rows} samples "
                        f"(parent_id={METADATA_PARENT_ID in metadata_package.levels[depth].schema.names})"
                    )

            # Write consolidated METADATA/levelX.parquet files
            temp_level_files = {}
            for i, level_table in enumerate(metadata_package.levels):
                arc_path = f"METADATA/level{i}.parquet"
                temp_path = self.temp_dir / f"{uuid.uuid4().hex}_level{i}.parquet"

                # Merge CDC defaults with user kwargs
                from tacotoolbox._constants import PARQUET_CDC_DEFAULT_CONFIG

                parquet_config = {**PARQUET_CDC_DEFAULT_CONFIG, **parquet_kwargs}
                parquet_config.pop("mode", None)

                pq.write_table(level_table, temp_path, **parquet_config)
                real_size = temp_path.stat().st_size
                virtual_zip.add_file(str(temp_path), arc_path, file_size=real_size)
                temp_level_files[arc_path] = temp_path
                self._register_temp_file(temp_path)

                logger.debug(f"Added {arc_path} ({real_size} bytes)")

            # Add COLLECTION.json
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
            self._register_temp_file(temp_json)

            # Final offset calculation
            virtual_zip.calculate_offsets()

            logger.debug("Preparing final file lists for ZIP creation")

            all_src_files = list(src_files)
            all_arc_files = list(arc_files)

            # Add __meta__ files (bottom-up order)
            for depth in range(max_depth, 0, -1):
                for folder_path in folders_by_depth[depth]:
                    meta_arc_path = f"{folder_path}__meta__"
                    if meta_arc_path in temp_meta_files:
                        temp_path = temp_meta_files[meta_arc_path]
                        all_src_files.append(str(temp_path))
                        all_arc_files.append(meta_arc_path)

            # Add METADATA/levelX.parquet files
            for i in range(len(metadata_package.levels)):
                arc_path = f"METADATA/level{i}.parquet"
                temp_path = temp_level_files[arc_path]
                all_src_files.append(str(temp_path))
                all_arc_files.append(arc_path)

            # Add COLLECTION.json
            all_src_files.append(str(temp_json))
            all_arc_files.append("COLLECTION.json")

            logger.info(f"Writing ZIP with {len(all_src_files)} total files")

            # Write ZIP with placeholder header
            header_entries = [(0, 0) for _ in range(num_entries)]

            tacozip.create(
                zip_path=str(self.output_path),
                src_files=all_src_files,
                arc_files=all_arc_files,
                entries=header_entries,
            )

            # Update TACO_HEADER with real offsets
            metadata_offsets, metadata_sizes = self._get_metadata_offsets()
            collection_offset, collection_size = self._get_collection_offset()

            real_entries = [
                *zip(metadata_offsets, metadata_sizes, strict=False),
                (collection_offset, collection_size),
            ]

            tacozip.update_header(zip_path=str(self.output_path), entries=real_entries)

            logger.info(f"ZIP created successfully: {self.output_path}")

        except Exception as e:
            logger.exception("Failed to create ZIP")
            raise TacoCreationError(
                f"Failed to create ZIP at '{self.output_path}': {e}"
            ) from e
        else:
            return self.output_path
        finally:
            self._cleanup()

    def _register_temp_file(self, path: pathlib.Path) -> None:
        """Register temporary file for automatic cleanup."""
        self._cleanup_stack.callback(path.unlink, missing_ok=True)

    def _extract_folder_order(self, arc_files: list[str]) -> list[str]:
        """Extract folder paths that need __meta__ files (Level 1+ only)."""
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
        """Group folders by depth for bottom-up processing."""
        by_depth: dict[int, list[str]] = {}

        for folder in folder_order:
            depth = folder.count("/") - 1

            if depth not in by_depth:
                by_depth[depth] = []
            by_depth[depth].append(folder)

        return by_depth

    def _write_single_parquet(
        self, table: pa.Table, folder_path: str, **parquet_kwargs: Any
    ) -> pathlib.Path:
        """Write a single __meta__ Parquet file to temp directory."""
        identifier = folder_path.replace("/", "_").strip("_")
        temp_path = self.temp_dir / f"{uuid.uuid4().hex}_{identifier}.parquet"

        filtered_table = self._filter_metadata_columns(table)

        # Default config for local __meta__ (simple, no CDC)
        default_config = {"compression": "zstd"}
        parquet_config = {**default_config, **parquet_kwargs}
        parquet_config.pop("mode", None)

        pq.write_table(filtered_table, temp_path, **parquet_config)

        self._register_temp_file(temp_path)
        return temp_path

    def _filter_metadata_columns(self, table: pa.Table) -> pa.Table:
        """Filter columns for __meta__ files (removes 'path' column)."""
        exclude_columns = {"path"}
        cols_to_keep = [col for col in table.schema.names if col not in exclude_columns]
        return table.select(cols_to_keep) if cols_to_keep else table

    def _get_metadata_offsets(self) -> tuple[list[int], list[int]]:
        """Get offsets and sizes for METADATA/levelX.parquet files."""
        offsets = []
        sizes = []

        with zipfile.ZipFile(self.output_path, "r") as zf, open(
            self.output_path, "rb"
        ) as f:
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
        """Get offset and size for COLLECTION.json."""
        with zipfile.ZipFile(self.output_path, "r") as zf, open(
            self.output_path, "rb"
        ) as f:
            info = zf.getinfo("COLLECTION.json")

            f.seek(info.header_offset)
            lfh = f.read(30)

            filename_len = int.from_bytes(lfh[26:28], "little")
            extra_len = int.from_bytes(lfh[28:30], "little")

            data_offset = info.header_offset + 30 + filename_len + extra_len
            data_size = info.compress_size

            return data_offset, data_size

    def _cleanup(self) -> None:
        """Clean up all temporary files created during ZIP creation."""
        logger.debug("Cleaning up temporary files")
        self._cleanup_stack.close()


def _add_zip_offsets(
    metadata_table: pa.Table,
    offsets_map: dict[str, tuple[int, int]],
    folder_path: str = "",
) -> pa.Table:
    """
    Add internal:offset and internal:size columns for ZIP containers.

    Both values come from VirtualZIP. For FOLDER samples, points to __meta__ file.
    For FILE samples, points to actual data file.

    Raises:
        TacoCreationError: If offsets are missing for non-padding samples
    """
    offsets: list[int | None] = []
    sizes: list[int | None] = []
    missing_offsets = []

    id_column = metadata_table.column("id")
    type_column = metadata_table.column("type")

    for i in range(metadata_table.num_rows):
        sample_id = id_column[i].as_py()
        sample_type = type_column[i].as_py()

        # Build archive path: FOLDERs point to __meta__, FILEs to data
        if sample_type == "FOLDER":
            arc_path = (
                f"{folder_path}{sample_id}/__meta__"
                if folder_path
                else f"DATA/{sample_id}/__meta__"
            )
        else:
            arc_path = (
                f"{folder_path}{sample_id}" if folder_path else f"DATA/{sample_id}"
            )

        # Get offset and size from VirtualZIP
        if arc_path in offsets_map:
            offset, size = offsets_map[arc_path]
            offsets.append(offset)
            sizes.append(size)
        else:
            if not is_padding_id(sample_id):
                missing_offsets.append(arc_path)
            offsets.append(None)
            sizes.append(None)

    # Fail fast if offsets are missing (indicates a bug in VirtualZIP calculation)
    if missing_offsets:
        raise TacoCreationError(
            f"Offsets not found for {len(missing_offsets)} non-padding samples. "
            f"First 5: {missing_offsets[:5]}"
        )

    # Add size column
    size_array = pa.array(sizes, type=pa.int64())
    size_field = pa.field(METADATA_SIZE, pa.int64())
    result_table = metadata_table.append_column(size_field, size_array)

    # Add offset column
    offset_array = pa.array(offsets, type=pa.int64())
    offset_field = pa.field(METADATA_OFFSET, pa.int64())
    result_table = result_table.append_column(offset_field, offset_array)

    # Place internal:* columns at the end for consistency
    return reorder_internal_columns(result_table)
