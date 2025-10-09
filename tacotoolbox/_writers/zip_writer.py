import contextlib
import json
import pathlib
import struct
import uuid
import zipfile
from typing import Any

import polars as pl
import pyarrow.parquet as pq
import tacozip

from tacotoolbox._metadata import OffsetEnricher
from tacotoolbox._types import DataLookup, LevelMetadata, MetadataPackage, OffsetPair


class ZipWriterError(Exception):
    """Raised when ZIP writing operations fail."""


class ZipOffsetReader:
    """Read internal offsets from ZIP files."""

    @staticmethod
    def read_file_offsets(zip_path: pathlib.Path, prefix: str) -> pl.DataFrame:
        """
        Read offsets for files matching prefix.

        Args:
            zip_path: Path to ZIP file
            prefix: Prefix to filter files (e.g., 'DATA/', 'METADATA/')

        Returns:
            DataFrame with columns: filename, internal:offset, internal:size
        """
        files_info = []

        with zipfile.ZipFile(zip_path, "r") as zip_file, open(zip_path, "rb") as f:
            for info in zip_file.infolist():
                # Skip TACO_HEADER and directories
                if info.filename == "TACO_HEADER" or info.filename.endswith("/"):
                    continue

                # Filter by prefix
                if not info.filename.startswith(prefix):
                    continue

                # Read local file header
                f.seek(info.header_offset)
                header = f.read(30)

                if len(header) < 30:
                    continue

                # Validate signature
                signature = struct.unpack("<I", header[0:4])[0]
                if signature != 0x04034B50:  # Local file header signature
                    continue

                # Calculate actual data offset
                filename_len = struct.unpack("<H", header[26:28])[0]
                extra_len = struct.unpack("<H", header[28:30])[0]
                actual_offset = info.header_offset + 30 + filename_len + extra_len

                files_info.append(
                    {
                        "filename": info.filename,
                        "internal:offset": actual_offset,
                        "internal:size": info.compress_size,
                    }
                )

        return pl.DataFrame(files_info) if files_info else pl.DataFrame()

    @staticmethod
    def create_data_lookup(df: pl.DataFrame) -> DataLookup:
        """
        Create lookup: archive_path -> (offset, size) for DATA/ files.

        Args:
            df: DataFrame from read_file_offsets() with DATA/ files

        Returns:
            Dictionary mapping archive path (e.g., "DATA/sample.tif") to (offset, size)
        """
        lookup: DataLookup = {}

        if df.is_empty():
            return lookup

        for row in df.iter_rows(named=True):
            # Use full archive path as key (not just sample_id)
            # This avoids collisions when multiple files have same ID in different paths
            lookup[row["filename"]] = (row["internal:offset"], row["internal:size"])

        return lookup

    @staticmethod
    def get_metadata_offsets(zip_path: pathlib.Path) -> list[OffsetPair]:
        """
        Get offsets for METADATA/levelX.parquet files in order.

        Args:
            zip_path: Path to ZIP file

        Returns:
            List of (offset, size) tuples sorted by level number
        """
        entries = []

        with zipfile.ZipFile(zip_path, "r") as zf, open(zip_path, "rb") as f:
            # Find all metadata parquets
            parquet_files = [
                info
                for info in zf.infolist()
                if info.filename.startswith("METADATA/")
                and info.filename.endswith(".parquet")
            ]

            # Sort by filename (level0.parquet, level1.parquet, ...)
            parquet_files.sort(key=lambda x: x.filename)

            for info in parquet_files:
                f.seek(info.header_offset)
                header = f.read(30)

                if len(header) >= 30:
                    signature = struct.unpack("<I", header[0:4])[0]
                    if signature == 0x04034B50:
                        filename_len = struct.unpack("<H", header[26:28])[0]
                        extra_len = struct.unpack("<H", header[28:30])[0]
                        actual_offset = (
                            info.header_offset + 30 + filename_len + extra_len
                        )
                        entries.append((actual_offset, info.compress_size))

        return entries

    @staticmethod
    def get_collection_offset(zip_path: pathlib.Path) -> OffsetPair:
        """
        Get offset for COLLECTION.json.

        Args:
            zip_path: Path to ZIP file

        Returns:
            (offset, size) tuple

        Raises:
            ZipWriterError: If COLLECTION.json not found
        """
        with zipfile.ZipFile(zip_path, "r") as zf, open(zip_path, "rb") as f:
            for info in zf.infolist():
                if info.filename == "COLLECTION.json":
                    f.seek(info.header_offset)
                    header = f.read(30)

                    if len(header) >= 30:
                        signature = struct.unpack("<I", header[0:4])[0]
                        if signature == 0x04034B50:
                            filename_len = struct.unpack("<H", header[26:28])[0]
                            extra_len = struct.unpack("<H", header[28:30])[0]
                            actual_offset = (
                                info.header_offset + 30 + filename_len + extra_len
                            )
                            return (actual_offset, info.compress_size)

        raise ZipWriterError("COLLECTION.json not found in ZIP file")


class ZipWriter:
    """Handle creation of .tacozip container files."""

    def __init__(self, output_path: pathlib.Path, quiet: bool = False) -> None:
        """
        Initialize ZIP writer.

        Args:
            output_path: Path for output .tacozip file
            quiet: Suppress progress messages
        """
        self.output_path = output_path
        self.quiet = quiet
        self.working_dir = output_path.parent
        self._temp_files: list[pathlib.Path] = []

    def create_complete_zip(
        self,
        src_files: list[str],
        arc_files: list[str],
        metadata_package: MetadataPackage,
        **kwargs: Any,
    ) -> pathlib.Path:
        """
        Create complete .tacozip file.

        ZIP format uses Parquet for metadata (immutable, columnar).

        Args:
            src_files: Source file paths for DATA/
            arc_files: Archive paths for DATA/
            metadata_package: Complete metadata bundle
            **kwargs: Additional kwargs for pyarrow.parquet.write_table()
                Examples: row_group_size, compression, use_dictionary, etc.

        Returns:
            Path to created .tacozip file

        Raises:
            ZipWriterError: If creation fails
        """
        try:
            # Save arc_files for enricher
            self.arc_files = arc_files

            # Step 1: Create ZIP with DATA/
            self._create_with_data(src_files, arc_files, metadata_package["max_depth"])

            # Step 2: Read DATA/ offsets
            data_df = ZipOffsetReader.read_file_offsets(self.output_path, "DATA/")
            data_lookup = ZipOffsetReader.create_data_lookup(data_df)

            # Step 3: Append metadata parquets
            self._append_metadata_parquets(
                metadata_package["levels"], data_lookup, **kwargs
            )

            # Step 4: Append COLLECTION.json
            self._append_collection_json(metadata_package["collection"])

            # Step 5: Update TACO_HEADER
            self._update_header()

            if not self.quiet:
                print(f"ZIP container created: {self.output_path}")

        except Exception as e:
            raise ZipWriterError(f"Failed to create ZIP container: {e}") from e
        else:
            return self.output_path
        finally:
            self._cleanup_temp_files()

    def _create_with_data(
        self, src_files: list[str], arc_files: list[str], max_depth: int
    ) -> None:
        """
        Create initial ZIP with DATA/ and placeholder header entries.

        Args:
            src_files: Source file paths
            arc_files: Archive paths
            max_depth: Maximum hierarchy depth (for placeholder count)
        """
        # Create placeholders: one per level + one for COLLECTION.json
        num_entries = max_depth + 2  # levels 0..max_depth + collection
        placeholder_entries = [(0, 0) for _ in range(num_entries)]

        tacozip.create(
            zip_path=str(self.output_path),
            src_files=src_files,
            arc_files=arc_files,
            entries=placeholder_entries,
        )

    def _append_metadata_parquets(
        self,
        levels: list[LevelMetadata],
        data_lookup: DataLookup,
        **kwargs: Any,
    ) -> None:
        """
        Write and append metadata parquets to ZIP.

        Args:
            levels: List of LevelMetadata dicts
            data_lookup: Lookup for DATA/ offsets by archive path
            **kwargs: Additional kwargs for pyarrow.parquet.write_table()
        """
        # Create enricher with arc_files for positional matching
        enricher = OffsetEnricher(self.output_path, self.arc_files, self.quiet)
        entries = []

        for i, level_meta in enumerate(levels):
            df = level_meta["dataframe"]

            # Enrich with internal:offset/size/header using positional matching
            df = enricher.enrich_metadata(df, data_lookup)

            # Write temporary parquet using PyArrow API
            temp_file = self._write_temp_parquet(df, i, **kwargs)
            self._temp_files.append(temp_file)

            entries.append((str(temp_file), f"METADATA/level{i}.parquet"))

        # Append all metadata files to ZIP
        tacozip.append_files(zip_path=str(self.output_path), entries=entries)

    def _append_collection_json(self, collection: dict[str, object]) -> None:
        """
        Write and append COLLECTION.json to ZIP.

        Args:
            collection: Dictionary for COLLECTION.json
        """
        temp_json = self.working_dir / f"{uuid.uuid4().hex}.json"
        self._temp_files.append(temp_json)

        with open(temp_json, "w", encoding="utf-8") as f:
            json.dump(collection, f, indent=4, ensure_ascii=False)

        tacozip.append_files(
            zip_path=str(self.output_path),
            entries=[(str(temp_json), "COLLECTION.json")],
        )

    def _update_header(self) -> None:
        """Update TACO_HEADER with final offsets."""
        metadata_offsets = ZipOffsetReader.get_metadata_offsets(self.output_path)
        collection_offset = ZipOffsetReader.get_collection_offset(self.output_path)

        all_entries = [*metadata_offsets, collection_offset]

        tacozip.update_header(zip_path=str(self.output_path), entries=all_entries)

    def _write_temp_parquet(
        self,
        df: pl.DataFrame,
        level: int,
        **kwargs: Any,
    ) -> pathlib.Path:
        """
        Write temporary parquet file using PyArrow API.

        Args:
            df: DataFrame to write
            level: Level number for filename
            **kwargs: Additional kwargs for pyarrow.parquet.write_table()

        Returns:
            Path to temporary file
        """
        temp_file = self.working_dir / f"{uuid.uuid4().hex}_level{level}.parquet"

        # Convert to Arrow table and write using PyArrow API
        arrow_table = df.to_arrow()
        pq.write_table(arrow_table, temp_file, **kwargs)

        return temp_file

    def _cleanup_temp_files(self) -> None:
        """Delete all temporary files."""
        for temp_file in self._temp_files:
            with contextlib.suppress(Exception):
                temp_file.unlink(missing_ok=True)
        self._temp_files.clear()
