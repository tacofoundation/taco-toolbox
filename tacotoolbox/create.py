import functools
import json
import pathlib
import shutil
import struct
import tempfile
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, suppress
from pathlib import Path

import polars as pl

from tacotoolbox.taco.datamodel import Taco
from tacotoolbox.tortilla.datamodel import Tortilla


def requires_tacotiff(func):
    """Simple decorator to ensure tacotiff is available."""
    _checked = False

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal _checked
        if not _checked:
            try:
                import tacotiff
            except ImportError as err:
                raise ImportError("tacotiff required. Install: pip install tacotiff") from err
            _checked = True
        return func(*args, **kwargs)

    return wrapper


class TacoCreationError(Exception):
    """Custom exception for TACO creation errors."""

    pass


def parse_size(size_str: str) -> int:
    """Parse human-readable size to bytes."""
    units = {"TB": 1024**4, "GB": 1024**3, "MB": 1024**2, "KB": 1024, "B": 1}

    size_str = size_str.upper().strip()

    # Check units in descending order by length to avoid "GB" matching "B" first
    for unit in sorted(units.keys(), key=len, reverse=True):
        if size_str.endswith(unit):
            number_part = size_str[: -len(unit)].strip()
            if not number_part:
                raise ValueError(f"Invalid size format: {size_str}")
            return int(float(number_part) * units[unit])

    raise ValueError(f"Invalid size format: {size_str}")


def calculate_sample_size(sample) -> int:
    """Calculate total size of a sample (recursive for TORTILLA types)."""
    if sample.type == "TORTILLA":
        # Recursive case: sum all nested samples
        return sum(calculate_sample_size(s) for s in sample.path.samples)
    else:
        # Base case: file size
        return sample.path.stat().st_size


def group_samples_by_size(samples: list, split_size: int) -> list[list]:
    """Group consecutive samples into chunks that don't exceed split_size."""
    if not samples:
        return []

    chunks = []
    current_chunk = []
    current_size = 0

    for sample in samples:
        sample_size = calculate_sample_size(sample)

        # If single sample exceeds limit, put it in its own chunk
        if sample_size > split_size:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_size = 0
            chunks.append([sample])
            continue

        # If adding this sample would exceed limit, start new chunk
        if current_size + sample_size > split_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = [sample]
            current_size = sample_size
        else:
            current_chunk.append(sample)
            current_size += sample_size

    # Add final chunk if not empty
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


class FileExtractor:
    """Handles file extraction for TACO creation - DATA/ files only."""

    @staticmethod
    def extract_files_recursive(
        samples,
        data_root: str = "DATA/",
        src_files: list[str] | None = None,
        arc_files: list[str] | None = None,
        path_prefix: str = "",
    ) -> tuple[list[str], list[str]]:
        """Recursively extract ONLY data files (no metadata.parquet anywhere)."""
        if src_files is None:
            src_files = []
        if arc_files is None:
            arc_files = []

        for sample in samples:
            if sample.type == "TORTILLA":
                # Build the new path prefix by appending current sample id
                new_path_prefix = (
                    f"{path_prefix}{sample.id}/" if path_prefix else f"{sample.id}/"
                )

                # Recurse into nested samples - NO metadata.parquet generation
                FileExtractor.extract_files_recursive(
                    sample.path.samples,
                    data_root,
                    src_files,
                    arc_files,
                    path_prefix=new_path_prefix,
                )
            else:
                # Only extract actual data files
                src_files.append(str(sample.path))
                file_suffix = Path(sample.path).suffix
                arc_files.append(f"{data_root}{path_prefix}{sample.id}{file_suffix}")

        return src_files, arc_files


class HierarchyGenerator:
    """Handles generation of METADATA/HIERARCHY/ structure with internal positioning to DATA/ files."""

    def __init__(
        self,
        taco: Taco,
        zip_path: pathlib.Path,
        quiet: bool = False,
        remove_path_column: bool = True,
    ):
        self.taco = taco
        self.zip_path = zip_path
        self.quiet = quiet
        self.remove_path_column = remove_path_column

    def generate_hierarchy_metadata(
        self, temp_dir: pathlib.Path, max_workers: int | None = None
    ) -> dict[str, pathlib.Path]:
        """Generate HIERARCHY/ metadata.parquet files with internal positioning to DATA/ files."""

        # Get file positions from ZIP (DATA/ files only)
        data_file_positions = self._get_zip_data_offsets()
        data_lookup = self._create_data_position_lookup(data_file_positions)

        # Collect all TORTILLA paths and their samples
        tortilla_map = self._collect_tortilla_paths(self.taco.tortilla.samples)

        if not tortilla_map:
            return {}

        # Generate metadata.parquet files in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(
                    self._generate_single_hierarchy_metadata,
                    tortilla_path,
                    samples,
                    data_lookup,
                    temp_dir,
                ): tortilla_path
                for tortilla_path, samples in tortilla_map.items()
            }

            results = {}
            for future in as_completed(future_to_path):
                tortilla_path = future_to_path[future]
                try:
                    parquet_path = future.result()
                    results[tortilla_path] = parquet_path
                except Exception as e:
                    raise TacoCreationError(
                        f"Failed to generate hierarchy metadata for {tortilla_path}: {e}"
                    ) from e

            return results

    def _get_zip_data_offsets(self) -> pl.DataFrame:
        """Get actual data offsets for DATA/ files only."""
        files_info = []

        with zipfile.ZipFile(self.zip_path, "r") as zip_file, open(self.zip_path, "rb") as f:
            for info in zip_file.infolist():
                if info.filename == "TACO_HEADER" or info.filename.endswith("/"):
                    continue

                # Only process DATA/ files
                if not info.filename.startswith("DATA/"):
                    continue

                # Seek to local file header
                f.seek(info.header_offset)
                header = f.read(30)

                if len(header) < 30:
                    continue

                # Validate local file header signature
                signature = struct.unpack("<I", header[0:4])[0]
                if signature != 0x04034B50:
                    continue

                filename_len = struct.unpack("<H", header[26:28])[0]
                extra_len = struct.unpack("<H", header[28:30])[0]

                files_info.append(
                    {
                        "id": info.filename,
                        "internal:offset": info.header_offset
                        + 30
                        + filename_len
                        + extra_len,
                        "internal:size": info.compress_size,
                    }
                )

        return pl.DataFrame(files_info)

    def _create_data_position_lookup(
        self, file_positions: pl.DataFrame
    ) -> dict[tuple[str, str], tuple[int, int]]:
        """Create lookup table mapping (folder_path, sample_id) to (offset, size) for DATA/ files."""
        lookup = {}

        for row in file_positions.iter_rows(named=True):
            file_path = row["id"]  # e.g., "DATA/tortilla_a/sample1.tif"
            path_parts = file_path.split("/")

            if len(path_parts) >= 3:  # DATA/folder/file
                folder_path = "/".join(path_parts[1:-1]) + "/"  # tortilla_a/
                filename = path_parts[-1]  # sample1.tif
                sample_id = filename.split(".")[0]  # sample1

                lookup[(folder_path, sample_id)] = (
                    row["internal:offset"],
                    row["internal:size"],
                )

            elif len(path_parts) == 2:  # DATA/file (level 0 direct files)
                filename = path_parts[-1]
                sample_id = filename.split(".")[0]

                # Level 0 files use empty folder path
                lookup[("", sample_id)] = (row["internal:offset"], row["internal:size"])

        return lookup

    def _collect_tortilla_paths(
        self, samples: list, current_path: str = ""
    ) -> dict[str, list]:
        """Recursively collect all TORTILLA samples and their paths."""
        tortilla_map = {}

        for sample in samples:
            if sample.type == "TORTILLA":
                # Build path for this TORTILLA
                sample_path = (
                    f"{current_path}{sample.id}/" if current_path else f"{sample.id}/"
                )

                # Store this TORTILLA and its samples
                tortilla_map[sample_path] = sample.path.samples

                # Recurse into nested levels
                nested_tortillas = self._collect_tortilla_paths(
                    sample.path.samples, sample_path
                )
                tortilla_map.update(nested_tortillas)

        return tortilla_map

    def _generate_single_hierarchy_metadata(
        self,
        tortilla_path: str,
        samples: list,
        data_lookup: dict,
        temp_dir: pathlib.Path,
    ) -> pathlib.Path:
        """Generate single metadata.parquet for a TORTILLA with internal positioning to DATA/ files."""

        # Export metadata from all samples
        metadata_dfs = []
        for sample in samples:
            sample_df = sample.export_metadata()
            metadata_dfs.append(sample_df)

        # Concatenate all sample metadata
        consolidated_df = pl.concat(metadata_dfs, how="vertical")

        # Add internal positioning columns
        consolidated_df = consolidated_df.with_columns(
            [
                pl.lit(None, dtype=pl.Int64).alias("internal:offset"),
                pl.lit(None, dtype=pl.Int64).alias("internal:size"),
                pl.lit(None, dtype=pl.Binary).alias("internal:header"),
            ]
        )

        # Process each row to add positioning to DATA/ files
        rows_data = []
        for row in consolidated_df.iter_rows(named=True):
            row_dict = dict(row)
            sample_id = row_dict["id"]

            # Look up position using folder path and sample_id (points to DATA/ files)
            key = (tortilla_path, sample_id)
            if key in data_lookup:
                row_dict["internal:offset"], row_dict["internal:size"] = data_lookup[
                    key
                ]

            # Add TACOTIFF header if needed
            if row_dict["type"] == "TACOTIFF" and "path" in row_dict:
                row_dict["internal:header"] = self._get_tacotiff_header(
                    row_dict["path"]
                )

            rows_data.append(row_dict)

        # Recreate DataFrame with positioning
        result_df = pl.DataFrame(rows_data, schema=consolidated_df.schema)

        # Remove path column if configured to do so
        if self.remove_path_column and "path" in result_df.columns:
            result_df = result_df.drop("path")

        # Generate UUID-based temporary filename
        temp_file = temp_dir / f"{uuid.uuid4().hex}.parquet"

        # Write parquet file
        result_df.write_parquet(temp_file)

        return temp_file

    @requires_tacotiff
    def _get_tacotiff_header(self, path: str) -> bytes | None:
        """Get TACOTIFF header information as binary."""
        import tacotiff  # Safe after decorator check

        try:
            header_data = tacotiff.metadata_from_tiff(path)
            if header_data is None:
                return None
            # Handle both bytes and string returns from tacotiff.metadata_from_tiff
            return (
                header_data
                if isinstance(header_data, bytes)
                else header_data.encode("utf-8")
            )
        except Exception as e:
            if not self.quiet:
                print(f"Warning: Could not extract header for {path}: {e}")
            return None


class ConsolidatedMetadataProcessor:
    """Handles consolidated metadata processing with references to HIERARCHY/ metadata.parquet files."""

    def __init__(
        self,
        taco: Taco,
        zip_path: pathlib.Path,
        quiet: bool = False,
        remove_path_column: bool = True,
    ):
        self.taco = taco
        self.zip_path = zip_path
        self.quiet = quiet
        self.remove_path_column = remove_path_column
        self.original_depth = taco.tortilla._current_depth

    def generate_consolidated_metadata(self) -> list[pl.DataFrame]:
        """Generate consolidated metadata tables with TORTILLA samples pointing to HIERARCHY/ files."""

        # Get positions of both DATA/ files and HIERARCHY/ metadata.parquet files
        data_file_positions = self._get_zip_data_offsets()
        hierarchy_file_positions = self._get_zip_hierarchy_offsets()

        # Create lookups
        data_lookup = self._create_data_position_lookup(data_file_positions)
        hierarchy_lookup = self._create_hierarchy_position_lookup(
            hierarchy_file_positions
        )

        results = []

        # Generate consolidated metadata for all depth levels
        for depth in range(self.original_depth + 1):
            self.taco.tortilla._current_depth = depth
            df = self.taco.tortilla.export_metadata(deep=depth)

            if df is not None and len(df) > 0:
                processed_df = self._process_metadata_depth(
                    df, data_lookup, hierarchy_lookup
                )
                processed_df = self._remove_null_columns(processed_df)
                results.append(processed_df)

        self.taco.tortilla._current_depth = self.original_depth
        return results

    def _get_zip_data_offsets(self) -> pl.DataFrame:
        """Get actual data offsets for DATA/ files only."""
        files_info = []

        with zipfile.ZipFile(self.zip_path, "r") as zip_file, open(self.zip_path, "rb") as f:
            for info in zip_file.infolist():
                if info.filename == "TACO_HEADER" or info.filename.endswith("/"):
                    continue

                # Only process DATA/ files
                if not info.filename.startswith("DATA/"):
                    continue

                # Seek to local file header
                f.seek(info.header_offset)
                header = f.read(30)

                if len(header) < 30:
                    continue

                # Validate local file header signature
                signature = struct.unpack("<I", header[0:4])[0]
                if signature != 0x04034B50:
                    continue

                filename_len = struct.unpack("<H", header[26:28])[0]
                extra_len = struct.unpack("<H", header[28:30])[0]

                files_info.append(
                    {
                        "id": info.filename,
                        "internal:offset": info.header_offset
                        + 30
                        + filename_len
                        + extra_len,
                        "internal:size": info.compress_size,
                    }
                )

        return pl.DataFrame(files_info)

    def _get_zip_hierarchy_offsets(self) -> pl.DataFrame:
        """Get actual offsets for METADATA/HIERARCHY/ metadata.parquet files."""
        files_info = []

        with zipfile.ZipFile(self.zip_path, "r") as zip_file, open(self.zip_path, "rb") as f:
            for info in zip_file.infolist():
                if info.filename == "TACO_HEADER" or info.filename.endswith("/"):
                    continue

                # Only process METADATA/HIERARCHY/ metadata.parquet files
                if not (
                    info.filename.startswith("METADATA/HIERARCHY/")
                    and info.filename.endswith("metadata.parquet")
                ):
                    continue

                # Seek to local file header
                f.seek(info.header_offset)
                header = f.read(30)

                if len(header) < 30:
                    continue

                # Validate local file header signature
                signature = struct.unpack("<I", header[0:4])[0]
                if signature != 0x04034B50:
                    continue

                filename_len = struct.unpack("<H", header[26:28])[0]
                extra_len = struct.unpack("<H", header[28:30])[0]

                files_info.append(
                    {
                        "id": info.filename,
                        "internal:offset": info.header_offset
                        + 30
                        + filename_len
                        + extra_len,
                        "internal:size": info.compress_size,
                    }
                )

        return pl.DataFrame(files_info)

    def _create_data_position_lookup(
        self, file_positions: pl.DataFrame
    ) -> dict[str, tuple[int, int]]:
        """Create lookup table mapping sample_id to (offset, size) for DATA/ files."""
        lookup = {}

        for row in file_positions.iter_rows(named=True):
            file_path = row["id"]  # e.g., "DATA/tortilla_a/sample1.tif"
            path_parts = file_path.split("/")

            if len(path_parts) >= 3:  # DATA/folder/file
                filename = path_parts[-1]  # sample1.tif
                sample_id = filename.split(".")[0]  # sample1
                lookup[sample_id] = (row["internal:offset"], row["internal:size"])

            elif len(path_parts) == 2:  # DATA/file (level 0 direct files)
                filename = path_parts[-1]
                sample_id = filename.split(".")[0]
                lookup[sample_id] = (row["internal:offset"], row["internal:size"])

        return lookup

    def _create_hierarchy_position_lookup(
        self, file_positions: pl.DataFrame
    ) -> dict[str, tuple[int, int]]:
        """Create lookup table mapping TORTILLA sample_id to (offset, size) for HIERARCHY/ metadata.parquet files."""
        lookup = {}

        for row in file_positions.iter_rows(named=True):
            file_path = row[
                "id"
            ]  # e.g., "METADATA/HIERARCHY/tortilla_a/metadata.parquet"
            path_parts = file_path.split("/")

            if len(path_parts) >= 4:  # METADATA/HIERARCHY/tortilla_a/metadata.parquet
                tortilla_id = path_parts[-2]  # tortilla_a (TORTILLA sample_id)
                lookup[tortilla_id] = (row["internal:offset"], row["internal:size"])

        return lookup

    def _process_metadata_depth(
        self, df: pl.DataFrame, data_lookup: dict, hierarchy_lookup: dict
    ) -> pl.DataFrame:
        """Process metadata for consolidated levels with proper references."""

        # Add new columns with default values first
        df = df.with_columns(
            [
                pl.lit(None, dtype=pl.Int64).alias("internal:offset"),
                pl.lit(None, dtype=pl.Int64).alias("internal:size"),
                pl.lit(None, dtype=pl.Binary).alias("internal:header"),
            ]
        )

        # Process each row
        rows_data = []
        for row in df.iter_rows(named=True):
            row_dict = dict(row)
            sample_id = row_dict["id"]

            if row_dict["type"] == "TORTILLA":
                # For TORTILLA samples: point to HIERARCHY/ metadata.parquet file
                if sample_id in hierarchy_lookup:
                    row_dict["internal:offset"], row_dict["internal:size"] = (
                        hierarchy_lookup[sample_id]
                    )

                # TORTILLA doesn't need internal:header

            else:
                # For non-TORTILLA samples: point to DATA/ files
                if sample_id in data_lookup:
                    row_dict["internal:offset"], row_dict["internal:size"] = (
                        data_lookup[sample_id]
                    )

                # Add TACOTIFF header if needed
                if row_dict["type"] == "TACOTIFF" and "path" in row_dict:
                    row_dict["internal:header"] = self._get_tacotiff_header(
                        row_dict["path"]
                    )

            rows_data.append(row_dict)

        # Recreate DataFrame with updated data
        result_df = pl.DataFrame(rows_data, schema=df.schema)

        # Remove path column if configured to do so
        if self.remove_path_column and "path" in result_df.columns:
            result_df = result_df.drop("path")

        return result_df

    def _remove_null_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Remove columns that are completely null or empty."""
        cols_to_keep = []
        for col in df.columns:
            # First check if column has any non-null values
            if not df[col].is_null().all():
                # For string columns, also check for empty strings and "None"
                if df[col].dtype == pl.Utf8:
                    non_empty_count = df.filter(
                        (pl.col(col).is_not_null())
                        & (pl.col(col) != "")
                        & (pl.col(col) != "None")
                    ).height

                    if non_empty_count > 0:
                        cols_to_keep.append(col)
                else:
                    # For non-string columns, just check if not all null
                    cols_to_keep.append(col)

        return (
            df.select(cols_to_keep) if cols_to_keep else df.select([df.columns[0]])
        )  # Keep at least one column

    @requires_tacotiff
    def _get_tacotiff_header(self, path: str) -> bytes | None:
        """Get TACOTIFF header information as binary."""
        import tacotiff  # Safe after decorator check

        try:
            header_data = tacotiff.metadata_from_tiff(path)
            if header_data is None:
                return None
            # Handle both bytes and string returns from tacotiff.metadata_from_tiff
            return (
                header_data
                if isinstance(header_data, bytes)
                else header_data.encode("utf-8")
            )
        except Exception as e:
            if not self.quiet:
                print(f"Warning: Could not extract header for {path}: {e}")
            return None


class MetadataManager:
    """Manages metadata parquet file operations."""

    @staticmethod
    def get_parquet_offsets(zip_path: pathlib.Path) -> tuple[list[int], list[int]]:
        """Return offsets and lengths for consolidated metadata parquet files in METADATA/."""
        offsets = []
        lengths = []

        with zipfile.ZipFile(zip_path, "r") as zf:
            parquet_files = [
                info
                for info in zf.infolist()
                if info.filename.startswith("METADATA/")
                and info.filename.endswith(".parquet")
                and not info.filename.startswith(
                    "METADATA/HIERARCHY/"
                )  # Exclude HIERARCHY files
            ]
            parquet_files.sort(key=lambda x: x.filename)

            for info in parquet_files:
                offsets.append(
                    info.header_offset + len(info.extra) + len(info.filename) + 30
                )
                lengths.append(info.file_size)

        return offsets, lengths

    @staticmethod
    def get_collection_json_offset(zip_path: pathlib.Path) -> tuple[int, int]:
        """Return offset and length for COLLECTION.json file."""
        with zipfile.ZipFile(zip_path, "r") as zf, open(zip_path, "rb") as f:
            for info in zf.infolist():
                if info.filename == "COLLECTION.json":
                    # Seek to local file header
                    f.seek(info.header_offset)
                    header = f.read(30)

                    if len(header) < 30:
                        continue

                    # Validate local file header signature
                    signature = struct.unpack("<I", header[0:4])[0]
                    if signature != 0x04034B50:
                        continue

                    filename_len = struct.unpack("<H", header[26:28])[0]
                    extra_len = struct.unpack("<H", header[28:30])[0]

                    actual_offset = (
                        info.header_offset + 30 + filename_len + extra_len
                    )
                    return actual_offset, info.compress_size

        raise ValueError("COLLECTION.json not found in ZIP file")

    @staticmethod
    def setup_temp_directory(temp_dir: pathlib.Path | None = None) -> pathlib.Path:
        """Create secure temporary directory with UUID."""
        base_temp = temp_dir or pathlib.Path(tempfile.gettempdir())
        safe_folder = base_temp / uuid.uuid4().hex
        safe_folder.mkdir(parents=True, exist_ok=True)
        return safe_folder

    @staticmethod
    @contextmanager
    def temporary_parquet_files(
        metadata_tables: list[pl.DataFrame], output_path: pathlib.Path
    ):
        """Context manager for temporary parquet files."""
        temp_files = []
        try:
            for df in metadata_tables:
                temp_file = output_path.parent / f"{uuid.uuid4()}.parquet"
                df.write_parquet(temp_file)
                temp_files.append(temp_file)

            yield temp_files

        finally:
            for temp_file in temp_files:
                with suppress(Exception):
                    temp_file.unlink()

    @staticmethod
    @contextmanager
    def temporary_json_file(data: dict, output_path: pathlib.Path):
        """Context manager for temporary JSON file."""
        temp_file = output_path.parent / f"{uuid.uuid4()}.json"
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            yield temp_file
        finally:
            with suppress(Exception):
                temp_file.unlink()


def _create_single_taco(
    taco: Taco,
    output_path: pathlib.Path,
    quiet: bool,
    remove_path_column: bool,
    temp_dir: pathlib.Path | None = None,
) -> pathlib.Path:
    """Create a single TACO file with correct HIERARCHY-first workflow."""
    # Setup secure temporary directory
    safe_temp_dir = MetadataManager.setup_temp_directory(temp_dir)

    try:
        # 1. Extract DATA/ files only (no metadata.parquet anywhere)
        src_files, arc_files = FileExtractor.extract_files_recursive(
            samples=taco.tortilla.samples
        )

        # 2. Create initial ZIP with DATA/ files only
        import tacozip

        tacozip.create(
            zip_path=str(output_path),
            src_files=src_files,
            arc_files=arc_files,
            entries=[(0, 0) for _ in range(taco.tortilla._current_depth + 1)],
        )

        # 3. Generate HIERARCHY/ metadata.parquet files FIRST (independent datasets)
        hierarchy_generator = HierarchyGenerator(
            taco, output_path, quiet, remove_path_column
        )
        hierarchy_cache = hierarchy_generator.generate_hierarchy_metadata(safe_temp_dir)

        # 4. Append HIERARCHY/ structure to ZIP
        if hierarchy_cache:
            hierarchy_entries = []
            for tortilla_path, temp_parquet in hierarchy_cache.items():
                hierarchy_entries.append(
                    (
                        str(temp_parquet),
                        f"METADATA/HIERARCHY/{tortilla_path}metadata.parquet",
                    )
                )

            tacozip.append_files(zip_path=str(output_path), entries=hierarchy_entries)

        # 5. Generate consolidated metadata WITH references to HIERARCHY/
        consolidated_processor = ConsolidatedMetadataProcessor(
            taco, output_path, quiet, remove_path_column
        )
        consolidated_metadata = consolidated_processor.generate_consolidated_metadata()

        # 6. Append consolidated metadata to ZIP
        with MetadataManager.temporary_parquet_files(
            consolidated_metadata, output_path
        ) as temp_files:
            tacozip.append_files(
                zip_path=str(output_path),
                entries=[
                    (str(temp_file), f"METADATA/level{i}.parquet")
                    for i, temp_file in enumerate(temp_files)
                ],
            )

        # 7. Add COLLECTION.json FIRST
        taco_json = taco.model_dump()
        taco_json.pop("tortilla", None)
        with MetadataManager.temporary_json_file(taco_json, output_path) as temp_json:
            tacozip.append_files(
                zip_path=str(output_path), entries=[(str(temp_json), "COLLECTION.json")]
            )

        # 8. Update ghost header with BOTH entries (metadata consolidada + COLLECTION.json)
        metadata_offsets, metadata_lengths = MetadataManager.get_parquet_offsets(
            output_path
        )
        collection_offset, collection_length = (
            MetadataManager.get_collection_json_offset(output_path)
        )

        # Combinar ambos entries para el header
        all_entries = [
            *zip(metadata_offsets, metadata_lengths),
            (collection_offset, collection_length)
        ]

        tacozip.update_header(zip_path=str(output_path), entries=all_entries)

        if not quiet:
            print(f"TACO file created successfully: {output_path}")

    except Exception as e:
        raise TacoCreationError(f"Failed to create TACO file: {e}") from e
    else:
        return output_path
    finally:
        # 9. Cleanup temporary directory
        with suppress(Exception):
            shutil.rmtree(safe_temp_dir, ignore_errors=True)


def create(
    taco: Taco,
    output: str | pathlib.Path,
    split_size: str | None = None,
    quiet: bool = False,
    remove_path_column: bool = True,
    temp_dir: pathlib.Path | None = None,
) -> list[pathlib.Path]:
    """
    Create TACO file(s) with HIERARCHY-first architecture and proper referencing.

    A TACO is a ZIP64 container optimized for storing large datasets
    that require partial reading and random access.

    Args:
        taco: A TACO object containing the collection metadata and samples
        output: The path where the TACO file(s) will be saved
        split_size: Optional size limit for splitting (e.g., "5GB", "100MB", "2TB")
        quiet: Whether to suppress output messages
        remove_path_column: Whether to remove path column from metadata
        temp_dir: Optional custom temporary directory path

    Returns:
        list[pathlib.Path]: List of created TACO file paths

    Raises:
        TacoCreationError: If TACO creation fails
    """
    output_path = pathlib.Path(output)

    # If no split_size, use single-file logic
    if split_size is None:
        result = _create_single_taco(
            taco, output_path, quiet, remove_path_column, temp_dir
        )
        return [result]

    try:
        # Parse split size and group samples
        max_size = parse_size(split_size)
        sample_chunks = group_samples_by_size(taco.tortilla.samples, max_size)

        if len(sample_chunks) == 1:
            # No splitting needed
            result = _create_single_taco(
                taco, output_path, quiet, remove_path_column, temp_dir
            )
            return [result]

        # Create multiple TACO files
        created_files = []
        base_name = output_path.stem
        extension = output_path.suffix
        parent_dir = output_path.parent

        for i, chunk_samples in enumerate(sample_chunks, 1):
            # Create new Tortilla with chunk samples
            chunk_tortilla = Tortilla(samples=chunk_samples)

            # Create new Taco with same metadata but new tortilla
            chunk_taco_data = taco.model_dump()
            chunk_taco_data["tortilla"] = chunk_tortilla
            chunk_taco = Taco(**chunk_taco_data)

            # Generate chunk filename
            chunk_filename = f"{base_name}_part{i:03d}{extension}"
            chunk_path = parent_dir / chunk_filename

            # Create chunk TACO file
            result = _create_single_taco(
                chunk_taco, chunk_path, quiet, remove_path_column, temp_dir
            )
            created_files.append(result)

        if not quiet:
            print(
                f"Created {len(created_files)} TACO chunks with target size {split_size}"
            )

    except Exception as e:
        raise TacoCreationError(f"Failed to create TACO file: {e}") from e
    else:
        return created_files
