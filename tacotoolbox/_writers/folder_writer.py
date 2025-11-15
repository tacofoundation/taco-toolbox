"""
Folder Writer - Create TACO datasets in FOLDER format.

This module handles the creation of FOLDER-format TACO containers with
dual metadata system:
- Local metadata: __meta__ files in each DATA/ subfolder (PARQUET format)
- Consolidated metadata: level*.parquet files in METADATA/ directory

The FOLDER format provides:
- Direct filesystem access to individual files
- Human-readable directory structure
- Parquet __meta__ for local access
- Parquet consolidated metadata with CDC for efficient queries
- Preserved internal:parent_id for hierarchical navigation

Structure example (level 0 = FILEs only):
    dataset/
    ├── DATA/
    │   ├── sample_001.tif
    │   ├── sample_002.tif
    │   └── sample_003.tif
    ├── METADATA/
    │   └── level0.parquet
    └── COLLECTION.json

Structure example (level 0 = FOLDERs, level 1 = FILEs):
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

Structure example (Change Detection - 3 levels deep):
    dataset/
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
import shutil
from typing import Any

import polars as pl

from tacotoolbox._constants import (
    FOLDER_COLLECTION_FILENAME,
    FOLDER_DATA_DIR,
    FOLDER_META_FILENAME,
    FOLDER_METADATA_DIR,
    METADATA_PARENT_ID,
)
from tacotoolbox._column_utils import write_parquet_file, write_parquet_file_with_cdc
from tacotoolbox._logging import get_logger
from tacotoolbox._metadata import MetadataPackage
from tacotoolbox._progress import ProgressContext, progress_bar, progress_scope  # ← NEW: Import progress utilities

logger = get_logger(__name__)


class FolderWriterError(Exception):
    """Raised when folder writing operations fail."""


class FolderWriter:
    """Handle creation of folder container structures with dual metadata."""

    def __init__(
        self, output_dir: pathlib.Path, quiet: bool = False, debug: bool = False
    ) -> None:
        """
        Initialize folder writer.

        Args:
            output_dir: Output directory path
            quiet: If True, hide progress bars (default: False)
            debug: If True, enable debug logging (default: False)
        """
        self.output_dir = output_dir
        self.quiet = quiet
        self.data_dir = output_dir / FOLDER_DATA_DIR
        self.metadata_dir = output_dir / FOLDER_METADATA_DIR

        logger.debug(f"FolderWriter initialized: output={output_dir}")

    def create_complete_folder(
        self,
        samples: list[Any],
        metadata_package: MetadataPackage,
        **kwargs: Any,
    ) -> pathlib.Path:
        """
        Create complete FOLDER TACO container.

        Args:
            samples: List of Sample objects
            metadata_package: Complete metadata package
            **kwargs: Additional Parquet writer parameters (e.g., compression_level)

        Returns:
            pathlib.Path: Path to created folder

        Raises:
            FolderWriterError: If folder creation fails
        """
        try:
            logger.info(f"Creating FOLDER container: {self.output_dir}")

            with ProgressContext(quiet=self.quiet):
                self._create_structure()
                self._copy_data_files(samples)
                self._write_local_metadata(metadata_package, **kwargs)
                self._write_consolidated_metadata(metadata_package, **kwargs)
                self._write_collection_json(metadata_package)

            logger.info(f"Folder container created: {self.output_dir}/")
            return self.output_dir

        except Exception as e:
            logger.error(f"Failed to create folder container: {e}")
            raise FolderWriterError(f"Failed to create folder container: {e}") from e

    def _create_structure(self) -> None:
        """Create base DATA/ and METADATA/ folders."""
        self.data_dir.mkdir(parents=True, exist_ok=False)
        self.metadata_dir.mkdir(parents=True, exist_ok=False)

        logger.debug(f"Created {FOLDER_DATA_DIR}/ and {FOLDER_METADATA_DIR}/")

    def _copy_data_files(self, samples: list[Any]) -> None:
        """
        Copy data files recursively with progress bar for large datasets.
        
        Shows progress bar if total size > 1GB.
        """
        # ← NEW: Calculate total size
        logger.debug("Calculating total size of samples")
        total_size = sum(_estimate_sample_size(sample) for sample in samples)
        
        # ← NEW: Show progress bar if > 1GB
        size_threshold = 1_000_000_000  # 1GB
        
        if total_size > size_threshold:
            logger.debug(f"Total size: {total_size / (1024**3):.2f} GB (showing progress)")
            with progress_scope(
                desc="Copying files",
                total=total_size,
                unit="B",
                colour="cyan"
            ) as pbar:
                self._copy_samples_recursive(samples, path_prefix="", pbar=pbar)
        else:
            logger.debug(f"Total size: {total_size / (1024**2):.2f} MB (no progress bar)")
            self._copy_samples_recursive(samples, path_prefix="")

    def _copy_samples_recursive(
        self, samples: list[Any], path_prefix: str = "", pbar=None  # ← NEW: Added pbar parameter
    ) -> None:
        """
        Recursively copy samples to DATA/.

        Args:
            samples: List of Sample objects
            path_prefix: Current path prefix in structure
            pbar: Optional progress bar to update
        """
        for sample in samples:
            if sample.type == "FOLDER":
                new_prefix = (
                    f"{path_prefix}{sample.id}/" if path_prefix else f"{sample.id}/"
                )
                nested_dir = self.data_dir / new_prefix.rstrip("/")
                nested_dir.mkdir(parents=True, exist_ok=True)

                self._copy_samples_recursive(
                    sample.path.samples, path_prefix=new_prefix, pbar=pbar  # ← NEW: Pass pbar
                )
            else:
                src_path = sample.path
                dst_path = self.data_dir / f"{path_prefix}{sample.id}"

                dst_path.parent.mkdir(parents=True, exist_ok=True)

                # ← NEW: Get file size before copying
                file_size = 0
                if hasattr(src_path, "stat"):
                    file_size = src_path.stat().st_size

                shutil.copy2(src_path, dst_path)
                
                # ← NEW: Update progress bar after copy
                if pbar is not None and file_size > 0:
                    pbar.update(file_size)

    def _write_local_metadata(
        self,
        metadata_package: MetadataPackage,
        **kwargs: Any,
    ) -> None:
        """
        Write local __meta__ files for each folder in PARQUET format.

        These files do NOT contain internal:parent_id (navigation is implicit
        via folder structure), but they preserve all other metadata fields.
        
        Shows progress bar if > 10 folders.

        Args:
            metadata_package: Complete metadata package
            **kwargs: Additional Parquet writer parameters (passed to write_parquet_file)

        Example:
            >>> writer._write_local_metadata(package, compression_level=22)
        """
        num_folders = len(metadata_package.local_metadata)
        logger.debug(f"Writing {num_folders} local __meta__ files")

        # ← NEW: Show progress bar if > 10 folders
        folder_threshold = 10
        
        if num_folders > folder_threshold:
            logger.debug(f"{num_folders} folders (showing progress)")
            items = progress_bar(
                metadata_package.local_metadata.items(),
                desc="Writing local __meta__",
                total=num_folders,
                unit="folder",
                colour="green"
            )
        else:
            logger.debug(f"{num_folders} folders (no progress bar)")
            items = metadata_package.local_metadata.items()

        for folder_path, local_df in items:
            meta_path = self.output_dir / f"{folder_path}{FOLDER_META_FILENAME}"

            meta_path.parent.mkdir(parents=True, exist_ok=True)

            # Write as Parquet with user-provided kwargs
            write_parquet_file(local_df, meta_path, **kwargs)

            logger.debug(f"Created {folder_path}{FOLDER_META_FILENAME}")

    def _write_consolidated_metadata(
        self,
        metadata_package: MetadataPackage,
        **kwargs: Any,
    ) -> None:
        """
        Write consolidated METADATA/levelX.parquet files with CDC.

        These files preserve ALL columns including internal:parent_id
        for hierarchical navigation via JOINs.

        Args:
            metadata_package: Complete metadata package
            **kwargs: Additional Parquet writer parameters (passed to write_parquet_file_with_cdc)

        Example:
            >>> writer._write_consolidated_metadata(package, compression_level=22)
        """
        logger.debug(
            f"Writing {len(metadata_package.levels)} consolidated metadata files"
        )

        for i, level_df in enumerate(metadata_package.levels):
            output_path = self.metadata_dir / f"level{i}.parquet"

            # Write consolidated as Parquet with CDC and user-provided kwargs
            write_parquet_file_with_cdc(level_df, output_path, **kwargs)

            has_parent_id = METADATA_PARENT_ID in level_df.columns
            logger.debug(
                f"Created {FOLDER_METADATA_DIR}/level{i}.parquet "
                f"({len(level_df)} rows, parent_id={has_parent_id})"
            )

    def _write_collection_json(self, metadata_package: MetadataPackage) -> None:
        """
        Write COLLECTION.json with pit_schema and field_schema embedded.

        Args:
            metadata_package: Complete metadata package
        """
        collection_path = self.output_dir / FOLDER_COLLECTION_FILENAME

        collection_with_schema = metadata_package.collection.copy()
        collection_with_schema["taco:pit_schema"] = metadata_package.pit_schema
        collection_with_schema["taco:field_schema"] = metadata_package.field_schema

        with open(collection_path, "w", encoding="utf-8") as f:
            json.dump(collection_with_schema, f, indent=4, ensure_ascii=False)

        logger.debug(f"Created {FOLDER_COLLECTION_FILENAME}")


# ← NEW: Helper function copied from create.py
def _estimate_sample_size(sample) -> int:
    """
    Estimate total size of a sample including nested samples.

    For FILE samples, returns actual file size from filesystem.
    For FOLDER samples, recursively sums all nested file sizes.

    Args:
        sample: Sample to estimate (FILE or FOLDER)

    Returns:
        int: Estimated size in bytes (0 if path doesn't support stat)
    """
    if sample.type == "FILE":
        if hasattr(sample.path, "stat"):
            return sample.path.stat().st_size
        return 0

    elif sample.type == "FOLDER":
        return sum(_estimate_sample_size(child) for child in sample.path.samples)

    return 0