"""
TACO Format Translation - Convert between ZIP and FOLDER formats.

This module provides bidirectional conversion between TACO container formats:
- ZIP → FOLDER: Extract complete structure with metadata
- FOLDER → ZIP: Package with offset recalculation

Key features:
- Preserves all metadata and hierarchical structure
- Handles AVRO colon sanitization (_COLON_ ↔ :)
- Regenerates __meta__ files with correct offsets for ZIP
- Supports all PIT schema patterns (FILES only, FOLDERs, mixed depths)

Functions:
    zip2folder(): Extract ZIP to FOLDER format
    folder2zip(): Package FOLDER into ZIP format

Example:
    >>> from tacotoolbox.translate import zip2folder, folder2zip
    >>> 
    >>> # Convert ZIP to FOLDER
    >>> zip2folder("dataset.tacozip", "dataset_folder/")
    >>> 
    >>> # Convert FOLDER to ZIP
    >>> folder2zip("dataset_folder/", "dataset.tacozip")
"""

import json
from pathlib import Path

import polars as pl
from tacoreader import TacoDataset

from tacotoolbox._constants import AVRO_COLON_REPLACEMENT
from tacotoolbox._metadata import MetadataPackage
from tacotoolbox._writers.export_writer import ExportWriter
from tacotoolbox._writers.zip_writer import ZipWriter


class TranslateError(Exception):
    """Raised when translation operations fail."""


# =============================================================================
# PUBLIC API
# =============================================================================


def zip2folder(
    zip_path: str | Path,
    folder_output: str | Path,
    nworkers: int = 4,
    quiet: bool = False,
) -> Path:
    """
    Convert ZIP format TACO to FOLDER format.

    This is a simple wrapper around ExportWriter that converts a complete
    ZIP container to FOLDER format without any filtering.

    Args:
        zip_path: Path to input .tacozip file
        folder_output: Path to output folder
        nworkers: Number of parallel workers for file copying
        quiet: If True, suppress progress messages

    Returns:
        Path to created FOLDER container

    Raises:
        TranslateError: If conversion fails

    Example:
        >>> zip2folder("dataset.tacozip", "dataset_folder/")
        PosixPath('dataset_folder')
    """
    try:
        dataset = TacoDataset(str(zip_path))
        
        writer = ExportWriter(
            dataset=dataset,
            output=Path(folder_output),
            nworkers=nworkers,
            quiet=quiet,
        )
        
        return writer.create_folder()
        
    except Exception as e:
        raise TranslateError(f"Failed to convert ZIP to FOLDER: {e}") from e


def folder2zip(
    folder_path: str | Path,
    zip_output: str | Path,
    quiet: bool = True,
    temp_dir: str | Path | None = None,
    **kwargs,
) -> Path:
    """
    Convert FOLDER format TACO to ZIP format.

    Reads existing metadata from FOLDER structure and reconstructs ZIP
    container with correct offsets. The __meta__ files are regenerated
    with internal:offset and internal:size columns.

    Process:
    1. Read COLLECTION.json
    2. Read consolidated metadata (METADATA/levelX.avro)
    3. Reconstruct local_metadata from consolidated (for __meta__ generation)
    4. Scan DATA/ for physical files
    5. Use ZipWriter to create ZIP (regenerates __meta__ with offsets)

    Args:
        folder_path: Path to input FOLDER container
        zip_output: Path to output .tacozip file
        quiet: If True, suppress progress messages
        temp_dir: Temporary directory for ZIP creation
        **kwargs: Additional arguments passed to ZipWriter

    Returns:
        Path to created .tacozip file

    Raises:
        TranslateError: If conversion fails

    Example:
        >>> folder2zip("dataset_folder/", "dataset.tacozip")
        PosixPath('dataset.tacozip')
    """
    folder_path = Path(folder_path)
    zip_output = Path(zip_output)

    try:
        if not quiet:
            print("Converting FOLDER to ZIP...")

        # 1. Read COLLECTION.json
        if not quiet:
            print("  [1/5] Reading COLLECTION.json...")
        collection = _read_collection(folder_path)

        # 2. Read consolidated metadata from METADATA/levelX.avro
        if not quiet:
            print("  [2/5] Reading consolidated metadata...")
        levels = _read_consolidated_metadata(folder_path)

        # 3. Reconstruct local_metadata from consolidated
        if not quiet:
            print("  [3/5] Reconstructing local metadata...")
        local_metadata = _reconstruct_local_metadata_from_levels(levels)

        # 4. Scan DATA/ for physical files
        if not quiet:
            print("  [4/5] Scanning data files...")
        src_files, arc_files = _scan_data_files(folder_path)

        # 5. Create MetadataPackage
        metadata_package = MetadataPackage(
            levels=levels,
            local_metadata=local_metadata,
            collection=collection,
            pit_schema=collection["taco:pit_schema"],
            field_schema=collection["taco:field_schema"],
            max_depth=len(levels) - 1,
        )

        # 6. Use ZipWriter to create ZIP (regenerates __meta__ with offsets)
        if not quiet:
            print("  [5/5] Creating ZIP container...")
        writer = ZipWriter(output_path=zip_output, quiet=quiet, temp_dir=temp_dir)
        result = writer.create_complete_zip(
            src_files=src_files,
            arc_files=arc_files,
            metadata_package=metadata_package,
            **kwargs,
        )

        if not quiet:
            print(f"Conversion complete: {result}")

        return result

    except Exception as e:
        raise TranslateError(f"Failed to convert FOLDER to ZIP: {e}") from e


# =============================================================================
# INTERNAL FUNCTIONS
# =============================================================================


def _read_collection(folder_path: Path) -> dict:
    """
    Read COLLECTION.json from FOLDER.

    Args:
        folder_path: Root path to FOLDER container

    Returns:
        Collection dictionary

    Raises:
        TranslateError: If COLLECTION.json missing or invalid
    """
    collection_path = folder_path / "COLLECTION.json"

    if not collection_path.exists():
        raise TranslateError(f"COLLECTION.json not found in {folder_path}")

    try:
        with open(collection_path, "r", encoding="utf-8") as f:
            collection = json.load(f)
    except json.JSONDecodeError as e:
        raise TranslateError(f"Invalid COLLECTION.json: {e}") from e

    # Validate required fields
    if "taco:pit_schema" not in collection:
        raise TranslateError("COLLECTION.json missing 'taco:pit_schema'")

    if "taco:field_schema" not in collection:
        raise TranslateError("COLLECTION.json missing 'taco:field_schema'")

    return collection


def _read_consolidated_metadata(folder_path: Path) -> list[pl.DataFrame]:
    """
    Read consolidated metadata from METADATA/levelX.avro files.

    CRITICAL: AVRO files use _COLON_ replacement for colons in column names.
    This function unsanitizes them back to standard format (internal:parent_id).

    Args:
        folder_path: Root path to FOLDER container

    Returns:
        List of DataFrames, one per level (sorted by level number)

    Raises:
        TranslateError: If metadata files missing or invalid
    """
    metadata_dir = folder_path / "METADATA"

    if not metadata_dir.exists():
        raise TranslateError(f"METADATA directory not found in {folder_path}")

    # Find all levelX.avro files
    level_files = sorted(metadata_dir.glob("level*.avro"))

    if not level_files:
        raise TranslateError(f"No level*.avro files found in {metadata_dir}")

    levels = []
    for level_file in level_files:
        try:
            df = pl.read_avro(level_file)
            df = _unsanitize_colons(df)
            levels.append(df)
        except Exception as e:
            raise TranslateError(f"Failed to read {level_file}: {e}") from e

    return levels


def _unsanitize_colons(df: pl.DataFrame) -> pl.DataFrame:
    """
    Convert AVRO sanitized column names back to standard format.

    AVRO does not support colons in field names, so they are replaced
    with _COLON_ during serialization. This function reverses that.

    Example:
        internal_COLON_parent_id → internal:parent_id
        stac_COLON_crs → stac:crs

    Args:
        df: DataFrame with sanitized column names

    Returns:
        DataFrame with standard column names
    """
    rename_map = {
        col: col.replace(AVRO_COLON_REPLACEMENT, ":")
        for col in df.columns
        if AVRO_COLON_REPLACEMENT in col
    }

    if rename_map:
        return df.rename(rename_map)
    return df


def _reconstruct_local_metadata_from_levels(
    levels: list[pl.DataFrame],
) -> dict[str, pl.DataFrame]:
    """
    Reconstruct local_metadata from consolidated levels.

    The __meta__ files in FOLDER format do NOT contain offset/size columns
    (they're filesystem-based, not byte-offset based). When converting to ZIP,
    we need to regenerate these files with the correct structure so ZipWriter
    can add offset/size columns.

    This function rebuilds the local_metadata dictionary by:
    1. Iterating through each level (except the last, which has no children)
    2. Finding FOLDER samples at each level
    3. Extracting their children from the next level using internal:parent_id
    4. Removing columns not needed in __meta__ (internal:relative_path)

    Args:
        levels: List of DataFrames (consolidated metadata per level)

    Returns:
        Dictionary mapping folder_path → children DataFrame

    Example:
        >>> levels = [level0_df, level1_df, level2_df]
        >>> local_metadata = _reconstruct_local_metadata_from_levels(levels)
        >>> local_metadata.keys()
        dict_keys(['DATA/folder_A/', 'DATA/folder_A/subfolder_B/', ...])
    """
    local_metadata = {}

    # Iterate over all levels except the last (it has no children)
    for level_idx in range(len(levels) - 1):
        current_level = levels[level_idx]
        next_level = levels[level_idx + 1]

        # Find all FOLDERs at this level
        folders = current_level.filter(pl.col("type") == "FOLDER")

        for folder_row in folders.iter_rows(named=True):
            parent_id = folder_row["internal:parent_id"]

            # Construct folder_path based on level
            if level_idx == 0:
                # Level 0: use 'id' directly (no relative_path column)
                folder_path = f"DATA/{folder_row['id']}/"
            else:
                # Level 1+: use 'internal:relative_path'
                folder_path = f"DATA/{folder_row['internal:relative_path']}/"

            # Get children from next level that belong to this folder
            children = next_level.filter(pl.col("internal:parent_id") == parent_id)

            # Remove columns that don't belong in __meta__ local files
            cols_to_drop = []
            if "internal:relative_path" in children.columns:
                cols_to_drop.append("internal:relative_path")

            if cols_to_drop:
                children = children.drop(cols_to_drop)

            local_metadata[folder_path] = children

    return local_metadata


def _scan_data_files(folder_path: Path) -> tuple[list[str], list[str]]:
    """
    Scan DATA/ directory for physical files.

    Builds parallel lists of source paths (absolute filesystem paths) and
    archive paths (paths within ZIP container). Excludes __meta__ files
    since they're regenerated during ZIP creation.

    Args:
        folder_path: Root path to FOLDER container

    Returns:
        Tuple of (src_files, arc_files):
            - src_files: List of absolute file paths
            - arc_files: List of archive paths (e.g., "DATA/folder_A/file.tif")

    Raises:
        TranslateError: If DATA directory missing or empty

    Example:
        >>> src_files, arc_files = _scan_data_files(Path("dataset/"))
        >>> src_files[0]
        '/absolute/path/to/dataset/DATA/sample_001.tif'
        >>> arc_files[0]
        'DATA/sample_001.tif'
    """
    data_dir = folder_path / "DATA"

    if not data_dir.exists():
        raise TranslateError(f"DATA directory not found in {folder_path}")

    src_files = []
    arc_files = []

    # Recursively find all files (excluding __meta__)
    for file_path in data_dir.rglob("*"):
        if file_path.is_file() and file_path.name != "__meta__":
            src_files.append(str(file_path))

            # Calculate relative path from DATA/
            relative = file_path.relative_to(data_dir)
            arc_files.append(f"DATA/{relative}")

    if not src_files:
        raise TranslateError(f"No data files found in {data_dir}")

    return src_files, arc_files