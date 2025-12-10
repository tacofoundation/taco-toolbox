"""
TACO Format Translation - Convert between ZIP and FOLDER formats.

This module provides bidirectional conversion between TACO container formats:
- ZIP -> FOLDER: Extract complete structure with metadata
- FOLDER -> ZIP: Package with offset recalculation

Key features:
- Preserves all metadata and hierarchical structure
- Regenerates __meta__ files with correct offsets for ZIP
- Supports all PIT schema patterns
"""

import asyncio
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tacoreader import TacoDataset

from tacotoolbox._exceptions import TacoCreationError
from tacotoolbox._logging import get_logger
from tacotoolbox._metadata import MetadataPackage
from tacotoolbox._writers.export_writer import ExportWriter
from tacotoolbox._writers.zip_writer import ZipWriter

logger = get_logger(__name__)


def zip2folder(
    input: str | Path,
    output: str | Path,
    limit: int = 100,
) -> Path:
    """
    Convert ZIP format TACO to FOLDER format.

    This is a wrapper around ExportWriter that converts a complete
    ZIP container to FOLDER format without any filtering.

    Args:
        input: Path to input .tacozip file
        output: Path to output folder
        limit: Maximum concurrent async operations (default: 100)

    Returns:
        Path: Path to created FOLDER

    Raises:
        TacoCreationError: If conversion fails

    Example:
        >>> import tacotoolbox
        >>> tacotoolbox.verbose(False)  # Hide progress
        >>> zip2folder("dataset.tacozip", "dataset_folder/")
        PosixPath('dataset_folder')
    """
    return asyncio.run(_zip2folder_async(input, output, limit))


async def _zip2folder_async(
    input: str | Path,
    output: str | Path,
    limit: int = 100,
) -> Path:
    """Internal async implementation for zip2folder."""
    try:
        logger.info(f"Converting ZIP to FOLDER: {input} → {output}")

        dataset = TacoDataset(str(input))

        writer = ExportWriter(
            dataset=dataset,
            output=Path(output),
            limit=limit,
        )

        result = await writer.create_folder()

        logger.info(f"Conversion complete: {result}")

    except Exception as e:
        logger.exception("Failed to convert ZIP to FOLDER")
        raise TacoCreationError(f"Failed to convert ZIP to FOLDER: {e}") from e

    else:
        return result


def folder2zip(
    input: str | Path,
    output: str | Path,
    temp_dir: str | Path | None = None,
    **kwargs,
) -> Path:
    """
    Convert FOLDER format TACO to ZIP format.

    Reads existing metadata from FOLDER structure and reconstructs ZIP
    container with correct offsets. The __meta__ files are regenerated
    with internal:offset and internal:size columns.

    Args:
        input: Path to input FOLDER directory
        output: Path to output .tacozip file
        temp_dir: Directory for temporary files (default: system temp)
        **kwargs: Additional Parquet writer parameters

    Returns:
        Path: Path to created .tacozip file

    Raises:
        TacoCreationError: If conversion fails

    Example:
        >>> import tacotoolbox
        >>> tacotoolbox.verbose(False)  # Hide progress
        >>> folder2zip("dataset_folder/", "dataset.tacozip")
        PosixPath('dataset.tacozip')

    Process:
        1. Read COLLECTION.json
        2. Read consolidated metadata (METADATA/levelX.parquet)
        3. Reconstruct local_metadata from consolidated (for __meta__ generation)
        4. Scan DATA/ for physical files
        5. Use ZipWriter to create ZIP (regenerates __meta__ with offsets)
    """
    input = Path(input)
    output = Path(output)

    # Convert temp_dir to Path if string
    if isinstance(temp_dir, str):
        temp_dir = Path(temp_dir)

    try:
        logger.info(f"Converting FOLDER to ZIP: {input} → {output}")

        # 1. Read COLLECTION.json
        logger.debug("Reading COLLECTION.json")
        collection = _read_collection(input)

        # 2. Read consolidated metadata from METADATA/levelX.parquet
        logger.debug("Reading consolidated metadata")
        levels = _read_consolidated_metadata(input)

        # 3. Reconstruct local_metadata from consolidated
        logger.debug("Reconstructing local metadata")
        local_metadata = _reconstruct_local_metadata_from_levels(levels)

        # 4. Scan DATA/ for physical files
        logger.debug("Scanning data files")
        src_files, arc_files = _scan_data_files(input)

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
        logger.debug("Creating ZIP container")

        # Progress bars controlled by logging level
        writer = ZipWriter(output_path=output, temp_dir=temp_dir)
        result = writer.create_complete_zip(
            src_files=src_files,
            arc_files=arc_files,
            metadata_package=metadata_package,
            **kwargs,
        )

        logger.info(f"Conversion complete: {result}")

    except Exception as e:
        logger.exception("Failed to convert FOLDER to ZIP")
        raise TacoCreationError(f"Failed to convert FOLDER to ZIP: {e}") from e

    else:
        return result


def _read_collection(folder_path: Path) -> dict:
    """
    Read COLLECTION.json from FOLDER.

    Raises:
        TacoCreationError: If file not found, invalid JSON, or missing required fields
    """
    collection_path = folder_path / "COLLECTION.json"

    if not collection_path.exists():
        raise TacoCreationError(f"COLLECTION.json not found in {folder_path}")

    try:
        with open(collection_path, encoding="utf-8") as f:
            collection = json.load(f)
    except json.JSONDecodeError as e:
        raise TacoCreationError(f"Invalid COLLECTION.json: {e}") from e

    # Validate required fields
    if "taco:pit_schema" not in collection:
        raise TacoCreationError("COLLECTION.json missing 'taco:pit_schema'")

    if "taco:field_schema" not in collection:
        raise TacoCreationError("COLLECTION.json missing 'taco:field_schema'")

    return collection


def _read_consolidated_metadata(folder_path: Path) -> list[pa.Table]:
    """
    Read consolidated metadata from METADATA/levelX.parquet files.

    Raises:
        TacoCreationError: If METADATA directory not found, no level files, or read fails
    """
    metadata_dir = folder_path / "METADATA"

    if not metadata_dir.exists():
        raise TacoCreationError(f"METADATA directory not found in {folder_path}")

    # Find all levelX.parquet files
    level_files = sorted(metadata_dir.glob("level*.parquet"))

    if not level_files:
        raise TacoCreationError(f"No level*.parquet files found in {metadata_dir}")

    logger.debug(f"Found {len(level_files)} metadata files")

    levels = []
    for level_file in level_files:
        try:
            table = pq.read_table(level_file)
            levels.append(table)
            logger.debug(f"Read {level_file.name}: {table.num_rows} rows")
        except Exception as e:
            raise TacoCreationError(f"Failed to read {level_file}: {e}") from e

    return levels


def _reconstruct_local_metadata_from_levels(
    levels: list[pa.Table],
) -> dict[str, pa.Table]:
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
    """
    local_metadata = {}

    # Iterate over all levels except the last (it has no children)
    for level_idx in range(len(levels) - 1):
        current_level = levels[level_idx]
        next_level = levels[level_idx + 1]

        # Find all FOLDERs at this level using PyArrow compute
        type_column = current_level.column("type")
        folders_mask = pc.equal(type_column, pa.scalar("FOLDER"))
        folders = current_level.filter(folders_mask)

        # Convert to list of dicts for iteration
        folders_list = folders.to_pylist()

        for folder_row in folders_list:
            parent_id = folder_row["internal:parent_id"]

            # Construct folder_path based on level
            if level_idx == 0:
                # Level 0: use 'id' directly (no relative_path column)
                folder_path = f"DATA/{folder_row['id']}/"
            else:
                # Level 1+: use 'internal:relative_path'
                folder_path = f"DATA/{folder_row['internal:relative_path']}/"

            # Get children from next level that belong to this folder
            parent_id_column = next_level.column("internal:parent_id")
            children_mask = pc.equal(parent_id_column, pa.scalar(parent_id))
            children = next_level.filter(children_mask)

            # Remove columns that don't belong in __meta__ local files
            cols_to_drop = []
            if "internal:relative_path" in children.schema.names:
                cols_to_drop.append("internal:relative_path")

            if cols_to_drop:
                children = children.drop(cols_to_drop)

            local_metadata[folder_path] = children

    logger.debug(f"Reconstructed {len(local_metadata)} local metadata entries")
    return local_metadata


def _scan_data_files(folder_path: Path) -> tuple[list[str], list[str]]:
    """
    Scan DATA/ directory for physical files.

    Builds parallel lists of source paths (absolute filesystem paths) and
    archive paths (paths within ZIP container). Excludes __meta__ files
    since they're regenerated during ZIP creation.

    Raises:
        TacoCreationError: If DATA directory not found or no data files found
    """
    data_dir = folder_path / "DATA"

    if not data_dir.exists():
        raise TacoCreationError(f"DATA directory not found in {folder_path}")

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
        raise TacoCreationError(f"No data files found in {data_dir}")

    logger.debug(f"Scanned {len(src_files)} data files")
    return src_files, arc_files
