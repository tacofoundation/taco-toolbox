"""
TACO Dataset Extraction.

Extract filtered subsets from TacoDataset to new FOLDER containers.
Handles parallel file copying, metadata reindexing, and collection generation.

This module enables creating new TACO datasets from filtered queries on existing
datasets. It supports:
- Parallel file extraction with configurable workers
- Automatic parent_id reindexing for relational queries
- Recursive FOLDER handling with local metadata
- ZIP-specific column removal (offset, size)
- Collection metadata updates with subset provenance

Key features:
- Works with filtered TacoDataset (e.g., from .sql() queries)
- Preserves hierarchical structure in FOLDER format
- Reindexes internal:parent_id for consistency
- Copies both FILEs and FOLDERs recursively
- Generates dual metadata system (local + consolidated)

Limitations:
- Only level0 filters supported (no level1+ joins)
- Output format is always FOLDER (not ZIP)
- Requires non-empty filtered dataset

Main function:
    extract(): Export filtered TacoDataset to FOLDER format

Private functions:
    _validate_dataset(): Pre-flight checks
    _create_folder_structure(): Create DATA/ and METADATA/ dirs
    _copy_single_file(): Copy individual file from vsi path
    _copy_folder_recursive(): Recursively copy FOLDER and children
    _copy_all_bytes(): Orchestrate parallel file copying
    _generate_consolidated_metadata(): Create METADATA/levelX.avro with reindexing
    _generate_collection_json(): Update collection with subset info

Example:
    >>> from tacoreader import TacoDataset
    >>> from tacotoolbox import extract
    >>> 
    >>> # Load and filter dataset
    >>> dataset = TacoDataset("big_dataset.tacozip")
    >>> filtered = dataset.sql("SELECT * FROM level0 WHERE country = 'ZA'")
    >>> 
    >>> # Extract to new FOLDER TACO
    >>> extract(filtered, output="south_africa/", nworkers=8)
    PosixPath('south_africa')
"""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import json
import polars as pl
import obstore as obs

from tacoreader import TacoDataset
from tacoreader.utils.vsi import strip_vsi_prefix, create_obstore_from_url

from tacotoolbox._column_utils import (
    reorder_internal_columns,
    write_avro_file,
    write_parquet_file,
    read_metadata_file,
)


def extract(
    dataset: TacoDataset,
    output: str | Path,
    nworkers: int = 4,
    overwrite: bool = False,
    quiet: bool = False,
) -> Path:
    """
    Extract filtered subset from TacoDataset to new FOLDER TACO.

    Creates a new FOLDER-format TACO container from a filtered TacoDataset.
    Copies all data files, regenerates metadata with reindexed parent_id
    values, and updates collection metadata with subset provenance.

    The extraction process:
    1. Validates dataset (no level1+ joins, not empty)
    2. Creates FOLDER structure (DATA/, METADATA/)
    3. Copies files in parallel (FILEs and FOLDERs)
    4. Generates consolidated metadata with reindexed parent_id
    5. Creates COLLECTION.json with subset metadata

    Args:
        dataset: TacoDataset with applied filters (e.g., from .sql())
        output: Path to output folder
        nworkers: Number of parallel workers for copying files
        overwrite: If True, remove existing output folder
        quiet: If True, suppress progress messages

    Returns:
        Path to created FOLDER TACO

    Raises:
        ValueError: If dataset has level1+ joins or is empty
        FileExistsError: If output exists and overwrite=False

    Example:
        >>> from tacoreader import TacoDataset
        >>> from tacotoolbox import extract
        >>>
        >>> # Filter by country
        >>> dataset = TacoDataset("global.tacozip")
        >>> filtered = dataset.sql("SELECT * FROM level0 WHERE country = 'BR'")
        >>> extract(filtered, "brazil_subset/", nworkers=8)
        PosixPath('brazil_subset')
        >>>
        >>> # Filter by date range
        >>> recent = dataset.sql("SELECT * FROM level0 WHERE year >= 2020")
        >>> extract(recent, "recent_data/", overwrite=True)
        PosixPath('recent_data')
    """
    output = Path(output)

    _validate_dataset(dataset, output, overwrite)

    _create_folder_structure(output)

    _copy_all_bytes(dataset, output, nworkers, quiet)

    _generate_consolidated_metadata(dataset, output)
    _generate_collection_json(dataset, output)

    if not quiet:
        print(f"Extraction complete: {output}")

    return output


def _validate_dataset(
    dataset: TacoDataset,
    output: Path,
    overwrite: bool,
) -> None:
    """
    Validate dataset and output path before extraction.

    Checks that:
    - Dataset has no level1+ joins (only level0 filters supported)
    - Dataset is not empty (at least one sample)
    - Output path doesn't exist or overwrite is True

    Args:
        dataset: TacoDataset to validate
        output: Target output path
        overwrite: Whether to allow overwriting existing path

    Raises:
        ValueError: If dataset validation fails
        FileExistsError: If output exists and overwrite=False
    """

    if dataset._has_level1_joins:
        raise ValueError(
            "Cannot extract dataset with level1+ joins. "
            "Only level0 filters are supported."
        )

    count = len(dataset.data._data)
    if count == 0:
        raise ValueError("Cannot extract empty dataset")

    if output.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output path already exists: {output}. "
                "Use overwrite=True to replace it."
            )
        shutil.rmtree(output)


def _create_folder_structure(output: Path) -> None:
    """
    Create base DATA/ and METADATA/ folders.

    Args:
        output: Root path for TACO container
    """
    (output / "DATA").mkdir(parents=True)
    (output / "METADATA").mkdir(parents=True)


def _copy_single_file(vsi_path: str, dest_path: Path) -> None:
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

    Example:
        >>> # Copy from ZIP entry
        >>> _copy_single_file(
        ...     "/vsisubfile/100_50,/data/archive.zip",
        ...     Path("output/sample.tif")
        ... )
        >>>
        >>> # Copy from filesystem
        >>> _copy_single_file("/data/image.tif", Path("output/image.tif"))
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if vsi_path.startswith("/vsisubfile/"):
        # Parse vsisubfile format: /vsisubfile/offset_size,source_path
        parts = vsi_path.replace("/vsisubfile/", "").split(",", 1)
        offset, size = map(int, parts[0].split("_"))
        zip_path = parts[1]

        clean_url = strip_vsi_prefix(zip_path)

        if Path(clean_url).exists():
            # Local file: direct read
            with open(clean_url, "rb") as f:
                f.seek(offset)
                data = f.read(size)
        else:
            # Remote file: use obstore for range read
            # Store created with from_url() already contains full path as prefix
            store = create_obstore_from_url(clean_url)

            # Use empty path since store already contains full file path
            data = bytes(obs.get_range(store, "", start=offset, end=offset + size))

        with open(dest_path, "wb") as f:
            f.write(data)
    else:
        # Regular file copy
        shutil.copy2(vsi_path, dest_path)


def _copy_folder_recursive(
    folder_row: dict,
    output: Path,
    dataset: TacoDataset,
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
        output: Output base path
        dataset: Source dataset
        level: Current level of this folder (0, 1, 2, ...)

    Example:
        >>> # Given level0 FOLDER "scene_001" with 2 FILEs in level1
        >>> folder_row = {
        ...     "id": "scene_001",
        ...     "type": "FOLDER",
        ...     "internal:parent_id": 0,
        ...     "internal:relative_path": "scene_001/"
        ... }
        >>> _copy_folder_recursive(folder_row, Path("output/"), dataset, level=0)
        >>> # Creates:
        >>> # output/DATA/scene_001/
        >>> # output/DATA/scene_001/__meta__
        >>> # output/DATA/scene_001/before.tif
        >>> # output/DATA/scene_001/after.tif
    """
    if level == 0:
        relative_path = folder_row["id"]
    else:
        relative_path = folder_row["internal:relative_path"]

    folder_path = output / "DATA" / relative_path
    folder_path.mkdir(parents=True, exist_ok=True)

    parent_id = folder_row["internal:parent_id"]
    next_level = level + 1

    view_name = f"level{next_level}"
    if view_name not in dataset.consolidated_files:
        return

    # Query children from next level
    if "internal:source_file" in folder_row and folder_row["internal:source_file"]:
        # TacoCat case: need to filter by both parent_id AND source_file
        children_df = pl.from_arrow(
            dataset._duckdb.execute(
                f'SELECT * FROM {view_name} WHERE "internal:parent_id" = ? AND "internal:source_file" = ?',
                [parent_id, folder_row["internal:source_file"]],
            ).fetch_arrow_table()
        )
    else:
        # Single TACO case: only filter by parent_id
        children_df = pl.from_arrow(
            dataset._duckdb.execute(
                f'SELECT * FROM {view_name} WHERE "internal:parent_id" = ?', [parent_id]
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

        child_dest = output / "DATA" / child_rel

        if child_type == "FILE":
            _copy_single_file(child_vsi, child_dest)
        elif child_type == "FOLDER":
            _copy_folder_recursive(child_row, output, dataset, next_level)

    # Write local __meta__ for this folder in PARQUET format
    meta_path = folder_path / "__meta__"
    write_parquet_file(children_df, meta_path)


def _copy_all_bytes(
    dataset: TacoDataset,
    output: Path,
    nworkers: int,
    quiet: bool,
) -> None:
    """
    Copy all bytes from dataset to output/DATA/.

    Uses parallel workers for FILE copying. FOLDER copying is recursive
    and single-threaded per folder tree.

    Args:
        dataset: Source dataset
        output: Output base path
        nworkers: Number of parallel workers for FILEs
        quiet: If True, suppress progress messages
    """
    df = dataset.data._data

    files_df = df.filter(pl.col("type") == "FILE")
    folders_df = df.filter(pl.col("type") == "FOLDER")

    # Copy FILEs in parallel
    if len(files_df) > 0:
        with ThreadPoolExecutor(max_workers=nworkers) as executor:
            futures = []
            for row in files_df.iter_rows(named=True):
                vsi_path = row["internal:gdal_vsi"]

                if "internal:relative_path" in row and row["internal:relative_path"]:
                    relative_path = row["internal:relative_path"]
                else:
                    relative_path = row["id"]

                dest = output / "DATA" / relative_path

                future = executor.submit(_copy_single_file, vsi_path, dest)
                futures.append(future)

            # Wait for all FILEs to complete
            for future in as_completed(futures):
                future.result()  # Raises exception if copy failed

        if not quiet:
            print(f"Copied {len(files_df)} FILES")

    # Copy FOLDERs recursively (single-threaded per tree)
    if len(folders_df) > 0:
        for row in folders_df.iter_rows(named=True):
            _copy_folder_recursive(row, output, dataset, level=0)

        if not quiet:
            print(f"Copied {len(folders_df)} FOLDERs")


def _generate_consolidated_metadata(dataset: TacoDataset, output: Path) -> None:
    """
    Generate consolidated metadata files (METADATA/levelX.avro).

    Filters existing consolidated metadata to selected samples and reindexes
    internal:parent_id to maintain relational consistency in the new dataset.

    The reindexing process:
    1. Level0: Filter to selected samples, assign new parent_id = 0, 1, 2, ...
    2. Level1+: Filter to children of selected parents, remap parent_id values

    Also removes ZIP-specific columns (internal:offset, internal:size) since
    FOLDER format doesn't use them.

    Args:
        dataset: Source dataset with filters applied
        output: Output base path

    Example:
        >>> # Original level0 has samples at indices [5, 12, 18]
        >>> # After filtering, new level0 has indices [0, 1, 2]
        >>> # Level1 parent_id values are remapped: 5→0, 12→1, 18→2
        >>> _generate_consolidated_metadata(dataset, Path("output/"))
        >>> # Creates output/METADATA/level0.avro, level1.avro, etc.
    """
    selected_ids = set(dataset.data._data["id"].to_list())

    level_names = sorted(
        dataset.consolidated_files.keys(), key=lambda x: int(x.replace("level", ""))
    )

    parent_id_mapping = None

    for level_name in level_names:
        file_path = dataset.consolidated_files[level_name]

        # Read metadata file (auto-detects parquet/avro)
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

        # Write consolidated metadata as AVRO
        output_path = output / "METADATA" / f"{level_name}.avro"
        write_avro_file(filtered_df, output_path)


def _generate_collection_json(dataset: TacoDataset, output: Path) -> None:
    """
    Generate COLLECTION.json with updated counts, extent, and subset info.

    Updates collection metadata to reflect the extracted subset:
    - Updates sample count (taco:pit_schema.root.n)
    - Adds taco:subset_of field with original dataset ID
    - Adds taco:subset_date with extraction timestamp

    Args:
        dataset: Source dataset
        output: Output base path

    Example:
        >>> _generate_collection_json(dataset, Path("output/"))
        >>> # Creates output/COLLECTION.json with:
        >>> # {
        >>> #   "id": "original_dataset",
        >>> #   "taco:pit_schema": {"root": {"n": 150, ...}, ...},
        >>> #   "taco:subset_of": "original_dataset",
        >>> #   "taco:subset_date": "2024-11-07T12:34:56Z"
        >>> # }
    """
    collection = dataset.collection.copy()

    # Update sample count
    new_count = len(dataset.data._data)
    collection["taco:pit_schema"]["root"]["n"] = new_count

    # Add subset provenance
    collection["taco:subset_of"] = collection.get("id", "unknown")

    from datetime import datetime, timezone

    collection["taco:subset_date"] = datetime.now(timezone.utc).isoformat()

    # Write to file
    output_path = output / "COLLECTION.json"
    with open(output_path, "w") as f:
        json.dump(collection, f, indent=4)
