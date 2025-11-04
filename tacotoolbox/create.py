"""
TACO container creation.

This module provides the main create() function for generating TACO containers
in ZIP or FOLDER format. It handles:

- Single container creation
- Dataset splitting (multi-part ZIPs)
- Metadata generation
- Automatic cleanup of temporary files

Temporary files created from bytes are ALWAYS cleaned up after
successful container creation. This is non-configurable to prevent disk space issues.

Main workflow:
    1. Validate parameters (output path, format, split size)
    2. Generate metadata package (consolidated + local)
    3. Extract file paths with archive paths
    4. Create container(s) using appropriate writer
    5. Cleanup temporary files automatically

Example:
    >>> from tacotoolbox import create, Taco, Sample
    >>> 
    >>> # Create basic ZIP container
    >>> taco = Taco(tortilla=Tortilla(samples=[...]), ...)
    >>> paths = create(taco, "output.tacozip")
    >>> 
    >>> # Create with splitting
    >>> paths = create(taco, "output.tacozip", split_size="4GB")
    >>> # Returns: [output_part0001.tacozip, output_part0002.tacozip, ...]
    >>> 
    >>> # Create FOLDER container
    >>> paths = create(taco, "output_folder", output_format="folder")
"""

import pathlib
from typing import Any, Literal

from tacotoolbox._metadata import MetadataGenerator
from tacotoolbox._validation import (
    TacoValidationError,
    validate_format_value,
    validate_output_path,
    validate_split_size,
)
from tacotoolbox._writers.folder_writer import FolderWriter
from tacotoolbox._writers.zip_writer import ZipWriter
from tacotoolbox.datamodel import Taco
from tacotoolbox.tortilla.datamodel import Tortilla


class TacoCreationError(Exception):
    """Raised when TACO creation fails."""


def create(
    taco: Taco,
    output: str | pathlib.Path,
    output_format: Literal["zip", "folder"] = "zip",
    split_size: str | None = None,
    sort_by: str | None = None,
    temp_dir: str | pathlib.Path | None = None,
    quiet: bool = True,
    **kwargs: Any,
) -> list[pathlib.Path]:
    """
    Create TACO container from Taco object.

    This is the main entry point for creating TACO containers. It supports:
    - ZIP format (.tacozip) with optional splitting
    - FOLDER format (directory structure)
    - Spatial sorting for geographic clustering in splits

    Temporary files created from bytes (via Sample(path=bytes))
    are ALWAYS cleaned up automatically after successful container creation.

    Args:
        taco: TACO object with tortilla and metadata
        output: Output path for container
        output_format: "zip" or "folder" (default: "zip")
        split_size: Optional size limit for splitting (ZIP only, e.g., "4GB")
        sort_by: Optional column name to sort samples before splitting.
                Useful for spatial clustering (e.g., "majortom:code")
        temp_dir: Temporary directory for ZIP creation
        quiet: If True, suppress progress messages (default: True)
        **kwargs: Additional arguments passed to writers

    Returns:
        List of created container paths (multiple if split)

    Raises:
        TacoValidationError: If validation fails
        TacoCreationError: If container creation fails

    Example:
        >>> # Basic usage
        >>> sample = Sample(id="s1", path=image_bytes, type="FILE")
        >>> taco = Taco(tortilla=Tortilla(samples=[sample]), ...)
        >>> paths = create(taco, "output.tacozip")

        >>> # With spatial sorting and splitting
        >>> majortom = MajorTOM(dist_km=100)
        >>> tortilla.extend_with(majortom)
        >>> taco = Taco(tortilla=tortilla, ...)
        >>> paths = create(
        ...     taco,
        ...     "output.tacozip",
        ...     split_size="4GB",
        ...     sort_by="majortom:code"
        ... )
        >>> # Creates chunks with spatial clustering:
        >>> # - output_part0001.tacozip: Europe samples
        >>> # - output_part0002.tacozip: Asia samples
        >>> # - output_part0003.tacozip: Americas samples

        >>> # Multiple sort strategies
        >>> create(taco, "out.tacozip", sort_by="stac:time_start")  # Temporal
        >>> create(taco, "out.tacozip", sort_by="custom:priority")  # Custom field
    """
    output_path = pathlib.Path(output)
    validate_format_value(output_format)

    if output_format == "folder" and output_path.suffix in (".zip", ".tacozip"):
        output_path = output_path.with_suffix("")

    validate_output_path(output_path, output_format)

    if output_format == "folder":
        split_size = None
        temp_dir = None

    # Sort samples if requested
    if sort_by is not None:
        if not quiet:
            print(f"Sorting samples by '{sort_by}'...")
        taco = _sort_taco_samples(taco, sort_by, quiet)

    try:
        if split_size is not None:
            max_size = validate_split_size(split_size)
            result = _create_with_splitting(
                taco, output_path, max_size, quiet, temp_dir, **kwargs
            )
        elif output_format == "zip":
            result = [_create_zip(taco, output_path, quiet, temp_dir, **kwargs)]
        else:
            result = [_create_folder(taco, output_path, quiet, **kwargs)]

        # Always cleanup temp files after SUCCESS
        _cleanup_tortilla_temp_files(taco.tortilla)

        return result

    except Exception:
        # On error, don't cleanup - files may be needed for debugging/retry
        raise


def _sort_taco_samples(taco: Taco, sort_column: str, quiet: bool) -> Taco:
    """
    Sort taco samples according to metadata column.

    Creates a new Taco object with samples reordered based on the specified
    metadata column. This enables spatial/temporal clustering when creating
    split containers.

    Common use cases:
    - "majortom:code": Geographic clustering (nearby samples together)
    - "stac:time_start": Temporal ordering
    - "custom:priority": User-defined ordering

    Args:
        taco: Original Taco object
        sort_column: Metadata column name to sort by
        quiet: Suppress messages

    Returns:
        New Taco object with sorted samples

    Raises:
        TacoCreationError: If sort column doesn't exist or sorting fails

    Example:
        >>> # Spatial clustering
        >>> sorted_taco = _sort_taco_samples(taco, "majortom:code", False)
        >>> # Now samples from same geographic region are consecutive
    """
    try:
        # Export metadata to get sort column
        df = taco.tortilla.export_metadata(deep=0)

        # Validate sort column exists
        if sort_column not in df.columns:
            available = sorted(df.columns)
            raise TacoCreationError(
                f"Sort column '{sort_column}' not found in metadata.\n"
                f"Available columns: {available}"
            )

        # Sort by column
        df_sorted = df.sort(sort_column)

        # Get sorted IDs
        sorted_ids = df_sorted["id"].to_list()

        # Create ID -> Sample mapping
        sample_map = {s.id: s for s in taco.tortilla.samples}

        # Reorder samples
        sorted_samples = [sample_map[sid] for sid in sorted_ids]

        if not quiet:
            print(f"   Sorted {len(sorted_samples)} samples")
            # Show first and last values for context
            first_val = df_sorted[sort_column][0]
            last_val = df_sorted[sort_column][-1]
            print(f"   Range: {first_val} → {last_val}")

        # Create new Tortilla with sorted samples
        sorted_tortilla = Tortilla(samples=sorted_samples)

        # Preserve all extension metadata from original tortilla
        sorted_tortilla._metadata_df = df_sorted        

        # Create new Taco with sorted tortilla (preserve all other metadata)
        taco_data = taco.model_dump()
        taco_data["tortilla"] = sorted_tortilla
        return Taco(**taco_data)

    except KeyError as e:
        raise TacoCreationError(f"Failed to sort samples: {e}") from e
    except Exception as e:
        raise TacoCreationError(f"Failed to sort samples: {e}") from e


def _extract_files_with_ids(samples: list, path_prefix: str = "") -> dict[str, Any]:
    """
    Extract files using sample IDs as-is in archive paths.

    Recursively traverses sample hierarchy and builds parallel lists of:
    - Source file paths (absolute paths on filesystem)
    - Archive paths (relative paths in ZIP/FOLDER container)

    Sample IDs are used directly without modification. If user wants
    extensions, they include them in the ID.

    Args:
        samples: List of Sample objects (can contain nested FOLDERs)
        path_prefix: Current path prefix in archive (e.g., "DATA/folder_A/")

    Returns:
        Dictionary with 'src_files' and 'arc_files' lists

    Example:
        >>> samples = [
        ...     Sample(id="img1", path="/data/img1.tif", type="FILE"),
        ...     Sample(id="folder_A", type="FOLDER", path=Tortilla([
        ...         Sample(id="nested", path="/data/nested.tif", type="FILE")
        ...     ]))
        ... ]
        >>> result = _extract_files_with_ids(samples, "DATA/")
        >>> result["src_files"]
        ["/data/img1.tif", "/data/nested.tif"]
        >>> result["arc_files"]
        ["DATA/img1", "DATA/folder_A/nested"]
    """
    src_files = []
    arc_files = []

    for sample in samples:
        if sample.type == "FOLDER":
            new_prefix = f"{path_prefix}{sample.id}/"
            nested = _extract_files_with_ids(sample.path.samples, new_prefix)
            src_files.extend(nested["src_files"])
            arc_files.extend(nested["arc_files"])
        else:
            src_path = str(sample.path)
            arc_path = f"{path_prefix}{sample.id}"

            src_files.append(src_path)
            arc_files.append(arc_path)

    return {"src_files": src_files, "arc_files": arc_files}


def _estimate_sample_size(sample) -> int:
    """
    Estimate total size of a sample including nested samples.

    For FILE samples, returns actual file size from filesystem.
    For FOLDER samples, recursively sums all nested file sizes.

    Used by group_samples_by_size() to calculate chunk sizes
    when splitting datasets into multiple containers.

    Args:
        sample: Sample to estimate (FILE or FOLDER)

    Returns:
        Estimated size in bytes (0 if path doesn't support stat)

    Example:
        >>> file = Sample(id="img", path=Path("/data/img.tif"), type="FILE")
        >>> _estimate_sample_size(file)
        10485760  # 10MB

        >>> folder = Sample(id="dataset", type="FOLDER", path=Tortilla([
        ...     Sample(id="img1", path=Path("/data/1.tif")),  # 5MB
        ...     Sample(id="img2", path=Path("/data/2.tif")),  # 5MB
        ... ]))
        >>> _estimate_sample_size(folder)
        10485760  # Sum of nested files
    """
    if sample.type == "FILE":
        # Get file size from filesystem
        if hasattr(sample.path, "stat"):
            return sample.path.stat().st_size
        return 0

    elif sample.type == "FOLDER":
        # Sum sizes of all nested samples recursively
        return sum(_estimate_sample_size(child) for child in sample.path.samples)

    return 0


def _group_samples_by_size(samples: list, max_size: int) -> list[list]:
    """
    Group samples into chunks based on size limit.

    Splits a list of samples into multiple chunks where each chunk's total
    size does not exceed max_size. Used for creating multi-part ZIP archives.

    Algorithm: Greedy packing
    - Add samples to current chunk until limit reached
    - Start new chunk when adding next sample would exceed limit
    - Each chunk contains complete samples (no partial splits)

    IMPORTANT: Samples should be pre-sorted (e.g., by majortom:code) before
    calling this function to ensure geographic/temporal clustering in chunks.

    Note: Individual samples larger than max_size will be placed alone
    in their own chunk (cannot split a single sample).

    Args:
        samples: List of Sample objects to group (should be pre-sorted)
        max_size: Maximum size per chunk in bytes

    Returns:
        List of sample chunks, each under max_size (if possible)

    Example:
        >>> # With spatial pre-sorting
        >>> samples = [eu1, eu2, eu3, asia1, asia2, us1, us2]
        >>> chunks = _group_samples_by_size(samples, max_size=20_000_000)
        >>> # Chunk 1: [eu1, eu2, eu3]
        >>> # Chunk 2: [asia1, asia2]
        >>> # Chunk 3: [us1, us2]
    """
    chunks = []
    current_chunk = []
    current_size = 0

    for sample in samples:
        sample_size = _estimate_sample_size(sample)

        # Check if adding this sample would exceed limit
        if current_size + sample_size > max_size and current_chunk:
            # Start new chunk
            chunks.append(current_chunk)
            current_chunk = [sample]
            current_size = sample_size
        else:
            # Add to current chunk
            current_chunk.append(sample)
            current_size += sample_size

    # Add remaining samples
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def _create_zip(
    taco: Taco,
    output_path: pathlib.Path,
    quiet: bool,
    temp_dir: pathlib.Path | str | None,
    **kwargs: Any,
) -> pathlib.Path:
    """
    Create single ZIP container.

    Workflow:
    1. Generate metadata package (consolidated + local)
    2. Extract file paths with archive paths
    3. Create ZIP using ZipWriter

    Args:
        taco: TACO object
        output_path: Output .tacozip path
        quiet: Suppress messages
        temp_dir: Temporary directory
        **kwargs: Additional writer arguments

    Returns:
        Path to created .tacozip file

    Raises:
        TacoCreationError: If ZIP creation fails
    """
    try:
        if not quiet:
            print("Generating metadata...")

        generator = MetadataGenerator(taco, quiet)
        metadata_package = generator.generate_all_levels()

        if not quiet:
            print(f"   Consolidated levels: {len(metadata_package.levels)}")
            print(f"   Local metadata folders: {len(metadata_package.local_metadata)}")

        if not quiet:
            print("Extracting file paths...")

        extracted = _extract_files_with_ids(taco.tortilla.samples, "DATA/")

        if not quiet:
            print(f"   {len(extracted['src_files'])} data files")

        writer = ZipWriter(output_path, quiet, temp_dir=temp_dir)
        return writer.create_complete_zip(
            src_files=extracted["src_files"],
            arc_files=extracted["arc_files"],
            metadata_package=metadata_package,
            **kwargs,
        )

    except Exception as e:
        raise TacoCreationError(f"Failed to create ZIP container: {e}") from e


def _create_folder(
    taco: Taco,
    output_path: pathlib.Path,
    quiet: bool,
    **kwargs: Any,
) -> pathlib.Path:
    """
    Create FOLDER container.

    Workflow:
    1. Generate metadata package (consolidated + local)
    2. Create folder structure using FolderWriter
    3. Copy files and write metadata

    Args:
        taco: TACO object
        output_path: Output folder path
        quiet: Suppress messages
        **kwargs: Additional writer arguments

    Returns:
        Path to created folder

    Raises:
        TacoCreationError: If folder creation fails
    """
    try:
        generator = MetadataGenerator(taco, quiet)
        metadata_package = generator.generate_all_levels()

        writer = FolderWriter(output_path, quiet)
        return writer.create_complete_folder(
            samples=taco.tortilla.samples,
            metadata_package=metadata_package,
            **kwargs,
        )

    except Exception as e:
        raise TacoCreationError(f"Failed to create folder container: {e}") from e


def _validate_chunk_paths(
    sample_chunks: list[Any],
    base_name: str,
    extension: str,
    parent_dir: pathlib.Path,
) -> None:
    """
    Validate that chunk output paths don't already exist.

    Prevents overwriting existing chunk files when splitting datasets.

    Args:
        sample_chunks: List of sample chunks to create
        base_name: Base filename (without extension)
        extension: File extension (e.g., ".tacozip")
        parent_dir: Parent directory for chunks

    Raises:
        TacoValidationError: If any chunk file already exists
    """
    for i in range(1, len(sample_chunks) + 1):
        chunk_filename = f"{base_name}_part{i:04d}{extension}"
        chunk_path = parent_dir / chunk_filename
        if chunk_path.exists():
            raise TacoValidationError(
                f"Chunk file already exists: {chunk_path}\n"
                f"Remove existing chunk files or choose a different output path."
            )


def _create_with_splitting(
    taco: Taco,
    output_path: pathlib.Path,
    max_size: int,
    quiet: bool,
    temp_dir: pathlib.Path | str | None,
    **kwargs: Any,
) -> list[pathlib.Path]:
    """
    Create multiple ZIP containers by splitting samples.

    Workflow:
    1. Group samples into chunks based on max_size
    2. Validate chunk output paths don't exist
    3. Create separate TACO for each chunk
    4. Create ZIP for each chunk

    Chunk naming: base_part0001.tacozip, base_part0002.tacozip, etc.

    Note: Samples should be pre-sorted (via sort_by parameter in create())
    to ensure spatial/temporal clustering in chunks.

    Args:
        taco: TACO object (potentially with pre-sorted samples)
        output_path: Base output path
        max_size: Maximum size per chunk in bytes
        quiet: Suppress messages
        temp_dir: Temporary directory
        **kwargs: Additional writer arguments

    Returns:
        List of created .tacozip paths

    Raises:
        TacoCreationError: If splitting or creation fails
    """
    try:
        sample_chunks = _group_samples_by_size(taco.tortilla.samples, max_size)

        if len(sample_chunks) == 1:
            # No splitting needed
            return [_create_zip(taco, output_path, quiet, temp_dir, **kwargs)]

        base_name = output_path.stem
        extension = output_path.suffix
        parent_dir = output_path.parent

        _validate_chunk_paths(sample_chunks, base_name, extension, parent_dir)

        created_files = []

        for i, chunk_samples in enumerate(sample_chunks, 1):
            # Create separate TACO for this chunk
            chunk_tortilla = Tortilla(samples=chunk_samples)
            chunk_taco_data = taco.model_dump()
            chunk_taco_data["tortilla"] = chunk_tortilla
            chunk_taco = Taco(**chunk_taco_data)

            # Generate chunk filename
            chunk_filename = f"{base_name}_part{i:04d}{extension}"
            chunk_path = parent_dir / chunk_filename

            if not quiet:
                print(f"Creating chunk {i}/{len(sample_chunks)}: {chunk_filename}")

            # Create chunk ZIP
            created_path = _create_zip(
                chunk_taco, chunk_path, quiet, temp_dir, **kwargs
            )
            created_files.append(created_path)

        if not quiet:
            print(
                f"Created {len(created_files)} ZIP chunks "
                f"with max size {max_size / (1024**3):.1f}GB"
            )

    except Exception as e:
        raise TacoCreationError(f"Failed to create split containers: {e}") from e
    else:
        return created_files


def _cleanup_tortilla_temp_files(tortilla: Tortilla) -> None:
    """
    Recursively cleanup temporary files from all samples in tortilla.

    This is called automatically after successful container creation to
    free up disk space used by temporary files created from bytes.

    The cleanup is:
    - Automatic: Always runs after successful create()
    - Silent: No messages (happens in background)
    - Recursive: Cleans nested samples in FOLDERs
    - Safe: Ignores errors (files may already be deleted)

    Args:
        tortilla: Tortilla containing samples to cleanup
    """
    for sample in tortilla.samples:
        if sample.type == "FILE":
            # Cleanup this sample's temp files (if any)
            sample.cleanup()

        elif sample.type == "FOLDER":
            # Recurse into nested tortilla
            _cleanup_tortilla_temp_files(sample.path)
