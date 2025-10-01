import pathlib
from typing import Any, Literal

from tacotoolbox._helpers import FileExtractor, group_samples_by_size
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
    split_size: str | None = "3.9GB",
    quiet: bool = True,
    **kwargs: Any,
) -> list[pathlib.Path]:
    """
    Create TACO dataset in ZIP or folder container format.

    Args:
        taco: TACO object with tortilla samples
        output: Output path
            - format="zip": Path to .tacozip file (e.g., "dataset.tacozip")
            - format="folder": Path to directory (e.g., "dataset/")
        output_format: Container format
            - "zip": Single .tacozip file with Parquet metadata (immutable)
            - "folder": Directory with Avro metadata (mutable, appendable)
        split_size: Optional size limit for splitting (format="zip" only)
            - Maximum: "3.9GB" (reserves space for metadata in 4GB ZIP limit)
            - Creates multiple parts: "dataset_part001.tacozip", etc.
        quiet: Suppress progress messages
        **kwargs: Format-specific kwargs
            - ZIP format: kwargs for pyarrow.parquet.write_table()
              (e.g., row_group_size, compression, use_dictionary)
            - Folder format: kwargs for fastavro.writer()
              (e.g., codec, sync_interval, metadata)

    Returns:
        List of created file/directory paths

    Raises:
        TacoValidationError: If arguments are invalid
        TacoCreationError: If creation fails

    Examples:
        # Standard ZIP container with Parquet metadata
        >>> create(taco, "dataset.tacozip")
        [PosixPath('dataset.tacozip')]

        # ZIP with custom Parquet settings
        >>> create(taco, "dataset.tacozip", row_group_size=10000, compression="snappy")
        [PosixPath('dataset.tacozip')]

        # ZIP with splitting
        >>> create(taco, "dataset.tacozip", split_size="2GB")
        [PosixPath('dataset_part001.tacozip'), PosixPath('dataset_part002.tacozip')]

        # Folder container with Avro metadata
        >>> create(taco, "dataset/", output_format="folder")
        [PosixPath('dataset')]

        # Folder with custom Avro codec
        >>> create(taco, "dataset/", output_format="folder", codec="snappy")
        [PosixPath('dataset')]
    """
    # Validate arguments
    output_path = pathlib.Path(output)
    validate_format_value(output_format)

    # Normalize output path for folder format
    if output_format == "folder" and output_path.suffix in (".zip", ".tacozip"):
        output_path = output_path.with_suffix("")

    validate_output_path(output_path, output_format)

    # Ignore split_size for folder format
    if output_format == "folder" and split_size is not None:
        split_size = None

    # Handle splitting if requested
    if split_size is not None:
        max_size = validate_split_size(split_size)
        return _create_with_splitting(
            taco, output_path, max_size, quiet, **kwargs
        )

    # Single container creation
    if output_format == "zip":
        return [_create_zip(taco, output_path, quiet, **kwargs)]

    # format == "folder"
    return [_create_folder(taco, output_path, quiet, **kwargs)]


def _create_zip(
    taco: Taco,
    output_path: pathlib.Path,
    quiet: bool,
    **kwargs: Any,
) -> pathlib.Path:
    """
    Create single ZIP container.

    Args:
        taco: TACO object
        output_path: Path to output .tacozip file
        quiet: Suppress messages
        **kwargs: Additional kwargs for pyarrow.parquet.write_table()

    Returns:
        Path to created .tacozip file

    Raises:
        TacoCreationError: If creation fails
    """
    try:
        # Generate metadata and PIT schema
        generator = MetadataGenerator(taco, quiet)
        metadata_package = generator.generate_all_levels()

        # Add PIT schema to collection
        collection = metadata_package["collection"]
        collection["taco:pit_schema"] = metadata_package["pit_schema"]

        # Extract files for DATA/
        extracted = FileExtractor.extract_files_recursive(taco.tortilla.samples)

        # Create ZIP
        writer = ZipWriter(output_path, quiet)
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
    Create folder container.

    Args:
        taco: TACO object
        output_path: Path to output directory
        quiet: Suppress messages
        **kwargs: Additional kwargs for fastavro.writer()

    Returns:
        Path to created directory

    Raises:
        TacoCreationError: If creation fails
    """
    try:
        # Generate metadata and PIT schema
        generator = MetadataGenerator(taco, quiet)
        metadata_package = generator.generate_all_levels()

        # Add PIT schema to collection
        collection = metadata_package["collection"]
        collection["taco:pit_schema"] = metadata_package["pit_schema"]

        # Create folder
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
    Validate that no chunk files already exist.

    Args:
        sample_chunks: List of sample chunks
        base_name: Base filename without extension
        extension: File extension
        parent_dir: Parent directory for chunk files

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
    **kwargs: Any,
) -> list[pathlib.Path]:
    """
    Create multiple ZIP containers via splitting.

    Args:
        taco: TACO object
        output_path: Base path for output files
        max_size: Maximum size per chunk in bytes
        quiet: Suppress messages
        **kwargs: Additional kwargs for pyarrow.parquet.write_table()

    Returns:
        List of created .tacozip file paths

    Raises:
        TacoCreationError: If creation fails
        TacoValidationError: If any chunk files already exist
    """
    try:
        # Group samples by size
        sample_chunks = group_samples_by_size(taco.tortilla.samples, max_size)

        # If only one chunk, create normally
        if len(sample_chunks) == 1:
            return [_create_zip(taco, output_path, quiet, **kwargs)]

        # Validate that no chunk files exist BEFORE starting
        base_name = output_path.stem
        extension = output_path.suffix
        parent_dir = output_path.parent

        _validate_chunk_paths(sample_chunks, base_name, extension, parent_dir)

        # Create multiple chunks
        created_files = []

        for i, chunk_samples in enumerate(sample_chunks, 1):
            # Create partial Taco for this chunk
            chunk_tortilla = Tortilla(samples=chunk_samples)
            chunk_taco_data = taco.model_dump()
            chunk_taco_data["tortilla"] = chunk_tortilla
            chunk_taco = Taco(**chunk_taco_data)

            # Generate chunk filename
            chunk_filename = f"{base_name}_part{i:04d}{extension}"
            chunk_path = parent_dir / chunk_filename

            # Create chunk
            created_path = _create_zip(chunk_taco, chunk_path, quiet, **kwargs)
            created_files.append(created_path)

        if not quiet:
            print(
                f"Created {len(created_files)} ZIP chunks "
                f"with max size {max_size / (1024**3):.1f}GB"
            )

        return created_files

    except Exception as e:
        raise TacoCreationError(f"Failed to create split containers: {e}") from e