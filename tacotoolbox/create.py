"""
TACO container creation with ZIP/FOLDER support and dataset splitting.

Main workflow:
1. Auto-detect format from extension (.zip/.tacozip → zip, else → folder)
2. Validate inputs
3. Generate metadata package
4. Create container(s) using appropriate writer
5. Cleanup temp files automatically

Example:
    >>> taco = Taco(tortilla=Tortilla(samples=[...]), ...)
    >>> paths = create(taco, "output.tacozip")  # auto-detects ZIP
    >>> paths = create(taco, "output_dataset")  # auto-detects FOLDER
"""

import pathlib
from typing import TYPE_CHECKING, Any, Literal, cast

from tacotoolbox._logging import get_logger
from tacotoolbox._metadata import MetadataGenerator
from tacotoolbox._progress import ProgressContext, progress_bar
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

if TYPE_CHECKING:
    from tacotoolbox.sample.datamodel import Sample

logger = get_logger(__name__)


class TacoCreationError(Exception):
    """Raised when TACO creation fails."""


def create(
    taco: Taco,
    output: str | pathlib.Path,
    output_format: Literal["zip", "folder", "auto"] = "auto",
    split_size: str | None = None,
    sort_by: str | None = None,
    temp_dir: str | pathlib.Path | None = None,
    quiet: bool = False,
    **kwargs: Any,
) -> list[pathlib.Path]:
    """
    Create TACO container from Taco object.

    Format auto-detection (output_format="auto"):
    - .zip/.tacozip → ZIP format
    - anything else → FOLDER format

    Temp files from Sample(path=bytes) are always cleaned up after success.
    """
    output_path = pathlib.Path(output)

    # Auto-detect format
    final_format: Literal["zip", "folder"]

    if output_format == "auto":
        if output_path.suffix.lower() in (".zip", ".tacozip"):
            final_format = "zip"
            logger.debug(
                f"Auto-detected format='zip' from extension: {output_path.suffix}"
            )
        else:
            final_format = "folder"
            logger.debug("Auto-detected format='folder' (no .zip/.tacozip extension)")
    else:
        # Explicit cast because we validated the literal in signature,
        # but mypy doesn't know "auto" is handled above for the variable assignment
        final_format = cast(Literal["zip", "folder"], output_format)

    _validate_all_inputs(taco, output_path, final_format, split_size)

    # Adjust output path for folder format
    if final_format == "folder" and output_path.suffix.lower() in (".zip", ".tacozip"):
        output_path = output_path.with_suffix("")
        logger.debug(f"Adjusted output path for folder format: {output_path}")

    # Force folder-specific constraints
    if final_format == "folder":
        split_size = None
        temp_dir = None
        logger.debug("Folder format: split_size and temp_dir ignored")

    # Convert temp_dir to Path if provided
    temp_path_dir = pathlib.Path(temp_dir) if temp_dir else None

    # Sort samples if requested
    if sort_by is not None:
        logger.info(f"Sorting samples by column: {sort_by}")
        taco = _sort_taco_samples(taco, sort_by, quiet)

    with ProgressContext(quiet=quiet):
        try:
            if split_size is not None:
                max_size = validate_split_size(split_size)
                logger.info(f"Creating split containers (max_size={split_size})")
                result = _create_with_splitting(
                    taco,
                    output_path,
                    max_size,
                    temp_path_dir,
                    quiet=quiet,
                    **kwargs,
                )
            elif final_format == "zip":
                logger.info(f"Creating single ZIP: {output_path}")
                result = [
                    _create_zip(taco, output_path, temp_path_dir, quiet=quiet, **kwargs)
                ]
            else:
                logger.info(f"Creating FOLDER: {output_path}")
                result = [_create_folder(taco, output_path, quiet=quiet, **kwargs)]

            logger.debug("Cleaning up temporary files from tortilla")
            _cleanup_tortilla_temp_files(taco.tortilla)

        except Exception:
            logger.exception("Container creation failed")
            raise
        else:
            logger.info(f"Successfully created {len(result)} container(s)")
            return result


def _validate_all_inputs(
    taco: Taco,
    output_path: pathlib.Path,
    output_format: Literal["zip", "folder"],
    split_size: str | None,
) -> None:
    """Validate all inputs before starting. Fails fast."""
    logger.debug("Validating inputs")

    validate_format_value(output_format)
    validate_output_path(output_path, output_format)

    if split_size is not None:
        if output_format == "folder":
            raise TacoValidationError(
                "split_size is not supported with format='folder'. "
                "Splitting is only available for format='zip'."
            )
        validate_split_size(split_size)

    if not taco.tortilla.samples:
        raise TacoValidationError("Cannot create container from empty tortilla")

    logger.debug("All inputs validated successfully")


def _validate_sort_column(sort_column: str, df_columns: list[str]) -> None:
    """Validate that sort column exists in metadata."""
    if sort_column not in df_columns:
        available = sorted(df_columns)
        raise TacoCreationError(
            f"Sort column '{sort_column}' not found in metadata.\n"
            f"Available columns: {available}"
        )


def _sort_taco_samples(taco: Taco, sort_column: str, quiet: bool) -> Taco:
    """
    Sort taco samples by metadata column for spatial/temporal clustering.

    Common use cases:
    - "majortom:code": Geographic clustering
    - "stac:time_start": Temporal ordering
    - "custom:priority": User-defined ordering
    """
    try:
        df = taco.tortilla.export_metadata(deep=0)

        _validate_sort_column(sort_column, df.columns)

        df_sorted = df.sort(sort_column)
        sorted_ids = df_sorted["id"].to_list()

        sample_map = {s.id: s for s in taco.tortilla.samples}
        sorted_samples = [sample_map[sid] for sid in sorted_ids]

        first_val = df_sorted[sort_column][0]
        last_val = df_sorted[sort_column][-1]
        logger.debug(
            f"Sorted {len(sorted_samples)} samples by '{sort_column}': "
            f"{first_val} → {last_val}"
        )

        sorted_tortilla = Tortilla(samples=sorted_samples)
        sorted_tortilla._metadata_df = df_sorted

        taco_data = taco.model_dump()
        taco_data["tortilla"] = sorted_tortilla
        return Taco(**taco_data)

    except KeyError as e:
        raise TacoCreationError(f"Failed to sort samples: {e}") from e
    except Exception as e:
        raise TacoCreationError(f"Failed to sort samples: {e}") from e


def _extract_files_with_ids(samples: list, path_prefix: str = "") -> dict[str, Any]:
    """
    Extract file paths with sample IDs as archive paths.

    Recursively builds parallel lists:
    - src_files: absolute filesystem paths
    - arc_files: relative ZIP/FOLDER paths

    Sample IDs are used directly without modification.
    """
    src_files = []
    arc_files = []

    for sample in samples:
        if sample.type == "FOLDER":
            new_prefix = f"{path_prefix}{sample.id}/"
            # Cast to Tortilla to access .samples (implied by type="FOLDER")
            tortilla = cast(Tortilla, sample.path)
            nested = _extract_files_with_ids(tortilla.samples, new_prefix)
            src_files.extend(nested["src_files"])
            arc_files.extend(nested["arc_files"])
        else:
            src_path = str(sample.path)
            arc_path = f"{path_prefix}{sample.id}"

            src_files.append(src_path)
            arc_files.append(arc_path)

    return {"src_files": src_files, "arc_files": arc_files}


def _estimate_sample_size(sample: "Sample") -> int:
    """
    Estimate total size of sample including nested samples.

    For FILE: returns actual file size from filesystem.
    For FOLDER: recursively sums all nested file sizes.
    Used by _group_samples_by_size() for chunk calculation.
    """
    if sample.type == "FILE":
        if isinstance(sample.path, pathlib.Path) and sample.path.exists():
            return sample.path.stat().st_size
        return 0

    elif sample.type == "FOLDER":
        # Cast to Tortilla to access .samples
        tortilla = cast(Tortilla, sample.path)
        return sum(_estimate_sample_size(child) for child in tortilla.samples)

    return 0


def _group_samples_by_size(samples: list["Sample"], max_size: int) -> list[list]:
    """
    Group samples into chunks based on size limit. Greedy packing algorithm.

    IMPORTANT: Samples should be pre-sorted (e.g., by majortom:code) before
    calling this to ensure geographic/temporal clustering in chunks.

    Individual samples larger than max_size will be placed alone in their chunk.
    """
    chunks = []
    # Annotate list to avoid mypy error
    current_chunk: list[Sample] = []
    current_size = 0

    for sample in samples:
        sample_size = _estimate_sample_size(sample)

        if current_size + sample_size > max_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = [sample]
            current_size = sample_size
        else:
            current_chunk.append(sample)
            current_size += sample_size

    if current_chunk:
        chunks.append(current_chunk)

    logger.debug(f"Grouped {len(samples)} samples into {len(chunks)} chunks")
    return chunks


def _create_zip(
    taco: Taco,
    output_path: pathlib.Path,
    temp_dir: pathlib.Path | None,
    quiet: bool = False,
    **kwargs: Any,
) -> pathlib.Path:
    """Create single ZIP container: metadata → extract paths → write ZIP."""
    try:
        logger.debug("Generating metadata package")
        generator = MetadataGenerator(taco, debug=False)
        metadata_package = generator.generate_all_levels()

        logger.debug(
            f"Metadata: {len(metadata_package.levels)} consolidated levels, "
            f"{len(metadata_package.local_metadata)} local folders"
        )

        logger.debug("Extracting file paths")
        extracted = _extract_files_with_ids(taco.tortilla.samples, "DATA/")
        logger.debug(f"Extracted {len(extracted['src_files'])} data files")

        logger.debug(f"Creating ZIP: {output_path}")
        # Now passing Path | None for temp_dir
        writer = ZipWriter(output_path, quiet=quiet, temp_dir=temp_dir)
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
    quiet: bool = False,
    **kwargs: Any,
) -> pathlib.Path:
    """Create FOLDER container: metadata → write folder structure."""
    try:
        logger.debug("Generating metadata package")
        generator = MetadataGenerator(taco, debug=False)
        metadata_package = generator.generate_all_levels()

        logger.debug(f"Creating FOLDER: {output_path}")
        writer = FolderWriter(output_path, quiet=quiet)
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
    """Validate chunk output paths don't exist. Prevents overwriting."""
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
    temp_dir: pathlib.Path | None,
    quiet: bool = False,
    **kwargs: Any,
) -> list[pathlib.Path]:
    """
    Create multiple ZIP containers by splitting samples.

    Chunk naming: base_part0001.tacozip, base_part0002.tacozip, etc.
    Samples should be pre-sorted (via sort_by) for clustering.
    """
    logger.debug(f"Grouping samples by size (max_size={max_size})")
    sample_chunks = _group_samples_by_size(taco.tortilla.samples, max_size)

    if len(sample_chunks) == 1:
        logger.info("Only one chunk needed, creating single container")
        return [_create_zip(taco, output_path, temp_dir, quiet=quiet, **kwargs)]

    base_name = output_path.stem
    extension = output_path.suffix
    parent_dir = output_path.parent

    logger.debug(f"Validating {len(sample_chunks)} chunk paths")
    _validate_chunk_paths(sample_chunks, base_name, extension, parent_dir)

    created_files = []

    for i, chunk_samples in enumerate(
        progress_bar(
            sample_chunks, desc="Creating ZIP chunks", unit="chunk", colour="cyan"
        ),
        1,
    ):
        chunk_tortilla = Tortilla(samples=chunk_samples)
        chunk_taco_data = taco.model_dump()
        chunk_taco_data["tortilla"] = chunk_tortilla
        chunk_taco_data.pop("extent", None)

        chunk_taco = Taco(**chunk_taco_data)

        chunk_filename = f"{base_name}_part{i:04d}{extension}"
        chunk_path = parent_dir / chunk_filename

        logger.info(f"Creating chunk {i}/{len(sample_chunks)}: {chunk_filename}")

        try:
            created_path = _create_zip(
                chunk_taco, chunk_path, temp_dir, quiet=quiet, **kwargs
            )
            created_files.append(created_path)
        except Exception as e:
            raise TacoCreationError(f"Failed to create split containers: {e}") from e

    logger.info(
        f"Created {len(created_files)} ZIP chunks with max size "
        f"{max_size / (1024**3):.1f}GB"
    )
    return created_files


def _cleanup_tortilla_temp_files(tortilla: Tortilla) -> None:
    """
    Recursively cleanup temp files from all samples.

    Called automatically after successful create() to free disk space.
    Silent, recursive, safe (ignores errors).
    """
    for sample in tortilla.samples:
        if sample.type == "FILE":
            sample.cleanup()
        elif sample.type == "FOLDER":
            # Cast to Tortilla to avoid mypy error: "Path | Tortilla | bytes" has no attribute "samples"
            # Since type is FOLDER, we know path is Tortilla
            child_tortilla = cast(Tortilla, sample.path)
            _cleanup_tortilla_temp_files(child_tortilla)
