"""
TACO container creation with ZIP/FOLDER support and dataset splitting.

Main workflow:
1. Auto-detect format from extension (.zip/.tacozip → zip, else → folder)
2. Validate inputs
3. Generate metadata package
4. Create container(s) using appropriate writer
5. Cleanup temp files automatically

Example:
    >>> import tacotoolbox
    >>> tacotoolbox.verbose(True)  # Show progress
    >>> taco = Taco(tortilla=Tortilla(samples=[...]), ...)
    >>> paths = create(taco, "output.tacozip")  # auto-detects ZIP
    >>> paths = create(taco, "output_dataset")  # auto-detects FOLDER
"""

import pathlib
import warnings
from typing import TYPE_CHECKING, Any, Literal, cast

from tacotoolbox._logging import get_logger
from tacotoolbox._metadata import MetadataGenerator
from tacotoolbox._progress import progress_bar
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
    split_size: str | None = "4GB",
    group_by: str | list[str] | None = None,
    temp_dir: str | pathlib.Path | None = None,
    **kwargs: Any,
) -> list[pathlib.Path]:
    """
    Create TACO container from Taco object.

    Format auto-detection (output_format="auto"):
    - .zip/.tacozip → ZIP format
    - anything else → FOLDER format

    Grouping behavior:
    - If group_by is set: Each unique group value creates one ZIP file
      (split_size is ignored, entire group goes into single file)
    - If group_by is None: Split by size using split_size parameter

    Temp files from Sample(path=bytes) are always cleaned up after success.

    Args:
        taco: Taco object to write
        output: Output path (file for ZIP, directory for FOLDER)
        output_format: Container format ("zip", "folder", or "auto")
        split_size: Max size per ZIP file (e.g., "4GB"), None disables splitting
        group_by: Column(s) to group by (creates one ZIP per unique value)
        temp_dir: Directory for temporary files (default: system temp)
        **kwargs: Additional Parquet writer parameters

    Returns:
        list[pathlib.Path]: List of created container paths

    Raises:
        TacoCreationError: If container creation fails
        TacoValidationError: If inputs are invalid
    """
    output_path = pathlib.Path(output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

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

    _validate_all_inputs(taco, output_path, final_format, split_size, group_by)

    # Adjust output path for folder format
    if final_format == "folder" and output_path.suffix.lower() in (".zip", ".tacozip"):
        output_path = output_path.with_suffix("")
        logger.debug(f"Adjusted output path for folder format: {output_path}")

    # Force folder-specific constraints
    if final_format == "folder":
        split_size = None
        group_by = None
        temp_dir = None
        logger.debug("Folder format: split_size, group_by, and temp_dir ignored")

    # Convert temp_dir to Path if provided
    temp_path_dir = pathlib.Path(temp_dir) if temp_dir else None

    # Progress bars controlled by logging level
    try:
        if group_by is not None:
            logger.info(f"Creating grouped containers by column: {group_by}")
            result = _create_grouped_zips(
                taco,
                output_path,
                group_by,
                temp_path_dir,
                **kwargs,
            )
        elif split_size is not None:
            max_size = validate_split_size(split_size)
            logger.info(f"Creating split containers (max_size={split_size})")
            result = _create_with_splitting(
                taco,
                output_path,
                max_size,
                temp_path_dir,
                **kwargs,
            )
        elif final_format == "zip":
            logger.info(f"Creating single ZIP: {output_path}")
            result = [_create_zip(taco, output_path, temp_path_dir, **kwargs)]
        else:
            logger.info(f"Creating FOLDER: {output_path}")
            result = [_create_folder(taco, output_path, **kwargs)]

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
    group_by: str | list[str] | None,
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

    if group_by is not None and output_format == "folder":
        raise TacoValidationError(
            "group_by is not supported with format='folder'. "
            "Grouping is only available for format='zip'."
        )

    if not taco.tortilla.samples:
        raise TacoValidationError("Cannot create container from empty tortilla")

    logger.debug("All inputs validated successfully")


def _validate_group_column(group_column: str, df_columns: list[str]) -> None:
    """Validate that group column exists in metadata."""
    if group_column not in df_columns:
        available = sorted(df_columns)
        raise TacoCreationError(
            f"Group column '{group_column}' not found in metadata.\n"
            f"Available columns: {available}"
        )


def _group_samples_by_column(
    taco: Taco, group_by: str | list[str]
) -> dict[str, list["Sample"]]:
    """
    Group samples by metadata column(s).

    Returns dict mapping group value(s) to list of samples.
    Converts integer values to strings with warning.

    Common use cases:
    - "spatial_group": Geographic grouping (e.g., "g0000", "g0001")
    - "majortom:code": MajorTOM grid cells
    - ["region", "sensor"]: Multiple columns combined with underscore
    """
    try:
        df = taco.tortilla.export_metadata(deep=0)

        # Handle single column or list of columns
        group_columns = [group_by] if isinstance(group_by, str) else group_by

        # Validate all columns exist
        for col in group_columns:
            _validate_group_column(col, df.columns)

        # Check if any column contains integers
        for col in group_columns:
            sample_value = df[col][0]
            if isinstance(sample_value, int):
                warnings.warn(
                    f"Group column '{col}' contains integer values. "
                    f"Converting to strings for file naming.",
                    UserWarning,
                    stacklevel=3,
                )

        # Create group keys by combining column values
        group_keys = []
        for row in df.iter_rows(named=True):
            values = [str(row[col]) for col in group_columns]
            group_key = "_".join(values)
            group_keys.append(group_key)

        # Build groups dictionary
        groups: dict[str, list[Sample]] = {}
        sample_map = {s.id: s for s in taco.tortilla.samples}

        for group_key, row in zip(group_keys, df.iter_rows(named=True), strict=True):
            sample_id = row["id"]
            sample = sample_map[sample_id]

            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(sample)

        logger.debug(
            f"Grouped {len(taco.tortilla.samples)} samples into {len(groups)} groups"
        )

    except KeyError as e:
        raise TacoCreationError(f"Failed to group samples: {e}") from e
    except Exception as e:
        raise TacoCreationError(f"Failed to group samples: {e}") from e
    else:
        return groups


def _create_grouped_zips(
    taco: Taco,
    output_path: pathlib.Path,
    group_by: str | list[str],
    temp_dir: pathlib.Path | None,
    **kwargs: Any,
) -> list[pathlib.Path]:
    """
    Create one ZIP file per group.

    Each group becomes a single ZIP file regardless of size.
    split_size is ignored when using group_by.

    File naming: {base}_{group_key}.tacozip
    Example: dataset_g0000.tacozip, dataset_g0001.tacozip
    """
    groups = _group_samples_by_column(taco, group_by)

    base_name = output_path.stem
    extension = output_path.suffix
    parent_dir = output_path.parent

    created_files = []

    for group_key, group_samples in progress_bar(
        groups.items(), desc="Creating grouped ZIPs", unit="group", colour="cyan"
    ):
        chunk_tortilla = Tortilla(samples=group_samples)
        chunk_taco_data = taco.model_dump()
        chunk_taco_data["tortilla"] = chunk_tortilla
        chunk_taco_data.pop("extent", None)

        chunk_taco = Taco(**chunk_taco_data)

        group_filename = f"{base_name}_{group_key}{extension}"
        group_path = parent_dir / group_filename

        if group_path.exists():
            raise TacoValidationError(
                f"Group file already exists: {group_path}\n"
                f"Remove existing files or choose a different output path."
            )

        logger.info(
            f"Creating group {group_key}: {len(group_samples)} samples → {group_filename}"
        )

        try:
            created_path = _create_zip(chunk_taco, group_path, temp_dir, **kwargs)
            created_files.append(created_path)
        except Exception as e:
            raise TacoCreationError(f"Failed to create grouped containers: {e}") from e

    logger.info(f"Created {len(created_files)} grouped ZIP files")
    return created_files


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
    **kwargs: Any,
) -> pathlib.Path:
    """
    Create single ZIP container: metadata -> extract paths -> write ZIP.
    """
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
        # Progress bars controlled by logging level
        writer = ZipWriter(output_path=output_path, temp_dir=temp_dir)
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
    **kwargs: Any,
) -> pathlib.Path:
    """
    Create FOLDER container: metadata → write folder structure.
    """
    try:
        logger.debug("Generating metadata package")
        generator = MetadataGenerator(taco, debug=False)
        metadata_package = generator.generate_all_levels()

        logger.debug(f"Creating FOLDER: {output_path}")
        writer = FolderWriter(output_path)
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
    **kwargs: Any,
) -> list[pathlib.Path]:
    """
    Create multiple ZIP containers by splitting samples.

    Chunk naming: base_part0001.tacozip, base_part0002.tacozip, etc.
    """
    logger.debug(f"Grouping samples by size (max_size={max_size})")
    sample_chunks = _group_samples_by_size(taco.tortilla.samples, max_size)

    if len(sample_chunks) == 1:
        logger.info("Only one chunk needed, creating single container")
        return [_create_zip(taco, output_path, temp_dir, **kwargs)]

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
            created_path = _create_zip(chunk_taco, chunk_path, temp_dir, **kwargs)
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
