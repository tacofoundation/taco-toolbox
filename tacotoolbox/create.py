"""
TACO container creation with ZIP/FOLDER support and dataset splitting.

Main workflow:
1. Auto-detect format from extension (.zip/.tacozip → zip, else → folder)
2. Validate inputs
3. Generate metadata package
4. Create container(s) using appropriate writer
5. Auto-consolidate into .tacocat/ when multiple ZIPs are created
6. Cleanup temp files automatically

Example:
    >>> import tacotoolbox
    >>> tacotoolbox.verbose(True)  # Show progress
    >>> taco = Taco(tortilla=Tortilla(samples=[...]), ...)
    >>> paths = create(taco, "output.tacozip")  # auto-detects ZIP
    >>> paths = create(taco, "output_dataset")  # auto-detects FOLDER
    >>> paths = create(taco, "data.tacozip", split_size="4GB")  # auto-creates .tacocat/
"""

import pathlib
import re
import warnings
from typing import TYPE_CHECKING, Any, Literal, cast

import pyarrow as pa
import pyarrow.compute as pc

from tacotoolbox._exceptions import TacoCreationError, TacoValidationError
from tacotoolbox._logging import get_logger
from tacotoolbox._metadata import MetadataGenerator
from tacotoolbox._progress import progress_bar
from tacotoolbox._validation import (
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


def create(  # noqa: C901
    taco: Taco,
    output: str | pathlib.Path,
    output_format: Literal["zip", "folder", "auto"] = "auto",
    split_size: str | None = "4GB",
    group_by: str | list[str] | None = None,
    consolidate: bool = True,
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

    Auto-consolidation (consolidate=True, default):
    - When multiple ZIPs are created (split or grouped), automatically creates
      .tacocat/ folder with consolidated parquets + COLLECTION.json metadata
    - Enables querying all ZIPs together with TacoCatReader
    - If consolidation fails, logs warning but keeps individual ZIPs

    Temp files from Sample(path=bytes) are always cleaned up after success.

    Args:
        taco: Taco object to write
        output: Output path (file for ZIP, directory for FOLDER)
        output_format: Container format ("zip", "folder", or "auto")
        split_size: Max size per ZIP file (e.g., "4GB"), None disables splitting
        group_by: Column(s) to group by (creates one ZIP per unique value)
        consolidate: Auto-create .tacocat/ when multiple ZIPs generated (default: True)
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

    _validate_all_inputs(
        taco, output_path, final_format, split_size, group_by, consolidate
    )

    # Adjust output path for folder format
    if final_format == "folder" and output_path.suffix.lower() in (".zip", ".tacozip"):
        output_path = output_path.with_suffix("")
        logger.debug(f"Adjusted output path for folder format: {output_path}")

    # Force folder-specific constraints
    if final_format == "folder":
        split_size = None
        group_by = None
        temp_dir = None
        consolidate = False
        logger.debug(
            "Folder format: split_size, group_by, temp_dir, and consolidate ignored"
        )

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

        # Auto-consolidate when multiple ZIPs are created
        if consolidate and len(result) > 1 and final_format == "zip":
            try:
                _create_consolidated_tacocat(result, output_path.parent)
            except Exception as e:
                logger.warning(
                    f"Failed to create .tacocat/ consolidation: {e}\n"
                    f"Individual ZIP files were created successfully."
                )

    except (TacoCreationError, TacoValidationError):
        raise
    except Exception as e:
        logger.exception("Container creation failed")
        raise TacoCreationError(f"Failed to create TACO container: {e}") from e
    else:
        logger.info(f"Successfully created {len(result)} container(s)")
        return result


def _create_consolidated_tacocat(
    zip_paths: list[pathlib.Path],
    output_dir: pathlib.Path,
) -> None:
    """
    Create .tacocat/ folder from multiple ZIP files.

    Automatically consolidates multiple ZIPs into a single queryable folder
    with unified parquet files and COLLECTION.json metadata.

    Assumes validation has already passed (no .tacocat/ conflict).
    """
    from tacotoolbox.tacocat import create_tacocat

    logger.info(f"Creating .tacocat/ consolidation for {len(zip_paths)} ZIPs...")

    create_tacocat(
        inputs=zip_paths,
        output=output_dir,
        validate_schema=True,
    )

    logger.info(f"Consolidation complete: {output_dir / '.tacocat'}")


def _validate_all_inputs(
    taco: Taco,
    output_path: pathlib.Path,
    output_format: Literal["zip", "folder"],
    split_size: str | None,
    group_by: str | list[str] | None,
    consolidate: bool,
) -> None:
    """Validate all inputs before starting. Fails fast."""
    logger.debug("Validating inputs")

    validate_format_value(output_format)
    validate_output_path(output_path, output_format)

    # Check reserved folder names for FOLDER format
    if output_format == "folder":
        from tacotoolbox._constants import RESERVED_FOLDER_NAMES

        folder_name = output_path.name
        if folder_name in RESERVED_FOLDER_NAMES:
            raise TacoValidationError(
                f"Output folder name '{folder_name}' is reserved by TACO specification.\n"
                f"Reserved names: {', '.join(sorted(RESERVED_FOLDER_NAMES))}\n"
                f"These names conflict with TACO container structure.\n"
                f"Please choose a different output name."
            )

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

    # Check .tacocat/ conflict BEFORE creating any ZIPs (fail fast)
    if consolidate and output_format == "zip" and (split_size or group_by):
        tacocat_path = output_path.parent / ".tacocat"
        if tacocat_path.exists():
            raise TacoValidationError(
                f".tacocat/ already exists in {output_path.parent}\n"
                f"This conflicts with automatic consolidation.\n"
                f"Options:\n"
                f"  1. Remove existing .tacocat/ directory: rm -rf {tacocat_path}\n"
                f"  2. Set consolidate=False to skip consolidation\n"
                f"  3. Use a different output directory"
            )

    logger.debug("All inputs validated successfully")


def _validate_group_column(group_column: str, table_columns: list[str]) -> None:
    """Validate that group column exists in metadata."""
    if group_column not in table_columns:
        available = sorted(table_columns)
        raise TacoCreationError(
            f"Group column '{group_column}' not found in metadata.\n"
            f"Available columns: {available}"
        )


def _sanitize_filename(name: str) -> str:
    r"""
    Sanitize string for use in filename.

    Replaces problematic characters with underscores:
    - Forward/backward slashes (/, \)
    - Colons (:)
    - Wildcards (*, ?)
    - Quotes (", ')
    - Angle brackets (<, >)
    - Pipes (|)
    - Multiple spaces/underscores collapsed to single underscore

    Examples:
        "Ocean/Sea/Lakes" → "Ocean_Sea_Lakes"
        "data:2024-01-01" → "data_2024-01-01"
        "file<test>" → "file_test_"
    """
    # Replace problematic characters with underscore
    sanitized = re.sub(r'[/\\:*?"<>|\']', "_", name)
    # Collapse multiple underscores/spaces to single underscore
    sanitized = re.sub(r"[_\s]+", "_", sanitized)
    # Strip leading/trailing underscores
    sanitized = sanitized.strip("_")
    return sanitized


def _group_samples_by_column(
    taco: Taco, group_by: str | list[str]
) -> dict[str, list["Sample"]]:
    """
    Group samples by metadata column(s).

    Returns dict mapping group value(s) to list of samples.
    Converts integer values to strings with warning.

    Common use cases:
    - "spatialgroup:code": Geographic grouping (e.g., "sg0000", "sg0001")
    - "majortom:code": MajorTOM grid cells
    - "geoenrich:admin_countries": Country names
    - ["region", "sensor"]: Multiple columns combined with underscore
    """
    try:
        table = taco.tortilla.export_metadata(deep=0)

        # Handle single column or list of columns
        group_columns = [group_by] if isinstance(group_by, str) else group_by

        # Validate all columns exist
        table_column_names = table.schema.names
        for col in group_columns:
            _validate_group_column(col, table_column_names)

        # Check if any column contains integers
        for col in group_columns:
            sample_value = table.column(col).to_pylist()[0]
            if isinstance(sample_value, int):
                warnings.warn(
                    f"Group column '{col}' contains integer values. "
                    f"Converting to strings for file naming.",
                    UserWarning,
                    stacklevel=3,
                )

        # Create group keys by combining column values
        group_keys = []
        for row in table.to_pylist():
            values = [str(row[col]) for col in group_columns]
            group_key = "_".join(values)
            group_keys.append(group_key)

        # Build groups dictionary
        groups: dict[str, list[Sample]] = {}
        sample_map = {s.id: s for s in taco.tortilla.samples}

        for group_key, row in zip(group_keys, table.to_pylist(), strict=True):
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
    r"""
    Create one ZIP file per group.

    Each group becomes a single ZIP file regardless of size.
    split_size is ignored when using group_by.

    File naming: {base}_{sanitized_group_key}.tacozip
    Example: dataset_sg0000.tacozip, dataset_Ocean_Sea_Lakes.tacozip

    Group keys are sanitized to remove invalid filename characters
    (/, \, :, *, ?, ", <>, |, ') which are replaced with underscores.
    """
    groups = _group_samples_by_column(taco, group_by)

    base_name = output_path.stem
    extension = output_path.suffix
    parent_dir = output_path.parent

    created_files = []

    for group_key, group_samples in progress_bar(
        groups.items(), desc="Creating grouped ZIPs", unit="group", colour="cyan"
    ):
        # Filter existing metadata table to preserve TortillaExtension fields
        sample_ids = [s.id for s in group_samples]
        id_column = taco.tortilla._metadata_table.column("id")
        mask = pc.is_in(id_column, pa.array(sample_ids))
        filtered_table = taco.tortilla._metadata_table.filter(mask)

        chunk_tortilla = Tortilla(samples=group_samples, _metadata_table=filtered_table)
        # Preserve TortillaExtension field descriptions from parent
        chunk_tortilla._field_descriptions.update(taco.tortilla._field_descriptions)

        chunk_taco_data = taco.model_dump()
        chunk_taco_data["tortilla"] = chunk_tortilla
        chunk_taco_data.pop("extent", None)

        chunk_taco = Taco(**chunk_taco_data)

        # Sanitize group_key for filename
        safe_group_key = _sanitize_filename(group_key)
        group_filename = f"{base_name}_{safe_group_key}{extension}"
        group_path = parent_dir / group_filename

        if group_path.exists():
            raise TacoValidationError(
                f"Group file already exists: {group_path}\n"
                f"Remove existing files or choose a different output path."
            )

        logger.info(
            f"Creating group '{group_key}': {len(group_samples)} samples → {group_filename}"
        )

        created_path = _create_zip(chunk_taco, group_path, temp_dir, **kwargs)
        created_files.append(created_path)

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
        sample_size = sample._size_bytes

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
    """Create single ZIP container: metadata -> extract paths -> write ZIP."""
    logger.debug("Generating metadata package")
    generator = MetadataGenerator(taco)
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


def _create_folder(
    taco: Taco,
    output_path: pathlib.Path,
    **kwargs: Any,
) -> pathlib.Path:
    """Create FOLDER container: metadata → write folder structure."""
    logger.debug("Generating metadata package")
    generator = MetadataGenerator(taco)
    metadata_package = generator.generate_all_levels()

    logger.debug(f"Creating FOLDER: {output_path}")
    writer = FolderWriter(output_path)
    return writer.create_complete_folder(
        samples=taco.tortilla.samples,
        metadata_package=metadata_package,
        **kwargs,
    )


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
        # Filter existing metadata table to preserve TortillaExtension fields
        sample_ids = [s.id for s in chunk_samples]
        id_column = taco.tortilla._metadata_table.column("id")
        mask = pc.is_in(id_column, pa.array(sample_ids))
        filtered_table = taco.tortilla._metadata_table.filter(mask)

        chunk_tortilla = Tortilla(samples=chunk_samples, _metadata_table=filtered_table)
        # Preserve TortillaExtension field descriptions from parent
        chunk_tortilla._field_descriptions.update(taco.tortilla._field_descriptions)

        chunk_taco_data = taco.model_dump()
        chunk_taco_data["tortilla"] = chunk_tortilla
        chunk_taco_data.pop("extent", None)

        chunk_taco = Taco(**chunk_taco_data)

        chunk_filename = f"{base_name}_part{i:04d}{extension}"
        chunk_path = parent_dir / chunk_filename

        logger.info(f"Creating chunk {i}/{len(sample_chunks)}: {chunk_filename}")

        created_path = _create_zip(chunk_taco, chunk_path, temp_dir, **kwargs)
        created_files.append(created_path)

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
