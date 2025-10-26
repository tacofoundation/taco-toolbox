import pathlib
from typing import Any, Literal

from tacotoolbox._helpers import group_samples_by_size
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
    temp_dir: str | pathlib.Path | None = None,
    quiet: bool = True,
    **kwargs: Any,
) -> list[pathlib.Path]:
    output_path = pathlib.Path(output)
    validate_format_value(output_format)
    
    if output_format == "folder" and output_path.suffix in (".zip", ".tacozip"):
        output_path = output_path.with_suffix("")
    
    validate_output_path(output_path, output_format)
    
    if output_format == "folder":
        split_size = None
        temp_dir = None
    
    if split_size is not None:
        max_size = validate_split_size(split_size)
        return _create_with_splitting(taco, output_path, max_size, quiet, temp_dir, **kwargs)
    
    if output_format == "zip":
        return [_create_zip(taco, output_path, quiet, temp_dir, **kwargs)]
    
    return [_create_folder(taco, output_path, quiet, **kwargs)]


def _extract_files_with_ids(samples: list, path_prefix: str = "") -> dict[str, Any]:
    """
    Extract files using sample IDs as-is in archive paths.
    
    Sample IDs are used directly without any modification.
    If user wants extension, they include it in the ID.
    
    Returns:
        dict with 'src_files' and 'arc_files' lists
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


def _create_zip(
    taco: Taco,
    output_path: pathlib.Path,
    quiet: bool,
    temp_dir: pathlib.Path | str | None,
    **kwargs: Any,
) -> pathlib.Path:
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
    try:
        sample_chunks = group_samples_by_size(taco.tortilla.samples, max_size)
        
        if len(sample_chunks) == 1:
            return [_create_zip(taco, output_path, quiet, temp_dir, **kwargs)]
        
        base_name = output_path.stem
        extension = output_path.suffix
        parent_dir = output_path.parent
        
        _validate_chunk_paths(sample_chunks, base_name, extension, parent_dir)
        
        created_files = []
        
        for i, chunk_samples in enumerate(sample_chunks, 1):
            chunk_tortilla = Tortilla(samples=chunk_samples)
            chunk_taco_data = taco.model_dump()
            chunk_taco_data["tortilla"] = chunk_tortilla
            chunk_taco = Taco(**chunk_taco_data)
            
            chunk_filename = f"{base_name}_part{i:04d}{extension}"
            chunk_path = parent_dir / chunk_filename
            
            created_path = _create_zip(chunk_taco, chunk_path, quiet, temp_dir, **kwargs)
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