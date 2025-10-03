import json
import pathlib
import shutil
import tempfile
import uuid
import zipfile
from contextlib import suppress
from typing import Any

import tacozip

from tacotoolbox._writers.zip_writer import ZipOffsetReader


class TacoEditError(Exception):
    """Raised when TACO edit operations fail."""


def edit_collection(
    taco_path: str | pathlib.Path,
    collection_data: dict[str, Any],
    temp_dir: pathlib.Path | None = None,
) -> None:
    """
    Edit COLLECTION.json in an existing TACO ZIP archive.

    Only for ZIP format (.tacozip). For folder containers, edit the JSON file directly.

    IMPORTANT: The field "taco:pit_schema" is automatically generated and protected.
    It CANNOT be modified. Attempting to include it in collection_data will raise an error.
    The original schema is always preserved automatically.

    Args:
        taco_path: Path to .tacozip file
        collection_data: New COLLECTION.json content
            Must include required TACO fields (id, dataset_version, etc.)
            MUST NOT include "taco:pit_schema" (raises error)
        temp_dir: Optional custom temporary directory

    Raises:
        TacoEditError: If operation fails or if collection_data contains "taco:pit_schema"

    Examples:
        # Correct usage
        >>> new_data = {
        ...     "id": "my_dataset",
        ...     "dataset_version": "2.0.0",
        ...     "description": "Updated",
        ...     "licenses": ["CC-BY-4.0"],
        ...     "extent": {...},
        ...     "providers": [...],
        ...     "tasks": ["classification"],
        ... }
        >>> edit_collection("dataset.tacozip", new_data)

        # Load and modify pattern (recommended)
        >>> from tacotoolbox import Taco
        >>> taco = Taco.load("dataset.tacozip")
        >>> data = taco.model_dump()
        >>> data.pop("tortilla")  # Remove tortilla
        >>> data.pop("taco:pit_schema", None)  # Remove protected field
        >>> data["description"] = "Updated"
        >>> edit_collection("dataset.tacozip", data)
    """
    taco_path = pathlib.Path(taco_path)

    # Validate it's a valid ZIP file
    _validate_zip_path(taco_path)

    # REJECT if user tries to modify protected schema
    if "taco:pit_schema" in collection_data:
        raise TacoEditError(
            "Cannot modify 'taco:pit_schema' field.\n"
            "This field is auto-generated from metadata and cannot be edited.\n"
            "Remove 'taco:pit_schema' from your collection_data.\n"
            "The original schema will be preserved automatically."
        )

    # Validate required fields
    _validate_collection_data(collection_data)

    # Read original schema to preserve it
    original_schema = _read_original_pit_schema(taco_path)

    # Build final data: user's data + original schema
    final_data = collection_data.copy()
    if original_schema is not None:
        final_data["taco:pit_schema"] = original_schema

    # Setup temp directory
    base_temp = temp_dir or pathlib.Path(tempfile.gettempdir())
    temp_subdir = base_temp / f"taco_edit_{uuid.uuid4().hex}"

    try:
        temp_subdir.mkdir(parents=True, exist_ok=True)

        # Write new COLLECTION.json
        temp_json = temp_subdir / "collection.json"
        with open(temp_json, "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=4, ensure_ascii=False)

        # Remove old COLLECTION.json
        tacozip.trim_from(zip_path=str(taco_path), target="COLLECTION.json")

        # Append new COLLECTION.json
        tacozip.append_files(
            zip_path=str(taco_path), entries=[(str(temp_json), "COLLECTION.json")]
        )

        # Update TACO_HEADER
        _update_taco_header(taco_path)

    except TacoEditError:
        raise
    except Exception as e:
        raise TacoEditError(f"Failed to edit collection: {e}") from e
    finally:
        with suppress(Exception):
            shutil.rmtree(temp_subdir)


def _validate_zip_path(path: pathlib.Path) -> None:
    """Validate that path is a valid ZIP file."""
    if not path.exists():
        raise TacoEditError(f"File does not exist: {path}")

    if path.is_dir():
        raise TacoEditError(
            f"edit_collection() is only for ZIP files (.tacozip).\n"
            f"Path is a directory: {path}\n"
            f"For folder containers, edit COLLECTION.json directly."
        )

    if not path.is_file():
        raise TacoEditError(f"Path must be a file: {path}")


def _validate_collection_data(data: dict[str, Any]) -> None:
    """Validate collection data has required TACO fields."""
    required_fields = ["id", "dataset_version"]

    missing = [f for f in required_fields if f not in data]
    if missing:
        raise TacoEditError(
            f"Collection data missing required fields: {missing}\n"
            f"Required: {required_fields}"
        )


def _read_original_pit_schema(taco_path: pathlib.Path) -> dict[str, Any] | None:
    """
    Read taco:pit_schema from existing COLLECTION.json.

    Args:
        taco_path: Path to .tacozip file

    Returns:
        Original PIT schema or None if not present
    """
    try:
        with zipfile.ZipFile(taco_path, "r") as zf, zf.open("COLLECTION.json") as f:
            collection = json.load(f)
            return collection.get("taco:pit_schema")
    except Exception:
        return None


def _update_taco_header(taco_path: pathlib.Path) -> None:
    """Update TACO_HEADER with new offsets."""
    try:
        metadata_offsets = ZipOffsetReader.get_metadata_offsets(taco_path)
        collection_offset = ZipOffsetReader.get_collection_offset(taco_path)
        all_entries = [*metadata_offsets, collection_offset]
        tacozip.update_header(zip_path=str(taco_path), entries=all_entries)
    except Exception as e:
        raise TacoEditError(f"Failed to update TACO_HEADER: {e}") from e
