import json
import pathlib
import tempfile
import uuid
from contextlib import suppress

from tacotoolbox.create import MetadataManager
from tacotoolbox.taco.datamodel import Taco


class TacoEditError(Exception):
    """Custom exception for TACO edit operations."""

    pass


def edit_collection(taco_path: str | pathlib.Path, new_collection_data: dict) -> None:
    """
    Edit the COLLECTION.json file in an existing TACO archive.

    Args:
        taco_path: Path to the TACO file to edit
        new_collection_data: Dictionary containing new TACO collection metadata

    Raises:
        TacoEditError: If validation fails or operation encounters errors

    Example:
        >>> new_data = {
        ...     "id": "my_dataset",
        ...     "dataset_version": "2.0.0",
        ...     "description": "Updated description",
        ...     # ... other required TACO fields
        ... }
        >>> edit_collection("my_dataset.taco", new_data)
    """
    taco_path = pathlib.Path(taco_path)

    # Step 1: Validate new collection data is a valid TACO
    try:
        # Add dummy tortilla for validation since COLLECTION.json doesn't include tortilla
        validation_data = new_collection_data.copy()
        if "tortilla" not in validation_data:
            # Create minimal dummy tortilla for validation
            from tacotoolbox.tortilla.datamodel import Tortilla

            validation_data["tortilla"] = Tortilla(samples=[])

        # Validate using Pydantic model
        validated_taco = Taco.model_validate(validation_data)

        # Remove tortilla from the data that will be saved (COLLECTION.json shouldn't include it)
        collection_data = validated_taco.model_dump()
        collection_data.pop("tortilla", None)

    except Exception as e:
        raise TacoEditError(f"Invalid TACO collection data: {e}") from e

    # Step 2: Create temporary JSON file
    temp_dir = pathlib.Path(tempfile.gettempdir())
    temp_json_path = temp_dir / f"collection_{uuid.uuid4().hex}.json"

    try:
        # Write new collection data to temporary file
        with open(temp_json_path, "w", encoding="utf-8") as f:
            json.dump(collection_data, f, indent=4, ensure_ascii=False)

        # Step 3: Replace COLLECTION.json in TACO archive
        import tacozip

        result = tacozip.replace_file(
            zip_path=str(taco_path),
            file_name="COLLECTION.json",
            new_src_path=str(temp_json_path),
        )

        if result != 0:
            raise TacoEditError(
                f"Failed to replace COLLECTION.json in TACO file (error code: {result})"
            )

        # Step 4: Update TACO_HEADER with new COLLECTION.json position
        try:
            # Get current metadata offsets (first part of header)
            metadata_offsets, metadata_lengths = MetadataManager.get_parquet_offsets(
                taco_path
            )

            # Get new COLLECTION.json position
            collection_offset, collection_length = (
                MetadataManager.get_collection_json_offset(taco_path)
            )

            # Reconstruct header entries: metadata + collection
            all_entries = [
                *zip(metadata_offsets, metadata_lengths),
                (collection_offset, collection_length),
            ]

            # Update header
            tacozip.update_header(zip_path=str(taco_path), entries=all_entries)

        except Exception as e:
            raise TacoEditError(f"Failed to update TACO header: {e}") from e

    finally:
        # Step 5: Cleanup temporary file
        with suppress(Exception):
            temp_json_path.unlink()