"""
Folder Writer - Create TACO datasets in FOLDER format.

This module handles the creation of FOLDER-format TACO containers with
dual metadata system:
- Local metadata: __meta__ files in each DATA/ subfolder
- Consolidated metadata: level*.avro files in METADATA/ directory

The FOLDER format provides:
- Direct filesystem access to individual files
- Human-readable directory structure
- Avro metadata for efficient queries
- Preserved internal:parent_id for hierarchical navigation

Structure example (level 0 = FILEs only):
    dataset/
    ├── DATA/
    │   ├── sample_001.tif
    │   ├── sample_002.tif
    │   └── sample_003.tif
    ├── METADATA/
    │   └── level0.avro
    └── COLLECTION.json

Structure example (level 0 = FOLDERs, level 1 = FILEs):
    dataset/
    ├── DATA/
    │   ├── folder_A/
    │   │   ├── __meta__
    │   │   ├── nested_001.tif
    │   │   └── nested_002.tif
    │   └── folder_B/
    │       ├── __meta__
    │       ├── nested_001.tif
    │       └── nested_002.tif
    ├── METADATA/
    │   ├── level0.avro
    │   └── level1.avro
    └── COLLECTION.json

Structure example (Change Detection - 3 levels deep):
    dataset/
    ├── DATA/
    │   ├── Landslide_001/
    │   │   ├── __meta__
    │   │   ├── label.json
    │   │   └── imagery/
    │   │       ├── __meta__
    │   │       ├── before.tif
    │   │       └── after.tif
    │   ├── Landslide_002/
    │   │   ├── __meta__
    │   │   ├── label.json
    │   │   └── imagery/
    │   │       ├── __meta__
    │   │       ├── before.tif
    │   │       └── after.tif
    │   └── ...
    ├── METADATA/
    │   ├── level0.avro
    │   ├── level1.avro
    │   └── level2.avro
    └── COLLECTION.json

PIT Schema for Change Detection example:
    {
        "root": {"n": 500, "type": ["FOLDER"]},
        "hierarchy": {
            "1": [{"n": 1000, "type": ["FILE", "FOLDER"], "id": ["label", "imagery"]}],
            "2": [{"n": 1000, "type": ["FILE", "FILE"], "id": ["before", "after"]}]
        }
    }
"""

import json
import pathlib
import shutil
from typing import Any

import fastavro
import polars as pl
import pyarrow as pa

from tacotoolbox._constants import (
    FOLDER_COLLECTION_FILENAME,
    FOLDER_DATA_DIR,
    FOLDER_META_FILENAME,
    FOLDER_METADATA_DIR,
    METADATA_PARENT_ID,
)
from tacotoolbox._metadata import MetadataPackage


class FolderWriterError(Exception):
    """Raised when folder writing operations fail."""


class FolderWriter:
    """Handle creation of folder container structures with dual metadata."""

    def __init__(self, output_dir: pathlib.Path, quiet: bool = True) -> None:
        self.output_dir = output_dir
        self.quiet = quiet
        self.data_dir = output_dir / FOLDER_DATA_DIR
        self.metadata_dir = output_dir / FOLDER_METADATA_DIR

    def create_complete_folder(
        self,
        samples: list[Any],
        metadata_package: MetadataPackage,
        **kwargs: Any,
    ) -> pathlib.Path:
        try:
            if not self.quiet:
                print("Creating folder structure...")

            self._create_structure()

            if not self.quiet:
                print("Copying data files...")
            self._copy_data_files(samples)

            if not self.quiet:
                print("Writing local metadata...")
            self._write_local_metadata_avro(metadata_package, **kwargs)

            if not self.quiet:
                print("Writing consolidated metadata...")
            self._write_consolidated_metadata_avro(metadata_package, **kwargs)

            if not self.quiet:
                print("Writing COLLECTION.json...")
            self._write_collection_json(metadata_package)

            if not self.quiet:
                print(f"Folder container created: {self.output_dir}/")

        except Exception as e:
            raise FolderWriterError(f"Failed to create folder container: {e}") from e
        else:
            return self.output_dir

    def _create_structure(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=False)
        self.metadata_dir.mkdir(parents=True, exist_ok=False)

    def _copy_data_files(self, samples: list[Any]) -> None:
        self._copy_samples_recursive(samples, path_prefix="")

    def _copy_samples_recursive(
        self, samples: list[Any], path_prefix: str = ""
    ) -> None:
        for sample in samples:
            if sample.type == "FOLDER":
                new_prefix = (
                    f"{path_prefix}{sample.id}/" if path_prefix else f"{sample.id}/"
                )
                nested_dir = self.data_dir / new_prefix.rstrip("/")
                nested_dir.mkdir(parents=True, exist_ok=True)

                self._copy_samples_recursive(
                    sample.path.samples, path_prefix=new_prefix
                )
            else:
                src_path = sample.path
                dst_path = self.data_dir / f"{path_prefix}{sample.id}"

                dst_path.parent.mkdir(parents=True, exist_ok=True)

                shutil.copy2(src_path, dst_path)

    def _write_local_metadata_avro(
        self,
        metadata_package: MetadataPackage,
        **kwargs: Any,
    ) -> None:
        """
        Write local __meta__ files for each folder.

        These files do NOT contain internal:parent_id (navigation is implicit
        via folder structure), but they preserve all other metadata fields.
        """
        for folder_path, local_df in metadata_package.local_metadata.items():
            meta_path = self.output_dir / f"{folder_path}{FOLDER_META_FILENAME}"

            meta_path.parent.mkdir(parents=True, exist_ok=True)

            self._write_avro_file(local_df, meta_path, **kwargs)

            if not self.quiet:
                print(f"  Created {folder_path}{FOLDER_META_FILENAME}")

    def _write_consolidated_metadata_avro(
        self,
        metadata_package: MetadataPackage,
        **kwargs: Any,
    ) -> None:
        """
        Write consolidated METADATA/levelX.avro files.

        These files preserve ALL columns from metadata_package.levels,
        including internal:parent_id for hierarchical navigation via JOINs.

        Example query enabled by parent_id:
          SELECT * FROM level0 l0
          LEFT JOIN level1 l1 ON l1."internal:parent_id" = l0."internal:id"

        FolderWriter does NOT modify the DataFrames, so parent_id is preserved.
        """
        for i, level_df in enumerate(metadata_package.levels):
            output_path = self.metadata_dir / f"level{i}.avro"

            # Direct write - preserves all columns including internal:parent_id
            self._write_avro_file(level_df, output_path, **kwargs)

            if not self.quiet:
                has_parent_id = METADATA_PARENT_ID in level_df.columns
                print(
                    f"  Created {FOLDER_METADATA_DIR}/level{i}.avro "
                    f"({len(level_df)} rows, parent_id={has_parent_id})"
                )

    def _write_avro_file(
        self,
        df: pl.DataFrame,
        output_path: pathlib.Path,
        **kwargs: Any,
    ) -> None:
        """Write DataFrame to Avro file, preserving all columns and types."""
        arrow_table = df.to_arrow()
        avro_schema = self._arrow_schema_to_avro(arrow_table.schema)
        records = arrow_table.to_pylist()

        with open(output_path, "wb") as f:
            fastavro.writer(f, avro_schema, records, **kwargs)

    def _arrow_schema_to_avro(self, arrow_schema: pa.Schema) -> dict[str, Any]:
        fields = []

        for field in arrow_schema:
            avro_type = self._arrow_type_to_avro(field.type)
            fields.append({"name": field.name, "type": ["null", avro_type]})

        return {"type": "record", "name": "TacoMetadata", "fields": fields}

    def _arrow_type_to_avro(self, arrow_type: pa.DataType) -> Any:
        simple_type = self._convert_simple_type(arrow_type)
        if simple_type:
            return simple_type

        if pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type):
            return self._convert_list_type(arrow_type)

        if pa.types.is_struct(arrow_type):
            return self._convert_struct_type(arrow_type)

        return "string"

    def _convert_simple_type(self, arrow_type: pa.DataType) -> str | None:
        if pa.types.is_int64(arrow_type):
            return "long"
        if (
            pa.types.is_int32(arrow_type)
            or pa.types.is_int16(arrow_type)
            or pa.types.is_int8(arrow_type)
        ):
            return "int"
        if pa.types.is_float64(arrow_type):
            return "double"
        if pa.types.is_float32(arrow_type):
            return "float"
        if pa.types.is_boolean(arrow_type):
            return "boolean"
        if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
            return "string"
        if pa.types.is_binary(arrow_type) or pa.types.is_large_binary(arrow_type):
            return "bytes"
        return None

    def _convert_list_type(self, arrow_type: pa.DataType) -> dict[str, Any]:
        value_type = arrow_type.value_type
        avro_value_type = self._arrow_type_to_avro(value_type)
        return {"type": "array", "items": avro_value_type}

    def _convert_struct_type(self, arrow_type: pa.DataType) -> dict[str, Any]:
        struct_fields = []
        for i in range(arrow_type.num_fields):
            struct_field = arrow_type.field(i)
            avro_field_type = self._arrow_type_to_avro(struct_field.type)
            struct_fields.append(
                {"name": struct_field.name, "type": ["null", avro_field_type]}
            )
        return {"type": "record", "name": "StructField", "fields": struct_fields}

    def _write_collection_json(self, metadata_package: MetadataPackage) -> None:
        """Write COLLECTION.json with pit_schema embedded."""
        collection_path = self.output_dir / FOLDER_COLLECTION_FILENAME

        collection_with_schema = metadata_package.collection.copy()
        collection_with_schema["taco:pit_schema"] = metadata_package.pit_schema

        with open(collection_path, "w", encoding="utf-8") as f:
            json.dump(collection_with_schema, f, indent=4, ensure_ascii=False)
