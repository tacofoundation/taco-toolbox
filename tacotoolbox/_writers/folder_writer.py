import json
import pathlib
import shutil
from typing import Any

import fastavro
import polars as pl
import pyarrow as pa

from tacotoolbox._metadata import RelativePathEnricher
from tacotoolbox._types import LevelMetadata, MetadataPackage


class FolderWriterError(Exception):
    """Raised when folder writing operations fail."""


class FolderWriter:
    """Handle creation of folder container structures."""

    def __init__(self, output_dir: pathlib.Path, quiet: bool = True) -> None:
        """
        Initialize folder writer.

        Args:
            output_dir: Path for output directory
            quiet: Suppress progress messages
        """
        self.output_dir = output_dir
        self.quiet = quiet
        self.data_dir = output_dir / "DATA"
        self.metadata_dir = output_dir / "METADATA"

    def create_complete_folder(
        self,
        samples: list[Any],
        metadata_package: MetadataPackage,
        **avro_kwargs: Any,
    ) -> pathlib.Path:
        """
        Create complete folder container.

        Folder format uses Avro for metadata (mutable, appendable).

        Args:
            samples: List of Sample objects from tortilla
            metadata_package: Complete metadata bundle
            **avro_kwargs: Additional kwargs for fastavro.writer()
                Examples: codec='snappy', sync_interval, metadata, etc.

        Returns:
            Path to created directory

        Raises:
            FolderWriterError: If creation fails
        """
        try:
            # Save samples for enricher
            self.samples = samples

            # Step 1: Create directory structure
            self._create_structure()

            # Step 2: Copy data files
            self._copy_data_files(samples)

            # Step 3: Write metadata as Avro files
            self._write_metadata_avro(metadata_package["levels"], **avro_kwargs)

            # Step 4: Write COLLECTION.json
            self._write_collection_json(metadata_package["collection"])

            if not self.quiet:
                print(f"Folder container created: {self.output_dir}/")

        except Exception as e:
            raise FolderWriterError(f"Failed to create folder container: {e}") from e
        else:
            return self.output_dir

    def _create_structure(self) -> None:
        """Create DATA/ and METADATA/ directories."""
        self.data_dir.mkdir(parents=True, exist_ok=False)
        self.metadata_dir.mkdir(parents=True, exist_ok=False)

    def _copy_data_files(self, samples: list[Any]) -> None:
        """
        Copy data files to DATA/ directory.

        Recursively handles TORTILLA structures, preserving hierarchy.

        Args:
            samples: List of Sample objects
        """
        self._copy_samples_recursive(samples, path_prefix="")

    def _copy_samples_recursive(
        self, samples: list[Any], path_prefix: str = ""
    ) -> None:
        """
        Recursively copy samples to DATA/.

        Args:
            samples: List of Sample objects
            path_prefix: Current path prefix for nested samples
        """
        for sample in samples:
            if sample.type == "FOLDER":
                # Create subdirectory for nested tortilla
                new_prefix = (
                    f"{path_prefix}{sample.id}/" if path_prefix else f"{sample.id}/"
                )
                nested_dir = self.data_dir / new_prefix.rstrip("/")
                nested_dir.mkdir(parents=True, exist_ok=True)

                # Recurse into children
                self._copy_samples_recursive(
                    sample.path.samples, path_prefix=new_prefix
                )
            else:
                # Copy leaf node file
                src_path = sample.path
                file_suffix = src_path.suffix
                dst_path = self.data_dir / f"{path_prefix}{sample.id}{file_suffix}"

                # Ensure parent directory exists
                dst_path.parent.mkdir(parents=True, exist_ok=True)

                # Copy file
                shutil.copy2(src_path, dst_path)

    def _write_metadata_avro(
        self,
        levels: list[LevelMetadata],
        **avro_kwargs: Any,
    ) -> None:
        """
        Write metadata as Avro files to METADATA/ directory.

        Avro format is used for folder containers because it's mutable
        and supports appending new records efficiently.

        Args:
            levels: List of LevelMetadata dicts
            **avro_kwargs: Additional kwargs for fastavro.writer()
        """
        enricher = RelativePathEnricher(self.samples, self.quiet)

        for i, level_meta in enumerate(levels):
            df = level_meta["dataframe"]

            # Enrich with internal:relative_path
            df = enricher.enrich_metadata(df)

            # Write Avro file
            output_path = self.metadata_dir / f"level{i}.avro"
            self._write_avro_file(df, output_path, **avro_kwargs)

    def _write_avro_file(
        self, df: pl.DataFrame, output_path: pathlib.Path, **avro_kwargs: Any
    ) -> None:
        """
        Write DataFrame to Avro file using Arrow schema.

        Args:
            df: DataFrame to write
            output_path: Output file path
            **avro_kwargs: Additional kwargs for fastavro.writer()
        """
        # Convert to Arrow (preserves all type information)
        arrow_table = df.to_arrow()

        # Convert Arrow schema to Avro schema
        avro_schema = self._arrow_schema_to_avro(arrow_table.schema)

        # Convert Arrow table to Python records
        records = arrow_table.to_pylist()

        # Write Avro file
        with open(output_path, "wb") as f:
            fastavro.writer(f, avro_schema, records, **avro_kwargs)

    def _arrow_schema_to_avro(self, arrow_schema: pa.Schema) -> dict[str, Any]:
        """
        Convert Arrow schema to Avro schema.

        Args:
            arrow_schema: PyArrow schema

        Returns:
            Avro schema dict
        """
        fields = []

        for field in arrow_schema:
            avro_type = self._arrow_type_to_avro(field.type)

            # Make all fields nullable
            fields.append({"name": field.name, "type": ["null", avro_type]})

        return {"type": "record", "name": "TacoMetadata", "fields": fields}

    def _arrow_type_to_avro(self, arrow_type: pa.DataType) -> Any:
        """
        Convert Arrow type to Avro type.

        Args:
            arrow_type: PyArrow data type

        Returns:
            Avro type (string or dict)
        """
        # Handle simple scalar types
        simple_type = self._convert_simple_type(arrow_type)
        if simple_type:
            return simple_type

        # Handle complex types
        if pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type):
            return self._convert_list_type(arrow_type)

        if pa.types.is_struct(arrow_type):
            return self._convert_struct_type(arrow_type)

        # Default to string for unknown types
        return "string"

    def _convert_simple_type(self, arrow_type: pa.DataType) -> str | None:
        """Convert simple Arrow types to Avro types."""
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
        """Convert Arrow list type to Avro array type."""
        value_type = arrow_type.value_type
        avro_value_type = self._arrow_type_to_avro(value_type)
        return {"type": "array", "items": avro_value_type}

    def _convert_struct_type(self, arrow_type: pa.DataType) -> dict[str, Any]:
        """Convert Arrow struct type to Avro record type."""
        struct_fields = []
        for i in range(arrow_type.num_fields):
            struct_field = arrow_type.field(i)
            avro_field_type = self._arrow_type_to_avro(struct_field.type)
            struct_fields.append(
                {"name": struct_field.name, "type": ["null", avro_field_type]}
            )
        return {"type": "record", "name": "StructField", "fields": struct_fields}

    def _write_collection_json(self, collection: dict[str, object]) -> None:
        """
        Write COLLECTION.json to container root.

        Args:
            collection: Dictionary for COLLECTION.json
        """
        collection_path = self.output_dir / "COLLECTION.json"

        with open(collection_path, "w", encoding="utf-8") as f:
            json.dump(collection, f, indent=4, ensure_ascii=False)
