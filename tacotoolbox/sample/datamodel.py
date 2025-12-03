"""
Sample datamodel for TACO framework.

The Sample is the fundamental data unit, combining raw data with structured metadata
for training, validation, and testing workflows.

Key features:
- Supports FILE and FOLDER asset types
- Automatic type inference from path (type="auto" default)
- Dynamic extension system for adding metadata
- Format validation via validators (TacotiffValidator, etc.)
- Automatic cleanup of temporary files from bytes
- Pop metadata fields as Arrow Tables for reuse
- Private _size_bytes attribute auto-calculated at construction
"""

import contextlib
import pathlib
import re
import tempfile
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

import pyarrow as pa
import pydantic

from tacotoolbox._constants import SHARED_CORE_FIELDS
from tacotoolbox.tortilla.datamodel import Tortilla

if TYPE_CHECKING:
    from tacotoolbox.sample.validators._base import SampleValidator

# Asset types
AssetType = Literal["FILE", "FOLDER"]

# Key validation pattern
VALID_KEY_PATTERN = re.compile(r"^[a-zA-Z0-9_]+(?:[:][\w]+)?$")


class SampleExtension(ABC, pydantic.BaseModel):
    """Abstract base class for Sample extensions that compute metadata."""

    schema_only: bool = pydantic.Field(
        False,
        description="If True, return None values while preserving schema",
        validation_alias="return_none",
    )

    @abstractmethod
    def get_schema(self) -> pa.Schema:
        """Return the expected Arrow schema for this extension."""
        pass

    @abstractmethod
    def get_field_descriptions(self) -> dict[str, str]:
        """Return field descriptions for each field."""
        pass

    @abstractmethod
    def _compute(self, sample: "Sample") -> pa.Table:
        """
        Actual computation logic.

        Returns:
            PyArrow Table with single row containing computed metadata.
        """
        pass

    def __call__(self, sample: "Sample") -> pa.Table:
        """Process Sample and return computed metadata as single-row Arrow Table."""
        if self.schema_only:
            pa_schema = self.get_schema()
            none_data = {col_name: [None] for col_name in pa_schema.names}
            return pa.Table.from_pydict(none_data, schema=pa_schema)

        return self._compute(sample)


class Sample(pydantic.BaseModel):
    """
    The fundamental data unit in the TACO framework, combining raw data with
    structured metadata for training, validation, and testing.

    Supported data asset types:
    - FILE: Any file-based format (e.g., GeoTIFF, NetCDF, HDF5, PDF, CSV, Zarr)
    - FOLDER: A nested collection of samples (Tortilla)

    Type inference:
    By default, type is automatically inferred from path:
    - Path or bytes → FILE
    - Tortilla → FOLDER

    You can explicitly specify type for validation, but it's optional.

    Format-specific validation is handled by validators (applied via validate_with):
    - Use TacotiffValidator for TACOTIFF format validation
    - Use TacozarrValidator for TACOZARR format validation (future)
    - Use TacogeoparquetValidator for TACOGEOPARQUET format validation (future)

    Bytes Support:
    When passing bytes as path, temporary files are created in the system temp
    directory. For large datasets with many samples, the temp directory may fill
    up quickly. Use temp_dir parameter to specify a directory with adequate space.

    Temporary files are automatically cleaned up when the Sample is
    garbage collected or when cleanup() is called explicitly.

    Private Attributes:
    - _size_bytes: Total size in bytes (file size or folder sum), auto-calculated
    - _temp_files: List of temp file paths for cleanup
    - _extension_schemas: Arrow schemas for extensions
    - _field_descriptions: Field descriptions for documentation
    """

    # Core attributes
    id: str  # Unique identifier following TACO naming conventions
    path: (
        pathlib.Path | Tortilla | bytes
    )  # Location of data (file, container, or bytes)
    type: Literal["FILE", "FOLDER", "auto"] = (
        "auto"  # Type of geospatial data asset (auto-inferred by default)
    )

    # Private attributes
    _size_bytes: int = pydantic.PrivateAttr(default=0)
    _temp_files: list[pathlib.Path] = pydantic.PrivateAttr(default_factory=list)
    _extension_schemas: dict[str, pa.DataType] = pydantic.PrivateAttr(
        default_factory=dict
    )
    _field_descriptions: dict[str, str] = pydantic.PrivateAttr(default_factory=dict)

    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True,  # Support Tortilla dataclass
        extra="allow",  # Allow dynamic fields from extensions
    )

    def __init__(self, temp_dir: pathlib.Path | None = None, **data: Any) -> None:
        """Initialize Sample with optional temp_dir for bytes conversion."""
        # Handle temp_dir for bytes conversion without storing it
        if "path" in data and isinstance(data["path"], bytes):
            temp_dir = (
                pathlib.Path(tempfile.gettempdir())
                if temp_dir is None
                else pathlib.Path(temp_dir)
            )

            # Create temp directory if it doesn't exist
            temp_dir.mkdir(parents=True, exist_ok=True)

            # Generate UUID-based filename
            temp_filename = uuid.uuid4().hex
            temp_path = temp_dir / temp_filename

            # Write bytes to temp file (even if empty - needed for padding samples)
            # Empty bytes create 0-byte files that can be copied to ZIP/FOLDER
            with open(temp_path, "wb") as f:
                f.write(data["path"])

            # Replace bytes with path
            data["path"] = temp_path.absolute()

            # Initialize _temp_files BEFORE super().__init__
            # This ensures it's available when pydantic validates
            object.__setattr__(self, "_temp_files", [temp_path])

        # Extract extension fields (anything not a core field)
        extension_fields = {
            k: v for k, v in data.items() if k not in SHARED_CORE_FIELDS
        }

        # Initialize with all fields (Pydantic accepts them due to extra="allow")
        super().__init__(**data)

        # Calculate size AFTER validation (when type is FILE/FOLDER, not "auto")
        object.__setattr__(self, "_size_bytes", self._calculate_size())

        # Auto-extend with extension fields to track their schemas
        if extension_fields:
            self.extend_with(extension_fields)

    @classmethod
    def _create_padding(
        cls, index: int, temp_dir: pathlib.Path | None = None
    ) -> "Sample":
        """
        Internal factory for creating padding samples.
        Bypasses ID validation for __TACOPAD__ prefix.

        Padding samples use empty bytes (b"") which creates a 0-byte temporary file.
        This file can be copied to ZIP/FOLDER containers like any other file.
        """
        # Create temp directory
        temp_dir = (
            pathlib.Path(tempfile.gettempdir())
            if temp_dir is None
            else pathlib.Path(temp_dir)
        )
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Generate UUID-based filename for 0-byte file
        temp_filename = uuid.uuid4().hex
        temp_path = temp_dir / temp_filename

        # Write empty bytes to create 0-byte temp file
        with open(temp_path, "wb") as f:
            f.write(b"")

        # Use model_construct to bypass validators
        sample = cls.model_construct(
            id=f"__TACOPAD__{index}", type="FILE", path=temp_path.absolute()
        )

        # Manually initialize private attributes (model_construct doesn't do this)
        object.__setattr__(sample, "_temp_files", [temp_path])
        object.__setattr__(sample, "_extension_schemas", {})
        object.__setattr__(sample, "_field_descriptions", {})
        object.__setattr__(sample, "_size_bytes", 0)  # 0-byte file

        return sample

    def _calculate_size(self) -> int:
        """
        Calculate total size in bytes.

        For FILE: returns file size from filesystem or bytes length
        For FOLDER: reads _size_bytes from Tortilla (already calculated)
        """
        if self.type == "FILE":
            if isinstance(self.path, pathlib.Path) and self.path.exists():
                return self.path.stat().st_size
            elif isinstance(self.path, bytes):
                return len(self.path)
            return 0
        elif self.type == "FOLDER":
            # Read from Tortilla (already calculated)
            from typing import cast

            tortilla = cast(Tortilla, self.path)
            return tortilla._size_bytes
        return 0

    def cleanup(self) -> None:
        """
        Clean up temporary files created from bytes.

        Call explicitly to clean up immediately instead of waiting for garbage collection.
        """
        if not self._temp_files:
            return

        for temp_file in self._temp_files:
            with contextlib.suppress(Exception):
                if temp_file.exists():
                    temp_file.unlink()

        self._temp_files.clear()

    def __del__(self) -> None:
        """Cleanup on garbage collection."""
        with contextlib.suppress(Exception):
            self.cleanup()

    @pydantic.field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        r"""
        Validate sample ID format.

        Rules:
        - NO slashes (/, \) - breaks file paths in ZIP and FOLDER containers
        - NO colons (:) - invalid on Windows, conflicts with extension namespaces
        - NO double underscore prefix (__) - reserved for __TACOPAD__ system
        """
        # Check for slashes
        if "/" in v or "\\" in v:
            raise ValueError(
                f"Sample ID cannot contain slashes: '{v}'\n"
                f"Slashes break file paths in ZIP and FOLDER containers."
            )

        # Check for colons
        if ":" in v:
            raise ValueError(
                f"Sample ID cannot contain colons: '{v}'\n"
                f"Colons are invalid in Windows filenames and conflict with namespace syntax."
            )

        # Check for double underscore prefix (reserved for internal use)
        if v.startswith("__"):
            raise ValueError(
                f"Sample ID cannot start with '__': '{v}'\n"
                f"Double underscore prefix is reserved for internal use (__TACOPAD__)."
            )

        return v

    @pydantic.field_validator("path")
    @classmethod
    def validate_path(
        cls, v: pathlib.Path | Tortilla | bytes
    ) -> pathlib.Path | Tortilla:
        """Validate and normalize the data path."""
        if isinstance(v, Tortilla):
            return v

        if isinstance(v, pathlib.Path):
            if not v.exists():
                raise ValueError(f"Path {v} does not exist.")
            return v.absolute()

        raise ValueError(
            "Path must be pathlib.Path or Tortilla (bytes handled in __init__)"
        )

    @pydantic.model_validator(mode="after")
    def infer_and_validate_type(self) -> "Sample":
        """
        Infer type from path or validate explicit type.

        If type="auto" (default), infers type from path:
        - Path or bytes → FILE
        - Tortilla → FOLDER

        If type is explicit ("FILE" or "FOLDER"), validates it matches path type.

        After this validator, self.type is ALWAYS "FILE" or "FOLDER" (never "auto").
        """
        # Infer type from path
        inferred_type: AssetType = (
            "FOLDER" if isinstance(self.path, Tortilla) else "FILE"
        )

        if self.type == "auto":
            # Auto mode - use inferred type
            self.type = inferred_type
        else:
            # Explicit mode - validate consistency
            if self.type != inferred_type:
                path_type_str = (
                    "Tortilla" if isinstance(self.path, Tortilla) else "Path/bytes"
                )
                raise ValueError(
                    f"Type mismatch: specified type='{self.type}' but path type ({path_type_str}) "
                    f"implies type='{inferred_type}'"
                )

        return self

    def validate_with(self, validator: "SampleValidator") -> None:
        """
        Validate sample using provided validator.

        Validators enforce format-specific requirements (e.g., TACOTIFF, TACOZARR).
        """
        validator.validate(self)

    def extend_with(
        self, extension: pa.Table | dict[str, Any] | Any, name: str | None = None
    ) -> None:
        """Add extension to sample by adding fields directly to the model."""
        if isinstance(extension, pa.Table):
            self._handle_arrow_table_extension(extension)
        elif callable(extension) and hasattr(extension, "model_dump"):
            self._handle_sample_extension(extension)
        elif isinstance(extension, dict):
            self._handle_dict_extension(extension)
        else:
            self._handle_pydantic_extension(extension, name)

        return None

    def pop(self, field: str) -> pa.Table:
        """
        Remove and return a metadata field as a single-row Arrow Table.

        Args:
            field: Name of the extension field to pop (e.g., "split", "stac:crs")

        Returns:
            Single-row Arrow Table with the field value and proper schema

        Raises:
            ValueError: If field is a core field
            KeyError: If field doesn't exist
        """
        # Validate field is not core
        if field in SHARED_CORE_FIELDS:
            raise ValueError(
                f"Cannot pop core field: '{field}'. "
                f"Core fields are: {', '.join(SHARED_CORE_FIELDS)}"
            )

        # Check field exists
        if not hasattr(self, field):
            raise KeyError(
                f"Field '{field}' does not exist in sample. "
                f"Available extension fields: {list(self._extension_schemas.keys())}"
            )

        # Get value
        value = getattr(self, field)

        # Create Arrow Table with proper schema
        if field in self._extension_schemas:
            # Use tracked schema
            arrow_schema = pa.schema([pa.field(field, self._extension_schemas[field])])
            arrow_table = pa.Table.from_pydict({field: [value]}, schema=arrow_schema)
        else:
            # Let PyArrow infer type
            arrow_table = pa.Table.from_pydict({field: [value]})

        # Remove from sample
        delattr(self, field)

        # Remove from schema tracking
        if field in self._extension_schemas:
            del self._extension_schemas[field]

        # Remove from descriptions if present
        if field in self._field_descriptions:
            del self._field_descriptions[field]

        return arrow_table

    def _handle_arrow_table_extension(self, arrow_table: pa.Table) -> None:
        """Handle direct Arrow Table extension."""
        if arrow_table.num_rows != 1:
            raise ValueError("Arrow Table extension must have exactly one row")

        # Capture Arrow schemas directly
        for field in arrow_table.schema:
            self._extension_schemas[field.name] = field.type

        # Convert to dict and add fields
        metadata_dict = arrow_table.to_pydict()
        metadata_dict = {k: v[0] for k, v in metadata_dict.items()}

        self._add_metadata_fields(metadata_dict)

    def _handle_sample_extension(self, extension: Any) -> None:
        """Handle SampleExtension."""
        computed_metadata = extension(self)

        if not isinstance(computed_metadata, pa.Table):
            raise TypeError(
                f"SampleExtension must return pa.Table, got {type(computed_metadata)}"
            )

        # Convert single-row Table to dict
        if computed_metadata.num_rows != 1:
            raise ValueError("SampleExtension must return single-row Table")

        # Capture Arrow schemas directly
        for field in computed_metadata.schema:
            self._extension_schemas[field.name] = field.type

        # Capture field descriptions if extension provides them
        if hasattr(extension, "get_field_descriptions"):
            descriptions = extension.get_field_descriptions()
            self._field_descriptions.update(descriptions)

        # Convert to dict
        metadata_dict = computed_metadata.to_pydict()
        # Extract values (each column is list of 1 element)
        metadata_dict = {k: v[0] for k, v in metadata_dict.items()}

        self._add_metadata_fields(metadata_dict)

    def _handle_dict_extension(self, extension: dict[str, Any]) -> None:
        """Handle dictionary extension."""
        # Create Arrow Table and let PyArrow infer types automatically
        arrow_table = pa.Table.from_pydict({k: [v] for k, v in extension.items()})

        # Capture Arrow schemas directly from inferred table
        for field in arrow_table.schema:
            self._extension_schemas[field.name] = field.type

        # Then add the fields
        self._add_metadata_fields(extension)

    def _handle_pydantic_extension(self, extension: Any, name: str | None) -> None:
        """Handle Pydantic model extension."""
        namespace = name if name else extension.__class__.__name__.lower()
        if hasattr(extension, "model_dump"):
            extension_data = extension.model_dump()
            namespaced_data = {}
            for key, value in extension_data.items():
                namespaced_key = f"{namespace}:{key}"
                namespaced_data[namespaced_key] = value
            self._add_metadata_fields(namespaced_data)
        else:
            raise ValueError(f"Invalid extension type: {type(extension)}")

    def _add_metadata_fields(self, metadata_dict: dict[str, Any]) -> None:
        """Add metadata fields to the sample with validation."""
        for key, value in metadata_dict.items():
            self._validate_key(key)
            if key in SHARED_CORE_FIELDS:
                raise ValueError(f"Cannot override core field: {key}")
            setattr(self, key, value)

    def _validate_key(self, key: str) -> None:
        """Validate key format."""
        if not VALID_KEY_PATTERN.match(key):
            raise ValueError(
                f"Invalid key format '{key}'. Use alphanumeric + underscore, "
                f"optionally with colon (e.g., 'key', 'my_key', 'stac:title')"
            )

    def export_metadata(self) -> pa.Table:
        """
        Export complete Sample metadata as single-row Arrow Table.

        Core fields (id, type, path) always use String type for schema consistency,
        even when path is None (FOLDER samples). This ensures all samples have
        compatible schemas for concatenation.

        The 'type' column always contains "FILE" or "FOLDER" (never "auto")
        because type inference happens during validation.
        """
        data = self.model_dump()

        # Handle path serialization
        if isinstance(self.path, pathlib.Path):
            data["path"] = self.path.as_posix()
        elif isinstance(self.path, Tortilla):
            data["path"] = None

        # Build Arrow schema (core + extensions)
        arrow_fields = [
            pa.field("id", pa.string()),
            pa.field("type", pa.string()),
            pa.field("path", pa.string()),
        ]

        # Add extension fields (already Arrow types)
        for field_name, arrow_dtype in self._extension_schemas.items():
            arrow_fields.append(pa.field(field_name, arrow_dtype))

        arrow_schema = pa.schema(arrow_fields)

        # Create Arrow Table
        return pa.Table.from_pydict(
            {k: [v] for k, v in data.items()}, schema=arrow_schema
        )
