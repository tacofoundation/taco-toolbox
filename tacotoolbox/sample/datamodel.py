import pathlib
import re
import tempfile
import uuid
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

import polars as pl
import pydantic

# Dependencies
from tacotoolbox.tortilla.datamodel import Tortilla

if TYPE_CHECKING:
    from tacotoolbox.sample.validators._base import SampleValidator

# Asset types
AssetType = Literal["FILE", "FOLDER"]

# Key validation pattern
VALID_KEY_PATTERN = re.compile(r"^[a-zA-Z0-9_]+(?:[:][\w]+)?$")

# Core fields that cannot be overwritten by extensions
PROTECTED_CORE_FIELDS = {"id", "type", "path"}


class SampleExtension(ABC, pydantic.BaseModel):
    """Abstract base class for Sample extensions that compute metadata."""

    return_none: bool = pydantic.Field(
        False, description="If True, return None values while preserving schema"
    )

    @abstractmethod
    def get_schema(self) -> dict[str, pl.DataType]:
        """Return the expected schema for this extension."""
        pass

    @abstractmethod
    def _compute(self, sample: "Sample") -> pl.DataFrame:
        """Actual computation logic - only called when return_none=False."""
        pass

    def __call__(self, sample: "Sample") -> pl.DataFrame:
        """
        Process Sample and return computed metadata.

        Args:
            sample: Input Sample object

        Returns:
            pl.DataFrame: Single-row DataFrame with computed metadata
        """
        # Check return_none FIRST for performance
        if self.return_none:
            schema = self.get_schema()
            none_data = {col_name: [None] for col_name in schema}
            return pl.DataFrame(none_data, schema=schema)

        # Only do actual computation if needed
        return self._compute(sample)


class Sample(pydantic.BaseModel):
    """
    The fundamental data unit in the TACO framework, combining raw data with
    structured metadata for training, validation, and testing.

    Supported data asset types:
    - FILE: Any file-based format (e.g., GeoTIFF, NetCDF, HDF5, PDF, CSV, Zarr)
    - FOLDER: A nested collection of samples (Tortilla)

    Format-specific validation is handled by validators (applied via validate_with):
    - Use TacotiffValidator for TACOTIFF format validation
    - Use TacozarrValidator for TACOZARR format validation (future)
    - Use TacogeoparquetValidator for TACOGEOPARQUET format validation (future)

    Bytes Support:
    When passing bytes as path, temporary files are created in the system temp
    directory. For large datasets with many samples, the temp directory may fill
    up quickly. Use temp_dir parameter to specify a directory with adequate space.

    Example:
        >>> from tacotoolbox import Sample
        >>> from tacotoolbox.sample.validators import TacotiffValidator
        >>>
        >>> # Basic sample
        >>> sample = Sample(
        ...     id="soyuntaco",
        ...     path=Path("/home/lxlx/sentinel2.tif"),
        ...     type="FILE"
        ... )
        >>>
        >>> # Validate with TACOTIFF format
        >>> sample.validate_with(TacotiffValidator())
        >>>
        >>> # Bytes support
        >>> sample = Sample(
        ...     id="bytesample",
        ...     path=image_bytes,
        ...     type="FILE",
        ...     temp_dir="/data/workspace"
        ... )
        >>>
        >>> # Extensions
        >>> sample.extend_with(stac_obj)
        >>> sample.extend_with({"s2:mgrs_tile": "T30UYA"})
        >>> sample.extend_with(scaling_extension)
        >>>
        >>> # Nested samples (FOLDER)
        >>> nested = Sample(
        ...     id="multitemporal",
        ...     path=Tortilla(samples=[s1, s2, s3]),
        ...     type="FOLDER"
        ... )
    """

    # Core attributes
    id: str  # Unique identifier following TACO naming conventions
    path: (
        pathlib.Path | Tortilla | bytes
    )  # Location of data (file, container, or bytes)
    type: AssetType  # Type of geospatial data asset

    # Private attribute to store extension schemas
    _extension_schemas: dict[str, pl.DataType] = pydantic.PrivateAttr(
        default_factory=dict
    )

    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True,  # Support Tortilla dataclass
        extra="allow",  # Allow dynamic fields from extensions
    )

    def __init__(self, temp_dir: pathlib.Path | None = None, **data):
        """Initialize Sample with optional temp_dir that doesn't get stored."""
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

            # Write bytes to temp file
            with open(temp_path, "wb") as f:
                f.write(data["path"])

            # Replace bytes with path
            data["path"] = temp_path.absolute()

        # Extract extension fields (anything not a core field)
        core_fields = {"id", "path", "type"}
        extension_fields = {k: v for k, v in data.items() if k not in core_fields}

        # Initialize with all fields (Pydantic accepts them due to extra="allow")
        super().__init__(**data)

        # Auto-extend with extension fields to track their schemas
        if extension_fields:
            self.extend_with(extension_fields)

    @pydantic.field_validator("path")
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
    def global_validation(self):
        """Cross-field validation ensuring path type matches asset type."""
        # FOLDER type must have Tortilla path
        if self.type == "FOLDER" and not isinstance(self.path, Tortilla):
            raise ValueError("FOLDER type must have a Tortilla instance as path")

        # FILE type must have pathlib.Path (validated in validate_path)
        if self.type == "FILE" and isinstance(self.path, Tortilla):
            raise ValueError("FILE type must have a pathlib.Path, not Tortilla")

        return self

    def validate_with(self, validator: "SampleValidator") -> None:
        """
        Validate sample using provided validator.

        Validators enforce format-specific requirements (e.g., TACOTIFF, TACOZARR).
        This method allows applying strict format validation after sample creation.

        Args:
            validator: SampleValidator instance (e.g., TacotiffValidator())

        Raises:
            ValidationError: If validation fails

        Example:
            >>> from tacotoolbox.sample.validators import TacotiffValidator
            >>>
            >>> sample = Sample(id="s2", path=Path("data.tif"), type="FILE")
            >>> sample.validate_with(TacotiffValidator())  # Validates TACOTIFF format
        """
        validator.validate(self)

    def extend_with(
        self, extension: Any | dict[str, Any], name: str | None = None
    ) -> None:
        """
        Add extension to sample by adding fields directly to the model.

        Args:
            extension: SampleExtension, Pydantic model, or dictionary to add
            name: Optional custom namespace (defaults to class name for objects)

        Returns:
            Sample: Self for method chaining
        """
        # Check if this is a computational SampleExtension
        if callable(extension) and hasattr(extension, "model_dump"):
            self._handle_sample_extension(extension)
        elif isinstance(extension, pl.DataFrame):
            self._handle_dataframe_extension(extension)
        elif isinstance(extension, dict):
            self._handle_dict_extension(extension)
        else:
            self._handle_pydantic_extension(extension, name)

        return None

    def _handle_sample_extension(self, extension) -> None:
        """Handle SampleExtension (callable with model_dump)."""
        computed_metadata = extension(self)
        if isinstance(computed_metadata, pl.DataFrame):
            # Convert single-row DataFrame to dict
            if len(computed_metadata) != 1:
                raise ValueError("SampleExtension must return single-row DataFrame")

            # Capture schemas before converting to dict
            for col_name, dtype in computed_metadata.schema.items():
                self._extension_schemas[col_name] = dtype

            metadata_dict = computed_metadata.to_dicts()[0]
            self._add_metadata_fields(metadata_dict)

    def _handle_dataframe_extension(self, extension: pl.DataFrame) -> None:
        """Handle direct DataFrame extension."""
        if len(extension) != 1:
            raise ValueError("DataFrame extension must have exactly one row")

        # Capture schemas before converting to dict
        for col_name, dtype in extension.schema.items():
            self._extension_schemas[col_name] = dtype

        metadata_dict = extension.to_dicts()[0]
        self._add_metadata_fields(metadata_dict)

    def _handle_dict_extension(self, extension: dict) -> None:
        """Handle dictionary extension."""
        # First, infer schemas from dict values and update _extension_schemas
        for key, value in extension.items():
            # Infer Polars dtype from Python value
            self._extension_schemas[key] = self._infer_polars_dtype(value)

        # Then add the fields
        self._add_metadata_fields(extension)

    def _handle_pydantic_extension(self, extension, name: str | None) -> None:
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
            raise ValueError(
                f"Extension must be pydantic model or dict, got: {type(extension)}"
            )

    def _add_metadata_fields(self, metadata_dict: dict) -> None:
        """Add metadata fields to the sample with validation."""
        for key, value in metadata_dict.items():
            self._validate_key(key)
            if key in PROTECTED_CORE_FIELDS:
                raise ValueError(f"Cannot override core field: {key}")
            setattr(self, key, value)

    def _validate_key(self, key: str) -> None:
        """Validate key format."""
        if not VALID_KEY_PATTERN.match(key):
            raise ValueError(
                f"Invalid key format '{key}'. Use alphanumeric + underscore, "
                f"optionally with colon (e.g., 'key', 'my_key', 'stac:title')"
            )

    def _infer_polars_dtype(self, value: Any) -> pl.DataType:
        """Infer Polars DataType from Python value."""
        if value is None:
            return pl.Utf8()  # Default to String for None
        elif isinstance(value, str):
            return pl.Utf8()
        elif isinstance(value, bool):
            return pl.Boolean()
        elif isinstance(value, int):
            return pl.Int64()
        elif isinstance(value, float):
            return pl.Float64()
        elif isinstance(value, list):
            if not value:
                return pl.List(pl.Utf8())  # Default list type
            # Infer from first element
            inner_type = self._infer_polars_dtype(value[0])
            return pl.List(inner_type)
        elif isinstance(value, dict):
            return pl.Struct(
                [pl.Field(k, self._infer_polars_dtype(v)) for k, v in value.items()]
            )
        else:
            warnings.warn(
                f"Could not infer Polars dtype for value: {value}. Defaulting to String.",
                stacklevel=2,
            )
            return pl.Utf8()  # Fallback to string

    def export_metadata(self) -> pl.DataFrame:
        """
        Export complete Sample metadata as a single-row DataFrame with proper schemas.

        Returns all fields in the model, including core attributes and
        extension metadata with proper data types preserved.

        Core fields (id, type, path) always use String type for schema consistency,
        even when path is None (FOLDER samples). This ensures all samples have
        compatible schemas for concatenation.

        Returns:
            pl.DataFrame: Single-row DataFrame with complete sample metadata
        """
        data = self.model_dump()

        # Handle path serialization
        if isinstance(self.path, pathlib.Path):
            data["path"] = self.path.as_posix()
        elif isinstance(self.path, Tortilla):
            data["path"] = None  # None value is OK, but type will be String

        # Define core schema - ensures schema consistency across all samples
        # path is String even for FOLDERs (value=None, but type=String)
        core_schema: dict[str, pl.DataType] = {
            "id": pl.String(),
            "type": pl.String(),
            "path": pl.String(),
        }

        # Build complete schema: core + extensions
        complete_schema = {**core_schema, **self._extension_schemas}

        # Create DataFrame with explicit schema
        df = pl.DataFrame([data], schema=complete_schema)

        return df
