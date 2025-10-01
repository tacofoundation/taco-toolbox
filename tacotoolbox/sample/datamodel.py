import functools
import pathlib
import re
import tempfile
import uuid
from abc import ABC, abstractmethod
from typing import Any, Literal

import polars as pl
import pydantic

# Dependencies
from tacotoolbox.tortilla.datamodel import Tortilla

# Asset types
AssetType = Literal["TACOTIFF", "TACOGEOPARQUET", "TORTILLA", "OTHER"]

# Key validation pattern
VALID_KEY_PATTERN = re.compile(r"^[a-zA-Z0-9_]+(?:[:][\w]+)?$")

# Core fields that cannot be overwritten by extensions
PROTECTED_CORE_FIELDS = {"id", "type", "path"}


class SampleExtension(ABC, pydantic.BaseModel):
    """Abstract base class for Sample extensions that compute metadata."""

    return_none: bool = pydantic.Field(False, description="If True, return None values while preserving schema")

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


def requires_gdal(min_version="3.11"):
    """Decorator to ensure GDAL is available with minimum version."""

    def decorator(func):
        # Cache the check result to make subsequent calls fast
        _gdal_checked = False
        _gdal_module = None

        def _raise_gdal_version_error(current_version, min_version):
            """Raise ImportError for GDAL version mismatch."""
            raise ImportError(f"GDAL {min_version}+ required. Current: {current_version}")

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal _gdal_checked, _gdal_module

            if not _gdal_checked:
                try:
                    from osgeo import gdal # type: ignore[import-untyped]

                    _gdal_module = gdal

                    # Simple version comparison using tuple comparison
                    current = tuple(map(int, gdal.__version__.split(".")[:2]))
                    required = tuple(map(int, min_version.split(".")[:2]))

                    if current < required:
                        _raise_gdal_version_error(gdal.__version__, min_version)

                except ImportError as e:
                    if "GDAL" not in str(e):
                        raise ImportError(
                            f"GDAL {min_version}+ required for TACOTIFF validation. "
                            f"Install: conda install gdal>={min_version}"
                        ) from e
                    raise

                _gdal_checked = True

            return func(*args, **kwargs)

        return wrapper

    return decorator


class TacotiffValidator:
    """
    Validator for TACOTIFF sample using GDAL to enforce strict format requirements.

    TACOTIFF format requirements:
    - Driver: GDAL generated COG (Cloud Optimized GeoTIFF)
    - Compression: JXL (next-gen JPEG with the best performance and quality)
    - Interleave: TILE (for efficient access patterns)
    - Overviews: None (to avoid redundant data storage)
    - BIGTIFF: YES (to standardize between large and small files)
    - GEOTIFF version: 1.1 (for standard compliance)
    """

    @requires_gdal(min_version="3.11")
    def validate(self, path: pathlib.Path) -> None:
        """
        Validate a TACOTIFF file against format requirements.

        Example:
            >>> validator = TacotiffValidator()
            >>> validator.validate(Path("my_file.tif"))  # Raises ValueError if invalid
        """
        from osgeo import gdal

        # Open the dataset using GDAL
        ds = gdal.Open(str(path))

        # Check if GDAL can open the file
        if not ds:
            raise ValueError(f"Cannot open {path} with GDAL")

        try:
            # Get image structure metadata from GDAL
            # This contains compression, interleave, and other format info
            ds_args = ds.GetMetadata("IMAGE_STRUCTURE")

            # Validate ZSTD compression (5000)
            compression = ds_args.get("COMPRESSION", "").upper()
            if compression != "JXL":
                raise ValueError(f"TACOTIFF assets must use JXL compression, found: {compression or 'NONE'}")

            # Validate TILE interleave
            interleave = ds_args.get("INTERLEAVE", "").upper()
            if interleave != "TILE":
                raise ValueError(f"TACOTIFF assets must use TILE interleave, found: {interleave or 'PIXEL'}")

            # Validate no overviews present
            band = ds.GetRasterBand(1)
            overview_count = band.GetOverviewCount()
            if overview_count != 0:
                raise ValueError(f"TACOTIFF assets must not have overviews, found: {overview_count} overview levels")

        finally:
            # Always clean up GDAL dataset to free memory
            ds = None


class Sample(pydantic.BaseModel):
    """
    The fundamental data unit in the TACO framework, combining raw data with
    structured metadata for training, validation, and testing.

    Supported data asset types:
    - TACOTIFF: Cloud Optimized GeoTIFF with strict format requirements
    - TACOGEOPARQUET: GeoParquet format with strict format requirements
    - TORTILLA: A set of samples with similar characteristics
    - OTHER: Other file-based formats (e.g., TIFF, NetCDF, HDF5, PDF, CSV)

    Bytes Support:
    When passing bytes as path, temporary files are created in the system temp
    directory. For large datasets with many samples, the temp directory may fill
    up quickly. Use temp_dir parameter to specify a directory with adequate space.

    Example:
        >>> sample = Sample(
        ...     id="soyuntaco",
        ...     path=Path("/home/lxlx/sentinel2.tif"),
        ...     type="TACOTIFF"
        ... )
        >>> sample = Sample(
        ...     id="bytesample",
        ...     path=image_bytes,
        ...     type="TACOTIFF",
        ...     temp_dir="/data/workspace"
        ... )
        >>> sample.extend_with(stac_obj)
        >>> sample.extend_with({"s2:mgrs_tile": "T30UYA"})
        >>> sample.extend_with(scaling_extension)
    """

    # Core attributes
    id: str  # Unique identifier following TACO naming conventions
    path: pathlib.Path | Tortilla | bytes  # Location of data (file, container, or bytes)
    type: AssetType  # Type of geospatial data asset

    # Private attribute to store extension schemas
    _extension_schemas: dict[str, pl.DataType] = pydantic.PrivateAttr(default_factory=dict)

    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True,  # Support Tortilla dataclass
        extra="allow",  # Allow dynamic fields from extensions
    )

    def __init__(self, temp_dir: pathlib.Path | None = None, **data):
        """Initialize Sample with optional temp_dir that doesn't get stored."""
        # Handle temp_dir for bytes conversion without storing it
        if "path" in data and isinstance(data["path"], bytes):
            temp_dir = pathlib.Path(tempfile.gettempdir()) if temp_dir is None else pathlib.Path(temp_dir)

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

        super().__init__(**data)

    @pydantic.field_validator("path")
    def validate_path(cls, v: pathlib.Path | Tortilla | bytes) -> pathlib.Path | Tortilla:
        """Validate and normalize the data path."""
        if isinstance(v, Tortilla):
            return v

        if isinstance(v, pathlib.Path):
            if not v.exists():
                raise ValueError(f"Path {v} does not exist.")
            return v.absolute()

        raise ValueError("Path must be pathlib.Path or Tortilla (bytes handled in __init__)")

    @pydantic.model_validator(mode="after")
    def global_validation(self):
        """Cross-field validation ensuring path type matches asset type."""
        # TORTILLA type must have Tortilla path
        if self.type == "TORTILLA" and not isinstance(self.path, Tortilla):
            raise ValueError("TORTILLA type must have a Tortilla instance as path")

        # TACOTIFF specific validations
        if self.type == "TACOTIFF":
            TacotiffValidator().validate(self.path)

        return self

    def extend_with(self, extension: Any | dict[str, Any], name: str | None = None) -> None:
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
            raise ValueError(f"Extension must be pydantic model or dict, got: {type(extension)}")

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

    def export_metadata(self) -> pl.DataFrame:
        """
        Export complete Sample metadata as a single-row DataFrame with proper schemas.

        Returns all fields in the model, including core attributes and
        extension metadata with proper data types preserved.

        Returns:
            pl.DataFrame: Single-row DataFrame with complete sample metadata
        """
        data = self.model_dump()

        # Handle path serialization
        if isinstance(self.path, pathlib.Path):
            data["path"] = self.path.as_posix()
        elif isinstance(self.path, Tortilla):
            data["path"] = None

        # Create initial DataFrame
        df = pl.DataFrame([data])

        # Apply saved schemas
        cast_exprs = []
        for col_name, dtype in self._extension_schemas.items():
            if col_name in df.columns:
                cast_exprs.append(pl.col(col_name).cast(dtype))

        if cast_exprs:
            df = df.with_columns(cast_exprs)

        return df
