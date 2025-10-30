"""
TACO Dataset Metadata Model.

Provides the core TACO dataset descriptor with extensible metadata support.
Combines Tortilla (hierarchical structure) with standardized dataset metadata
and a flexible extension system for domain-specific enrichment.

Main components:
- Taco: Dataset metadata container
- TacoExtension: Base for computed metadata extensions
- Contact: Contributor information
- Extent: Spatial/temporal boundaries
- TaskType: ML task categories
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Literal

import polars as pl
import pydantic

from tacotoolbox.tortilla.datamodel import Tortilla


class TacoExtension(ABC, pydantic.BaseModel):
    """Base class for TACO metadata extensions."""

    @abstractmethod
    def get_schema(self) -> dict[str, pl.DataType]:
        """Return column names and Polars DataTypes for this extension."""
        pass

    @abstractmethod
    def _compute(self, taco: "Taco") -> pl.DataFrame:
        """Compute extension metadata, return single-row DataFrame."""
        pass

    def __call__(self, taco: "Taco") -> pl.DataFrame:
        """Execute extension computation."""
        return self._compute(taco)


# ML task types for dataset categorization
TaskType = Literal[
    "regression",
    "classification",
    "scene-classification",
    "detection",
    "object-detection",
    "segmentation",
    "semantic-segmentation",
    "instance-segmentation",
    "panoptic-segmentation",
    "similarity-search",
    "generative",
    "image-captioning",
    "super-resolution",
    "denoising",
    "inpainting",
    "colorization",
    "style-transfer",
    "deblurring",
    "dehazing",
    "foundation-model",
    "other",
]


class Contact(TacoExtension):
    """
    Dataset contributor contact information.

    At least one of 'name' or 'organization' required.
    Role examples: "principal-investigator", "data-curator", "maintainer"
    """

    name: str | None = None
    organization: str | None = None
    email: str | None = None
    role: str | None = None

    @pydantic.model_validator(mode="after")
    def check_name_or_organization(self):
        if not self.name and not self.organization:
            raise ValueError("Either 'name' or 'organization' must be provided")
        return self

    @pydantic.field_validator("email")
    @classmethod
    def validate_email(cls, v: str | None) -> str | None:
        if v and "@" not in v:
            raise ValueError("Invalid email format - must contain '@' symbol")
        return v

    def get_schema(self) -> dict[str, pl.DataType]:
        return {
            "name": pl.Utf8(),
            "organization": pl.Utf8(),
            "email": pl.Utf8(),
            "role": pl.Utf8(),
        }

    def _compute(self, taco: "Taco") -> pl.DataFrame:
        return pl.DataFrame([self.model_dump()])


class Extent(TacoExtension):
    """
    Spatial and temporal boundaries.

    Spatial: [min_lon, min_lat, max_lon, max_lat] in WGS84 decimal degrees
    Temporal: [start_iso, end_iso] in ISO 8601 format (optional)
    """

    spatial: list[float]  # [min_lon, min_lat, max_lon, max_lat]
    temporal: list[str] | None = None  # [start_iso, end_iso]

    @pydantic.field_validator("spatial")
    @classmethod
    def validate_spatial(cls, v: list[float]) -> list[float]:
        if len(v) != 4:
            raise ValueError(
                "Spatial extent must have exactly 4 values: [min_lon, min_lat, max_lon, max_lat]"
            )

        min_lon, min_lat, max_lon, max_lat = v

        if not (-180 <= min_lon <= 180) or not (-180 <= max_lon <= 180):
            raise ValueError("Longitude values must be between -180 and 180 degrees")

        if not (-90 <= min_lat <= 90) or not (-90 <= max_lat <= 90):
            raise ValueError("Latitude values must be between -90 and 90 degrees")

        if min_lon >= max_lon:
            raise ValueError("min_longitude must be less than max_longitude")

        if min_lat >= max_lat:
            raise ValueError("min_latitude must be less than max_latitude")

        return v

    @pydantic.field_validator("temporal")
    @classmethod
    def validate_temporal(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v

        if len(v) != 2:
            raise ValueError(
                "Temporal extent must have exactly 2 values: [start_datetime, end_datetime]"
            )

        start_str, end_str = v

        try:
            # Parse ISO 8601 datetime strings
            if start_str.endswith("Z"):
                start_dt = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
            else:
                start_dt = datetime.fromisoformat(start_str)

            if end_str.endswith("Z"):
                end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
            else:
                end_dt = datetime.fromisoformat(end_str)

        except ValueError as e:
            raise ValueError(
                f"Invalid datetime format. Use ISO 8601 format (e.g., '2023-01-01T00:00:00Z'): {e}"
            ) from e

        if start_dt >= end_dt:
            raise ValueError("Start datetime must be before end datetime")

        return v

    def get_schema(self) -> dict[str, pl.DataType]:
        return {"spatial": pl.List(pl.Float64()), "temporal": pl.List(pl.Utf8())}

    def _compute(self, taco: "Taco") -> pl.DataFrame:
        return pl.DataFrame([self.model_dump()])


class Taco(pydantic.BaseModel):
    """
    Core TACO dataset metadata container.

    Combines:
    - Tortilla: Hierarchical sample structure
    - Core metadata: Required identification fields
    - Extensions: Dynamic metadata via extend_with()

    Required fields:
        tortilla: Hierarchical sample structure
        id: Unique identifier (lowercase, alphanumeric + _ -)
        dataset_version: Version string (e.g., "1.0.0")
        description: Dataset description
        licenses: License identifiers (e.g., ["CC-BY-4.0"])
        extent: Spatial/temporal boundaries
        providers: Dataset creators (Contact list)
        tasks: ML task types

    Optional fields:
        taco_version: TACO format version (default: "0.5.0")
        title: Human-friendly title (max 250 chars)
        curators: Dataset curators (Contact list)
        keywords: Searchable tags
    """

    # Required core fields
    tortilla: Tortilla
    id: str
    dataset_version: str
    description: str
    licenses: list[str]
    extent: Extent
    providers: list[Contact]
    tasks: list[TaskType]

    # Optional core fields
    taco_version: str = "0.5.0"
    title: str | None = None
    curators: list[Contact] | None = None
    keywords: list[str] | None = None

    model_config = pydantic.ConfigDict(extra="allow", arbitrary_types_allowed=True)

    @pydantic.field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        """Validate ID format: lowercase, alphanumeric + underscores/hyphens."""
        if not value.islower():
            raise ValueError("Dataset ID must be lowercase")

        if not value.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                "Dataset ID must be alphanumeric (underscores and hyphens allowed)"
            )

        return value

    @pydantic.field_validator("title")
    @classmethod
    def validate_title(cls, value: str | None) -> str | None:
        """Validate title length (max 250 characters)."""
        if value and len(value) > 250:
            raise ValueError("Title must be less than 250 characters")
        return value

    def extend_with(self, extension):
        """
        Add extension data to dataset.

        Supports:
        - TacoExtension instances (computed)
        - pl.DataFrame (single-row)
        - dict (key-value)
        - Pydantic models (via model_dump)
        """
        if callable(extension) and isinstance(extension, TacoExtension):
            self._handle_taco_extension(extension)
        elif isinstance(extension, pl.DataFrame):
            self._handle_dataframe_extension(extension)
        elif isinstance(extension, dict):
            self._handle_dict_extension(extension)
        else:
            self._handle_pydantic_extension(extension)

        return None

    def _handle_taco_extension(self, extension: TacoExtension) -> None:
        """Handle TacoExtension instance."""
        extension_df = extension(self)
        if not isinstance(extension_df, pl.DataFrame):
            raise TypeError("TacoExtension must return pl.DataFrame")
        if len(extension_df) != 1:
            raise ValueError("TacoExtension must return single-row DataFrame")

        extension_data = extension_df.to_dicts()[0]
        self._set_extension_attributes(extension_data)

    def _handle_dataframe_extension(self, extension: pl.DataFrame) -> None:
        """Handle DataFrame extension."""
        if len(extension) != 1:
            raise ValueError("DataFrame extension must have exactly one row")

        extension_data = extension.to_dicts()[0]
        self._set_extension_attributes(extension_data)

    def _handle_dict_extension(self, extension: dict) -> None:
        """Handle dictionary extension."""
        self._set_extension_attributes(extension)

    def _handle_pydantic_extension(self, extension) -> None:
        """Handle Pydantic model extension."""
        if hasattr(extension, "model_dump"):
            extension_data = extension.model_dump()
            self._set_extension_attributes(extension_data)
        else:
            raise ValueError(
                "Extension must be TacoExtension, DataFrame, dict, or pydantic model"
            )

    def _set_extension_attributes(self, extension_data: dict) -> None:
        """Set extension fields as instance attributes."""
        for key, value in extension_data.items():
            setattr(self, key, value)

    def export_metadata(self) -> pl.DataFrame:
        """
        Export complete metadata as single-row DataFrame.

        Preserves nested structures (Contact, Extent) without flattening.
        Includes core fields, Tortilla reference, and all extensions.
        """
        metadata: dict = {}

        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue

            if isinstance(value, Contact) or isinstance(value, Extent):
                metadata[key] = value.model_dump()

            elif isinstance(value, list) and value and isinstance(value[0], Contact):
                metadata[key] = [contact.model_dump() for contact in value]

            else:
                metadata[key] = value

        return pl.DataFrame([metadata])
