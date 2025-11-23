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

Positional-Invariance Tree (PIT):
    PIT validates that all root samples are isomorphic. For hierarchical datasets
    (all FOLDER samples), this means:
    - Same number of children at each position
    - Same ID at each position across all roots
    - Same type at each position across all roots
    - Recursive validation for nested FOLDERs

    Rule: Root[i].children[k] must be identical for all i at position k.

    Example (valid):
        Date_0/ -> [GLORYS/, L4/, L3/]
        Date_1/ -> [GLORYS/, L4/, L3/]
        Date_2/ -> [GLORYS/, L4/, L3/]

    Example (invalid):
        Date_0/ -> [GLORYS/, L4/]
        Date_1/ -> [L4/, GLORYS/]  # Position 0 must be GLORYS

    Validation occurs at Taco creation to ensure structural consistency.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Literal, cast

import polars as pl
import pydantic
from shapely.geometry import Point
from shapely.wkb import loads as wkb_loads

from tacotoolbox._constants import PADDING_PREFIX
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
    def check_name_or_organization(self) -> "Contact":
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

        if start_dt > end_dt:
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
        providers: Dataset creators (Contact list)
        tasks: ML task types

    Optional fields:
        taco_version: TACO format version (default: "0.5.0")
        title: Human-friendly title (max 250 chars)
        curators: Dataset curators (Contact list)
        keywords: Searchable tags
        extent: Spatial/temporal boundaries (auto-calculated from STAC/ISTAC)
    """

    # Required core fields
    tortilla: Tortilla
    id: str
    dataset_version: str
    description: str
    licenses: list[str]
    providers: list[Contact]
    tasks: list[TaskType]

    # Optional core fields
    taco_version: str = "0.5.0"
    title: str | None = None
    curators: list[Contact] | None = None
    keywords: list[str] | None = None
    extent: Extent | None = None

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

    @pydantic.model_validator(mode="after")
    def validate_and_initialize(self) -> "Taco":
        """Validate PIT compliance and auto-calculate extent."""
        self._validate_pit_compliance()
        self._auto_calculate_extent()
        return self

    def _validate_pit_compliance(self) -> None:  # noqa: C901
        """
        Validate PIT constraint for root tortilla.

        This is the only place where PIT validation occurs because Taco knows
        its tortilla is the root of the dataset tree. Validates that all root
        samples are structurally isomorphic.
        """
        samples = self.tortilla.samples

        # Validate all root samples have same type
        root_types = [s.type for s in samples]
        unique_types = set(root_types)

        if len(unique_types) > 1:
            raise ValueError(
                f"PIT violation: All root samples must have the same type.\n"
                f"Found mixed types at root level: {unique_types}\n"
                f"All root samples must be either all FILE or all FOLDER."
            )

        # If all FILEs, no further validation needed
        if "FOLDER" not in unique_types:
            return

        # Collect FOLDER samples
        folder_samples = [s for s in samples if s.type == "FOLDER"]

        if len(folder_samples) <= 1:
            return

        # Get non-padding children from each folder
        folder_children: list[tuple[str, list]] = []
        for folder in folder_samples:
            if hasattr(folder.path, "samples") and folder.path.samples:
                real_children = [
                    child
                    for child in folder.path.samples
                    if not child.id.startswith(PADDING_PREFIX)
                ]
                folder_children.append((folder.id, real_children))

        if len(folder_children) <= 1:
            return

        # Validate all folders have same number of children
        child_counts = [len(children) for _, children in folder_children]
        if len(set(child_counts)) > 1:
            count_report = ", ".join(
                f"'{folder_id}': {count}"
                for (folder_id, _), count in zip(
                    folder_children, child_counts, strict=False
                )
            )
            raise ValueError(
                f"PIT violation: All root samples must be isomorphic.\n"
                f"Child counts (excluding padding): {count_report}\n"
                f"All root samples must have the same internal structure."
            )

        # Validate PIT
        reference_folder_id, reference_children = folder_children[0]
        num_positions = len(reference_children)

        for pos in range(num_positions):
            reference_child = reference_children[pos]
            reference_id = reference_child.id
            reference_type = reference_child.type

            for folder_id, children in folder_children[1:]:
                current_child = children[pos]

                if current_child.id != reference_id:
                    raise ValueError(
                        f"PIT violation at position {pos}:\n"
                        f"Root sample '{reference_folder_id}' has child ID: '{reference_id}'\n"
                        f"Root sample '{folder_id}' has child ID: '{current_child.id}'\n"
                        f"All root samples must have the same child ID at position {pos}."
                    )

                if current_child.type != reference_type:
                    raise ValueError(
                        f"PIT violation at position {pos} (ID: '{reference_id}'):\n"
                        f"Root sample '{reference_folder_id}' has child type: '{reference_type}'\n"
                        f"Root sample '{folder_id}' has child type: '{current_child.type}'\n"
                        f"All root samples must have the same child type at position {pos}."
                    )

        # Recursive validation for nested FOLDERs
        for pos in range(num_positions):
            children_at_position = [children[pos] for _, children in folder_children]

            if all(child.type == "FOLDER" for child in children_at_position):
                try:
                    child_tortillas = [
                        cast(Tortilla, child.path) for child in children_at_position
                    ]
                    for i, child_tortilla in enumerate(child_tortillas[1:], start=1):
                        if len(child_tortilla.samples) != len(
                            child_tortillas[0].samples
                        ):
                            raise ValueError(
                                f"PIT violation in nested structure at position {pos}:\n"
                                f"Root[0].child[{pos}] ('{children_at_position[0].id}') has {len(child_tortillas[0].samples)} children\n"
                                f"Root[{i}].child[{pos}] ('{children_at_position[i].id}') has {len(child_tortilla.samples)} children\n"
                                f"All children at position {pos} must be isomorphic."
                            )
                except AttributeError:
                    pass

    def _auto_calculate_extent(self) -> None:
        """
        Calculate extent automatically from STAC/ISTAC metadata in samples.

        Searches incrementally through hierarchy levels (0, 1, 2...) until finding
        STAC or ISTAC columns. Uses that level's DataFrame to compute spatial and
        temporal extents.

        Priority: STAC > ISTAC

        Defaults if no spatiotemporal metadata found:
        - spatial: [-180, -90, 180, 90] (global)
        - temporal: None (atemporal dataset)
        """
        max_depth = self.tortilla._current_depth

        for depth in range(max_depth + 1):
            df = self.tortilla.export_metadata(deep=depth)

            has_stac = "stac:centroid" in df.columns and "stac:time_start" in df.columns
            has_istac = (
                "istac:centroid" in df.columns and "istac:time_start" in df.columns
            )

            if has_stac or has_istac:
                if has_stac:
                    spatial = _calculate_spatial_extent(df, "stac:centroid")
                    temporal = _calculate_temporal_extent(
                        df,
                        "stac:time_start",
                        "stac:time_end" if "stac:time_end" in df.columns else None,
                        (
                            "stac:time_middle"
                            if "stac:time_middle" in df.columns
                            else None
                        ),
                    )
                else:
                    spatial = _calculate_spatial_extent(df, "istac:centroid")
                    temporal = _calculate_temporal_extent(
                        df,
                        "istac:time_start",
                        "istac:time_end" if "istac:time_end" in df.columns else None,
                        (
                            "istac:time_middle"
                            if "istac:time_middle" in df.columns
                            else None
                        ),
                    )

                self.extent = Extent(spatial=spatial, temporal=temporal)
                return

        self.extent = Extent(spatial=[-180.0, -90.0, 180.0, 90.0], temporal=None)

    def extend_with(self, extension: Any) -> None:
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

    def _handle_dict_extension(self, extension: dict[str, Any]) -> None:
        """Handle dictionary extension."""
        self._set_extension_attributes(extension)

    def _handle_pydantic_extension(self, extension: Any) -> None:
        """Handle Pydantic model extension."""
        if hasattr(extension, "model_dump"):
            extension_data = extension.model_dump()
            self._set_extension_attributes(extension_data)
        else:
            raise ValueError(
                "Extension must be TacoExtension, DataFrame, dict, or pydantic model"
            )

    def _set_extension_attributes(self, extension_data: dict[str, Any]) -> None:
        """Set extension fields as instance attributes."""
        for key, value in extension_data.items():
            setattr(self, key, value)

    def export_metadata(self) -> pl.DataFrame:
        """
        Export complete metadata as single-row DataFrame.

        Preserves nested structures (Contact, Extent) without flattening.
        Includes core fields, Tortilla reference, and all extensions.
        """
        metadata: dict[str, Any] = {}

        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue

            if isinstance(value, Contact | Extent):
                metadata[key] = value.model_dump()

            elif isinstance(value, list) and value and isinstance(value[0], Contact):
                metadata[key] = [contact.model_dump() for contact in value]

            else:
                metadata[key] = value

        return pl.DataFrame([metadata])


def _calculate_spatial_extent(df: pl.DataFrame, centroid_col: str) -> list[float]:
    """
    Calculate spatial bounding box from centroid geometries.

    Parses WKB binary centroids and computes [min_lon, min_lat, max_lon, max_lat].
    Skips None values (padding samples).
    """
    centroids = [
        cast(Point, wkb_loads(wkb))
        for wkb in df[centroid_col].to_list()
        if wkb is not None
    ]

    if not centroids:
        return [-180.0, -90.0, 180.0, 90.0]

    lons = [point.x for point in centroids]
    lats = [point.y for point in centroids]

    return [min(lons), min(lats), max(lons), max(lats)]


def _calculate_temporal_extent(
    df: pl.DataFrame,
    time_start_col: str,
    time_end_col: str | None,
    time_middle_col: str | None,
) -> list[str] | None:
    """
    Calculate temporal interval from Datetime columns.

    Extracts datetime values, computes min/max in Python.
    Skips None values (padding samples).

    Priority cascade: time_middle > time_start > time_end
    Tries each column until finding one with valid values.
    """
    # Try columns in priority order until finding one with valid values
    candidate_cols = []
    if time_middle_col and time_middle_col in df.columns:
        candidate_cols.append(time_middle_col)
    if time_start_col in df.columns:
        candidate_cols.append(time_start_col)
    if time_end_col and time_end_col in df.columns:
        candidate_cols.append(time_end_col)

    if not candidate_cols:
        return None

    # Try each candidate until finding one with non-None values
    for time_col in candidate_cols:
        time_values = [dt for dt in df[time_col].to_list() if dt is not None]

        if time_values:
            # Found valid values, compute extent
            min_dt = min(time_values)
            max_dt = max(time_values)

            # Convert to UTC if not already (Polars Datetime without timezone is naive)
            if min_dt.tzinfo is None:
                min_dt = min_dt.replace(tzinfo=timezone.utc)
            if max_dt.tzinfo is None:
                max_dt = max_dt.replace(tzinfo=timezone.utc)

            # Convert to ISO 8601 strings with 'Z' suffix
            return [
                min_dt.isoformat().replace("+00:00", "Z"),
                max_dt.isoformat().replace("+00:00", "Z"),
            ]

    # No column had valid values
    return None
