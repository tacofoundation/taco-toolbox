"""
Spatial grouping extension for Tortilla using Z-order curve.

Groups samples by spatial proximity using space-filling curve algorithm
for compact bounding box generation without external dependencies.

Uses Morton encoding (Z-order curve) to preserve spatial locality:
1. Extract centroids from stac:centroid column (WKB binary)
2. Compute Z-order code by interleaving lon/lat bits
3. Sort samples by Z-order code
4. Group consecutive samples into chunks by target_count OR target_size
5. Assign spatial group IDs

Exports to Arrow Table:
- spatialgroup:code: String (format: 'sg0000', 'sg0001', ...)
"""

import logging

import pyarrow as pa
import pydantic
from pydantic import Field
from shapely.geometry import Point
from shapely.wkb import loads as wkb_loads

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from tacotoolbox._validation import parse_size
from tacotoolbox.tortilla.datamodel import TortillaExtension

logger = logging.getLogger(__name__)


def normalize_coords(lon: float, lat: float) -> tuple[float, float]:
    """
    Normalize lon/lat to [0, 1] range.

    lon: [-180, 180] → [0, 1]
    lat: [-90, 90] → [0, 1]
    """
    norm_lon = (lon + 180.0) / 360.0
    norm_lat = (lat + 90.0) / 180.0
    return norm_lon, norm_lat


def morton_encode(x: float, y: float) -> int:
    """
    Compute Morton code (Z-order) by interleaving bits.

    Takes normalized coordinates [0, 1] and returns integer Z-order code.
    Higher precision = more bits = better spatial resolution.
    """
    # Scale to 21-bit integers (covers Earth at ~5m resolution)
    max_val = (1 << 21) - 1
    ix = int(x * max_val)
    iy = int(y * max_val)

    # Interleave bits
    result = 0
    for i in range(21):
        result |= ((ix & (1 << i)) << i) | ((iy & (1 << i)) << (i + 1))

    return result


def compute_z_order(lon: float, lat: float) -> int:
    """
    Compute Z-order code for geographic coordinates.

    Combines normalization and Morton encoding.
    """
    norm_lon, norm_lat = normalize_coords(lon, lat)
    return morton_encode(norm_lon, norm_lat)


class SpatialGrouping(TortillaExtension):
    """
    Spatial grouping extension for Tortilla using Z-order space-filling curve.

    Groups samples by spatial proximity to create compact bounding boxes
    for split ZIP files. Uses Z-order curve to preserve locality without
    requiring external geospatial dependencies (only numpy).

    The algorithm:
    1. Extract centroids from stac:centroid column (WKB binary)
    2. Compute Z-order code for each sample (interleaved lon/lat bits)
    3. Sort samples by Z-order code
    4. Group consecutive samples into chunks by target_count OR target_size
    5. Assign spatial_group ID to each sample

    Each spatial group will have samples that are spatially close,
    resulting in compact bounding boxes when computing extents.

    Grouping modes (you MUST specify at least one):
    - target_count only: Group by number of samples
    - target_size only: Group by cumulative size ("1GB", "512MB", etc.)
    - Both: Cut group when EITHER limit is reached (whichever comes first)

    Examples:
        # Group by sample count
        spatial = SpatialGrouping(target_count=1000)

        # Group by size
        spatial = SpatialGrouping(target_size="1GB")

        # Hybrid: whichever limit is hit first
        spatial = SpatialGrouping(target_count=1000, target_size="512MB")
    """

    target_count: int | None = Field(
        default=None,
        gt=0,
        description="Maximum samples per spatial group (None = no limit)",
    )

    target_size: str | None = Field(
        default=None,
        description="Maximum bytes per spatial group (e.g., '1GB', '512MB', None = no limit)",
    )

    # Private attribute for parsed bytes
    _target_size_bytes: int | None = pydantic.PrivateAttr(default=None)

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    @pydantic.model_validator(mode="after")
    def parse_target_size(self) -> "SpatialGrouping":
        """Parse target_size string to bytes and validate at least one limit."""
        # Validate at least one limit is specified
        if self.target_count is None and self.target_size is None:
            raise ValueError(
                "Must specify at least one of 'target_count' or 'target_size'.\n\n"
                "Examples:\n"
                "  SpatialGrouping(target_count=1000)      # Group by count\n"
                "  SpatialGrouping(target_size='1GB')      # Group by size\n"
                "  SpatialGrouping(target_count=1000, target_size='512MB')  # Both"
            )

        # Parse target_size if provided
        if self.target_size is not None:
            try:
                self._target_size_bytes = parse_size(self.target_size)
            except ValueError as e:
                raise ValueError(
                    f"Invalid target_size format: {e}\n"
                    f"Use format like '1GB', '512MB', '1024KB'"
                ) from e

            if self._target_size_bytes <= 0:
                raise ValueError(
                    f"target_size must be positive. Got: {self.target_size}"
                )

        return self

    def get_schema(self) -> pa.Schema:
        """Return the expected schema for this extension."""
        return pa.schema([pa.field("spatialgroup:code", pa.string())])

    def get_field_descriptions(self) -> dict[str, str]:
        """Return field descriptions for each field."""
        return {
            "spatialgroup:code": "Spatial group identifier using Z-order curve for compact bounding boxes."
        }

    def _compute(self, tortilla) -> pa.Table:  # noqa: C901
        """Process Tortilla and return Arrow Table with spatial group codes."""
        if not HAS_NUMPY:
            raise ImportError(
                "SpatialGrouping requires numpy.\n" "Install with: pip install numpy"
            )

        table = tortilla._metadata_table

        # Check for centroid column
        centroid_column = "stac:centroid"

        if centroid_column not in table.schema.names:
            raise ValueError(
                f"Column '{centroid_column}' not found in tortilla metadata.\n"
                f"Available columns: {table.schema.names}\n"
                f"Ensure samples have STAC extension applied."
            )

        centroid_array = table.column(centroid_column)

        coords = []
        valid_indices = []

        for i in range(table.num_rows):
            centroid_wkb = centroid_array[i].as_py()
            if centroid_wkb is None:
                continue

            try:
                geom = wkb_loads(centroid_wkb)
                if isinstance(geom, Point):
                    lon, lat = geom.x, geom.y
                    coords.append((lon, lat))
                    valid_indices.append(i)
                else:
                    logger.debug(
                        f"Sample {i}: Expected Point geometry, got {type(geom).__name__}"
                    )
            except Exception as e:
                logger.debug(f"Sample {i}: Failed to parse centroid: {e}")
                continue

        if not coords:
            raise ValueError(f"No valid centroids found in column '{centroid_column}'")

        # Compute Z-order codes
        z_orders = np.array([compute_z_order(lon, lat) for lon, lat in coords])

        # Sort by Z-order
        sorted_indices = np.argsort(z_orders)

        # Initialize all as None
        group_codes: list[str | None] = [None] * table.num_rows

        # Group samples by Z-order with size/count limits
        current_group_id = 0
        current_group_count = 0
        current_group_size = 0

        for local_idx in sorted_indices:
            original_idx = valid_indices[local_idx]
            sample = tortilla.samples[original_idx]
            sample_size = sample._size_bytes

            # Check if we need to start a new group
            start_new_group = False

            if current_group_count > 0:  # Not the first sample
                # Check target_count limit
                if self.target_count and current_group_count >= self.target_count:
                    start_new_group = True

                # Check target_size limit (using parsed bytes)
                if (
                    self._target_size_bytes
                    and (current_group_size + sample_size) > self._target_size_bytes
                ):
                    start_new_group = True

            if start_new_group:
                # Start new group
                current_group_id += 1
                current_group_count = 0
                current_group_size = 0

            # Assign to current group
            group_codes[original_idx] = f"sg{current_group_id:04d}"
            current_group_count += 1
            current_group_size += sample_size

        return pa.Table.from_pydict(
            {"spatialgroup:code": group_codes}, schema=self.get_schema()
        )
