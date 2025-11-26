"""
Spatial grouping extension for Tortilla using Z-order curve.

Groups samples by spatial locality using space-filling curve algorithm
for compact bbox generation without external dependencies.
"""

import logging

import polars as pl
import pydantic
from pydantic import Field
from shapely.geometry import Point
from shapely.wkb import loads as wkb_loads

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

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
    4. Group consecutive samples into chunks of target_size
    5. Assign spatial_group ID to each sample

    Each spatial group will have samples that are spatially close,
    resulting in compact bounding boxes when computing extents.
    """

    target_size: int = Field(
        1000,
        description="Target number of samples per spatial group",
        gt=0,
    )

    centroid_column: str = Field(
        "stac:centroid",
        description="Column name containing WKB centroid geometry",
    )

    group_column: str = Field(
        "spatial_group",
        description="Output column name for spatial group IDs",
    )

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    def get_schema(self) -> dict[str, pl.DataType]:
        """Return the expected schema for this extension."""
        return {self.group_column: pl.Utf8()}

    def _compute(self, tortilla) -> pl.DataFrame:
        """
        Process Tortilla and return DataFrame with spatial group IDs.

        Returns DataFrame with single column containing group IDs like "g0000", "g0001", etc.
        """
        if not HAS_NUMPY:
            raise ImportError(
                "SpatialGrouping requires numpy.\n" "Install with: pip install numpy"
            )

        df = tortilla._metadata_df

        if self.centroid_column not in df.columns:
            raise ValueError(
                f"Column '{self.centroid_column}' not found in tortilla metadata.\n"
                f"Available columns: {df.columns}\n"
                f"Ensure samples have STAC extension applied."
            )

        centroids_binary = df[self.centroid_column].to_list()

        coords = []
        valid_indices = []

        for i, centroid_wkb in enumerate(centroids_binary):
            if centroid_wkb is None:
                continue

            try:
                geom = wkb_loads(bytes(centroid_wkb))
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
            raise ValueError(
                f"No valid centroids found in column '{self.centroid_column}'"
            )

        z_orders = np.array([compute_z_order(lon, lat) for lon, lat in coords])

        sorted_indices = np.argsort(z_orders)

        group_ids: list[str | None] = [None] * len(df)

        for group_id, chunk_start in enumerate(
            range(0, len(sorted_indices), self.target_size)
        ):
            chunk_indices = sorted_indices[chunk_start : chunk_start + self.target_size]

            for local_idx in chunk_indices:
                original_idx = valid_indices[local_idx]
                group_ids[original_idx] = f"g{group_id:04d}"

        return pl.DataFrame({self.group_column: group_ids}, schema=self.get_schema())
