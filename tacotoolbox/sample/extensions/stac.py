"""
STAC extension for spatiotemporal raster metadata.

Provides minimal SpatioTemporal Asset Catalog (STAC)-style metadata fields
for samples with regular raster data (affine geotransform).

Exports to DataFrame:
- stac:crs: String (WKT2/EPSG/PROJ)
- stac:tensor_shape: List[Int64]
- stac:geotransform: List[Float64]
- stac:time_start: Datetime[μs]
- stac:centroid: Binary (WKB, EPSG:4326)
- stac:time_end: Datetime[μs]
- stac:time_middle: Datetime[μs]
"""

import datetime
from typing import TypeAlias

import polars as pl
import pydantic
from pydantic import Field
from pyproj import CRS, Transformer
from shapely.geometry import Point, Polygon
from shapely.wkb import dumps as wkb_dumps

from tacotoolbox.sample.datamodel import SampleExtension

# Soft dependency - only imported when check_antimeridian=True
try:
    import antimeridian

    HAS_ANTIMERIDIAN = True
except ImportError:
    HAS_ANTIMERIDIAN = False


ShapeND: TypeAlias = tuple[int, ...]
GeoTransform6: TypeAlias = tuple[float, float, float, float, float, float]
TimestampLike: TypeAlias = datetime.datetime | int | float


def raster_centroid(
    crs: str,
    geotransform: GeoTransform6,
    raster_shape: ShapeND,
    check_antimeridian: bool = False,
) -> bytes:
    """
    Calculate raster centroid in EPSG:4326 and return as WKB binary.

    If check_antimeridian=True, detects and handles antimeridian crossings
    correctly (slower, requires 'antimeridian' package).
    """
    # Extract geotransform parameters
    origin_x, pixel_width, _, origin_y, _, pixel_height = geotransform
    rows, cols = raster_shape

    # Compute raster centroid in the raster CRS
    centroid_x = origin_x + (cols / 2) * pixel_width
    centroid_y = origin_y + (rows / 2) * pixel_height

    # Transform centroid to EPSG:4326 using pyproj
    transformer = Transformer.from_crs(
        CRS.from_string(crs), CRS.from_epsg(4326), always_xy=True
    )

    if not check_antimeridian:
        lon, lat = transformer.transform(centroid_x, centroid_y)
        point = Point(lon, lat)
        return wkb_dumps(point)

    # ANTIMERIDIAN MODE: Check if raster crosses ±180° longitude
    # This requires checking bbox corners to detect the crossing

    if not HAS_ANTIMERIDIAN:
        raise ImportError(
            "check_antimeridian=True requires the 'antimeridian' package.\n"
            "Install with: pip install antimeridian\n"
            "Or set check_antimeridian=False to use fast mode (works for most rasters)."
        )

    # Calculate bbox corners in source CRS
    min_x = origin_x
    max_x = origin_x + cols * pixel_width
    min_y = origin_y + rows * pixel_height  # pixel_height is typically negative
    max_y = origin_y

    # Transform all 4 corners to EPSG:4326
    corners_lon = []
    corners_lat = []
    for x, y in [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]:
        lon, lat = transformer.transform(x, y)
        corners_lon.append(lon)
        corners_lat.append(lat)

    # Check if bbox crosses antimeridian
    # Heuristic: if longitude span > 180°, it wraps around
    lon_span = max(corners_lon) - min(corners_lon)
    crosses_antimeridian = lon_span > 180

    if crosses_antimeridian:
        # Build polygon and use antimeridian.centroid()
        bbox_polygon = Polygon(
            [
                (corners_lon[0], corners_lat[0]),
                (corners_lon[1], corners_lat[1]),
                (corners_lon[2], corners_lat[2]),
                (corners_lon[3], corners_lat[3]),
                (corners_lon[0], corners_lat[0]),  # Close ring
            ]
        )

        centroid_point = antimeridian.centroid(bbox_polygon)
    else:
        # Bbox doesn't cross antimeridian - use simple centroid
        centroid_lon = sum(corners_lon) / 4
        centroid_lat = sum(corners_lat) / 4
        centroid_point = Point(centroid_lon, centroid_lat)

    return wkb_dumps(centroid_point)


class STAC(SampleExtension):
    """
    Minimal SpatioTemporal Asset Catalog (STAC)-style metadata for samples.

    For regular raster data with affine geotransform.

    Notes
    -----

    - Timestamps stored as Parquet TIMESTAMP with microsecond precision
    - datetime.datetime inputs are automatically converted to microseconds (int64)
    - int/float inputs in seconds are converted to microseconds
    - time_middle is auto-computed when both start and end times exist
    - Set check_antimeridian=True (requires: pip install antimeridian)
    """

    crs: str = Field(
        description="Coordinate Reference System in WKT2, EPSG code, or PROJ string (e.g., 'EPSG:4326')."
    )
    tensor_shape: ShapeND = Field(
        description="Shape of the data tensor (e.g., (bands, height, width) or (height, width))"
    )
    geotransform: GeoTransform6 = Field(
        description="GDAL geotransform tuple (origin_x, pixel_width, rotation_x, origin_y, rotation_y, pixel_height)"
    )
    time_start: TimestampLike = Field(
        description="Acquisition start timestamp as Datetime[μs] (microseconds since Unix epoch, timezone-naive UTC)."
    )
    centroid: bytes | None = Field(
        default=None,
        description="Raster centroid in EPSG:4326 as WKB binary (Well-Known Binary geometry format).",
    )
    time_end: TimestampLike | None = Field(
        default=None,
        description="Acquisition end timestamp as Datetime[μs] (microseconds since Unix epoch, timezone-naive UTC). Must be ≥ time_start.",
    )
    time_middle: int | None = Field(
        default=None,
        description="Middle timestamp as Datetime[μs] (microseconds since Unix epoch, timezone-naive UTC).",
    )
    check_antimeridian: bool = Field(
        default=False,
        description="Enable antimeridian (±180° longitude) crossing detection. Required for rasters spanning the dateline. Requires 'antimeridian' package.",
    )

    @pydantic.model_validator(mode="after")
    def check_times(cls, values):
        """Validates that time_start <= time_end and converts to microseconds."""
        # Convert datetime to microseconds (int64)
        if isinstance(values.time_start, datetime.datetime):
            values.time_start = int(values.time_start.timestamp() * 1_000_000)
        else:
            # Assume seconds, convert to microseconds
            values.time_start = int(values.time_start * 1_000_000)

        if values.time_end is not None:
            if isinstance(values.time_end, datetime.datetime):
                values.time_end = int(values.time_end.timestamp() * 1_000_000)
            else:
                # Assume seconds, convert to microseconds
                values.time_end = int(values.time_end * 1_000_000)

            if values.time_start > values.time_end:
                raise ValueError(
                    f"Invalid times: {values.time_start} > {values.time_end}"
                )

        return values

    @pydantic.model_validator(mode="after")
    def populate_time_middle(cls, values):
        """Auto-populate time_middle when both time_start and time_end exist."""
        if values.time_end is not None and values.time_middle is None:
            values.time_middle = (values.time_start + values.time_end) // 2

        return values

    @pydantic.model_validator(mode="after")
    def populate_centroid(cls, values):
        """
        Auto-populate centroid if not provided.

        Assumes spatial dimensions are the last two of tensor_shape.

        If check_antimeridian=True, requires 'antimeridian' package for
        correct handling of rasters crossing ±180° longitude.
        """
        if values.centroid is None:
            if len(values.tensor_shape) < 2:
                raise ValueError(
                    f"tensor_shape must have at least 2 dimensions (got {values.tensor_shape})"
                )
            # Extract (rows, cols) from the last two dims and compute centroid
            rows, cols = values.tensor_shape[-2], values.tensor_shape[-1]
            values.centroid = raster_centroid(
                crs=values.crs,
                geotransform=values.geotransform,
                raster_shape=(rows, cols),
                check_antimeridian=values.check_antimeridian,
            )
        return values

    def get_schema(self) -> dict[str, pl.DataType]:
        """Return the expected schema for this extension."""
        return {
            "stac:crs": pl.Utf8(),
            "stac:tensor_shape": pl.List(pl.Int64()),
            "stac:geotransform": pl.List(pl.Float64()),
            "stac:time_start": pl.Datetime(time_unit="us", time_zone=None),
            "stac:centroid": pl.Binary(),
            "stac:time_end": pl.Datetime(time_unit="us", time_zone=None),
            "stac:time_middle": pl.Datetime(time_unit="us", time_zone=None),
        }

    def _compute(self, sample) -> pl.DataFrame:
        """Actual computation logic - only called when schema_only=False."""
        return pl.DataFrame(
            {
                "stac:crs": [self.crs],
                "stac:tensor_shape": [list(self.tensor_shape)],
                "stac:geotransform": [list(self.geotransform)],
                "stac:time_start": [self.time_start],
                "stac:centroid": [self.centroid],
                "stac:time_end": [self.time_end],
                "stac:time_middle": [self.time_middle],
            },
            schema=self.get_schema(),
        )

    def get_field_descriptions(self) -> dict[str, str]:
        """Return field descriptions for each field."""
        return {
            "stac:crs": "Coordinate reference system (WKT2, EPSG, or PROJ string)",
            "stac:tensor_shape": "Raster dimensions e.g. [bands, height, width]",
            "stac:geotransform": "GDAL affine transform [origin_x, pixel_width, rot_x, origin_y, rot_y, pixel_height]",
            "stac:time_start": "Acquisition start timestamp (microseconds since Unix epoch, UTC)",
            "stac:centroid": "Raster center point in EPSG:4326 (WKB binary format)",
            "stac:time_end": "Acquisition end timestamp (microseconds since Unix epoch, UTC)",
            "stac:time_middle": "Midpoint between start and end timestamps (microseconds since Unix epoch, UTC)",
        }
