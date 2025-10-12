import datetime
from typing import TypeAlias

import polars as pl
import pydantic
from pyproj import CRS, Transformer
from shapely.geometry import Point
from shapely.wkb import dumps as wkb_dumps

from tacotoolbox.sample.datamodel import SampleExtension

ShapeND: TypeAlias = tuple[int, ...]
GeoTransform6: TypeAlias = tuple[float, float, float, float, float, float]
TimestampLike: TypeAlias = datetime.datetime | int | float


def raster_centroid(
    crs: str,
    geotransform: GeoTransform6,
    raster_shape: ShapeND,
) -> bytes:
    """
    Calculate the centroid of a raster in EPSG:4326 and return as WKB binary.

    Args:
        crs (str): The raster's Coordinate Reference System (e.g., "EPSG:32633").
        geotransform (Tuple[float, float, float, float, float, float]): The
            geotransform of the raster following the GDAL convention:
            (
                top left x,
                x resolution,
                x rotation,
                top left y,
                y rotation,
                y resolution
            )
        raster_shape (Tuple[int, int]): The shape of the raster as (rows, columns).

    Returns:
        bytes: The centroid coordinates in EPSG:4326 as WKB binary.
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
    lon, lat = transformer.transform(centroid_x, centroid_y)

    # Create shapely Point and convert to WKB
    point = Point(lon, lat)
    return wkb_dumps(point)


class STAC(SampleExtension):
    """
    Minimal SpatioTemporal Asset Catalog (STAC)-style metadata for samples

    Fields
    ------
    crs : str
        Coordinate reference system identifier or definition (e.g., "EPSG:4326",
        a PROJ string, or WKT).
    tensor_shape : ShapeND
        Shape of the tensor data, typically a 2D shape (height, width) for
        raster data. For n-dimensional data, the spatial dimensions are
        expected to be the last two dimensions.
    geotransform : GeoTransform6
        GDAL-style affine transform (x0, px_w, rot_x, y0, rot_y, px_h),
        where typical north-up rasters have rot_x = rot_y = 0 and px_h < 0.
    time_start : TimestampLike
        Start time. Accepts a timezone-aware or naive `datetime` (naive will be
        interpreted as system/unspecified timezone by `timestamp()`), or an
        epoch value in seconds (`int`/`float`).
    centroid : bytes | None
        Automatically computed centroid of the raster in EPSG:4326 as WKB binary.
        If not provided, it will be computed from `crs`, `geotransform`, and
        `tensor_shape`.
    time_end : TimestampLike | None
        Optional end time. Same accepted forms as `time_start`.
    time_middle : int | None
        Automatically computed midpoint between `time_start` and `time_end`.
        Only populated when `time_end` is provided. Calculated as the integer
        average of the two timestamps.

    Notes
    -----
    - During validation, any `datetime` provided for `time_start`/`time_end` is
      coerced to seconds since the Unix epoch (int) via `.timestamp()`.
    - The model enforces a non-decreasing temporal interval: `time_start <= time_end`
      (i.e., it rejects only `time_start > time_end`).
    - `time_middle` is automatically computed when both start and end times exist.
    """

    crs: str
    tensor_shape: ShapeND
    geotransform: GeoTransform6
    time_start: TimestampLike
    centroid: bytes | None = None
    time_end: TimestampLike | None = None
    time_middle: int | None = None

    @pydantic.model_validator(mode="after")
    def check_times(cls, values):
        """Validates that the time_start is before time_end."""
        # If time_start is a datetime object, convert it to a timestamp
        if isinstance(values.time_start, datetime.datetime):
            values.time_start = int(values.time_start.timestamp())
        else:
            values.time_start = int(values.time_start)

        # If time_end is a datetime object, convert it to a timestamp
        if values.time_end is not None:
            if isinstance(values.time_end, datetime.datetime):
                values.time_end = int(values.time_end.timestamp())
            else:
                values.time_end = int(values.time_end)

            if values.time_start > values.time_end:
                raise ValueError(
                    f"Invalid times: {values.time_start} > {values.time_end}"
                )

        return values

    @pydantic.model_validator(mode="after")
    def populate_time_middle(cls, values):
        """
        Auto-populate `time_middle` when both time_start and time_end exist.

        This validator runs after `check_times` to ensure timestamps are already
        converted to integers.
        """
        if values.time_end is not None and values.time_middle is None:
            values.time_middle = (values.time_start + values.time_end) // 2

        return values

    @pydantic.model_validator(mode="after")
    def populate_centroid(cls, values):
        """
        Auto-populate `centroid` if not provided.

        Assumes the spatial dimensions are the last two of `tensor_shape`.
        Raises a clear error if `tensor_shape` has fewer than two dims.
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
            )
        return values

    def get_schema(self) -> dict[str, pl.DataType]:
        """Return the expected schema for this extension."""
        return {
            "stac:crs": pl.Utf8(),
            "stac:tensor_shape": pl.List(pl.Int64()),
            "stac:geotransform": pl.List(pl.Float64()),
            "stac:time_start": pl.Int64(),
            "stac:centroid": pl.Binary(),
            "stac:time_end": pl.Int64(),
            "stac:time_middle": pl.Int64(),
        }

    def _compute(self, sample) -> pl.DataFrame:
        """Actual computation logic - only called when return_none=False."""
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
