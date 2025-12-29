"""
STAC extension for spatiotemporal raster metadata.

Provides SpatioTemporal Asset Catalog (STAC)-style metadata fields
for samples with regular raster data (affine geotransform).

For irregular geometries (swaths, vector data), use ISTAC instead.

Exports to DataFrame:
- stac:crs: String (WKT2/EPSG/PROJ)
- stac:tensor_shape: List[Int64]
- stac:geotransform: List[Float64]
- stac:time_start: Datetime[μs]
- stac:centroid: Binary (WKB, EPSG:4326)
- stac:time_end: Datetime[μs]
- stac:time_middle: Datetime[μs]
"""

import math
from typing import TypeAlias

import pyarrow as pa
import pydantic
from pydantic import Field
from pyproj import CRS, Transformer
from pyproj.exceptions import CRSError
from shapely.geometry import Point
from shapely.wkb import dumps as wkb_dumps

from tacotoolbox.sample.datamodel import SampleExtension

ShapeND: TypeAlias = tuple[int, ...]
GeoTransform6: TypeAlias = tuple[float, float, float, float, float, float]


def raster_centroid(
    crs: str,
    geotransform: GeoTransform6,
    raster_shape: ShapeND,
    sample_id: str | None = None,
) -> bytes:
    """Calculate raster centroid in EPSG:4326, return as WKB."""
    id_str = f"[{sample_id}] " if sample_id else ""

    origin_x, pixel_width, _, origin_y, _, pixel_height = geotransform
    rows, cols = raster_shape

    centroid_x = origin_x + (cols / 2) * pixel_width
    centroid_y = origin_y + (rows / 2) * pixel_height

    try:
        src_crs = CRS.from_string(crs)
    except CRSError as e:
        raise ValueError(f"{id_str}Invalid CRS: {crs}") from e

    transformer = Transformer.from_crs(src_crs, CRS.from_epsg(4326), always_xy=True)
    lon, lat = transformer.transform(centroid_x, centroid_y)

    if math.isnan(lon) or math.isnan(lat):
        raise ValueError(f"{id_str}Centroid is NaN. Source: ({centroid_x}, {centroid_y})")

    if math.isinf(lon) or math.isinf(lat):
        raise ValueError(f"{id_str}Centroid is Inf. Source: ({centroid_x}, {centroid_y})")

    if not (-180 <= lon <= 180):
        raise ValueError(f"{id_str}Longitude {lon} out of bounds [-180, 180]")

    if not (-90 <= lat <= 90):
        raise ValueError(f"{id_str}Latitude {lat} out of bounds [-90, 90]")

    return wkb_dumps(Point(lon, lat))


class STAC(SampleExtension):
    """
    SpatioTemporal Asset Catalog (STAC) metadata for regular raster data.

    Requirements:
    - Timestamps: int64 microseconds since Unix epoch (UTC)
    - Centroid: auto-computed from geotransform if not provided
    - time_middle: auto-computed as (time_start + time_end) // 2
    """

    sample_id: str | None = Field(default=None, description="Sample ID for error messages.")
    crs: str = Field(description="Coordinate Reference System (WKT2, EPSG, or PROJ).")
    tensor_shape: ShapeND = Field(description="Tensor shape (e.g., (bands, height, width)).")
    geotransform: GeoTransform6 = Field(description="GDAL geotransform tuple.")
    time_start: int = Field(description="Start timestamp (μs since Unix epoch, UTC).")
    centroid: bytes | None = Field(default=None, description="Centroid in EPSG:4326 as WKB.")
    time_end: int | None = Field(default=None, description="End timestamp (μs since Unix epoch, UTC).")
    time_middle: int | None = Field(default=None, description="Middle timestamp (μs since Unix epoch, UTC).")

    def _id_str(self) -> str:
        return f"[{self.sample_id}] " if self.sample_id else ""

    @pydantic.model_validator(mode="after")
    def check_times(self):
        if self.time_end is not None and self.time_start > self.time_end:
            raise ValueError(f"{self._id_str()}time_start ({self.time_start}) > time_end ({self.time_end})")
        return self

    @pydantic.model_validator(mode="after")
    def populate_time_middle(self):
        if self.time_end is not None and self.time_middle is None:
            self.time_middle = (self.time_start + self.time_end) // 2
        return self

    @pydantic.model_validator(mode="after")
    def populate_centroid(self):
        if self.centroid is None:
            if len(self.tensor_shape) < 2:
                raise ValueError(f"{self._id_str()}tensor_shape must have >= 2 dimensions")
            rows, cols = self.tensor_shape[-2], self.tensor_shape[-1]
            self.centroid = raster_centroid(
                crs=self.crs,
                geotransform=self.geotransform,
                raster_shape=(rows, cols),
                sample_id=self.sample_id,
            )
        return self

    def get_schema(self) -> pa.Schema:
        return pa.schema([
            pa.field("stac:crs", pa.string()),
            pa.field("stac:tensor_shape", pa.list_(pa.int64())),
            pa.field("stac:geotransform", pa.list_(pa.float64())),
            pa.field("stac:time_start", pa.timestamp("us", tz=None)),
            pa.field("stac:centroid", pa.binary()),
            pa.field("stac:time_end", pa.timestamp("us", tz=None)),
            pa.field("stac:time_middle", pa.timestamp("us", tz=None)),
        ])

    def get_field_descriptions(self) -> dict[str, str]:
        return {
            "stac:crs": "Coordinate reference system (WKT2, EPSG, or PROJ)",
            "stac:tensor_shape": "Raster dimensions [bands, height, width]",
            "stac:geotransform": "GDAL affine transform",
            "stac:time_start": "Start timestamp (μs since Unix epoch, UTC)",
            "stac:centroid": "Center point in EPSG:4326 (WKB)",
            "stac:time_end": "End timestamp (μs since Unix epoch, UTC)",
            "stac:time_middle": "Middle timestamp (μs since Unix epoch, UTC)",
        }

    def _compute(self, sample) -> pa.Table:
        return pa.Table.from_pydict(
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
