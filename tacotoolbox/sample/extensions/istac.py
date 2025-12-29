"""
ISTAC extension for irregular spatiotemporal geometries.

Provides Irregular SpatioTemporal Asset Catalog (ISTAC) metadata for
non-regular geospatial data without affine geotransform.

For regular raster data, use STAC instead.

Exports to DataFrame:
- istac:crs: String (WKT2/EPSG/PROJ)
- istac:geometry: Binary (WKB)
- istac:time_start: Datetime[μs]
- istac:time_end: Datetime[μs]
- istac:time_middle: Datetime[μs]
- istac:centroid: Binary (WKB, EPSG:4326)
"""

import math

import pyarrow as pa
import pydantic
from pydantic import Field
from pyproj import CRS, Transformer
from pyproj.exceptions import CRSError
from shapely.geometry import Point
from shapely.wkb import dumps as wkb_dumps
from shapely.wkb import loads as wkb_loads

from tacotoolbox.sample.datamodel import SampleExtension

try:
    import antimeridian

    HAS_ANTIMERIDIAN = True
except ImportError:
    HAS_ANTIMERIDIAN = False


def geometry_centroid( #noqa: C901
    crs: str,
    geometry: bytes,
    check_antimeridian: bool = False,
    sample_id: str | None = None,
) -> bytes:
    """Calculate geometry centroid in EPSG:4326, return as WKB."""
    id_str = f"[{sample_id}] " if sample_id else ""

    # Load geometry
    try:
        geom = wkb_loads(geometry)
    except Exception as e:
        raise ValueError(f"{id_str}Invalid WKB geometry") from e

    if geom.is_empty:
        raise ValueError(f"{id_str}Geometry is empty")

    # Parse CRS
    try:
        src_crs = CRS.from_string(crs)
    except CRSError as e:
        raise ValueError(f"{id_str}Invalid CRS: {crs}") from e

    # Compute centroid
    if src_crs.is_geographic and check_antimeridian:
        if not HAS_ANTIMERIDIAN:
            raise ImportError(
                f"{id_str}check_antimeridian=True requires 'antimeridian' package. pip install antimeridian"
            )
        centroid_geom = antimeridian.centroid(geom)
    else:
        centroid_geom = geom.centroid

    # Transform to WGS84 if needed
    if src_crs.is_geographic:
        lon, lat = centroid_geom.x, centroid_geom.y
    else:
        transformer = Transformer.from_crs(src_crs, CRS.from_epsg(4326), always_xy=True)
        lon, lat = transformer.transform(centroid_geom.x, centroid_geom.y)

    # Validate
    if math.isnan(lon) or math.isnan(lat):
        raise ValueError(f"{id_str}Centroid is NaN")

    if math.isinf(lon) or math.isinf(lat):
        raise ValueError(f"{id_str}Centroid is Inf")

    if not (-180 <= lon <= 180):
        raise ValueError(f"{id_str}Longitude {lon} out of bounds [-180, 180]")

    if not (-90 <= lat <= 90):
        raise ValueError(f"{id_str}Latitude {lat} out of bounds [-90, 90]")

    return wkb_dumps(Point(lon, lat))


class ISTAC(SampleExtension):
    """
    Irregular SpatioTemporal Asset Catalog (ISTAC) metadata for non-regular geometries.

    Requirements:
    - Timestamps: int64 microseconds since Unix epoch (UTC)
    - Geometry: WKB binary in source CRS
    - Centroid: auto-computed from geometry if not provided
    - time_middle: auto-computed as (time_start + time_end) // 2
    - check_antimeridian: only relevant for geographic CRS (EPSG:4326)
    """

    sample_id: str | None = Field(default=None, description="Sample ID for error messages.")
    crs: str = Field(description="Coordinate Reference System (WKT2, EPSG, or PROJ).")
    geometry: bytes = Field(description="Spatial footprint in source CRS as WKB.")
    time_start: int = Field(description="Start timestamp (μs since Unix epoch, UTC).")
    time_end: int | None = Field(default=None, description="End timestamp (μs since Unix epoch, UTC).")
    time_middle: int | None = Field(default=None, description="Middle timestamp (μs since Unix epoch, UTC).")
    centroid: bytes | None = Field(default=None, description="Centroid in EPSG:4326 as WKB.")
    check_antimeridian: bool = Field(default=False, description="Handle antimeridian crossing (geographic CRS only).")

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
            self.centroid = geometry_centroid(
                crs=self.crs,
                geometry=self.geometry,
                check_antimeridian=self.check_antimeridian,
                sample_id=self.sample_id,
            )
        return self

    def get_schema(self) -> pa.Schema:
        return pa.schema([
            pa.field("istac:crs", pa.string()),
            pa.field("istac:geometry", pa.binary()),
            pa.field("istac:time_start", pa.timestamp("us", tz=None)),
            pa.field("istac:time_end", pa.timestamp("us", tz=None)),
            pa.field("istac:time_middle", pa.timestamp("us", tz=None)),
            pa.field("istac:centroid", pa.binary()),
        ])

    def get_field_descriptions(self) -> dict[str, str]:
        return {
            "istac:crs": "Coordinate reference system (WKT2, EPSG, or PROJ)",
            "istac:geometry": "Spatial footprint in source CRS (WKB)",
            "istac:time_start": "Start timestamp (μs since Unix epoch, UTC)",
            "istac:time_end": "End timestamp (μs since Unix epoch, UTC)",
            "istac:time_middle": "Middle timestamp (μs since Unix epoch, UTC)",
            "istac:centroid": "Center point in EPSG:4326 (WKB)",
        }

    def _compute(self, sample) -> pa.Table:
        return pa.Table.from_pydict(
            {
                "istac:crs": [self.crs],
                "istac:geometry": [self.geometry],
                "istac:time_start": [self.time_start],
                "istac:time_end": [self.time_end],
                "istac:time_middle": [self.time_middle],
                "istac:centroid": [self.centroid],
            },
            schema=self.get_schema(),
        )
