"""
ISTAC extension for irregular spatiotemporal geometries.

Provides Irregular SpatioTemporal Asset Catalog (ISTAC) metadata for
non-regular geospatial data without affine geotransform.

Use cases: satellite swaths, flight paths, vector data, irregular sensor networks.

Exports to DataFrame:
- istac:crs: String (WKT2/EPSG/PROJ)
- istac:geometry: Binary (WKB)
- istac:time_start: Datetime[μs]
- istac:time_end: Datetime[μs]
- istac:time_middle: Datetime[μs]
- istac:centroid: Binary (WKB, EPSG:4326)
"""

import pyarrow as pa
import pydantic
from pydantic import Field
from pyproj import CRS, Transformer
from shapely.geometry import Point
from shapely.wkb import dumps as wkb_dumps
from shapely.wkb import loads as wkb_loads

from tacotoolbox.sample.datamodel import SampleExtension

# Soft dependency - only imported when check_antimeridian=True
try:
    import antimeridian

    HAS_ANTIMERIDIAN = True
except ImportError:
    HAS_ANTIMERIDIAN = False


class ISTAC(SampleExtension):
    """
    Irregular SpatioTemporal Asset Catalog (ISTAC) metadata for non-regular geometries.

    For geospatial data that cannot be represented with an affine geotransform:
    - Satellite swaths: CloudSat, CALIPSO, GPM orbital tracks
    - Flight paths: Aircraft or drone trajectories
    - Vector data: Polygons, lines, or points without underlying raster
    - Irregular samplings: Weather stations, buoy arrays, sensor networks

    Unlike STAC (designed for regular rasters with geotransform), ISTAC stores
    the complete geometry directly as WKB binary for arbitrary spatial footprints.

    Notes
    -----
    - Timestamps stored as Parquet TIMESTAMP with microsecond precision
    - int inputs are treated as epoch seconds and converted to microseconds
    - Centroid is always in EPSG:4326 regardless of source geometry CRS
    - For regular raster grids, use the STAC extension instead
    - WKB binary format for efficient storage and GeoParquet compatibility
    - time_middle is auto-computed when both start and end times exist
    """

    crs: str = Field(
        description="Coordinate Reference System in WKT2, EPSG code, or PROJ string (e.g., 'EPSG:4326')."
    )
    geometry: bytes = Field(
        description="Spatial footprint geometry in source CRS as WKB binary (Well-Known Binary format). Can be Point, LineString, Polygon, or MultiPolygon."
    )
    time_start: int = Field(
        description="Acquisition start timestamp as Datetime[μs] (microseconds since Unix epoch, timezone-naive UTC)."
    )
    time_end: int | None = Field(
        default=None,
        description="Acquisition end timestamp as Datetime[μs] (microseconds since Unix epoch, timezone-naive UTC). Must be ≥ time_start.",
    )
    time_middle: int | None = Field(
        default=None,
        description="Middle timestamp as Datetime[μs] (microseconds since Unix epoch, timezone-naive UTC). Auto-computed as (time_start + time_end) // 2.",
    )
    centroid: bytes | None = Field(
        default=None,
        description="Geometry centroid in EPSG:4326 as WKB binary. Auto-computed from geometry if not provided.",
    )
    check_antimeridian: bool = Field(
        default=False,
        description="Enable antimeridian (±180° longitude) crossing detection. Required for geometries spanning the dateline. Requires 'antimeridian' package.",
    )

    @pydantic.model_validator(mode="after")
    def check_times(self) -> "ISTAC":
        """Validate that time_start <= time_end if time_end is provided and convert to microseconds."""
        # Convert seconds to microseconds if needed
        # Assume input is in seconds (epoch), convert to microseconds
        self.time_start = int(self.time_start * 1_000_000)

        if self.time_end is not None:
            self.time_end = int(self.time_end * 1_000_000)

            if self.time_start > self.time_end:
                raise ValueError(
                    f"Invalid temporal interval: time_start ({self.time_start}) "
                    f"> time_end ({self.time_end})"
                )
        return self

    @pydantic.model_validator(mode="after")
    def populate_time_middle(self) -> "ISTAC":
        """Auto-populate time_middle when both time_start and time_end exist."""
        if self.time_end is not None and self.time_middle is None:
            self.time_middle = (self.time_start + self.time_end) // 2

        return self

    @pydantic.model_validator(mode="after")
    def populate_centroid(self) -> "ISTAC":
        """
        Auto-compute centroid in EPSG:4326 if not provided.

        If check_antimeridian=True, uses 'antimeridian' package to correctly
        handle geometries crossing ±180° longitude (e.g., Pacific swaths).
        """
        if self.centroid is None:
            # Load geometry from WKB
            geom = wkb_loads(self.geometry)

            # Compute centroid with optional antimeridian handling
            if self.check_antimeridian:
                if not HAS_ANTIMERIDIAN:
                    raise ImportError(
                        "check_antimeridian=True requires the 'antimeridian' package.\n"
                        "Install with: pip install antimeridian\n"
                    )
                # Use antimeridian-aware centroid calculation
                # Correctly handles geometries crossing ±180° longitude
                centroid_geom = antimeridian.centroid(geom)
            else:
                centroid_geom = geom.centroid

            # Transform to EPSG:4326 if needed
            if self.crs.upper() != "EPSG:4326":
                transformer = Transformer.from_crs(
                    CRS.from_string(self.crs), CRS.from_epsg(4326), always_xy=True
                )
                x, y = transformer.transform(centroid_geom.x, centroid_geom.y)
                centroid_geom = Point(x, y)

            # Store as WKB
            self.centroid = wkb_dumps(centroid_geom)

        return self

    def get_schema(self) -> pa.Schema:
        """Return the expected Arrow schema for this extension."""
        return pa.schema(
            [
                pa.field("istac:crs", pa.string()),
                pa.field("istac:geometry", pa.binary()),
                pa.field("istac:time_start", pa.timestamp("us", tz=None)),
                pa.field("istac:time_end", pa.timestamp("us", tz=None)),
                pa.field("istac:time_middle", pa.timestamp("us", tz=None)),
                pa.field("istac:centroid", pa.binary()),
            ]
        )

    def get_field_descriptions(self) -> dict[str, str]:
        """Return field descriptions for each field."""
        return {
            "istac:crs": "Coordinate reference system (WKT2, EPSG, or PROJ string)",
            "istac:geometry": "Spatial footprint geometry in source CRS (WKB binary format)",
            "istac:time_start": "Acquisition start timestamp (microseconds since Unix epoch, UTC)",
            "istac:time_end": "Acquisition end timestamp (microseconds since Unix epoch, UTC)",
            "istac:time_middle": "Midpoint between start and end timestamps (microseconds since Unix epoch, UTC)",
            "istac:centroid": "Geometry center point in EPSG:4326 (WKB binary format)",
        }

    def _compute(self, sample) -> pa.Table:
        """Actual computation logic - returns PyArrow Table."""
        data = {
            "istac:crs": [self.crs],
            "istac:geometry": [self.geometry],
            "istac:time_start": [self.time_start],
            "istac:time_end": [self.time_end],
            "istac:time_middle": [self.time_middle],
            "istac:centroid": [self.centroid],
        }

        return pa.Table.from_pydict(data, schema=self.get_schema())
