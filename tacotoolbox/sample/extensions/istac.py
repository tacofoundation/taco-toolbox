import polars as pl
import pydantic
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

    This extension exists to handle geospatial data that cannot be represented with
    a affine geotransform. Common use cases include:

    - **Satellite swaths**: CloudSat, CALIPSO, GPM orbital tracks with irregular coverage
    - **Flight paths**: Aircraft or drone trajectories with arbitrary geometries
    - **Vector data**: Polygons, lines, or points without underlying raster structure
    - **Irregular samplings**: Weather station networks, buoy arrays, sensor deployments

    Unlike the STAC extension (designed for regular rasters with geotransform),
    ISTAC stores the complete geometry directly as WKB binary, making it
    suitable for any arbitrary spatial footprint.

    Fields
    ------
    crs : str
        Coordinate reference system identifier (e.g., "EPSG:4326", PROJ string, or WKT).
        Most satellite swaths use EPSG:4326 (WGS84 lon/lat).
    geometry : bytes
        Complete spatial footprint as WKB (Well-Known Binary). Typically a Polygon or
        MultiPolygon representing the data coverage area.
    time_start : int
        Start time as seconds since Unix epoch (1970-01-01 00:00:00 UTC).
        Use `int(datetime.timestamp())` to convert from datetime objects.
    time_end : int | None
        Optional end time as seconds since Unix epoch. If provided, must be >= time_start.
        Useful for data representing a time interval (e.g., satellite pass duration).
    time_middle : int | None
        Automatically computed midpoint between `time_start` and `time_end`.
        Only populated when `time_end` is provided. Calculated as the integer
        average of the two timestamps.
    centroid : bytes | None
        Optional centroid point in EPSG:4326 as WKB binary. If not provided, it will
        be automatically computed from the geometry. Useful for quick spatial queries
        without loading full geometry.
    check_antimeridian : bool
        If True, detect and correctly handle geometries crossing the antimeridian
        (±180° longitude). Requires 'antimeridian' package. Default False (fast mode).

    Examples
    --------
    >>> from shapely.geometry import Polygon
    >>> from shapely.wkb import dumps as wkb_dumps
    >>> from datetime import datetime
    >>>
    >>> # CloudSat orbital swath polygon (default fast mode)
    >>> swath = Polygon([(-180, -60), (180, -60), (180, 60), (-180, 60)])
    >>> istac = ISTAC(
    ...     crs="EPSG:4326",
    ...     geometry=wkb_dumps(swath),
    ...     time_start=int(datetime(2025, 1, 15, 12, 30).timestamp()),
    ...     time_end=int(datetime(2025, 1, 15, 12, 45).timestamp())
    ... )
    >>> sample.extend_with(istac)
    >>>
    >>> # Swath crossing Pacific (antimeridian mode)
    >>> pacific_swath = Polygon([(170, -10), (-170, -10), (-170, 10), (170, 10)])
    >>> istac = ISTAC(
    ...     crs="EPSG:4326",
    ...     geometry=wkb_dumps(pacific_swath),
    ...     time_start=int(datetime(2025, 1, 15, 12, 30).timestamp()),
    ...     check_antimeridian=True  # Correct centroid for ±180° crossings
    ... )
    >>> sample.extend_with(istac)

    Notes
    -----
    - All timestamps are stored as int64 for efficiency and consistency with STAC
    - Centroid is always in EPSG:4326 regardless of source geometry CRS
    - For raster data with regular grids, use the STAC extension instead
    - WKB binary format is used for efficient storage and GeoParquet compatibility
    - `time_middle` is automatically computed when both start and end times exist
    - Set check_antimeridian=True for Pacific/Polar data (requires: pip install antimeridian)
    """

    crs: str
    geometry: bytes
    time_start: int
    time_end: int | None = None
    time_middle: int | None = None
    centroid: bytes | None = None
    check_antimeridian: bool = False

    @pydantic.model_validator(mode="after")
    def check_times(self) -> "ISTAC":
        """Validate that time_start <= time_end if time_end is provided."""
        if self.time_end is not None and self.time_start > self.time_end:
            raise ValueError(
                f"Invalid temporal interval: time_start ({self.time_start}) "
                f"> time_end ({self.time_end})"
            )
        return self

    @pydantic.model_validator(mode="after")
    def populate_time_middle(self) -> "ISTAC":
        """
        Auto-populate `time_middle` when both time_start and time_end exist.

        This validator runs after `check_times` to ensure the temporal interval
        is valid before computing the midpoint.
        """
        if self.time_end is not None and self.time_middle is None:
            self.time_middle = (self.time_start + self.time_end) // 2

        return self

    @pydantic.model_validator(mode="after")
    def populate_centroid(self) -> "ISTAC":
        """
        Auto-compute centroid in EPSG:4326 if not provided.

        Loads the geometry, computes its centroid, and reprojects to EPSG:4326
        if the source CRS is different.
        
        If check_antimeridian=True, uses the 'antimeridian' package to correctly
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
                        "Or set check_antimeridian=False to use fast mode (works for most geometries)."
                    )
                # Use antimeridian-aware centroid calculation
                # This correctly handles geometries crossing ±180° longitude
                centroid_geom = antimeridian.centroid(geom)
            else:
                # FAST PATH (default): Standard shapely centroid
                # Works correctly for 99% of geometries (those not crossing ±180°)
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

    def get_schema(self) -> dict[str, pl.DataType]:
        """Return the expected Polars schema for this extension."""
        return {
            "istac:crs": pl.Utf8(),
            "istac:geometry": pl.Binary(),
            "istac:time_start": pl.Int64(),
            "istac:time_end": pl.Int64(),
            "istac:time_middle": pl.Int64(),
            "istac:centroid": pl.Binary(),
        }

    def _compute(self, sample) -> pl.DataFrame:
        """Actual computation logic - only called when schema_only=False."""
        return pl.DataFrame(
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