import polars as pl
import pydantic
from pyproj import CRS, Transformer
from shapely.geometry import Point
from shapely.wkb import dumps as wkb_dumps
from shapely.wkb import loads as wkb_loads

from tacotoolbox.sample.datamodel import SampleExtension


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
    centroid : bytes | None
        Optional centroid point in EPSG:4326 as WKB binary. If not provided, it will
        be automatically computed from the geometry. Useful for quick spatial queries
        without loading full geometry.

    Examples
    --------
    >>> from shapely.geometry import Polygon
    >>> from shapely.wkb import dumps as wkb_dumps
    >>> from datetime import datetime
    >>>
    >>> # CloudSat orbital swath polygon
    >>> swath = Polygon([(-180, -60), (180, -60), (180, 60), (-180, 60)])
    >>>
    >>> istac = ISTAC(
    ...     crs="EPSG:4326",
    ...     geometry=wkb_dumps(swath),
    ...     time_start=int(datetime(2025, 1, 15, 12, 30).timestamp()),
    ...     time_end=int(datetime(2025, 1, 15, 12, 45).timestamp())
    ... )
    >>>
    >>> sample.extend_with(istac)

    Notes
    -----
    - All timestamps are stored as int64 for efficiency and consistency with STAC
    - Centroid is always in EPSG:4326 regardless of source geometry CRS
    - For raster data with regular grids, use the STAC extension instead
    - WKB binary format is used for efficient storage and GeoParquet compatibility
    """

    crs: str
    geometry: bytes
    time_start: int
    time_end: int | None = None
    centroid: bytes | None = None

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
    def populate_centroid(self) -> "ISTAC":
        """
        Auto-compute centroid in EPSG:4326 if not provided.

        Loads the geometry, computes its centroid, and reprojects to EPSG:4326
        if the source CRS is different.
        """
        if self.centroid is None:
            # Load geometry from WKB
            geom = wkb_loads(self.geometry)

            # Compute centroid in source CRS
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
            "istac:centroid": pl.Binary(),
        }

    def _compute(self, sample) -> pl.DataFrame:
        """Actual computation logic - only called when return_none=False."""
        return pl.DataFrame(
            {
                "istac:crs": [self.crs],
                "istac:geometry": [self.geometry],
                "istac:time_start": [self.time_start],
                "istac:time_end": [self.time_end],
                "istac:centroid": [self.centroid],
            },
            schema=self.get_schema(),
        )
