"""
GeoEnrich extension for Tortilla using Earth Engine.

Enriches samples with geospatial data from Google Earth Engine:
- Physical: elevation, topographic complexity
- Climate: precipitation, temperature
- Soil: clay, sand, carbon, bulk density, pH
- Socioeconomic: GDP, population density, human modification index
- Administrative: countries, states, districts (human-readable names)

Fetches data at sample centroids using Earth Engine reducers.
Uses spatial sorting (Morton encoding) for improved EE cache locality.

Exports to Polars DataFrame:
- geoenrich:elevation: Float32
- geoenrich:cisi: Float32
- geoenrich:precipitation: Float32
- geoenrich:temperature: Float32
- geoenrich:soil_clay: Float32
- geoenrich:soil_sand: Float32
- geoenrich:soil_carbon: Float32
- geoenrich:soil_bulk_density: Float32
- geoenrich:soil_ph: Float32
- geoenrich:gdp: Float32
- geoenrich:human_modification: Float32
- geoenrich:population: Float32
- geoenrich:admin_countries: String
- geoenrich:admin_states: String
- geoenrich:admin_districts: String

Requirements:
- earthengine-api: pip install earthengine-api
- earthengine authenticate (one-time setup)
"""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from importlib.resources import as_file, files
from typing import TYPE_CHECKING, Any, cast

import polars as pl
import pydantic
from pydantic import Field
from shapely.geometry import Point
from shapely.wkb import loads as wkb_loads
from tqdm import tqdm

from tacotoolbox.tortilla.datamodel import TortillaExtension

if TYPE_CHECKING:
    from tacotoolbox.tortilla.datamodel import Tortilla


# ============================================================================
# PRODUCT CONFIGURATION
# ============================================================================
#
# To add a new product, just add ONE line here. Everything else is automatic.
#
# Format: (name, path, reducer_type, band, collection_type, unmask_value, dtype)
#   - name: Variable name (will be used as column name with "geoenrich:" prefix)
#   - path: Earth Engine asset path
#   - reducer_type: "mean", "mode", or "sum"
#   - band: Band name to select (None for single-band or all bands)
#   - collection_type: "Image" or "ImageCollection"
#   - unmask_value: Value for masked pixels (0 for most, 65535 for admin)
#   - dtype: Polars data type (pl.Float32 for numeric, pl.Utf8 for text)
#
# Example: Add new product
#   ("soil_moisture", "NASA/SMAP/SPL4SMGP/007", "mean", "sm_surface", "ImageCollection", 0, pl.Float32),

PRODUCT_CONFIGS = [
    # Physical/topographic
    (
        "elevation",
        "projects/sat-io/open-datasets/GLO-30",
        "mean",
        None,
        "ImageCollection",
        0,
        pl.Float32,
    ),
    (
        "cisi",
        "projects/sat-io/open-datasets/CISI/global_CISI",
        "mean",
        None,
        "Image",
        0,
        pl.Float32,
    ),
    # Climate
    (
        "precipitation",
        "projects/ee-csaybar-real/assets/precipitation",
        "mean",
        None,
        "Image",
        0,
        pl.Float32,
    ),
    (
        "temperature",
        "projects/ee-csaybar-real/assets/temperature",
        "mean",
        None,
        "Image",
        0,
        pl.Float32,
    ),
    # Soil (OpenLandMap)
    (
        "soil_clay",
        "OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02",
        "mean",
        "b0",
        "Image",
        0,
        pl.Float32,
    ),
    (
        "soil_sand",
        "OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02",
        "mean",
        "b0",
        "Image",
        0,
        pl.Float32,
    ),
    (
        "soil_carbon",
        "OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02",
        "mean",
        "b0",
        "Image",
        0,
        pl.Float32,
    ),
    (
        "soil_bulk_density",
        "OpenLandMap/SOL/SOL_BULKDENS-FINEEARTH_USDA-4A1H_M/v02",
        "mean",
        "b0",
        "Image",
        0,
        pl.Float32,
    ),
    (
        "soil_ph",
        "OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02",
        "mean",
        "b0",
        "Image",
        0,
        pl.Float32,
    ),
    # Socioeconomic
    (
        "gdp",
        "projects/sat-io/open-datasets/GRIDDED_HDI_GDP/total_gdp_perCapita_1990_2022_5arcmin",
        "mean",
        "PPP_2022",
        "Image",
        0,
        pl.Float32,
    ),
    (
        "human_modification",
        "projects/sat-io/open-datasets/GHM/HM_1990_2020_OVERALL_300M",
        "mean",
        "constant",
        "ImageCollection",
        0,
        pl.Float32,
    ),
    (
        "population",
        "projects/sat-io/open-datasets/hrsl/hrslpop",
        "mean",
        None,
        "ImageCollection",
        0,
        pl.Float32,
    ),
    # Administrative (always last) - Utf8 because we resolve to human-readable names
    (
        "admin_countries",
        "projects/ee-csaybar-real/assets/admin0",
        "mode",
        None,
        "Image",
        65535,
        pl.Utf8,
    ),
    (
        "admin_states",
        "projects/ee-csaybar-real/assets/admin1",
        "mode",
        None,
        "Image",
        65535,
        pl.Utf8,
    ),
    (
        "admin_districts",
        "projects/ee-csaybar-real/assets/admin2",
        "mode",
        None,
        "Image",
        65535,
        pl.Utf8,
    ),
]

# Auto-generated schema from PRODUCT_CONFIGS (never edit this manually)
PRODUCT_SCHEMA: dict[str, pl.DataType] = {
    name: dtype()
    for name, _path, _reducer, _band, _coll_type, _unmask, dtype in PRODUCT_CONFIGS
}


# ============================================================================
# EARTH ENGINE UTILITIES
# ============================================================================


def _import_earth_engine() -> Any:
    """Lazy import of Earth Engine with helpful error message."""
    try:
        import ee

    except ImportError as e:
        raise ImportError(
            "Google Earth Engine API is required for the GeoEnrich extension.\n"
            "Install with: pip install earthengine-api\n"
            "Then authenticate: earthengine authenticate"
        ) from e
    else:
        return ee


def get_geoenrich_products() -> list[dict[str, Any]]:
    """
    Get all available Earth Engine products for geospatial enrichment.

    Automatically generated from PRODUCT_CONFIGS.
    """
    ee = _import_earth_engine()

    # Reducer mapping
    reducer_map = {
        "mean": ee.Reducer.mean(),
        "mode": ee.Reducer.mode(),
        "sum": ee.Reducer.sum(),
    }

    products = []
    for (
        name,
        path,
        reducer_type,
        band,
        collection_type,
        unmask_value,
        _dtype,
    ) in PRODUCT_CONFIGS:
        # Load image or collection
        if collection_type == "ImageCollection":
            image = ee.ImageCollection(path).mosaic().unmask(unmask_value)
        else:
            image = ee.Image(path).unmask(unmask_value)

        # Select band if specified
        if band:
            image = image.select(band)

        # Rename to product name
        image = image.rename(name)

        products.append(
            {"name": name, "image": image, "reducer": reducer_map[reducer_type]}
        )

    return products


# ============================================================================
# SPATIAL UTILITIES
# ============================================================================


def morton_key(lon: float, lat: float, bits: int = 24) -> int:
    """
    Generate Morton (Z-order) key for spatial coordinates.

    Interleaves longitude and latitude bits to create a single integer key
    that preserves spatial locality. Points close in 2D space have similar
    Morton keys, improving Earth Engine cache hits.
    """
    # Normalize to [0, 1]
    x = (lon + 180.0) / 360.0
    y = (lat + 90.0) / 180.0

    # Quantize to integer range
    maxv = (1 << bits) - 1
    xi = max(0, min(maxv, int(x * maxv)))
    yi = max(0, min(maxv, int(y * maxv)))

    def _part1by1(v: int) -> int:
        """Interleave bits by inserting zeros between them."""
        v &= 0x00000000FFFFFFFF
        v = (v | (v << 16)) & 0x0000FFFF0000FFFF
        v = (v | (v << 8)) & 0x00FF00FF00FF00FF
        v = (v | (v << 4)) & 0x0F0F0F0F0F0F0F0F
        v = (v | (v << 2)) & 0x3333333333333333
        v = (v | (v << 1)) & 0x5555555555555555
        return v

    return (_part1by1(xi) << 1) | _part1by1(yi)


def _chunks(seq: list, size: int) -> Any:
    """Yield consecutive chunks from sequence of specified size."""
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


# ============================================================================
# ADMIN RESOLUTION
# ============================================================================


@lru_cache(maxsize=3)
def _load_admin_layer(level: int) -> pl.DataFrame:
    """Load admin lookup table from embedded parquet file (cached)."""
    base = files("tacotoolbox").joinpath("tortilla/data/admin/")
    traversable = base / f"admin{level}.parquet"

    with as_file(traversable) as path:
        return pl.read_parquet(path)


def resolve_admin_names(df: pl.DataFrame, admin_vars: list[str]) -> pl.DataFrame:
    """
    Replace admin code columns with human-readable names using efficient joins.

    This replaces the old map_elements approach which was 10-100x slower.
    """
    # Map admin variable names to levels
    admin_level_map = {
        "admin_countries": 0,
        "admin_states": 1,
        "admin_districts": 2,
    }

    for admin_var in admin_vars:
        level = admin_level_map.get(admin_var)
        if level is None:
            continue

        # Load lookup table (cached via @lru_cache)
        df_admin = _load_admin_layer(level)
        admin_code_col = f"admin_code{level}"

        # Cast to Int64 for join compatibility
        df = df.with_columns(pl.col(admin_var).cast(pl.Int64, strict=False))

        # LEFT JOIN to get human-readable names
        df = df.join(
            df_admin.select([admin_code_col, "name"]),
            left_on=admin_var,
            right_on=admin_code_col,
            how="left",
        )

        # Replace code column with name, fill nulls for ocean/sea
        df = df.drop(admin_var).rename({"name": admin_var})
        df = df.with_columns(pl.col(admin_var).fill_null("Ocean/Sea/Lakes"))

    return df


# ============================================================================
# MAIN EXTENSION CLASS
# ============================================================================


class GeoEnrich(TortillaExtension):
    """
    Geographic enrichment extension for Tortilla.

    Fetches geospatial, climatic, socioeconomic, and administrative data
    from Earth Engine for sample centroids.

    Features:
        - Flexible variable selection via 'variables' parameter
        - Spatial ordering (Morton) for improved EE cache locality
        - Parallel processing with configurable batching
        - Fast admin name resolution using vectorized joins
        - Progress bars via tqdm

    Requirements:
        - earthengine-api: pip install earthengine-api
        - tqdm: pip install tqdm
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    # Public configuration parameters
    variables: list[str] | None = Field(
        default=None,
        description="List of variable names to fetch. If None, fetches all 15 available variables. See PRODUCT_CONFIGS for complete list.",
    )
    scale_m: float = Field(
        default=5120.0,
        ge=1.0,
        description="Earth Engine reducer spatial resolution in meters. Smaller values = higher resolution but slower queries and higher quota usage. Default 5120m (~5km).",
    )
    batch_size: int = Field(
        default=250,
        ge=1,
        description="Number of sample points per Earth Engine reduceRegions API call. Larger batches are faster but may hit EE memory limits. Default 250.",
    )
    max_concurrency: int = Field(
        default=8,
        ge=1,
        description="Maximum concurrent Earth Engine API requests via ThreadPoolExecutor. Higher values speed up processing but may hit EE rate limits. 8 is safe for most accounts.",
    )
    show_progress: bool = Field(
        default=True,
        description="Display tqdm progress bar during Earth Engine processing.",
    )

    def model_post_init(self, __context: Any) -> None:
        """Validate variables after model initialization."""
        if self.variables is not None:
            invalid = set(self.variables) - set(PRODUCT_SCHEMA.keys())
            if invalid:
                raise ValueError(
                    f"Invalid variables: {sorted(invalid)}\n"
                    f"Valid options: {sorted(PRODUCT_SCHEMA.keys())}"
                )

    def get_schema(self) -> dict[str, pl.DataType]:
        """Return the expected schema for this extension."""
        active_vars = self.variables or list(PRODUCT_SCHEMA.keys())
        return {f"geoenrich:{var}": PRODUCT_SCHEMA[var] for var in active_vars}

    def get_field_descriptions(self) -> dict[str, str]:
        """Return field descriptions for each field."""
        all_descriptions = {
            "geoenrich:elevation": "Mean elevation in meters (GLO-30 DEM)",
            "geoenrich:cisi": "Mean of harmonized global critical infrastructure & index (CISI)",
            "geoenrich:precipitation": "Mean annual precipitation in mm estimated from GPM data",
            "geoenrich:temperature": "Mean annual temperature in °C estimated from MODIS LST data",
            "geoenrich:soil_clay": "Clay content fraction (0-1) at surface layer, using the OpenLandMap dataset",
            "geoenrich:soil_sand": "Sand content fraction (0-1) at surface layer, using the OpenLandMap dataset",
            "geoenrich:soil_carbon": "Organic carbon content in g/kg at surface layer, using the OpenLandMap dataset",
            "geoenrich:soil_bulk_density": "Bulk density in kg/m³ at surface layer, using the OpenLandMap dataset",
            "geoenrich:soil_ph": "Soil pH (H2O) at surface layer, using the OpenLandMap dataset",
            "geoenrich:gdp": "GDP per capita in USD (PPP 2022), Kummu et al. 2025",
            "geoenrich:human_modification": "Human modification index v3 1990-2020, Theobald et al. 2024",
            "geoenrich:population": "Population density from HRSL. Facebook High Resolution Settlement Layer",
            "geoenrich:admin_countries": "Country name at centroid location",
            "geoenrich:admin_states": "State/province name at centroid location",
            "geoenrich:admin_districts": "District/county name at centroid location",
        }

        # Return only active variables
        active_vars = self.variables or list(PRODUCT_SCHEMA.keys())
        return {
            key: all_descriptions[key]
            for key in all_descriptions
            if key.replace("geoenrich:", "") in active_vars
        }

    def _get_active_products(self) -> list[dict]:
        """Get product configurations to process based on user selection."""
        all_products = get_geoenrich_products()
        if self.variables is None:
            return all_products
        return [p for p in all_products if p["name"] in self.variables]

    def _extract_points(self, df: pl.DataFrame) -> list[tuple[int, float, float]]:
        """Extract coordinate points from DataFrame and sort spatially."""
        points = []
        for i, row in enumerate(df.iter_rows(named=True)):
            # Cast to Point to access .x and .y attributes safely
            geom = cast(Point, wkb_loads(row["stac:centroid"]))
            points.append((i, float(geom.x), float(geom.y)))

        # Sort spatially using Morton key for better EE cache locality
        points.sort(key=lambda t: morton_key(t[1], t[2], bits=24))
        return points

    def _group_products_by_reducer(self, products: list[dict]) -> dict[Any, list[dict]]:
        """
        Group products by their Earth Engine reducer type.

        Allows processing all products with the same reducer in a single EE call.
        """
        groups = defaultdict(list)
        for product in products:
            groups[product["reducer"]].append(product)
        return dict(groups)

    def _fix_mode_columns(self, df: pl.DataFrame, products: list[dict]) -> pl.DataFrame:
        """
        Fix Earth Engine mode() reducer column naming issue.

        When using mode() reducer with multiple bands, Earth Engine returns
        columns named "mode", "mode_1", "mode_2" instead of the band names.
        This function renames them back to expected product names.
        """
        mode_cols = ["mode"] + [f"mode_{i}" for i in range(1, len(products))]
        rename_map = {
            mode_col: product["name"]
            for mode_col, product in zip(mode_cols, products, strict=False)
            if mode_col in df.columns
        }
        return df.rename(rename_map)

    def _reduce_chunk(
        self,
        chunk: list[tuple[int, float, float]],
        reducer_groups: dict[Any, list[dict]],
        ee: Any,
    ) -> pl.DataFrame:
        """Process a chunk of coordinate points with Earth Engine."""
        # Create Earth Engine FeatureCollection
        fc = ee.FeatureCollection(
            [
                ee.Feature(ee.Geometry.Point(lon, lat), {"idx": idx})
                for idx, lon, lat in chunk
            ]
        )

        # Process each reducer type separately
        all_dataframes = []

        for reducer, products in reducer_groups.items():
            # Combine all images for this reducer
            combined_image = ee.Image([p["image"] for p in products])

            # Apply reducer to all points
            data = combined_image.reduceRegions(
                collection=fc,
                reducer=reducer,
                scale=self.scale_m,
            ).getInfo()

            # Convert to DataFrame
            rows = [f["properties"] for f in data["features"]]
            df_chunk = pl.DataFrame(rows)

            # Fix mode() reducer column names if needed
            if "Reducer.mode" in str(reducer):
                df_chunk = self._fix_mode_columns(df_chunk, products)

            all_dataframes.append(df_chunk)

        # Merge all reducer results by idx
        result = all_dataframes[0]
        for df in all_dataframes[1:]:
            result = result.join(df, on="idx", how="inner")

        return result

    def _process_batches(
        self, points: list[tuple[int, float, float]], products: list[dict]
    ) -> pl.DataFrame:
        """Process all point batches in parallel and return consolidated DataFrame."""
        ee = _import_earth_engine()
        reducer_groups = self._group_products_by_reducer(products)
        chunks = list(_chunks(points, self.batch_size))

        # Get expected schema for casting
        product_names = [p["name"] for p in products]
        expected_schema = {name: PRODUCT_SCHEMA[name] for name in product_names}

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            futures = [
                executor.submit(self._reduce_chunk, chunk, reducer_groups, ee)
                for chunk in chunks
            ]

            results = []
            with tqdm(
                total=len(chunks),
                desc="GeoEnrich",
                disable=not self.show_progress,
            ) as pbar:
                for future in as_completed(futures):
                    df_chunk = future.result()

                    # Cast to expected schema BEFORE appending
                    df_chunk = df_chunk.select(
                        ["idx"]
                        + [
                            pl.col(name).cast(expected_schema[name], strict=False)
                            for name in product_names
                        ]
                    )

                    results.append(df_chunk)
                    pbar.update(1)

        return pl.concat(results, how="vertical")

    def _compute(self, tortilla: "Tortilla") -> pl.DataFrame:
        """Process Tortilla and return geographic enrichment."""
        df = tortilla._metadata_df

        # Get active products
        active_products = self._get_active_products()
        if not active_products:
            # Return empty DataFrame with same row count
            return pl.DataFrame({"__empty__": [None] * len(df)}).drop("__empty__")

        # Extract and spatially sort points
        points = self._extract_points(df)

        # Process in parallel batches
        raw_results = self._process_batches(points, active_products)

        # Fill nulls with 0 for numeric columns
        raw_results = raw_results.fill_null(0)

        # Resolve admin names if any admin variables
        admin_vars = [
            p["name"]
            for p in active_products
            if p["name"] in ["admin_countries", "admin_states", "admin_districts"]
        ]
        if admin_vars:
            raw_results = resolve_admin_names(raw_results, admin_vars)

        # Sort by idx to match original order, then drop idx
        raw_results = raw_results.sort("idx").drop("idx")

        # Add "geoenrich:" prefix
        product_names = [p["name"] for p in active_products]
        return raw_results.select(
            [pl.col(name).alias(f"geoenrich:{name}") for name in product_names]
        )
