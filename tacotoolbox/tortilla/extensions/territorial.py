import ee
from tqdm.auto import tqdm
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import polars as pl
import pydantic
from pydantic import Field
from shapely.wkb import loads as wkb_loads

from tacotoolbox.tortilla.datamodel import TortillaExtension
from tacotoolbox.tortilla.extensions import territorial_products
from tacotoolbox.tortilla.extensions import territorial_utils

TERRITORIAL_PRODUCTS : list[str] = [
    "elevation",
    "cisi",
    "precipitation",
    "temperature",
    "soil_clay",
    "soil_sand",
    "soil_carbon",
    "soil_bulk_density",
    "soil_ph",
    "gdp",
    "human_modification",
    "population",
    "admin_countries",
    "admin_states",
    "admin_districts"
]

class Territorial(TortillaExtension):
    """
    Territorial data enrichment extension for TORTILLA.
    
    Fetches geospatial, climatic, socioeconomic, and administrative data
    from Earth Engine for sample centroids using flexible variable selection.

    Features:
        - Flexible variable selection via 'variables' parameter
        - Spatial ordering (Morton) for improved EE cache locality  
        - Parallel processing with configurable batching
        - Support for multiple reducer types (mean, sum, mode)
        - Admin name resolution from local lookup tables
        - Retry logic for transient Earth Engine errors

    Example Usage:
        # All variables (default)
        territorial = Territorial(batch_size=100, max_concurrency=4)
        tortilla.extend_with(territorial)
        
        # Specific variables only
        territorial = Territorial(
            variables=['elevation', 'gdp', 'admin_countries', 'temp_jan'],
            scale_m=1000.0,
            batch_size=50
        )
        tortilla.extend_with(territorial)
        
        # Returns DataFrame with columns like:
        # territorial:elevation, territorial:gdp, territorial:admin_countries
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    # Private implementation constants
    _ADMIN_LAYERS_PKG = "tacotoolbox"

    # Public configuration parameters
    variables: list[str] | None = Field(
        None, 
        description="List of variable names to fetch (e.g., ['elevation', 'gdp', 'admin_countries']). If None, fetches all available variables."
    )
    scale_m: float = Field(
        5120.0,
        ge=1.0,
        description="Earth Engine reducer scale in meters. Smaller values = higher resolution but slower processing."
    )
    batch_size: int = Field(
        250,
        ge=1,
        description="Number of points per Earth Engine reduceRegions call. Larger batches = fewer API calls but higher memory usage."
    )
    max_concurrency: int = Field(
        8,
        ge=1, 
        description="Maximum number of concurrent Earth Engine requests. Higher = faster but may hit rate limits."
    )
    max_retries: int = Field(
        2, 
        ge=0, 
        description="Maximum retry attempts for failed Earth Engine requests."
    )
    retry_base_delay: float = Field(
        2.0, 
        ge=0.0, 
        description="Base delay in seconds for exponential backoff retry logic."
    )
    show_progress: bool = Field(
        True, 
        description="Whether to display progress bar during processing."
    )

    def get_schema(self) -> dict[str, pl.DataType]:
        """Return the expected schema for this extension."""
        active_products = self._get_active_products()
        schema = {}
        
        for product in active_products:
            var_name = product['name']
            column_name = f"territorial:{var_name}"
            
            # Admin variables are strings, others are numeric
            if var_name.startswith("admin"):
                schema[column_name] = pl.Utf8
            else:
                schema[column_name] = pl.Float32
                
        return schema

    def _get_active_products(self) -> list[dict]:
        """
        Get product configurations to process based on user selection.
        
        Returns:
            List of product dicts with 'name', 'image', and 'reducer' keys.
            If self.variables is None, returns all available products.
            Otherwise, filters products by user-specified variable names.
        """
        all_products = territorial_products.get_territorial_products()
        if self.variables is None:
            return all_products
        return [p for p in all_products if p['name'] in self.variables]

    def _extract_points(self, df: pl.DataFrame) -> list[tuple[int, float, float]]:
        """
        Extract coordinate points from DataFrame and sort spatially.
        
        Args:
            df: Input DataFrame with 'stac:centroid' WKB binary column
            
        Returns:
            List of tuples (row_index, longitude, latitude) sorted by Morton key
            for improved Earth Engine cache locality.
        """
        points = []
        for i, row in enumerate(df.iter_rows(named=True)):
            # Convert WKB binary to shapely geometry and extract coordinates
            geom = wkb_loads(row["stac:centroid"])
            points.append((i, float(geom.x), float(geom.y)))
        
        # Sort spatially using Morton (Z-order) key for better EE cache locality
        points.sort(key=lambda t: territorial_utils.morton_key(t[1], t[2], bits=24))
        return points

    def _group_products_by_reducer(self, products: list[dict]) -> dict:
        """
        Group products by their Earth Engine reducer type.
        
        Args:
            products: List of product configurations
            
        Returns:
            Dict mapping ee.Reducer objects to lists of products that use that reducer.
            This allows processing all products with the same reducer in a single EE call.
        """
        groups = {}
        for product in products:
            reducer = product['reducer']
            if reducer not in groups:
                groups[reducer] = []
            groups[reducer].append(product)
        return groups

    def _reduce_chunk(self, chunk: list[tuple[int, float, float]], reducer_groups: dict) -> list[dict]:
        """
        Process a chunk of coordinate points with Earth Engine.
        
        Args:
            chunk: List of (row_index, lon, lat) tuples to process
            reducer_groups: Dict mapping reducers to their associated products
            
        Returns:
            List of property dictionaries, one per input point, containing
            all requested variable values plus the original row index.
        """
        # Create Earth Engine FeatureCollection from coordinate points
        fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point(lon, lat), {"idx": i})
            for i, lon, lat in chunk
        ])
        
        # Process each reducer type separately
        all_results = []
        for reducer, products in reducer_groups.items():
            # Combine all images for this reducer into a single multi-band image
            combined_image = ee.Image([p['image'] for p in products])
            
            # Apply the reducer to all points
            data = combined_image.reduceRegions(
                collection=fc,
                reducer=reducer,
                scale=self.scale_m,
            ).getInfo()
            all_results.append(data["features"])

        # Merge results from different reducers by feature index
        merged = []
        for i in range(len(all_results[0])):
            props = {}
            # Combine properties from all reducer results for this point
            for feature_list in all_results:
                props.update(feature_list[i].get("properties", {}))
            merged.append(props)
        return merged

    def _reduce_chunk_with_retry(self, chunk: list[tuple[int, float, float]], reducer_groups: dict) -> list[dict]:
        """
        Reduce chunk with exponential backoff retry logic.
        
        Args:
            chunk: Coordinate points to process
            reducer_groups: Reducer groupings
            
        Returns:
            Processed results from Earth Engine
            
        Raises:
            Exception: If all retry attempts fail
        """
        for attempt in range(self.max_retries + 1):
            try:
                return self._reduce_chunk(chunk, reducer_groups)
            except Exception:
                if attempt == self.max_retries:
                    raise
                # Exponential backoff: 2^attempt * base_delay seconds
                time.sleep(self.retry_base_delay * (2 ** attempt))

    def _process_admin_variables(self, df_data: pl.DataFrame, admin_vars: list[str]) -> pl.DataFrame:
        """
        Add human-readable admin names to data for administrative variables.
        
        Args:
            df_data: DataFrame containing admin codes from Earth Engine
            admin_vars: List of admin variable names to process
            
        Returns:
            Updated DataFrame with additional {var_name}_name columns containing
            human-readable place names (e.g., "United States", "California").
        """
        # Map variable names to administrative levels
        admin_level_map = {
            "admin_countries": 0,  # Country level
            "admin_states": 1,     # State/province level  
            "admin_districts": 2   # District/county level
        }
        
        for admin_var in admin_vars:
            level = admin_level_map.get(admin_var)
            if level is None:
                continue
            
            # Load lookup table for this admin level
            df_admin = territorial_utils.load_admin_layer(level, self._ADMIN_LAYERS_PKG)
            admin_code_col = f"admin_code{level}"
            
            # Create mapping from numeric codes to human-readable names
            admin_map = dict(df_admin.select([admin_code_col, "name"]).iter_rows())
            
            # Add name column using the lookup mapping
            df_data = df_data.with_columns([
                pl.col(admin_var).cast(pl.Int64, strict=False).map_elements(
                    lambda x: admin_map.get(x, "Ocean/Sea/Lakes") if x is not None else "Ocean/Sea/Lakes",
                    return_dtype=pl.Utf8
                ).alias(f"{admin_var}_name")
            ])
        
        return df_data

    def _build_result_dataframe(self, entries: dict, products: list[dict]) -> pl.DataFrame:
        """
        Build final result DataFrame with proper schema and column naming.
        
        Args:
            entries: Dict mapping variable names to lists of values
            products: List of product configurations
            
        Returns:
            DataFrame with territorial:* columns and appropriate data types.
            Admin variables get Utf8 type, numeric variables get Float32.
        """
        result_data = {}
        schema = {}
        
        for product in products:
            var_name = product['name']
            column_name = f"territorial:{var_name}"
            result_data[column_name] = entries[var_name]
            
            # Set appropriate schema
            if var_name.startswith("admin"):
                schema[column_name] = pl.Utf8
            else:
                schema[column_name] = pl.Float32

        return pl.DataFrame(result_data, schema=schema)

    def _compute(self, tortilla: 'Tortilla') -> pl.DataFrame:
        """
        Process Tortilla and return territorial enrichment.
        
        Args:
            tortilla: Input Tortilla object containing STAC data
            
        Returns:
            DataFrame with territorial:* columns aligned with input DataFrame.
            Contains the same number of rows as input, with new columns for
            each requested territorial variable.
            
        Raises:
            ImportError: If earthengine-api not available
            Exception: Various Earth Engine or processing errors
            
        Example:
            >>> territorial = Territorial(variables=['elevation', 'gdp'])
            >>> result = territorial._compute(tortilla)
            >>> print(result.columns)
            ['territorial:elevation', 'territorial:gdp']
        """
        # Get DataFrame from tortilla
        df = tortilla._metadata_df
        
        # Step 1: Determine which products to process
        active_products = self._get_active_products()
        if not active_products:
            # Return empty DataFrame with same row count if no products selected
            return pl.DataFrame({"__empty__": [None] * len(df)}).drop("__empty__")
        
        # Step 2: Extract and spatially sort coordinate points
        points = self._extract_points(df)
        
        # Step 3: Initialize result storage
        n = len(df)
        entries = {product['name']: [None] * n for product in active_products}
        
        # Step 4: Group products by reducer type for efficient processing
        reducer_groups = self._group_products_by_reducer(active_products)

        # Step 5: Process points in parallel batches
        rows = []
        total_chunks = math.ceil(len(points) / self.batch_size)
        
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            # Submit all chunks for parallel processing
            futures = {
                executor.submit(self._reduce_chunk_with_retry, chunk, reducer_groups): chunk
                for chunk in territorial_utils._chunks(points, self.batch_size)
            }
            
            # Collect results with progress bar
            with tqdm(total=total_chunks, desc="Territorial", disable=not self.show_progress) as pbar:
                for future in as_completed(futures):
                    rows.extend(future.result())
                    pbar.update(1)

        # Step 6: Convert results to DataFrame and handle missing values
        df_data = pl.DataFrame(rows).fill_null(0)

        # Step 7: Process administrative variables to add human-readable names
        admin_vars = [p['name'] for p in active_products if p['name'].startswith("admin")]
        if admin_vars:
            df_data = self._process_admin_variables(df_data, admin_vars)

        # Step 8: Populate final entries from Earth Engine results
        for row in df_data.iter_rows(named=True):
            idx = int(row["idx"])  # Original DataFrame row index
            
            for product in active_products:
                var_name = product['name']
                if var_name.startswith("admin"):
                    # Use human-readable name for admin variables
                    name_col = f"{var_name}_name"
                    entries[var_name][idx] = str(row.get(name_col, "Unknown"))
                else:
                    # Convert numeric values, handling None/NaN
                    val = row.get(var_name)
                    entries[var_name][idx] = float(val) if val is not None else None

        # Step 9: Build and return final result DataFrame
        return self._build_result_dataframe(entries, active_products)


if __name__ == "__main__":
    """Test script demonstrating Territorial extension usage."""
    import random
    from shapely.geometry import Point
    from shapely.wkb import dumps as wkb_dumps

    # Initialize Earth Engine
    ee.Initialize()

    # Create test data with random US coordinates
    samples = []
    random.seed(42)
    for i in range(10):
        lon = random.uniform(-125.0, -66.0)  # US longitude range
        lat = random.uniform(24.0, 49.0)     # US latitude range
        point = Point(lon, lat)
        samples.append({
            "id": f"sample_{i}",
            "stac:centroid": wkb_dumps(point),
        })
    
    # Create test DataFrame
    df_samples = pl.DataFrame(samples, schema={
        "id": pl.Utf8,
        "stac:centroid": pl.Binary
    })

    # Test territorial extension with subset of variables
    print("Testing Territorial extension...")
    territorial = Territorial(
        batch_size=5,
        show_progress=True
    )
    
    # Create mock tortilla object for testing
    class MockTortilla:
        def __init__(self, df):
            self._metadata_df = df
    
    mock_tortilla = MockTortilla(df_samples)
    result = territorial._compute(mock_tortilla)
    
    print(f"Input shape: {df_samples.shape}")
    print(f"Result shape: {result.shape}")
    print(f"Result columns: {result.columns}")
    print("\nSample results:")
    print(result.head())