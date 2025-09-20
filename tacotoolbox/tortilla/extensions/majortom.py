import math
from typing import Iterable

import numpy as np
import polars as pl
import pydantic
from pydantic import Field, PrivateAttr
from shapely.wkb import loads as wkb_loads

# Import the ABC interface
from tacotoolbox.tortilla.datamodel import TortillaExtension

class MajorTOM(TortillaExtension):
    """
    MajorTOM-like spherical grid with ~`dist_km` spacing.

    API Usage:
        majortom = MajorTOM(dist_km=100)
        tortilla.extend_with(majortom)  # Now compatible with Tortilla ABC interface

    Matching legacy semantics:
        - Equator row is computed with searchsorted(..., side="left").
        - Column labels are generated BEFORE longitude filtering and then sliced,
          so R/L counts remain relative to the 0° meridian of the full row.
    """

    # ---- Inputs / knobs -----------------------------------------------------
    dist_km: float = Field(100, description="Target spacing (km) along meridians and parallels.")
    latitude_range: tuple[float, float] = Field(
        (-85.0, 85.0), description="Inclusive latitude limits for grid generation (deg)."
    )
    longitude_range: tuple[float, float] = Field(
        (-180.0, 180.0), description="Inclusive longitude limits used for filtering columns (deg)."
    )
    sep: str = Field("_", description="Separator for row/col codes")

    # ---- Internal derived state (private) -----------------------------------
    _lats: np.ndarray = PrivateAttr(default_factory=lambda: np.empty(0)) # sorted rings (deg)
    _row_labels: np.ndarray = PrivateAttr(default_factory=lambda: np.empty(0, dtype=object))
    _zero_row_idx: int = PrivateAttr(default=0)

    _row_lons: list[np.ndarray] = PrivateAttr(default_factory=list) # per-row sorted lons (filtered)
    _row_col_labels: list[np.ndarray] = PrivateAttr(default_factory=list) # per-row labels (filtered)

    # Constants
    R_EQUATOR_KM: float = 6378.137

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    # ======================================================================
    #                              BUILD GRID
    # ======================================================================
    @pydantic.model_validator(mode="after")
    def _build(self):
        """Compute latitudinal rings and per-row longitude subdivisions."""
        # --- 1) Latitudinal rings (south→north), ~dist_km spacing pole→pole
        # Match the original Grid class exactly
        arc_pole_to_pole_km = math.pi * self.R_EQUATOR_KM
        num_divisions_in_hemisphere = math.ceil(arc_pole_to_pole_km / self.dist_km)

        # Create latitudes exactly like the original Grid class
        lats_all = np.linspace(-90, 90, num_divisions_in_hemisphere + 1)[:-1]
        lats_all = np.mod(lats_all, 180) - 90
        lats_all = np.sort(lats_all)

        # Legacy equator row: side="left" (first index where lat >= 0)
        zero_row = int(np.searchsorted(lats_all, 0.0, side="left"))
        rows_all = np.empty_like(lats_all, dtype=object)
        rows_all[zero_row:] = [f"{i}U" for i in range(len(lats_all) - zero_row)]
        rows_all[:zero_row] = [f"{abs(i - zero_row)}D" for i in range(zero_row)]

        # Filter by latitude_range
        lat_lo, lat_hi = self.latitude_range
        lat_mask = (lats_all >= lat_lo) & (lats_all <= lat_hi)
        self._lats = lats_all[lat_mask]
        self._row_labels = rows_all[lat_mask]

        # Equator index within filtered view (clamped if filtered out)
        zero_in_filtered = np.searchsorted(self._lats, 0.0, side="left")
        self._zero_row_idx = int(np.clip(zero_in_filtered, 0, max(0, len(self._lats) - 1)))

        # --- 2) Per-row longitudes and labels (generate, then filter) -------
        lon_lo, lon_hi = self.longitude_range
        row_lons_filtered: list[np.ndarray] = []
        row_labels_filtered: list[np.ndarray] = []

        for lat in self._lats:
            circ_km = 2.0 * math.pi * self.R_EQUATOR_KM * math.cos(math.radians(lat))
            n_cols = max(1, int(math.ceil(circ_km / self.dist_km)))

            # Full longitudes for the row: [-180, 180), step = 360/n_cols
            # Match the original Grid class exactly
            lons_full = np.linspace(-180.0, 180.0, n_cols + 1)[:-1]
            lons_full = np.mod(lons_full, 360) - 180
            lons_full = np.sort(lons_full)

            # Legacy column labels built against ZERO at exact 0°, if present
            zero_hits = np.where(lons_full == 0.0)[0]
            if zero_hits.size > 0:
                zero_col_full = int(zero_hits[0])
            else:
                # Legacy code assumes 0 exists; to avoid a crash, use nearest-to-0
                zero_col_full = int(np.argmin(np.abs(lons_full)))

            cols_full = np.empty(lons_full.size, dtype=object)
            cols_full[zero_col_full:] = [f"{i}R" for i in range(lons_full.size - zero_col_full)]
            cols_full[:zero_col_full] = [f"{abs(i - zero_col_full)}L" for i in range(zero_col_full)]

            # Now apply longitude_range filter to BOTH arrays (preserving labels)
            if lon_lo > -180.0 or lon_hi < 180.0:
                mask = (lons_full >= lon_lo) & (lons_full <= lon_hi)
                lons = lons_full[mask]
                cols = cols_full[mask]
            else:
                lons = lons_full
                cols = cols_full

            if lons.size == 0:
                # Keep at least one column if filtering removed all
                lons = np.array([-180.0], dtype=np.float64)
                cols = np.array(["0R"], dtype=object)

            row_lons_filtered.append(lons)
            row_labels_filtered.append(cols)

        self._row_lons = row_lons_filtered
        self._row_col_labels = row_labels_filtered
        return self

    # ======================================================================
    #                         CORE CONVERSIONS
    # ======================================================================
    def latlon2rowcol(
        self,
        lats: Iterable[float] | float,
        lons: Iterable[float] | float,
        *,
        return_idx: bool = False,
        integer: bool = False,
    ):
        """
        Convert latitude/longitude to (row_label, col_label) of the **bottom-left** grid anchor.

        integer=True mimics legacy mapping:
          - rows:  "kU" → +k,  "kD" → -k
          - cols:  "kR" → +k,  "kL" → -k
        """
        lats_arr = np.atleast_1d(np.asarray(lats, dtype=np.float64))
        lons_arr = np.atleast_1d(np.asarray(lons, dtype=np.float64))
        if lats_arr.shape != lons_arr.shape:
            raise ValueError("lats and lons must have the same shape")

        # Row index: rightmost ring with lat <= value (bottom edge); clamp to bounds
        row_idx = np.searchsorted(self._lats, lats_arr, side="left") - 1
        row_idx = np.clip(row_idx, 0, len(self._lats) - 1)

        # Column index per unique row
        col_idx = np.empty_like(row_idx)
        for r in np.unique(row_idx):
            sel = (row_idx == r)
            row_lons = self._row_lons[int(r)]
            ci = np.searchsorted(row_lons, lons_arr[sel], side="left") - 1
            ci[ci < 0] = row_lons.size - 1  # wrap
            col_idx[sel] = ci

        # Labels from prebuilt per-row label arrays (legacy semantics)
        rows = np.array([self._row_labels[int(ri)] for ri in row_idx], dtype=object)
        cols = np.array([self._row_col_labels[int(ri)][int(ci)] for ri, ci in zip(row_idx, col_idx)], dtype=object)

        if integer:
            # Convert labels to signed ints like legacy Grid.latlon2rowcol(integer=True)
            def r_int(lbl: str) -> int:
                k = int(lbl[:-1]); s = lbl[-1]
                return k if s == "U" else -k
            def c_int(lbl: str) -> int:
                k = int(lbl[:-1]); s = lbl[-1]
                return k if s == "R" else -k
            rints = np.array([r_int(x) for x in rows], dtype=int)
            cints = np.array([c_int(x) for x in cols], dtype=int)
            if rows.size == 1:
                if return_idx:
                    return rints.item(), cints.item(), int(row_idx.item()), int(col_idx.item())
                return rints.item(), cints.item()
            if return_idx:
                return rints, cints, row_idx.astype(int), col_idx.astype(int)
            return rints, cints

        if rows.size == 1:
            if return_idx:
                return rows.item(), cols.item(), int(row_idx.item()), int(col_idx.item())
            return rows.item(), cols.item()
        if return_idx:
            return rows, cols, row_idx.astype(int), col_idx.astype(int)
        return rows, cols

    def rowcol2latlon(
        self,
        rows: Iterable[str | int] | str | int,
        cols: Iterable[str | int] | str | int,
    ):
        """
        Convert (row_label/row_int, col_label/col_int) -> (lat, lon) of the **bottom-left** grid anchor.

        If strings are given, they must match the stored labels (e.g., "3U", "2D", "5R", "4L").
        If integers are given, they are interpreted in legacy form:
            rows: +k → "kU",  −k → "kD"
            cols: +k → "kR",  −k → "kL"
        """
        # Normalize inputs
        to_obj_array = lambda x: np.atleast_1d(
            np.array(list(x) if isinstance(x, Iterable) and not isinstance(x, (str, bytes)) else [x], dtype=object)
        )
        rows_in = to_obj_array(rows)
        cols_in = to_obj_array(cols)
        if rows_in.shape != cols_in.shape:
            raise ValueError("rows and cols must have the same shape")

        # Decode rows to absolute row indices
        row_idx = np.empty(rows_in.shape[0], dtype=int)
        for i, rv in enumerate(rows_in):
            if isinstance(rv, (int, np.integer)):
                k = int(rv); label = f"{abs(k)}{'U' if k >= 0 else 'D'}"
            else:
                label = str(rv)
            # find the index in self._row_labels equal to label
            hits = np.where(self._row_labels == label)[0]
            if hits.size == 0:
                raise ValueError(f"Row label {label!r} not found in grid.")
            row_idx[i] = int(hits[0])
        row_idx = np.clip(row_idx, 0, len(self._lats) - 1)

        # Decode cols to absolute col indices (search label inside that row's label array)
        col_idx = np.empty_like(row_idx)
        for i, (ri, cv) in enumerate(zip(row_idx, cols_in)):
            if isinstance(cv, (int, np.integer)):
                k = int(cv); clabel = f"{abs(k)}{'R' if k >= 0 else 'L'}"
            else:
                clabel = str(cv)
            row_cols = self._row_col_labels[int(ri)]
            hits = np.where(row_cols == clabel)[0]
            if hits.size == 0:
                raise ValueError(f"Col label {clabel!r} not found in row {self._row_labels[int(ri)]}.")
            col_idx[i] = int(hits[0])

        # Emit coordinates (bottom-left anchor)
        lats = self._lats[row_idx]
        lons = np.array([self._row_lons[int(ri)][int(ci)] for ri, ci in zip(row_idx, col_idx)], dtype=float)

        if lats.size == 1:
            return float(lats.item()), float(lons.item())
        return lats.astype(float), lons.astype(float)

    def get_schema(self) -> dict[str, pl.DataType]:
        """Return the expected schema for this extension."""
        return {"majortom:code": pl.Utf8}

    def _compute(self, tortilla: 'Tortilla') -> pl.DataFrame:
        """
        Process Tortilla and return DataFrame with MajorTOM codes.
        
        Args:
            tortilla: Input Tortilla object
            
        Returns:
            pl.DataFrame: DataFrame with "majortom:code" column, 
                         aligned with input DataFrame (exact row count match)
        """
        # Get DataFrame from tortilla
        df = tortilla._metadata_df
        
        # Extract coordinates from WKB binary centroids
        lats, lons = [], []
        valid_indices = []
        
        for i, row in enumerate(df.iter_rows(named=True)):
            centroid_wkb = row.get("stac:centroid", None)
            if centroid_wkb is not None:
                try:
                    # Convert WKB binary to shapely geometry
                    geom = wkb_loads(centroid_wkb)
                    if hasattr(geom, 'x') and hasattr(geom, 'y'):  # Point geometry
                        lons.append(float(geom.x))
                        lats.append(float(geom.y))
                        valid_indices.append(i)
                except Exception:
                    # Skip invalid WKB data
                    continue
        
        # Initialize results array with None for all rows
        codes = [None] * len(df)
        
        if lats:
            # Convert coordinates to grid codes
            rows, cols = self.latlon2rowcol(lats, lons, integer=False)
            
            # Ensure rows and cols are iterable
            if not hasattr(rows, '__iter__') or isinstance(rows, str):
                rows = [rows]
            if not hasattr(cols, '__iter__') or isinstance(cols, str):
                cols = [cols]
            
            # Fill results at valid positions
            for valid_idx, r, c in zip(valid_indices, rows, cols):
                if r is not None and c is not None:
                    codes[valid_idx] = f"{r}{self.sep}{c}"
        
        # Return DataFrame with proper schema
        return pl.DataFrame(
            {"majortom:code": codes},
            schema={"majortom:code": pl.Utf8}
        )


if __name__ == "__main__":
    import random
    from shapely.geometry import Point
    from shapely.wkb import dumps as wkb_dumps

    # Create fake DataFrame with STAC WKB data (simulating what Tortilla provides)
    min_lon, min_lat, max_lon, max_lat = -125.0, 24.0, -66.0, 49.0
    
    data = []
    random.seed(42)
    for i in range(50):
        lon = random.uniform(min_lon, max_lon)
        lat = random.uniform(min_lat, max_lat)
        # Create WKB binary centroid
        point = Point(lon, lat)
        centroid_wkb = wkb_dumps(point)
        
        data.append({
            "id": f"sample_{i}",
            "type": "TACOTIFF", 
            "stac:crs": "EPSG:4326",
            "stac:centroid": centroid_wkb,
            "stac:tensor_shape": [512, 512],
        })
    
    # Create DataFrame input (what Tortilla would provide)
    input_df = pl.DataFrame(data, schema={
        "id": pl.Utf8,
        "type": pl.Utf8,
        "stac:crs": pl.Utf8,
        "stac:centroid": pl.Binary,
        "stac:tensor_shape": pl.List(pl.Int64)
    })

    # Test ABC-compatible API
    majortom = MajorTOM()
    
    # Create mock tortilla object for testing
    class MockTortilla:
        def __init__(self, df):
            self._metadata_df = df
    
    mock_tortilla = MockTortilla(input_df)
    result_df = majortom._compute(mock_tortilla)
    
    print("Input DataFrame shape:", input_df.shape)
    print("Result DataFrame shape:", result_df.shape)
    print("Result columns:", result_df.columns)
    print("Result schema:", result_df.schema)
    print("\nSample results:")
    print(result_df.head())