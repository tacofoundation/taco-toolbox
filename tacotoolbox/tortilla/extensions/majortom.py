"""
MajorTOM spherical grid extension.

Assigns samples to a hierarchical spherical grid with configurable spacing.
Uses latitude and longitude subdivisions to create grid cells with
approximately uniform spacing in kilometers.

Grid ID format: [DIST]km_[ROWID]_[COLID]
- Example: 0100km_0003U_0005R (100km spacing, row 3 Up, column 5 Right)
- Row labels: NNNNUD (e.g., 0003U for 3 degrees north, 0002D for 2 south)
- Column labels: NNNNRL (e.g., 0005R for 5 degrees east, 0004L for 4 west)

Exports to Arrow Table:
- majortom:code: String (format: '0100km_0003U_0005R')
"""

import math
from collections.abc import Iterable
from typing import TYPE_CHECKING

import pyarrow as pa
import pydantic
from pydantic import Field, PrivateAttr
from shapely.wkb import loads as wkb_loads

from tacotoolbox.tortilla.datamodel import TortillaExtension

if TYPE_CHECKING:
    from tacotoolbox.tortilla.datamodel import Tortilla

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class MajorTOM(TortillaExtension):
    """
    MajorTOM-like spherical grid with ~`dist_km` spacing.

    ID Format: [DIST]km_[ROWID]_[COLID]
        Example: 0100km_0003U_0005R
        All numeric values use %04d formatting for consistency.
    """

    dist_km: float = Field(
        default=100,
        description="Target spacing in kilometers along meridians and parallels. Grid cells have approximately this spacing.",
    )
    latitude_range: tuple[float, float] = Field(
        default=(-85.0, 85.0),
        description="Inclusive latitude limits in degrees for grid generation (min, max). Default excludes polar regions.",
    )
    longitude_range: tuple[float, float] = Field(
        default=(-180.0, 180.0),
        description="Inclusive longitude limits in degrees for filtering grid columns (min, max).",
    )
    sep: str = Field(
        default="_",
        description="Separator character for grid code components (e.g., '_' produces '0100km_0003U_0005R').",
    )

    _lats: np.ndarray = PrivateAttr(default_factory=lambda: np.empty(0))
    _row_labels: np.ndarray = PrivateAttr(
        default_factory=lambda: np.empty(0, dtype=object)
    )
    _zero_row_idx: int = PrivateAttr(default=0)
    _row_lons: list[np.ndarray] = PrivateAttr(default_factory=list)
    _row_col_labels: list[np.ndarray] = PrivateAttr(default_factory=list)

    R_EQUATOR_KM: float = 6378.137

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    @pydantic.model_validator(mode="after")
    def _build(self) -> "MajorTOM":
        """Compute latitudinal rings and per-row longitude subdivisions."""
        arc_pole_to_pole_km = math.pi * self.R_EQUATOR_KM
        num_divisions_in_hemisphere = math.ceil(arc_pole_to_pole_km / self.dist_km)

        lats_all = np.linspace(-90, 90, num_divisions_in_hemisphere + 1)[:-1]
        lats_all = np.mod(lats_all, 180) - 90  # type: ignore[assignment]
        lats_all = np.sort(lats_all)

        zero_row = int(np.searchsorted(lats_all, 0.0, side="left"))
        rows_all = np.empty_like(lats_all, dtype=object)

        # Use %04d formatting for row labels
        rows_all[zero_row:] = [f"{i:04d}U" for i in range(len(lats_all) - zero_row)]
        rows_all[:zero_row] = [f"{abs(i - zero_row):04d}D" for i in range(zero_row)]

        lat_lo, lat_hi = self.latitude_range
        lat_mask = (lats_all >= lat_lo) & (lats_all <= lat_hi)
        self._lats = lats_all[lat_mask]
        self._row_labels = rows_all[lat_mask]

        zero_in_filtered = np.searchsorted(self._lats, 0.0, side="left")
        self._zero_row_idx = int(
            np.clip(zero_in_filtered, 0, max(0, len(self._lats) - 1))
        )

        lon_lo, lon_hi = self.longitude_range
        row_lons_filtered: list[np.ndarray] = []
        row_labels_filtered: list[np.ndarray] = []

        for lat in self._lats:
            circ_km = 2.0 * math.pi * self.R_EQUATOR_KM * math.cos(math.radians(lat))
            n_cols = max(1, int(math.ceil(circ_km / self.dist_km)))

            lons_full = np.linspace(-180.0, 180.0, n_cols + 1)[:-1]
            lons_full = np.mod(lons_full, 360) - 180  # type: ignore[assignment]
            lons_full = np.sort(lons_full)

            zero_hits = np.where(lons_full == 0.0)[0]
            zero_col_full = (
                int(zero_hits[0])
                if zero_hits.size > 0
                else int(np.argmin(np.abs(lons_full)))
            )

            cols_full = np.empty(lons_full.size, dtype=object)

            # Use %04d formatting for column labels
            cols_full[zero_col_full:] = [
                f"{i:04d}R" for i in range(lons_full.size - zero_col_full)
            ]
            cols_full[:zero_col_full] = [
                f"{abs(i - zero_col_full):04d}L" for i in range(zero_col_full)
            ]

            if lon_lo > -180.0 or lon_hi < 180.0:
                mask = (lons_full >= lon_lo) & (lons_full <= lon_hi)
                lons = lons_full[mask]
                cols = cols_full[mask]
            else:
                lons = lons_full
                cols = cols_full

            if lons.size == 0:
                lons = np.array([-180.0], dtype=np.float64)
                cols = np.array(["0000R"], dtype=object)

            row_lons_filtered.append(lons)
            row_labels_filtered.append(cols)

        self._row_lons = row_lons_filtered
        self._row_col_labels = row_labels_filtered
        return self

    def latlon2rowcol(
        self,
        lats: Iterable[float] | float,
        lons: Iterable[float] | float,
        *,
        return_idx: bool = False,
        integer: bool = False,
    ):
        """
        Convert latitude/longitude to (row_label, col_label) of the bottom-left grid anchor.

        integer=True mimics legacy mapping with updated format:
          - rows:  "0003U" -> +3,  "0002D" -> -2
          - cols:  "0005R" -> +5,  "0004L" -> -4
        """
        lats_arr, lons_arr = self._validate_and_normalize_coords(lats, lons)
        row_idx, col_idx = self._compute_grid_indices(lats_arr, lons_arr)
        rows, cols = self._get_label_arrays(row_idx, col_idx)

        if integer:
            return self._convert_to_integer_format(
                rows, cols, row_idx, col_idx, return_idx
            )

        return self._format_output(rows, cols, row_idx, col_idx, return_idx)

    def _validate_and_normalize_coords(
        self, lats: Iterable[float] | float, lons: Iterable[float] | float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate and normalize coordinate inputs."""
        lats_arr = np.atleast_1d(np.asarray(lats, dtype=np.float64))
        lons_arr = np.atleast_1d(np.asarray(lons, dtype=np.float64))
        if lats_arr.shape != lons_arr.shape:
            raise ValueError("lats and lons must have the same shape")
        return lats_arr, lons_arr

    def _compute_grid_indices(
        self, lats_arr: np.ndarray, lons_arr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute row and column indices for coordinates."""
        row_idx = np.searchsorted(self._lats, lats_arr, side="left") - 1
        row_idx = np.clip(row_idx, 0, len(self._lats) - 1).astype(np.int64)

        col_idx = np.empty_like(row_idx, dtype=np.int64)
        for r in np.unique(row_idx):
            sel = row_idx == r
            row_lons = self._row_lons[int(r)]
            ci = np.searchsorted(row_lons, lons_arr[sel], side="left") - 1
            ci[ci < 0] = row_lons.size - 1
            col_idx[sel] = ci

        return row_idx, col_idx

    def _get_label_arrays(
        self, row_idx: np.ndarray, col_idx: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get row and column labels from indices."""
        rows = np.array([self._row_labels[int(ri)] for ri in row_idx], dtype=object)
        cols = np.array(
            [
                self._row_col_labels[int(ri)][int(ci)]
                for ri, ci in zip(row_idx, col_idx, strict=True)
            ],
            dtype=object,
        )
        return rows, cols

    def _convert_to_integer_format(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        row_idx: np.ndarray,
        col_idx: np.ndarray,
        return_idx: bool,
    ):
        """Convert labels to signed integers for legacy compatibility."""

        def r_int(lbl: str) -> int:
            k = int(lbl[:-1])
            s = lbl[-1]
            return k if s == "U" else -k

        def c_int(lbl: str) -> int:
            k = int(lbl[:-1])
            s = lbl[-1]
            return k if s == "R" else -k

        rints = np.array([r_int(x) for x in rows], dtype=int)
        cints = np.array([c_int(x) for x in cols], dtype=int)

        if rows.size == 1:
            if return_idx:
                return (
                    rints.item(),
                    cints.item(),
                    int(row_idx.item()),
                    int(col_idx.item()),
                )
            return rints.item(), cints.item()
        if return_idx:
            return rints, cints, row_idx.astype(int), col_idx.astype(int)
        return rints, cints

    def _format_output(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        row_idx: np.ndarray,
        col_idx: np.ndarray,
        return_idx: bool,
    ):
        """Format output for string label format."""
        if rows.size == 1:
            if return_idx:
                return (
                    rows.item(),
                    cols.item(),
                    int(row_idx.item()),
                    int(col_idx.item()),
                )
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
        Convert (row_label/row_int, col_label/col_int) -> (lat, lon) of the bottom-left grid anchor.

        Supports both new format ("0003U") and legacy integer format.
        """
        to_obj_array = lambda x: np.atleast_1d(
            np.array(
                (
                    list(x)
                    if isinstance(x, Iterable) and not isinstance(x, str | bytes)
                    else [x]
                ),
                dtype=object,
            )
        )
        rows_in = to_obj_array(rows)
        cols_in = to_obj_array(cols)
        if rows_in.shape != cols_in.shape:
            raise ValueError("rows and cols must have the same shape")

        row_idx = np.empty(rows_in.shape[0], dtype=int)
        for i, rv in enumerate(rows_in):
            if isinstance(rv, int | np.integer):
                k = int(rv)
                label = f"{abs(k):04d}{'U' if k >= 0 else 'D'}"
            else:
                label = str(rv)
            hits = np.where(self._row_labels == label)[0]
            if hits.size == 0:
                raise ValueError(f"Row label {label!r} not found in grid.")
            row_idx[i] = int(hits[0])
        row_idx = np.clip(row_idx, 0, len(self._lats) - 1)

        col_idx = np.empty_like(row_idx)
        for i, (ri, cv) in enumerate(zip(row_idx, cols_in, strict=True)):
            if isinstance(cv, int | np.integer):
                k = int(cv)
                clabel = f"{abs(k):04d}{'R' if k >= 0 else 'L'}"
            else:
                clabel = str(cv)
            row_cols = self._row_col_labels[int(ri)]
            hits = np.where(row_cols == clabel)[0]
            if hits.size == 0:
                raise ValueError(
                    f"Col label {clabel!r} not found in row {self._row_labels[int(ri)]}."
                )
            col_idx[i] = int(hits[0])

        lats = self._lats[row_idx]
        lons = np.array(
            [
                self._row_lons[int(ri)][int(ci)]
                for ri, ci in zip(row_idx, col_idx, strict=True)
            ],
            dtype=float,
        )

        if lats.size == 1:
            return float(lats.item()), float(lons.item())
        return lats.astype(float), lons.astype(float)

    def get_schema(self) -> pa.Schema:
        """Return the expected schema for this extension."""
        return pa.schema([pa.field("majortom:code", pa.string())])

    def get_field_descriptions(self) -> dict[str, str]:
        """Return field descriptions for each field."""
        return {
            "majortom:code": "MajorTOM spherical grid cell identifier (e.g., 0100km_0003U_0005R) with ~dist_km spacing"
        }

    def _compute(self, tortilla: "Tortilla") -> pa.Table:  # noqa: C901
        """
        Process Tortilla and return Arrow Table with MajorTOM codes.

        New format: [DIST]km_[ROWID]_[COLID]
        Example: 0100km_0003U_0005R
        """

        if not HAS_NUMPY:
            raise ImportError(
                "MajorTOM extension requires numpy.\n" "Install with: pip install numpy"
            )

        table = tortilla._metadata_table

        lats, lons = [], []
        valid_indices = []

        # Check if centroid column exists
        if "stac:centroid" not in table.schema.names:
            raise ValueError(
                "Column 'stac:centroid' not found in tortilla metadata.\n"
                f"Available columns: {table.schema.names}\n"
                "Ensure samples have STAC extension applied."
            )

        centroid_column = table.column("stac:centroid")

        for i in range(table.num_rows):
            centroid_wkb = centroid_column[i].as_py()
            if centroid_wkb is not None:
                try:
                    geom = wkb_loads(centroid_wkb)
                    if hasattr(geom, "x") and hasattr(geom, "y"):
                        lons.append(float(geom.x))
                        lats.append(float(geom.y))
                        valid_indices.append(i)
                except (ValueError, TypeError, AttributeError):
                    continue

        codes: list[str | None] = [None] * table.num_rows

        if lats:
            rows, cols = self.latlon2rowcol(lats, lons, integer=False)

            if not hasattr(rows, "__iter__") or isinstance(rows, str):
                rows = [rows]
            if not hasattr(cols, "__iter__") or isinstance(cols, str):
                cols = [cols]

            # Format dist_km as integer with %04d
            dist_km_formatted = f"{int(self.dist_km):04d}km"

            for valid_idx, r, c in zip(valid_indices, rows, cols, strict=True):
                if r is not None and c is not None:
                    # New format: [DIST]km_[ROWID]_[COLID]
                    codes[valid_idx] = f"{dist_km_formatted}{self.sep}{r}{self.sep}{c}"

        return pa.Table.from_pydict(
            {"majortom:code": codes},
            schema=pa.schema([pa.field("majortom:code", pa.string())]),
        )
