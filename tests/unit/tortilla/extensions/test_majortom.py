"""Tests for MajorTOM extension."""

import re

import pyarrow as pa
import pytest

from tacotoolbox.tortilla.extensions.majortom import MajorTOM


class TestGridCodeFormat:

    def test_code_matches_expected_pattern(self, make_tortilla_with_stac):
        t = make_tortilla_with_stac(coords=[(10.0, 45.0)])
        t.extend_with(MajorTOM(dist_km=100))

        code = t.metadata_table.column("majortom:code")[0].as_py()
        pattern = r"^\d{4}km_\d{4}[UD]_\d{4}[RL]$"
        assert re.match(pattern, code), f"Code '{code}' doesn't match pattern"

    def test_dist_km_appears_in_code(self, make_tortilla_with_stac):
        t = make_tortilla_with_stac(coords=[(0.0, 0.0)])
        t.extend_with(MajorTOM(dist_km=250))

        code = t.metadata_table.column("majortom:code")[0].as_py()
        assert code.startswith("0250km_")

    def test_custom_separator(self, make_tortilla_with_stac):
        t = make_tortilla_with_stac(coords=[(0.0, 0.0)])
        t.extend_with(MajorTOM(dist_km=100, sep="-"))

        code = t.metadata_table.column("majortom:code")[0].as_py()
        assert "-" in code
        assert "_" not in code


class TestLatLonToRowCol:

    @pytest.fixture
    def grid(self):
        return MajorTOM(dist_km=100)

    def test_equator_row_near_zero(self, grid):
        row, col = grid.latlon2rowcol(0.0, 0.0)
        # Grid discretization means (0,0) may fall in adjacent cell
        # Just verify it's near the equator (small row number)
        row_num = int(row[:-1])
        assert row_num <= 1

    def test_prime_meridian_col_near_zero(self, grid):
        row, col = grid.latlon2rowcol(0.0, 0.0)
        # Grid discretization means (0,0) may fall in adjacent cell
        col_num = int(col[:-1])
        assert col_num <= 1

    def test_northern_hemisphere_uses_U(self, grid):
        row, _ = grid.latlon2rowcol(45.0, 0.0)
        assert row.endswith("U")

    def test_southern_hemisphere_uses_D(self, grid):
        row, _ = grid.latlon2rowcol(-45.0, 0.0)
        assert row.endswith("D")

    def test_eastern_hemisphere_uses_R(self, grid):
        _, col = grid.latlon2rowcol(0.0, 90.0)
        assert col.endswith("R")

    def test_western_hemisphere_uses_L(self, grid):
        _, col = grid.latlon2rowcol(0.0, -90.0)
        assert col.endswith("L")

    def test_integer_mode_returns_signed_ints(self, grid):
        # Northern/Eastern → positive
        r_ne, c_ne = grid.latlon2rowcol(45.0, 90.0, integer=True)
        assert isinstance(r_ne, int)
        assert isinstance(c_ne, int)
        assert r_ne > 0
        assert c_ne > 0

        # Southern/Western → negative
        r_sw, c_sw = grid.latlon2rowcol(-45.0, -90.0, integer=True)
        assert r_sw < 0
        assert c_sw < 0

    def test_return_idx_gives_four_values(self, grid):
        result = grid.latlon2rowcol(45.0, 90.0, return_idx=True)
        assert len(result) == 4
        row, col, row_idx, col_idx = result
        assert isinstance(row_idx, int)
        assert isinstance(col_idx, int)

    def test_batch_coordinates(self, grid):
        lats = [0.0, 45.0, -30.0]
        lons = [0.0, 90.0, -60.0]
        rows, cols = grid.latlon2rowcol(lats, lons)

        assert len(rows) == 3
        assert len(cols) == 3


class TestRowColToLatLon:

    @pytest.fixture
    def grid(self):
        return MajorTOM(dist_km=100)

    def test_zero_returns_near_origin(self, grid):
        lat, lon = grid.rowcol2latlon("0000U", "0000R")
        assert abs(lat) < 5  # Within one cell of equator
        assert abs(lon) < 5  # Within one cell of prime meridian

    def test_integer_input_works(self, grid):
        lat_str, lon_str = grid.rowcol2latlon("0003U", "0005R")
        lat_int, lon_int = grid.rowcol2latlon(3, 5)
        assert lat_str == lat_int
        assert lon_str == lon_int

    def test_negative_integer_gives_south_west(self, grid):
        lat, lon = grid.rowcol2latlon(-3, -5)
        assert lat < 0
        assert lon < 0

    def test_invalid_row_label_raises(self, grid):
        with pytest.raises(ValueError, match="not found"):
            grid.rowcol2latlon("9999U", "0000R")


class TestRoundtrip:

    @pytest.fixture
    def grid(self):
        return MajorTOM(dist_km=100)

    def test_latlon_roundtrip_within_cell(self, grid):
        """Converting lat/lon → row/col → lat/lon should return cell anchor."""
        original_lat, original_lon = 45.5, 10.3

        row, col = grid.latlon2rowcol(original_lat, original_lon)
        recovered_lat, recovered_lon = grid.rowcol2latlon(row, col)

        # Recovered coords are cell anchor, not original point
        # But should be within one cell distance
        assert abs(recovered_lat - original_lat) < 2.0  # ~100km ≈ 0.9° lat
        assert abs(recovered_lon - original_lon) < 3.0  # varies with lat


class TestLatitudeRange:

    def test_default_excludes_poles(self):
        grid = MajorTOM(dist_km=100)
        # Default latitude_range=(-85, 85)
        assert grid._lats.min() >= -85
        assert grid._lats.max() <= 85

    def test_custom_range_filters(self):
        grid = MajorTOM(dist_km=100, latitude_range=(-60, 60))
        assert grid._lats.min() >= -60
        assert grid._lats.max() <= 60

    def test_out_of_range_point_clamps(self):
        grid = MajorTOM(dist_km=100, latitude_range=(-60, 60))
        # Point at 80°N is outside range, should clamp to edge
        row, _ = grid.latlon2rowcol(80.0, 0.0)
        # Should get a valid row (clamped)
        assert row.endswith("U")


class TestTortillaIntegration:

    def test_missing_centroid_raises(self, make_tortilla):
        t = make_tortilla(n_samples=2)  # No STAC extension

        with pytest.raises(ValueError, match="stac:centroid.*not found"):
            t.extend_with(MajorTOM())

    def test_multiple_samples_get_codes(self, make_tortilla_with_stac):
        coords = [
            (0.0, 0.0),  # Origin
            (10.0, 45.0),  # Europe
            (-74.0, 40.7),  # NYC
        ]
        t = make_tortilla_with_stac(coords=coords)
        t.extend_with(MajorTOM(dist_km=100))

        codes = t.metadata_table.column("majortom:code").to_pylist()
        assert len(codes) == 3
        assert all(c is not None for c in codes)
        # All codes should be unique for these distant points
        assert len(set(codes)) == 3

    def test_null_centroid_gives_null_code(self, make_sample):
        from tacotoolbox.tortilla.datamodel import Tortilla

        s = make_sample("no_centroid")
        # Add stac:centroid column with None value
        s.extend_with(
            pa.table(
                {"stac:centroid": [None]},
                schema=pa.schema([pa.field("stac:centroid", pa.binary())]),
            )
        )

        t = Tortilla(samples=[s])
        t.extend_with(MajorTOM())

        code = t.metadata_table.column("majortom:code")[0].as_py()
        assert code is None


class TestSchemaOnly:

    def test_return_none_preserves_schema(self, make_tortilla_with_stac):
        t = make_tortilla_with_stac(coords=[(0.0, 0.0)])
        t.extend_with(MajorTOM(return_none=True))

        assert "majortom:code" in t.metadata_table.schema.names
        code = t.metadata_table.column("majortom:code")[0].as_py()
        assert code is None

    def test_schema_only_via_internal_attr(self):
        """Verify schema_only is the internal attribute name."""
        ext = MajorTOM(return_none=True)
        assert ext.schema_only is True


class TestSchema:

    def test_get_schema_returns_expected_field(self):
        grid = MajorTOM()
        schema = grid.get_schema()
        assert len(schema) == 1
        assert schema.field(0).name == "majortom:code"
        assert schema.field(0).type == pa.string()

    def test_get_field_descriptions_has_code(self):
        grid = MajorTOM()
        desc = grid.get_field_descriptions()
        assert "majortom:code" in desc
        assert "MajorTOM" in desc["majortom:code"]
