"""Tests for SpatialGrouping extension."""

import re

import pytest

from tacotoolbox.tortilla.extensions.spatial_grouping import (
    SpatialGrouping,
    compute_z_order,
    morton_encode,
    normalize_coords,
)


class TestNormalizeCoords:

    def test_origin_maps_to_center(self):
        x, y = normalize_coords(0.0, 0.0)
        assert x == 0.5
        assert y == 0.5

    def test_min_bounds_map_to_zero(self):
        x, y = normalize_coords(-180.0, -90.0)
        assert x == 0.0
        assert y == 0.0

    def test_max_bounds_map_to_one(self):
        x, y = normalize_coords(180.0, 90.0)
        assert x == 1.0
        assert y == 1.0


class TestMortonEncode:

    def test_origin_returns_fixed_value(self):
        code = morton_encode(0.5, 0.5)
        assert isinstance(code, int)
        assert code > 0

    def test_deterministic(self):
        code1 = morton_encode(0.3, 0.7)
        code2 = morton_encode(0.3, 0.7)
        assert code1 == code2

    def test_different_inputs_different_codes(self):
        code1 = morton_encode(0.1, 0.1)
        code2 = morton_encode(0.9, 0.9)
        assert code1 != code2


class TestComputeZOrder:

    def test_deterministic(self):
        z1 = compute_z_order(10.0, 45.0)
        z2 = compute_z_order(10.0, 45.0)
        assert z1 == z2

    def test_nearby_points_have_similar_codes(self):
        # Paris and Lyon are ~400km apart
        paris = compute_z_order(2.35, 48.85)
        lyon = compute_z_order(4.83, 45.76)

        # Tokyo is ~10000km from Paris
        tokyo = compute_z_order(139.69, 35.68)

        # Paris-Lyon difference should be smaller than Paris-Tokyo
        diff_nearby = abs(paris - lyon)
        diff_far = abs(paris - tokyo)
        assert diff_nearby < diff_far

    def test_antipodal_points_very_different(self):
        point = compute_z_order(0.0, 0.0)
        antipode = compute_z_order(180.0, 0.0)
        assert abs(point - antipode) > 1_000_000


class TestSpatialGroupingValidation:

    def test_no_limits_raises(self):
        with pytest.raises(ValueError, match="Must specify at least one"):
            SpatialGrouping()

    def test_target_count_only_valid(self):
        sg = SpatialGrouping(target_count=100)
        assert sg.target_count == 100
        assert sg._target_size_bytes is None

    def test_target_size_only_valid(self):
        sg = SpatialGrouping(target_size="1GB")
        assert sg._target_size_bytes == 1024**3

    def test_both_limits_valid(self):
        sg = SpatialGrouping(target_count=100, target_size="512MB")
        assert sg.target_count == 100
        assert sg._target_size_bytes == 512 * 1024**2

    def test_invalid_size_format_raises(self):
        with pytest.raises(ValueError, match="Invalid target_size"):
            SpatialGrouping(target_size="1XB")

    def test_zero_target_count_raises(self):
        with pytest.raises(ValueError):
            SpatialGrouping(target_count=0)

    def test_negative_target_count_raises(self):
        with pytest.raises(ValueError):
            SpatialGrouping(target_count=-5)


class TestSpatialGroupingTargetSize:

    def test_parses_gigabytes(self):
        sg = SpatialGrouping(target_size="2GB")
        assert sg._target_size_bytes == 2 * 1024**3

    def test_parses_megabytes(self):
        sg = SpatialGrouping(target_size="512MB")
        assert sg._target_size_bytes == 512 * 1024**2

    def test_parses_kilobytes(self):
        sg = SpatialGrouping(target_size="1024KB")
        assert sg._target_size_bytes == 1024 * 1024

    def test_parses_bytes(self):
        sg = SpatialGrouping(target_size="1000B")
        assert sg._target_size_bytes == 1000


class TestGroupCodeFormat:

    def test_code_matches_pattern(self, make_tortilla_with_stac):
        t = make_tortilla_with_stac(coords=[(0.0, 0.0), (1.0, 1.0)])
        t.extend_with(SpatialGrouping(target_count=10))

        codes = t.metadata_table.column("spatialgroup:code").to_pylist()
        pattern = r"^sg\d{4}$"
        for code in codes:
            assert re.match(pattern, code), f"Code '{code}' doesn't match pattern"

    def test_first_group_is_sg0000(self, make_tortilla_with_stac):
        t = make_tortilla_with_stac(coords=[(0.0, 0.0)])
        t.extend_with(SpatialGrouping(target_count=10))

        code = t.metadata_table.column("spatialgroup:code")[0].as_py()
        assert code == "sg0000"


class TestGroupingByCount:

    def test_respects_target_count(self, make_tortilla_with_stac):
        # 10 samples, target_count=3 → should get 4 groups (3+3+3+1)
        coords = [(float(i), float(i)) for i in range(10)]
        t = make_tortilla_with_stac(coords=coords)
        t.extend_with(SpatialGrouping(target_count=3))

        codes = t.metadata_table.column("spatialgroup:code").to_pylist()
        unique_groups = set(codes)
        assert len(unique_groups) == 4

    def test_single_group_when_under_limit(self, make_tortilla_with_stac):
        coords = [(float(i), float(i)) for i in range(5)]
        t = make_tortilla_with_stac(coords=coords)
        t.extend_with(SpatialGrouping(target_count=10))

        codes = t.metadata_table.column("spatialgroup:code").to_pylist()
        assert all(c == "sg0000" for c in codes)


class TestGroupingBySize:

    def test_respects_target_size(self, make_sample_with_stac):
        from tacotoolbox.tortilla.datamodel import Tortilla

        # Create samples with known sizes
        samples = []
        for i in range(5):
            # Each sample ~500 bytes
            s = make_sample_with_stac(
                f"s{i}", lon=float(i), lat=float(i), content=b"x" * 500
            )
            samples.append(s)

        t = Tortilla(samples=samples)
        # target_size=1000B → should split after 2 samples (~1000B each pair)
        t.extend_with(SpatialGrouping(target_size="1000B"))

        codes = t.metadata_table.column("spatialgroup:code").to_pylist()
        unique_groups = set(codes)
        # Should have multiple groups since 5*500 > 1000
        assert len(unique_groups) >= 2


class TestHybridGrouping:

    def test_cuts_on_count_first(self, make_sample_with_stac):
        from tacotoolbox.tortilla.datamodel import Tortilla

        # Small samples, low count limit
        samples = [
            make_sample_with_stac(
                f"s{i}", lon=float(i), lat=float(i), content=b"x" * 10
            )
            for i in range(10)
        ]
        t = Tortilla(samples=samples)
        # Count limit will hit first (size limit too high)
        t.extend_with(SpatialGrouping(target_count=3, target_size="1GB"))

        codes = t.metadata_table.column("spatialgroup:code").to_pylist()
        unique_groups = set(codes)
        assert len(unique_groups) == 4  # 3+3+3+1


class TestMissingCentroid:

    def test_no_centroid_column_raises(self, make_tortilla):
        t = make_tortilla(n_samples=3)

        with pytest.raises(ValueError, match="stac:centroid.*not found"):
            t.extend_with(SpatialGrouping(target_count=10))


class TestSchemaOnly:

    def test_return_none_preserves_schema(self, make_tortilla_with_stac):
        t = make_tortilla_with_stac(coords=[(0.0, 0.0)])
        t.extend_with(SpatialGrouping(target_count=10, return_none=True))

        assert "spatialgroup:code" in t.metadata_table.schema.names
        code = t.metadata_table.column("spatialgroup:code")[0].as_py()
        assert code is None


class TestSchema:

    def test_get_schema_returns_expected_field(self):
        sg = SpatialGrouping(target_count=10)
        schema = sg.get_schema()
        assert len(schema) == 1
        assert schema.field(0).name == "spatialgroup:code"

    def test_get_field_descriptions(self):
        sg = SpatialGrouping(target_count=10)
        desc = sg.get_field_descriptions()
        assert "spatialgroup:code" in desc
        assert "Z-order" in desc["spatialgroup:code"]
