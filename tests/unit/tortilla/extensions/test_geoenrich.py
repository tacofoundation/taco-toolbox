"""Tests for GeoEnrich extension.

Tests are split into:
- Pure functions (no EE required): morton_key, resolve_admin_names
- Validation (no EE required): invalid variables, schema
- Integration (requires EE): marked with @pytest.mark.ee
"""

import pyarrow as pa
import pytest

from tacotoolbox.tortilla.extensions.geoenrich import (
    PRODUCT_SCHEMA,
    GeoEnrich,
    morton_key,
    resolve_admin_names,
)


class TestMortonKey:

    def test_deterministic(self):
        k1 = morton_key(10.0, 45.0)
        k2 = morton_key(10.0, 45.0)
        assert k1 == k2

    def test_returns_int(self):
        k = morton_key(0.0, 0.0)
        assert isinstance(k, int)

    def test_origin_produces_valid_key(self):
        k = morton_key(0.0, 0.0)
        assert k > 0

    def test_nearby_points_similar_keys(self):
        # Paris and Lyon (~400km)
        paris = morton_key(2.35, 48.85)
        lyon = morton_key(4.83, 45.76)

        # Tokyo (~10000km from Paris)
        tokyo = morton_key(139.69, 35.68)

        diff_nearby = abs(paris - lyon)
        diff_far = abs(paris - tokyo)
        assert diff_nearby < diff_far

    def test_different_bits_parameter(self):
        k16 = morton_key(10.0, 45.0, bits=16)
        k24 = morton_key(10.0, 45.0, bits=24)
        # Higher bits = larger range of values
        assert k24 > k16


class TestResolveAdminNames:

    def test_resolves_country_code_zero(self):
        # Code 0 = Afghanistan in admin0.parquet
        table = pa.table({"admin_countries": [0]})
        result = resolve_admin_names(table, ["admin_countries"])

        name = result.column("admin_countries")[0].as_py()
        assert name == "Afghanistan"

    def test_resolves_state_code_zero(self):
        # Code 0 = Kandahar in admin1.parquet
        table = pa.table({"admin_states": [0]})
        result = resolve_admin_names(table, ["admin_states"])

        name = result.column("admin_states")[0].as_py()
        assert name == "Kandahar"

    def test_resolves_district_code_zero(self):
        # Code 0 = Deh Bala in admin2.parquet
        table = pa.table({"admin_districts": [0]})
        result = resolve_admin_names(table, ["admin_districts"])

        name = result.column("admin_districts")[0].as_py()
        assert name == "Deh Bala"

    def test_ocean_code_resolved(self):
        # Code 65535 = Ocean/Sea/Lakes
        table = pa.table({"admin_countries": [65535]})
        result = resolve_admin_names(table, ["admin_countries"])

        name = result.column("admin_countries")[0].as_py()
        assert name == "Ocean/Sea/Lakes"

    def test_null_filled_with_ocean(self):
        table = pa.table({"admin_countries": pa.array([None], type=pa.int64())})
        result = resolve_admin_names(table, ["admin_countries"])

        name = result.column("admin_countries")[0].as_py()
        assert name == "Ocean/Sea/Lakes"

    def test_multiple_codes_resolved(self):
        table = pa.table({"admin_countries": [0, 65535]})
        result = resolve_admin_names(table, ["admin_countries"])

        names = result.column("admin_countries").to_pylist()
        assert names == ["Afghanistan", "Ocean/Sea/Lakes"]

    def test_multiple_admin_levels(self):
        table = pa.table(
            {
                "admin_countries": [0],
                "admin_states": [0],
                "admin_districts": [0],
            }
        )
        result = resolve_admin_names(
            table,
            ["admin_countries", "admin_states", "admin_districts"],
        )

        assert result.column("admin_countries")[0].as_py() == "Afghanistan"
        assert result.column("admin_states")[0].as_py() == "Kandahar"
        assert result.column("admin_districts")[0].as_py() == "Deh Bala"

    def test_preserves_other_columns(self):
        table = pa.table(
            {
                "admin_countries": [0],
                "other_col": ["keep_me"],
            }
        )
        result = resolve_admin_names(table, ["admin_countries"])

        assert "other_col" in result.schema.names
        assert result.column("other_col")[0].as_py() == "keep_me"

    def test_ignores_unknown_admin_var(self):
        table = pa.table({"admin_countries": [0]})
        # "unknown_admin" not in admin_level_map, should be ignored
        result = resolve_admin_names(table, ["admin_countries", "unknown_admin"])

        assert result.column("admin_countries")[0].as_py() == "Afghanistan"


class TestGeoEnrichValidation:

    def test_invalid_variable_raises(self):
        with pytest.raises(ValueError, match="Invalid variables.*fake_var"):
            GeoEnrich(variables=["fake_var"])

    def test_multiple_invalid_variables_listed(self):
        with pytest.raises(ValueError, match="fake1.*fake2"):
            GeoEnrich(variables=["fake1", "fake2"])

    def test_valid_single_variable(self):
        ge = GeoEnrich(variables=["elevation"])
        assert ge.variables == ["elevation"]

    def test_valid_multiple_variables(self):
        ge = GeoEnrich(variables=["elevation", "precipitation", "admin_countries"])
        assert len(ge.variables) == 3

    def test_none_variables_means_all(self):
        ge = GeoEnrich(variables=None)
        assert ge.variables is None

    def test_all_product_schema_keys_valid(self):
        # Sanity check: all keys in PRODUCT_SCHEMA are valid
        all_vars = list(PRODUCT_SCHEMA.keys())
        ge = GeoEnrich(variables=all_vars)
        assert ge.variables == all_vars


class TestGeoEnrichSchema:

    def test_get_schema_all_variables(self):
        ge = GeoEnrich(variables=None)
        schema = ge.get_schema()

        assert len(schema) == len(PRODUCT_SCHEMA)
        for field in schema:
            assert field.name.startswith("geoenrich:")

    def test_get_schema_subset(self):
        ge = GeoEnrich(variables=["elevation", "temperature"])
        schema = ge.get_schema()

        assert len(schema) == 2
        names = [f.name for f in schema]
        assert "geoenrich:elevation" in names
        assert "geoenrich:temperature" in names

    def test_get_schema_types_match_product_schema(self):
        ge = GeoEnrich(variables=["elevation", "admin_countries"])
        schema = ge.get_schema()

        elev_field = schema.field("geoenrich:elevation")
        admin_field = schema.field("geoenrich:admin_countries")

        assert elev_field.type == pa.float32()
        assert admin_field.type == pa.string()

    def test_get_field_descriptions_all(self):
        ge = GeoEnrich(variables=None)
        desc = ge.get_field_descriptions()

        assert len(desc) == len(PRODUCT_SCHEMA)
        assert "geoenrich:elevation" in desc
        assert "GLO-30" in desc["geoenrich:elevation"]

    def test_get_field_descriptions_subset(self):
        ge = GeoEnrich(variables=["elevation", "gdp"])
        desc = ge.get_field_descriptions()

        assert len(desc) == 2
        assert "geoenrich:elevation" in desc
        assert "geoenrich:gdp" in desc
        assert "geoenrich:temperature" not in desc


class TestGeoEnrichParameters:

    def test_default_scale(self):
        ge = GeoEnrich(variables=["elevation"])
        assert ge.scale_m == 5120.0

    def test_custom_scale(self):
        ge = GeoEnrich(variables=["elevation"], scale_m=1000.0)
        assert ge.scale_m == 1000.0

    def test_scale_validation_min(self):
        with pytest.raises(ValueError):
            GeoEnrich(variables=["elevation"], scale_m=0.5)

    def test_default_batch_size(self):
        ge = GeoEnrich(variables=["elevation"])
        assert ge.batch_size == 250

    def test_batch_size_validation(self):
        with pytest.raises(ValueError):
            GeoEnrich(variables=["elevation"], batch_size=0)

    def test_default_concurrency(self):
        ge = GeoEnrich(variables=["elevation"])
        assert ge.max_concurrency == 8

    def test_show_progress_default_true(self):
        ge = GeoEnrich(variables=["elevation"])
        assert ge.show_progress is True


class TestReturnNoneAlias:
    """Test that return_none alias works at attribute level (no EE required)."""

    def test_return_none_sets_schema_only(self):
        ext = GeoEnrich(variables=["elevation"], return_none=True)
        assert ext.schema_only is True

    def test_default_schema_only_is_false(self):
        ext = GeoEnrich(variables=["elevation"])
        assert ext.schema_only is False


class TestProductSchema:

    def test_all_expected_products_present(self):
        expected = [
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
            "admin_districts",
        ]
        for name in expected:
            assert name in PRODUCT_SCHEMA

    def test_admin_fields_are_string_type(self):
        assert PRODUCT_SCHEMA["admin_countries"] == pa.string()
        assert PRODUCT_SCHEMA["admin_states"] == pa.string()
        assert PRODUCT_SCHEMA["admin_districts"] == pa.string()

    def test_numeric_fields_are_float32(self):
        numeric = [
            "elevation",
            "cisi",
            "precipitation",
            "temperature",
            "gdp",
            "population",
        ]
        for name in numeric:
            assert PRODUCT_SCHEMA[name] == pa.float32()
