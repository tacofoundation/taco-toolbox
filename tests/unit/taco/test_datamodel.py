"""Core Taco datamodel tests."""

from datetime import datetime, timezone

import pyarrow as pa
import pytest
from shapely.geometry import Point
from shapely.wkb import dumps as wkb_dumps

from tacotoolbox.sample.datamodel import Sample
from tacotoolbox.taco.datamodel import (
    Curator,
    Extent,
    Provider,
    Taco,
    TacoExtension,
    _calculate_spatial_extent,
    _calculate_temporal_extent,
)
from tacotoolbox.tortilla.datamodel import Tortilla


class TestProvider:
    def test_url_requires_scheme(self):
        with pytest.raises(ValueError, match="http://"):
            Provider(name="Test", url="example.com")

    def test_accepts_https(self):
        p = Provider(name="Test", url="https://example.com")
        assert p.url == "https://example.com"


class TestCurator:
    def test_requires_name_or_organization(self):
        with pytest.raises(ValueError, match="name.*organization"):
            Curator(email="test@example.com")

    def test_email_requires_at_symbol(self):
        with pytest.raises(ValueError, match="@"):
            Curator(name="John", email="invalid")

    def test_organization_alone_suffices(self):
        c = Curator(organization="ACME")
        assert c.name is None


class TestExtent:
    def test_spatial_requires_four_values(self):
        with pytest.raises(ValueError, match="exactly 4"):
            Extent(spatial=[0, 0, 1])

    def test_south_cannot_exceed_north(self):
        with pytest.raises(ValueError, match="south.*north"):
            Extent(spatial=[0, 50, 10, 40])

    def test_longitude_bounds(self):
        with pytest.raises(ValueError, match="Longitude"):
            Extent(spatial=[-200, -10, 10, 10])

    def test_antimeridian_crossing_valid(self):
        e = Extent(spatial=[170, -10, -170, 10])
        assert e.spatial[0] > e.spatial[2]

    def test_temporal_order(self):
        with pytest.raises(ValueError, match="before"):
            Extent(spatial=[0, 0, 1, 1], temporal=["2024-01-01T00:00:00Z", "2023-01-01T00:00:00Z"])

    def test_temporal_invalid_format(self):
        with pytest.raises(ValueError, match="ISO 8601"):
            Extent(spatial=[0, 0, 1, 1], temporal=["not-a-date", "2023-01-01"])


class TestTacoIdValidation:
    def test_must_be_lowercase(self, minimal_taco_kwargs):
        minimal_taco_kwargs["id"] = "MyDataset"
        with pytest.raises(ValueError, match="lowercase"):
            Taco(**minimal_taco_kwargs)

    def test_allows_hyphens_underscores(self, minimal_taco_kwargs):
        minimal_taco_kwargs["id"] = "my-dataset_v1"
        taco = Taco(**minimal_taco_kwargs)
        assert taco.id == "my-dataset_v1"

    def test_rejects_special_chars(self, minimal_taco_kwargs):
        minimal_taco_kwargs["id"] = "my.dataset"
        with pytest.raises(ValueError, match="alphanumeric"):
            Taco(**minimal_taco_kwargs)


class TestTacoTitleValidation:
    def test_max_length_250(self, minimal_taco_kwargs):
        minimal_taco_kwargs["title"] = "x" * 251
        with pytest.raises(ValueError, match="250"):
            Taco(**minimal_taco_kwargs)


class TestRSUTValidation:
    """
    RSUT (Root-Sibling Uniform Tree) validation.
    Code uses _validate_pit_compliance internally (legacy naming).
    
    Invariant 2: Type uniformity at root level
    Invariant 3: Structural homogeneity among level-1 siblings (does NOT propagate deeper)
    """

    def test_mixed_types_at_root_rejected(self, minimal_taco_kwargs, make_sample, make_folder_sample):
        """Invariant 2: root cannot mix FILE and FOLDER."""
        file_sample = make_sample("file_0")
        folder_sample = make_folder_sample("folder_0", ["child_a", "child_b"])
        
        minimal_taco_kwargs["tortilla"] = Tortilla(samples=[file_sample, folder_sample])
        
        with pytest.raises(ValueError, match="same type"):
            Taco(**minimal_taco_kwargs)

    def test_different_child_counts_rejected(self, minimal_taco_kwargs, make_folder_sample):
        """Invariant 3: siblings must have same child count."""
        folder_a = make_folder_sample("date_0", ["glorys", "l4", "l3"])
        folder_b = make_folder_sample("date_1", ["glorys", "l4"])
        
        minimal_taco_kwargs["tortilla"] = Tortilla(samples=[folder_a, folder_b])
        
        with pytest.raises(ValueError, match="isomorphic"):
            Taco(**minimal_taco_kwargs)

    def test_different_child_ids_at_position_rejected(self, minimal_taco_kwargs, make_folder_sample):
        """Invariant 3: children at same position must have same ID."""
        folder_a = make_folder_sample("date_0", ["glorys", "l4", "l3"])
        folder_b = make_folder_sample("date_1", ["glorys", "l3", "l4"])  # swapped
        
        minimal_taco_kwargs["tortilla"] = Tortilla(samples=[folder_a, folder_b])
        
        with pytest.raises(ValueError, match="position"):
            Taco(**minimal_taco_kwargs)

    def test_isomorphic_folders_accepted(self, minimal_taco_kwargs, make_folder_sample):
        folder_a = make_folder_sample("date_0", ["glorys", "l4", "l3"])
        folder_b = make_folder_sample("date_1", ["glorys", "l4", "l3"])
        folder_c = make_folder_sample("date_2", ["glorys", "l4", "l3"])
        
        minimal_taco_kwargs["tortilla"] = Tortilla(samples=[folder_a, folder_b, folder_c])
        
        taco = Taco(**minimal_taco_kwargs)
        assert len(taco.tortilla.samples) == 3

    def test_single_folder_always_valid(self, minimal_taco_kwargs, make_folder_sample):
        folder = make_folder_sample("only_one", ["a", "b", "c"])
        minimal_taco_kwargs["tortilla"] = Tortilla(samples=[folder])
        
        taco = Taco(**minimal_taco_kwargs)
        assert taco.tortilla.samples[0].id == "only_one"

    def test_all_files_skips_structural_check(self, minimal_taco_kwargs, make_sample):
        samples = [make_sample(f"file_{i}") for i in range(5)]
        minimal_taco_kwargs["tortilla"] = Tortilla(samples=samples)
        
        taco = Taco(**minimal_taco_kwargs)
        assert len(taco.tortilla.samples) == 5


class TestAutoExtent:
    def test_default_global_without_stac(self, minimal_taco_kwargs):
        taco = Taco(**minimal_taco_kwargs)
        assert taco.extent.spatial == [-180.0, -90.0, 180.0, 90.0]
        assert taco.extent.temporal is None

    def test_computed_from_stac(self, minimal_taco_kwargs, make_tortilla_with_stac):
        coords_times = [
            (5, 0, datetime(2023, 1, 1, tzinfo=timezone.utc)),
            (10, 20, datetime(2023, 6, 15, tzinfo=timezone.utc)),
            (2, -10, datetime(2023, 12, 31, tzinfo=timezone.utc)),
        ]
        minimal_taco_kwargs["tortilla"] = make_tortilla_with_stac(coords_times)
        
        taco = Taco(**minimal_taco_kwargs)
        
        assert taco.extent.spatial == [2, -10, 10, 20]
        assert "2023-01-01" in taco.extent.temporal[0]
        assert "2023-12-31" in taco.extent.temporal[1]


class TestSpatialExtentCalculation:
    def _make_table(self, lon_lats: list[tuple[float, float]]) -> pa.Table:
        centroids = [wkb_dumps(Point(lon, lat)) for lon, lat in lon_lats]
        return pa.table({"centroid": centroids})

    def test_single_point(self):
        table = self._make_table([(10, 20)])
        bbox = _calculate_spatial_extent(table, "centroid")
        assert bbox == [10, 20, 10, 20]

    def test_all_positive_longitudes(self):
        table = self._make_table([(10, 0), (50, 10), (30, -5)])
        bbox = _calculate_spatial_extent(table, "centroid")
        assert bbox == [10, -5, 50, 10]

    def test_all_negative_longitudes(self):
        table = self._make_table([(-120, 30), (-80, 40), (-100, 35)])
        bbox = _calculate_spatial_extent(table, "centroid")
        assert bbox == [-120, 30, -80, 40]

    def test_mixed_keeps_larger_span(self):
        # positive span: 160, negative span: 10
        table = self._make_table([(10, 0), (170, 0), (-10, 0), (-20, 0)])
        bbox = _calculate_spatial_extent(table, "centroid")
        assert bbox[0] == 10 and bbox[2] == 170

    def test_empty_returns_global(self):
        table = pa.table({"centroid": pa.array([], type=pa.binary())})
        bbox = _calculate_spatial_extent(table, "centroid")
        assert bbox == [-180.0, -90.0, 180.0, 90.0]


class TestTemporalExtentCalculation:
    def test_basic_range(self):
        times = [
            datetime(2023, 1, 1, tzinfo=timezone.utc),
            datetime(2023, 6, 15, tzinfo=timezone.utc),
            datetime(2023, 12, 31, tzinfo=timezone.utc),
        ]
        table = pa.table({"time": times})
        
        result = _calculate_temporal_extent(table, "time", None, None)
        
        assert result[0] == "2023-01-01T00:00:00Z"
        assert result[1] == "2023-12-31T00:00:00Z"

    def test_skips_none_values(self):
        times = [
            datetime(2023, 1, 1, tzinfo=timezone.utc),
            None,
            datetime(2023, 12, 31, tzinfo=timezone.utc),
        ]
        table = pa.table({"time": pa.array(times, type=pa.timestamp("us", tz="UTC"))})
        
        result = _calculate_temporal_extent(table, "time", None, None)
        assert result is not None

    def test_time_middle_priority(self):
        table = pa.table({
            "start": [datetime(2020, 1, 1, tzinfo=timezone.utc)],
            "middle": [datetime(2023, 6, 15, tzinfo=timezone.utc)],
        })
        
        result = _calculate_temporal_extent(table, "start", None, "middle")
        assert "2023-06-15" in result[0]


class TestExtendWith:
    def test_dict(self, minimal_taco_kwargs):
        taco = Taco(**minimal_taco_kwargs)
        taco.extend_with({"custom_field": "value", "custom_count": 42})
        
        assert getattr(taco, "custom_field") == "value"
        assert getattr(taco, "custom_count") == 42

    def test_table(self, minimal_taco_kwargs):
        taco = Taco(**minimal_taco_kwargs)
        taco.extend_with(pa.table({"ext_name": ["test"], "ext_value": [123]}))
        
        assert getattr(taco, "ext_name") == "test"
        assert getattr(taco, "ext_value") == 123

    def test_table_rejects_multi_row(self, minimal_taco_kwargs):
        taco = Taco(**minimal_taco_kwargs)
        
        with pytest.raises(ValueError, match="one row"):
            taco.extend_with(pa.table({"ext_x": [1, 2, 3]}))

    def test_taco_extension(self, minimal_taco_kwargs):
        class DummyExtension(TacoExtension):
            def get_schema(self):
                return pa.schema([pa.field("dummy_value", pa.int32())])
            
            def get_field_descriptions(self):
                return {"dummy_value": "Test"}
            
            def _compute(self, taco):
                return pa.table({"dummy_value": [999]})
        
        taco = Taco(**minimal_taco_kwargs)
        taco.extend_with(DummyExtension())
        
        assert getattr(taco, "dummy_value") == 999