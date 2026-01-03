"""Unit tests for tacotoolbox._tacollection module.

Pure functions tested with dicts/Paths - no real tacozip files needed.
Integration tests with real files are in tests/integration/test_tacollection.py.
"""

from pathlib import Path

import pytest

from tacotoolbox._exceptions import TacoConsolidationError, TacoSchemaError
from tacotoolbox._tacollection import (
    _collect_partition_extents,
    _merge_spatial_extents,
    _merge_temporal_extents,
    _sum_pit_schemas,
    _validate_collection_fields,
    _validate_field_schema,
    _validate_pit_structure,
    _validate_tacozip_file,
)


class TestValidateTacozipFile:
    def test_raises_file_not_found(self):
        with pytest.raises(TacoConsolidationError, match="not found"):
            _validate_tacozip_file(Path("/nonexistent/path.tacozip"))

    def test_raises_not_a_file(self, tmp_path):
        with pytest.raises(TacoConsolidationError, match="not a file"):
            _validate_tacozip_file(tmp_path)

    def test_raises_empty_file(self, tmp_path):
        empty = tmp_path / "empty.tacozip"
        empty.touch()
        with pytest.raises(TacoConsolidationError, match="empty"):
            _validate_tacozip_file(empty)

    def test_valid_file_passes(self, tmp_path):
        valid = tmp_path / "valid.tacozip"
        valid.write_bytes(b"content")
        _validate_tacozip_file(valid)


class TestValidateCollectionFields:
    def test_raises_missing_pit_schema(self, tmp_path):
        collection = {"taco:field_schema": {"level0": []}}
        with pytest.raises(TacoConsolidationError, match="pit_schema"):
            _validate_collection_fields(collection, tmp_path / "test.tacozip")

    def test_raises_missing_field_schema(self, tmp_path):
        collection = {"taco:pit_schema": {"root": {"n": 1}}}
        with pytest.raises(TacoConsolidationError, match="field_schema"):
            _validate_collection_fields(collection, tmp_path / "test.tacozip")

    def test_valid_collection_passes(self, tmp_path):
        collection = {
            "taco:pit_schema": {"root": {"n": 1}},
            "taco:field_schema": {"level0": []},
        }
        _validate_collection_fields(collection, tmp_path / "test.tacozip")


class TestValidatePitStructure:
    def test_empty_collections_passes(self):
        _validate_pit_structure([])

    def test_single_collection_passes(self):
        collections = [{"taco:pit_schema": {"root": {"type": "FILE", "n": 5}}}]
        _validate_pit_structure(collections)

    def test_identical_schemas_pass(self):
        pit = {"root": {"type": "FOLDER", "n": 2}, "hierarchy": {"1": [{"type": ["FILE"], "id": ["a"], "n": 4}]}}
        collections = [{"taco:pit_schema": pit.copy()}, {"taco:pit_schema": pit.copy()}]
        _validate_pit_structure(collections)

    def test_different_root_type_raises(self):
        collections = [
            {"taco:pit_schema": {"root": {"type": "FILE", "n": 5}}},
            {"taco:pit_schema": {"root": {"type": "FOLDER", "n": 5}}},
        ]
        with pytest.raises(TacoSchemaError, match="different root type"):
            _validate_pit_structure(collections)

    def test_different_hierarchy_levels_raises(self):
        collections = [
            {"taco:pit_schema": {"root": {"type": "FOLDER"}, "hierarchy": {"1": []}}},
            {"taco:pit_schema": {"root": {"type": "FOLDER"}, "hierarchy": {"1": [], "2": []}}},
        ]
        with pytest.raises(TacoSchemaError, match="different hierarchy levels"):
            _validate_pit_structure(collections)

    def test_different_pattern_count_raises(self):
        collections = [
            {"taco:pit_schema": {"root": {"type": "FOLDER"}, "hierarchy": {"1": [{"type": ["FILE"], "id": ["a"]}]}}},
            {"taco:pit_schema": {"root": {"type": "FOLDER"}, "hierarchy": {"1": [{"type": ["FILE"], "id": ["a"]}, {"type": ["FILE"], "id": ["b"]}]}}},
        ]
        with pytest.raises(TacoSchemaError, match="different pattern count"):
            _validate_pit_structure(collections)

    def test_different_types_raises(self):
        collections = [
            {"taco:pit_schema": {"root": {"type": "FOLDER"}, "hierarchy": {"1": [{"type": ["FILE"], "id": ["a"]}]}}},
            {"taco:pit_schema": {"root": {"type": "FOLDER"}, "hierarchy": {"1": [{"type": ["FOLDER"], "id": ["a"]}]}}},
        ]
        with pytest.raises(TacoSchemaError, match="different types"):
            _validate_pit_structure(collections)

    def test_different_ids_raises(self):
        collections = [
            {"taco:pit_schema": {"root": {"type": "FOLDER"}, "hierarchy": {"1": [{"type": ["FILE"], "id": ["a"]}]}}},
            {"taco:pit_schema": {"root": {"type": "FOLDER"}, "hierarchy": {"1": [{"type": ["FILE"], "id": ["b"]}]}}},
        ]
        with pytest.raises(TacoSchemaError, match="different ids"):
            _validate_pit_structure(collections)

    def test_missing_pit_schema_raises(self):
        collections = [{"taco:pit_schema": {"root": {"type": "FILE"}}}, {}]
        with pytest.raises(TacoSchemaError, match="missing taco:pit_schema"):
            _validate_pit_structure(collections)

    def test_first_missing_pit_schema_raises(self):
        with pytest.raises(TacoSchemaError, match="missing taco:pit_schema"):
            _validate_pit_structure([{}])


class TestValidateFieldSchema:
    def test_empty_collections_passes(self):
        _validate_field_schema([])

    def test_identical_schemas_pass(self):
        schema = {"level0": [["id", "string", ""], ["type", "string", ""]]}
        collections = [{"taco:field_schema": schema}, {"taco:field_schema": schema}]
        _validate_field_schema(collections)

    def test_different_levels_raises(self):
        collections = [
            {"taco:field_schema": {"level0": []}},
            {"taco:field_schema": {"level0": [], "level1": []}},
        ]
        with pytest.raises(TacoSchemaError, match="different field schema levels"):
            _validate_field_schema(collections)

    def test_different_fields_raises(self):
        collections = [
            {"taco:field_schema": {"level0": [["id", "string", ""]]}},
            {"taco:field_schema": {"level0": [["name", "string", ""]]}},
        ]
        with pytest.raises(TacoSchemaError, match="different field schema"):
            _validate_field_schema(collections)

    def test_missing_field_schema_raises(self):
        collections = [{"taco:field_schema": {"level0": []}}, {}]
        with pytest.raises(TacoSchemaError, match="missing taco:field_schema"):
            _validate_field_schema(collections)


class TestSumPitSchemas:
    def test_sums_root_n(self):
        collections = [
            {"taco:pit_schema": {"root": {"n": 5, "type": "FILE"}}},
            {"taco:pit_schema": {"root": {"n": 3, "type": "FILE"}}},
        ]
        result = _sum_pit_schemas(collections)
        assert result["root"]["n"] == 8

    def test_sums_hierarchy_n(self):
        collections = [
            {"taco:pit_schema": {"root": {"n": 2}, "hierarchy": {"1": [{"n": 4, "type": ["FILE"]}]}}},
            {"taco:pit_schema": {"root": {"n": 2}, "hierarchy": {"1": [{"n": 6, "type": ["FILE"]}]}}},
        ]
        result = _sum_pit_schemas(collections)
        assert result["hierarchy"]["1"][0]["n"] == 10

    def test_empty_collections_raises(self):
        with pytest.raises(TacoConsolidationError, match="empty"):
            _sum_pit_schemas([])

    def test_missing_pit_schema_raises(self):
        with pytest.raises(TacoConsolidationError, match="missing taco:pit_schema"):
            _sum_pit_schemas([{}])

    def test_missing_root_raises(self):
        with pytest.raises(TacoConsolidationError, match="missing 'root'"):
            _sum_pit_schemas([{"taco:pit_schema": {}}])

    def test_zero_sum_raises(self):
        collections = [
            {"taco:pit_schema": {"root": {"n": 0}}},
            {"taco:pit_schema": {"root": {"n": 0}}},
        ]
        with pytest.raises(TacoConsolidationError, match="zero"):
            _sum_pit_schemas(collections)


class TestMergeSpatialExtents:
    def test_returns_union_bbox(self):
        collections = [
            {"extent": {"spatial": [-10.0, 30.0, 0.0, 40.0]}},
            {"extent": {"spatial": [100.0, -10.0, 110.0, 0.0]}},
        ]
        result = _merge_spatial_extents(collections)
        assert result == [-10.0, -10.0, 110.0, 40.0]

    def test_defaults_to_global(self):
        collections = [{"extent": {}}, {"extent": {}}]
        result = _merge_spatial_extents(collections)
        assert result == [-180.0, -90.0, 180.0, 90.0]

    def test_handles_missing_extent(self):
        collections = [{}, {"extent": {"spatial": [0.0, 0.0, 1.0, 1.0]}}]
        result = _merge_spatial_extents(collections)
        assert result == [0.0, 0.0, 1.0, 1.0]

    def test_handles_none_spatial(self):
        collections = [{"extent": {"spatial": None}}, {"extent": {"spatial": [0.0, 0.0, 1.0, 1.0]}}]
        result = _merge_spatial_extents(collections)
        assert result == [0.0, 0.0, 1.0, 1.0]

    def test_single_collection(self):
        collections = [{"extent": {"spatial": [1.0, 2.0, 3.0, 4.0]}}]
        result = _merge_spatial_extents(collections)
        assert result == [1.0, 2.0, 3.0, 4.0]


class TestMergeTemporalExtents:
    def test_returns_full_range(self):
        collections = [
            {"extent": {"temporal": ["2023-01-01T00:00:00Z", "2023-06-30T23:59:59Z"]}},
            {"extent": {"temporal": ["2023-07-01T00:00:00Z", "2023-12-31T23:59:59Z"]}},
        ]
        result = _merge_temporal_extents(collections)
        assert result[0] == "2023-01-01T00:00:00Z"
        assert result[1] == "2023-12-31T23:59:59Z"

    def test_returns_none_if_empty(self):
        collections = [{"extent": {}}, {"extent": {}}]
        result = _merge_temporal_extents(collections)
        assert result is None

    def test_handles_missing_temporal(self):
        collections = [
            {},
            {"extent": {"temporal": ["2023-01-01T00:00:00Z", "2023-12-31T23:59:59Z"]}},
        ]
        result = _merge_temporal_extents(collections)
        assert result == ["2023-01-01T00:00:00Z", "2023-12-31T23:59:59Z"]

    def test_handles_offset_timezone(self):
        collections = [
            {"extent": {"temporal": ["2023-01-01T00:00:00+00:00", "2023-06-30T23:59:59+00:00"]}},
            {"extent": {"temporal": ["2023-07-01T00:00:00Z", "2023-12-31T23:59:59Z"]}},
        ]
        result = _merge_temporal_extents(collections)
        assert "2023-01-01" in result[0]
        assert "2023-12-31" in result[1]


class TestCollectPartitionExtents:
    def test_collects_all_extents(self):
        collections = [
            {"id": "ds1", "extent": {"spatial": [0, 0, 1, 1], "temporal": ["2023-01-01Z", "2023-06-30Z"]}},
            {"id": "ds2", "extent": {"spatial": [2, 2, 3, 3]}},
        ]
        paths = [Path("/data/ds1.tacozip"), Path("/data/ds2.tacozip")]
        result = _collect_partition_extents(collections, paths)
        assert len(result) == 2
        assert result[0]["file"] == "ds1.tacozip"
        assert result[0]["id"] == "ds1"
        assert result[0]["spatial"] == [0, 0, 1, 1]
        assert "temporal" in result[0]
        assert result[1]["file"] == "ds2.tacozip"
        assert "temporal" not in result[1]

    def test_handles_missing_extent(self):
        collections = [{"id": "ds1"}]
        paths = [Path("/data/ds1.tacozip")]
        result = _collect_partition_extents(collections, paths)
        assert result[0]["id"] == "ds1"
        assert "spatial" not in result[0]