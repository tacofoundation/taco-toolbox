"""
Test _tacollection.py - consolidate COLLECTION.json from multiple tacozips.
"""

import json

import pytest

from tacotoolbox._tacollection import (
    create_tacollection,
    _read_single_collection,
    _sum_pit_schemas,
    _merge_spatial_extents,
    _merge_temporal_extents,
    _validate_pit_structure,
    _validate_field_schema,
)
from tacotoolbox._exceptions import TacoConsolidationError, TacoSchemaError


class TestReadCollection:
    """Test reading COLLECTION.json from tacozip."""

    def test_reads_collection_from_zip(self, zip_fixture):
        """Must read valid COLLECTION.json from tacozip."""
        collection = _read_single_collection(zip_fixture)

        assert "id" in collection
        assert "taco:pit_schema" in collection
        assert "taco:field_schema" in collection

    def test_reads_collection_from_nested_zip(self, nested_zip_a):
        """Must read COLLECTION.json from nested tacozip."""
        collection = _read_single_collection(nested_zip_a)

        assert collection["id"] == "nested-a"
        assert "taco:pit_schema" in collection
        assert "taco:field_schema" in collection

    def test_raises_on_nonexistent_file(self, tmp_path):
        """Must raise on missing file."""
        with pytest.raises(TacoConsolidationError, match="File not found"):
            _read_single_collection(tmp_path / "nonexistent.tacozip")


class TestValidatePitStructure:
    """Test PIT schema validation across collections."""

    def test_accepts_identical_pit_schemas(self, nested_zip_a, nested_zip_b):
        """Collections with same pit_schema must pass validation."""
        collections = [
            _read_single_collection(nested_zip_a),
            _read_single_collection(nested_zip_b),
        ]

        # Should not raise
        _validate_pit_structure(collections)

    def test_rejects_different_root_types(self, zip_fixture, nested_zip_a):
        """Collections with different root types must fail."""
        collections = [
            _read_single_collection(zip_fixture),  # FILE root
            _read_single_collection(nested_zip_a),  # FOLDER root
        ]

        with pytest.raises(TacoSchemaError, match="different root type"):
            _validate_pit_structure(collections)


class TestValidateFieldSchema:
    """Test field schema validation across collections."""

    def test_accepts_identical_field_schemas(self, nested_zip_a, nested_zip_b):
        """Collections with same field_schema must pass validation."""
        collections = [
            _read_single_collection(nested_zip_a),
            _read_single_collection(nested_zip_b),
        ]

        # Should not raise
        _validate_field_schema(collections)


class TestSumPitSchemas:
    """Test summing n values across pit_schemas."""

    def test_sums_root_n_values(self, nested_zip_a, nested_zip_b):
        """Must sum root.n across all collections."""
        collections = [
            _read_single_collection(nested_zip_a),
            _read_single_collection(nested_zip_b),
        ]

        result = _sum_pit_schemas(collections)

        # Each nested has 2 FOLDERs, total should be 4
        assert result["root"]["n"] == 4

    def test_sums_hierarchy_n_values(self, nested_zip_a, nested_zip_b):
        """Must sum hierarchy n values across all collections."""
        collections = [
            _read_single_collection(nested_zip_a),
            _read_single_collection(nested_zip_b),
        ]

        result = _sum_pit_schemas(collections)

        # Each nested has 2 children (1 per FOLDER), total should be 4
        if "hierarchy" in result and "level1" in result["hierarchy"]:
            level1_n = result["hierarchy"]["level1"][0]["n"]
            assert level1_n == 4


class TestMergeExtents:
    """Test merging spatial and temporal extents."""

    def test_merge_spatial_returns_global_bbox(self):
        """Must compute union of spatial extents."""
        collections = [
            {"extent": {"spatial": [-10.0, 30.0, 0.0, 40.0]}},
            {"extent": {"spatial": [100.0, -10.0, 110.0, 0.0]}},
        ]

        result = _merge_spatial_extents(collections)

        assert result == [-10.0, -10.0, 110.0, 40.0]

    def test_merge_spatial_defaults_to_global(self):
        """Must default to global extent if none found."""
        collections = [{"extent": {}}, {"extent": {}}]

        result = _merge_spatial_extents(collections)

        assert result == [-180.0, -90.0, 180.0, 90.0]

    def test_merge_temporal_returns_full_range(self):
        """Must return earliest start to latest end."""
        collections = [
            {"extent": {"temporal": ["2023-01-01T00:00:00Z", "2023-06-30T23:59:59Z"]}},
            {"extent": {"temporal": ["2023-07-01T00:00:00Z", "2023-12-31T23:59:59Z"]}},
        ]

        result = _merge_temporal_extents(collections)

        assert result[0] == "2023-01-01T00:00:00Z"
        assert result[1] == "2023-12-31T23:59:59Z"

    def test_merge_temporal_returns_none_if_empty(self):
        """Must return None if no temporal extents."""
        collections = [{"extent": {}}, {"extent": {}}]

        result = _merge_temporal_extents(collections)

        assert result is None


class TestCreateTacollection:
    """Test full consolidation workflow."""

    def test_creates_collection_json(self, nested_zips, tmp_path):
        """Must create consolidated COLLECTION.json."""
        output = tmp_path / "COLLECTION.json"

        create_tacollection(nested_zips, output=output)

        assert output.exists()

        with open(output) as f:
            collection = json.load(f)

        assert "taco:pit_schema" in collection
        assert "taco:field_schema" in collection
        assert "taco:sources" in collection

    def test_consolidated_has_summed_n(self, nested_zips, tmp_path):
        """Consolidated pit_schema must have summed n values."""
        output = tmp_path / "COLLECTION.json"

        create_tacollection(nested_zips, output=output)

        with open(output) as f:
            collection = json.load(f)

        # 2 + 2 = 4 FOLDERs
        assert collection["taco:pit_schema"]["root"]["n"] == 4

    def test_consolidated_has_sources_info(self, nested_zips, tmp_path):
        """Consolidated must track source files."""
        output = tmp_path / "COLLECTION.json"

        create_tacollection(nested_zips, output=output)

        with open(output) as f:
            collection = json.load(f)

        sources = collection["taco:sources"]
        assert sources["count"] == 2
        assert "nested_a.tacozip" in sources["files"]
        assert "nested_b.tacozip" in sources["files"]

    def test_raises_on_empty_inputs(self, tmp_path):
        """Must raise on empty inputs."""
        with pytest.raises(TacoConsolidationError, match="No datasets"):
            create_tacollection([], output=tmp_path / "out.json")

    def test_raises_on_existing_output(self, nested_zips, tmp_path):
        """Must raise if output already exists."""
        output = tmp_path / "COLLECTION.json"
        output.touch()

        with pytest.raises(TacoConsolidationError, match="already exists"):
            create_tacollection(nested_zips, output=output)
