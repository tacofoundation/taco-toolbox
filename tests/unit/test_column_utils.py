"""Tests for tacotoolbox._column_utils module."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tacotoolbox._column_utils import (
    align_arrow_schemas,
    ensure_columns_exist,
    is_internal_column,
    read_metadata_file,
    remove_empty_columns,
    reorder_internal_columns,
    validate_schema_consistency,
    write_parquet_file,
)


class TestIsInternalColumn:
    def test_internal_prefix_detected(self):
        assert is_internal_column("internal:offset") is True
        assert is_internal_column("internal:parent_id") is True

    def test_regular_columns_not_internal(self):
        assert is_internal_column("id") is False
        assert is_internal_column("stac:crs") is False
        assert is_internal_column("internal_note") is False  # no colon


class TestAlignArrowSchemas:
    """Core function for concatenating tables with different extensions."""

    def test_empty_list_returns_empty(self):
        assert align_arrow_schemas([]) == []

    def test_single_table_unchanged(self):
        t = pa.table({"id": ["a"], "value": [1]})
        result = align_arrow_schemas([t])
        assert result[0].equals(t)

    def test_identical_schemas_preserved(self):
        t1 = pa.table({"id": ["a"], "x": [1]})
        t2 = pa.table({"id": ["b"], "x": [2]})
        result = align_arrow_schemas([t1, t2])
        assert result[0].schema.equals(result[1].schema)
        assert result[0].column_names == ["id", "x"]

    def test_missing_columns_filled_with_nulls(self):
        t1 = pa.table({"id": ["a"], "x": [1]})
        t2 = pa.table({"id": ["b"], "y": [2.0]})
        result = align_arrow_schemas([t1, t2])

        # Both should have id, x, y
        assert set(result[0].column_names) == {"id", "x", "y"}
        assert set(result[1].column_names) == {"id", "x", "y"}

        # t1 should have null for y
        assert result[0].column("y").to_pylist() == [None]
        # t2 should have null for x
        assert result[1].column("x").to_pylist() == [None]

    def test_core_fields_ordered_first(self):
        t1 = pa.table({"z_ext": [1], "id": ["a"], "type": ["FILE"], "path": ["/x"]})
        t2 = pa.table({"a_ext": [2], "id": ["b"], "type": ["FILE"], "path": ["/y"]})
        result = align_arrow_schemas([t1, t2])

        # Core fields first, then extensions alphabetically
        assert result[0].column_names[:3] == ["id", "type", "path"]
        assert result[0].column_names[3:] == ["a_ext", "z_ext"]

    def test_custom_core_fields(self):
        t1 = pa.table({"custom": [1], "other": [2]})
        t2 = pa.table({"other": [3], "extra": [4]})
        result = align_arrow_schemas([t1, t2], core_fields=["custom"])

        assert result[0].column_names[0] == "custom"

    def test_schemas_concatenatable_after_alignment(self):
        t1 = pa.table({"id": ["a"], "stac:crs": ["EPSG:4326"]})
        t2 = pa.table({"id": ["b"], "scaling:factor": [1.0]})
        aligned = align_arrow_schemas([t1, t2])

        # Should not raise
        combined = pa.concat_tables(aligned)
        assert combined.num_rows == 2


class TestReorderInternalColumns:
    def test_internal_columns_moved_to_end(self):
        t = pa.table(
            {
                "internal:offset": [0],
                "id": ["a"],
                "internal:size": [100],
                "stac:crs": ["EPSG:4326"],
            }
        )
        result = reorder_internal_columns(t)
        names = result.column_names

        # Regular columns first
        assert names.index("id") < names.index("internal:offset")
        assert names.index("stac:crs") < names.index("internal:offset")

    def test_internal_columns_follow_metadata_order(self):
        t = pa.table(
            {
                "id": ["a"],
                "internal:size": [100],
                "internal:offset": [0],
                "internal:parent_id": [1],
            }
        )
        result = reorder_internal_columns(t)
        internal_cols = [c for c in result.column_names if c.startswith("internal:")]

        # parent_id before offset before size (per METADATA_COLUMNS_ORDER)
        assert internal_cols.index("internal:parent_id") < internal_cols.index(
            "internal:offset"
        )
        assert internal_cols.index("internal:offset") < internal_cols.index(
            "internal:size"
        )


class TestRemoveEmptyColumns:
    def test_all_null_column_removed(self):
        t = pa.table(
            {
                "id": ["a", "b"],
                "empty": pa.array([None, None], type=pa.int64()),
                "valid": [1, 2],
            }
        )
        result = remove_empty_columns(t, preserve_core=False, preserve_internal=False)
        assert "empty" not in result.column_names
        assert "valid" in result.column_names

    def test_empty_string_column_removed(self):
        t = pa.table(
            {
                "id": ["a"],
                "empty_str": [""],
                "valid_str": ["hello"],
            }
        )
        result = remove_empty_columns(t, preserve_core=False)
        assert "empty_str" not in result.column_names
        assert "valid_str" in result.column_names

    def test_none_string_column_removed(self):
        t = pa.table(
            {
                "id": ["a"],
                "none_str": ["None"],
                "valid": ["actual"],
            }
        )
        result = remove_empty_columns(t, preserve_core=False)
        assert "none_str" not in result.column_names

    def test_core_fields_preserved_when_empty(self):
        t = pa.table(
            {
                "id": pa.array([None], type=pa.string()),
                "other": pa.array([None], type=pa.int64()),
            }
        )
        result = remove_empty_columns(t, preserve_core=True)
        assert "id" in result.column_names

    def test_internal_columns_preserved_when_empty(self):
        t = pa.table(
            {
                "internal:offset": pa.array([None], type=pa.int64()),
                "other": pa.array([None], type=pa.int64()),
            }
        )
        result = remove_empty_columns(t, preserve_internal=True, preserve_core=False)
        assert "internal:offset" in result.column_names

    def test_keeps_at_least_one_column(self):
        t = pa.table(
            {
                "empty1": pa.array([None], type=pa.int64()),
                "empty2": pa.array([None], type=pa.int64()),
            }
        )
        result = remove_empty_columns(t, preserve_core=False, preserve_internal=False)
        assert len(result.column_names) >= 1


class TestValidateSchemaConsistency:
    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="empty Table list"):
            validate_schema_consistency([])

    def test_single_table_passes(self):
        t = pa.table({"id": ["a"]})
        validate_schema_consistency([t])  # should not raise

    def test_identical_schemas_pass(self):
        t1 = pa.table({"id": ["a"], "x": [1]})
        t2 = pa.table({"id": ["b"], "x": [2]})
        validate_schema_consistency([t1, t2])  # should not raise

    def test_missing_column_raises(self):
        t1 = pa.table({"id": ["a"], "x": [1]})
        t2 = pa.table({"id": ["b"]})
        with pytest.raises(ValueError, match="Missing columns"):
            validate_schema_consistency([t1, t2])

    def test_extra_column_raises(self):
        t1 = pa.table({"id": ["a"]})
        t2 = pa.table({"id": ["b"], "extra": [1]})
        with pytest.raises(ValueError, match="Extra columns"):
            validate_schema_consistency([t1, t2])

    def test_type_mismatch_raises(self):
        t1 = pa.table({"id": ["a"], "x": pa.array([1], type=pa.int64())})
        t2 = pa.table({"id": ["b"], "x": pa.array([1.0], type=pa.float64())})
        with pytest.raises(ValueError, match="Type mismatches"):
            validate_schema_consistency([t1, t2])


class TestEnsureColumnsExist:
    def test_all_columns_present_passes(self):
        t = pa.table({"id": ["a"], "x": [1], "y": [2]})
        ensure_columns_exist(t, ["id", "x"])  # should not raise

    def test_missing_column_raises(self):
        t = pa.table({"id": ["a"]})
        with pytest.raises(ValueError, match="Missing required columns.*missing"):
            ensure_columns_exist(t, ["id", "missing"])


class TestParquetIO:
    def test_write_and_read_roundtrip(self, tmp_path):
        t = pa.table({"id": ["a", "b"], "internal:offset": [0, 100]})
        path = tmp_path / "test.parquet"

        write_parquet_file(t, path)
        result = read_metadata_file(path)

        assert result.equals(t)

    def test_read_unsupported_format_raises(self, tmp_path):
        path = tmp_path / "test.csv"
        path.touch()
        with pytest.raises(ValueError, match="Unsupported metadata format"):
            read_metadata_file(path)
