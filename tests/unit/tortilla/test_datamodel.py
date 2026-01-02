"""Tests for Tortilla datamodel."""

import pyarrow as pa
import pytest

from tacotoolbox.sample.datamodel import Sample
from tacotoolbox.tortilla.datamodel import Tortilla, TortillaExtension


class TestTortillaConstruction:

    def test_empty_samples_raises(self):
        with pytest.raises(ValueError, match="empty samples list"):
            Tortilla(samples=[])

    def test_duplicate_ids_raises_with_count(self, make_sample):
        s1 = make_sample("dupe")
        s2 = make_sample("dupe")
        s3 = make_sample("unique")

        with pytest.raises(ValueError, match=r"'dupe' \(2x\)"):
            Tortilla(samples=[s1, s2, s3])

    def test_single_sample_valid(self, make_sample):
        t = Tortilla(samples=[make_sample("only_one")])
        assert len(t) == 1

    def test_metadata_table_has_core_fields(self, make_tortilla):
        t = make_tortilla(n_samples=2)
        schema_names = t.metadata_table.schema.names
        assert "id" in schema_names
        assert "type" in schema_names
        assert "path" in schema_names

    def test_size_bytes_accumulates(self, make_sample):
        s1 = make_sample("a", content=b"x" * 100)
        s2 = make_sample("b", content=b"x" * 200)
        t = Tortilla(samples=[s1, s2])
        assert t._size_bytes == s1._size_bytes + s2._size_bytes


class TestStrictSchema:

    def test_schema_mismatch_shows_missing_columns(self, make_sample):
        s1 = make_sample("s1")
        s1.extend_with(pa.table({"extra_col": [1]}))

        s2 = make_sample("s2")  # no extra_col

        with pytest.raises(ValueError, match="Missing columns.*extra_col"):
            Tortilla(samples=[s1, s2], strict_schema=True)

    def test_schema_mismatch_shows_extra_columns(self, make_sample):
        s1 = make_sample("s1")

        s2 = make_sample("s2")
        s2.extend_with(pa.table({"surprise": ["val"]}))

        with pytest.raises(ValueError, match="Extra columns.*surprise"):
            Tortilla(samples=[s1, s2], strict_schema=True)

    def test_loose_schema_fills_none(self, make_sample):
        s1 = make_sample("s1")
        s1.extend_with(pa.table({"only_in_s1": [42]}))

        s2 = make_sample("s2")

        t = Tortilla(samples=[s1, s2], strict_schema=False)

        col = t.metadata_table.column("only_in_s1")
        assert col[0].as_py() == 42
        assert col[1].as_py() is None


class TestPadding:

    def test_pad_to_makes_length_divisible(self, make_sample):
        samples = [make_sample(f"s{i}") for i in range(7)]
        t = Tortilla(samples=samples, pad_to=4)
        assert len(t) % 4 == 0
        assert len(t) == 8

    def test_no_padding_when_already_divisible(self, make_sample):
        samples = [make_sample(f"s{i}") for i in range(4)]
        t = Tortilla(samples=samples, pad_to=4)
        assert len(t) == 4

    def test_padding_ids_have_tacopad_prefix(self, make_sample):
        samples = [make_sample(f"s{i}") for i in range(3)]
        t = Tortilla(samples=samples, pad_to=4)

        padding_sample = t.samples[-1]
        assert padding_sample.id.startswith("__TACOPAD__")

    def test_padding_preserves_extension_schema(self, make_sample):
        s = make_sample("real")
        s.extend_with(pa.table({"custom_field": [123]}))

        t = Tortilla(samples=[s], pad_to=2)

        assert "custom_field" in t.metadata_table.schema.names
        # Padding sample has None for extension field
        assert t.metadata_table.column("custom_field")[1].as_py() is None


class TestDepth:

    def test_flat_tortilla_depth_zero(self, make_tortilla):
        t = make_tortilla(n_samples=5)
        assert t._current_depth == 0

    def test_nested_tortilla_depth_one(self, make_nested_tortilla):
        t = make_nested_tortilla(n_folders=2, n_children=3)
        assert t._current_depth == 1

    def test_double_nested_depth_two(self, make_sample):
        # level 2: leaf samples
        leaves = [make_sample(f"leaf_{i}") for i in range(2)]
        inner = Tortilla(samples=leaves)

        # level 1: folder containing inner
        mid_sample = Sample(id="mid", path=inner, type="FOLDER")
        middle = Tortilla(samples=[mid_sample])

        # level 0: folder containing middle
        outer_sample = Sample(id="outer", path=middle, type="FOLDER")
        outer = Tortilla(samples=[outer_sample])

        assert outer._current_depth == 2


class TestExportMetadata:

    def test_deep_zero_returns_current_level(self, make_tortilla):
        t = make_tortilla(n_samples=3)
        table = t.export_metadata(deep=0)
        assert table.num_rows == 3

    def test_deep_exceeds_structure_raises(self, make_tortilla):
        t = make_tortilla(n_samples=3)  # flat, depth=0
        with pytest.raises(ValueError, match="only has 0 levels"):
            t.export_metadata(deep=1)

    def test_deep_negative_raises(self, make_tortilla):
        t = make_tortilla(n_samples=2)
        with pytest.raises(ValueError, match="non-negative"):
            t.export_metadata(deep=-1)

    def test_deep_one_expands_children(self, make_nested_tortilla):
        t = make_nested_tortilla(n_folders=2, n_children=3)
        table = t.export_metadata(deep=1)
        # 2 folders * 3 children = 6 rows
        assert table.num_rows == 6
        assert "internal:current_id" in table.schema.names
        assert "internal:parent_id" in table.schema.names


class TestExtendWith:

    def test_row_count_mismatch_raises(self, make_tortilla):
        t = make_tortilla(n_samples=3)
        wrong_rows = pa.table({"col": [1, 2]})  # 2 rows, need 3

        class BadExtension(TortillaExtension):
            def get_schema(self):
                return pa.schema([pa.field("col", pa.int64())])

            def get_field_descriptions(self):
                return {"col": "test"}

            def _compute(self, tortilla):
                return wrong_rows

        with pytest.raises(ValueError, match="returned 2 rows.*expected 3"):
            t.extend_with(BadExtension())

    def test_column_conflict_raises(self, make_tortilla):
        t = make_tortilla(n_samples=2)

        class ConflictExtension(TortillaExtension):
            def get_schema(self):
                return pa.schema([pa.field("id", pa.string())])

            def get_field_descriptions(self):
                return {"id": "conflict"}

            def _compute(self, tortilla):
                return pa.table({"id": ["a", "b"]})

        with pytest.raises(ValueError, match="Column conflicts.*id"):
            t.extend_with(ConflictExtension())

    def test_extend_with_chains(self, make_tortilla):
        t = make_tortilla(n_samples=2)

        class Ext1(TortillaExtension):
            def get_schema(self):
                return pa.schema([pa.field("ext1", pa.int64())])

            def get_field_descriptions(self):
                return {"ext1": "first"}

            def _compute(self, tortilla):
                return pa.table({"ext1": [1, 2]})

        class Ext2(TortillaExtension):
            def get_schema(self):
                return pa.schema([pa.field("ext2", pa.int64())])

            def get_field_descriptions(self):
                return {"ext2": "second"}

            def _compute(self, tortilla):
                return pa.table({"ext2": [3, 4]})

        result = t.extend_with(Ext1()).extend_with(Ext2())
        assert result is t
        assert "ext1" in t.metadata_table.schema.names
        assert "ext2" in t.metadata_table.schema.names


class TestPop:

    def test_pop_returns_column_as_table(self, make_tortilla):
        t = make_tortilla(n_samples=2)

        class TestExt(TortillaExtension):
            def get_schema(self):
                return pa.schema([pa.field("removable", pa.int64())])

            def get_field_descriptions(self):
                return {"removable": "test"}

            def _compute(self, tortilla):
                return pa.table({"removable": [10, 20]})

        t.extend_with(TestExt())
        popped = t.pop("removable")

        assert popped.num_rows == 2
        assert popped.column("removable").to_pylist() == [10, 20]
        assert "removable" not in t.metadata_table.schema.names

    def test_pop_core_field_raises(self, make_tortilla):
        t = make_tortilla(n_samples=2)
        with pytest.raises(ValueError, match="Cannot pop core field"):
            t.pop("id")

    def test_pop_nonexistent_raises_keyerror(self, make_tortilla):
        t = make_tortilla(n_samples=2)
        with pytest.raises(KeyError, match="does not exist"):
            t.pop("ghost_field")


class TestSampleOrdering:

    def test_file_before_folder_warns(self, make_sample):
        file_sample = make_sample("file_first")

        inner = Tortilla(samples=[make_sample("child")])
        folder_sample = Sample(id="folder_second", path=inner, type="FOLDER")

        with pytest.warns(UserWarning, match="FILE at position 0 before FOLDER"):
            Tortilla(samples=[file_sample, folder_sample])

    def test_folder_before_file_no_warning(self, make_sample):
        inner = Tortilla(samples=[make_sample("child")])
        folder_sample = Sample(id="folder_first", path=inner, type="FOLDER")
        file_sample = make_sample("file_second")

        # Should not warn
        t = Tortilla(samples=[folder_sample, file_sample])
        assert len(t) == 2

    def test_all_files_no_warning(self, make_tortilla):
        # make_tortilla creates all FILE samples
        t = make_tortilla(n_samples=3)
        assert len(t) == 3


class TestPrebuiltMetadataTable:

    def test_prebuilt_table_preserves_columns(self, make_sample):
        samples = [make_sample(f"s{i}") for i in range(2)]

        # Build table with extra column (simulating extension)
        prebuilt = pa.table(
            {
                "id": ["s0", "s1"],
                "type": ["FILE", "FILE"],
                "path": ["/fake/s0", "/fake/s1"],
                "preserved_col": [100, 200],
            }
        )

        t = Tortilla(samples=samples, _metadata_table=prebuilt)
        assert "preserved_col" in t.metadata_table.schema.names
        assert t.metadata_table.column("preserved_col").to_pylist() == [100, 200]

    def test_prebuilt_table_row_mismatch_raises(self, make_sample):
        samples = [make_sample(f"s{i}") for i in range(3)]

        wrong_rows = pa.table(
            {
                "id": ["s0", "s1"],
                "type": ["FILE", "FILE"],
                "path": ["/a", "/b"],
            }
        )

        with pytest.raises(ValueError, match="2 rows but 3 samples"):
            Tortilla(samples=samples, _metadata_table=wrong_rows)
