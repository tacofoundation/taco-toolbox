"""Tests for tacotoolbox._metadata module."""

import pyarrow as pa
import pytest

from tacotoolbox._constants import (
    METADATA_CURRENT_ID,
    METADATA_PARENT_ID,
    METADATA_RELATIVE_PATH,
    PADDING_PREFIX,
)
from tacotoolbox._metadata import (
    MetadataGenerator,
    MetadataPackage,
    _compute_real_mask,
    _count_real_in_slice,
    _find_best_group,
    generate_collection_json,
    generate_field_schema,
    generate_pit_schema,
)


class TestComputeRealMask:
    def test_all_real_samples(self):
        ids = pa.chunked_array([["a", "b", "c"]])
        mask = _compute_real_mask(ids)
        assert mask.to_pylist() == [True, True, True]

    def test_all_padding_samples(self):
        ids = pa.chunked_array([[f"{PADDING_PREFIX}0", f"{PADDING_PREFIX}1"]])
        mask = _compute_real_mask(ids)
        assert mask.to_pylist() == [False, False]

    def test_mixed_samples(self):
        ids = pa.chunked_array([["real", f"{PADDING_PREFIX}0", "another"]])
        mask = _compute_real_mask(ids)
        assert mask.to_pylist() == [True, False, True]


class TestCountRealInSlice:
    def test_count_in_slice(self):
        mask = pa.chunked_array([[True, True, False, True, False, False]])
        assert _count_real_in_slice(mask, start=0, length=3) == 2
        assert _count_real_in_slice(mask, start=3, length=3) == 1


class TestFindBestGroup:
    def test_finds_group_with_most_real(self):
        table = pa.table(
            {
                "id": ["a", f"{PADDING_PREFIX}0", "b", "c"],
                "type": ["FILE", "FILE", "FILE", "FILE"],
            }
        )
        mask = _compute_real_mask(table.column("id"))

        ids, types, idx = _find_best_group(table, mask, group_size=2, num_groups=2)

        assert idx == 1  # second group has 2 real, first has 1
        assert ids == ["b", "c"]

    def test_early_exit_on_perfect_group(self):
        table = pa.table(
            {
                "id": ["a", "b", "c", "d"],
                "type": ["FILE"] * 4,
            }
        )
        mask = _compute_real_mask(table.column("id"))

        ids, types, idx = _find_best_group(table, mask, group_size=2, num_groups=2)

        assert idx == 0  # first group is perfect, no need to check second


class TestGeneratePitSchema:
    def test_empty_tables_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            generate_pit_schema([])

    def test_missing_type_column_raises(self):
        table = pa.table({"id": ["a"]})
        with pytest.raises(ValueError, match="missing 'type'"):
            generate_pit_schema([table])

    def test_single_level_flat(self):
        level0 = pa.table(
            {
                "id": ["s0", "s1", "s2"],
                "type": ["FILE", "FILE", "FILE"],
            }
        )
        schema = generate_pit_schema([level0])

        assert schema["root"]["n"] == 3
        assert schema["root"]["type"] == "FILE"
        assert schema["hierarchy"] == {}

    def test_two_level_hierarchy(self):
        level0 = pa.table(
            {
                "id": ["folder_0", "folder_1"],
                "type": ["FOLDER", "FOLDER"],
                METADATA_CURRENT_ID: [0, 1],
            }
        )
        level1 = pa.table(
            {
                "id": ["child_0", "child_1", "child_2", "child_3"],
                "type": ["FILE", "FILE", "FILE", "FILE"],
                METADATA_PARENT_ID: [0, 0, 1, 1],
            }
        )
        schema = generate_pit_schema([level0, level1])

        assert schema["root"]["n"] == 2
        assert schema["root"]["type"] == "FOLDER"
        assert "1" in schema["hierarchy"]
        assert schema["hierarchy"]["1"][0]["n"] == 4

    def test_padding_excluded_from_canonical(self):
        level0 = pa.table(
            {
                "id": ["folder_0", "folder_1"],
                "type": ["FOLDER", "FOLDER"],
                METADATA_CURRENT_ID: [0, 1],
            }
        )
        level1 = pa.table(
            {
                "id": [f"{PADDING_PREFIX}0", "real_0", "real_1", "real_2"],
                "type": ["FILE", "FILE", "FILE", "FILE"],
                METADATA_PARENT_ID: [0, 0, 1, 1],
            }
        )
        schema = generate_pit_schema([level0, level1])

        # Should pick folder_1's children (2 real) over folder_0's (1 real)
        canonical_ids = schema["hierarchy"]["1"][0]["id"]
        assert f"{PADDING_PREFIX}0" not in canonical_ids


class TestGenerateFieldSchema:
    def test_includes_all_columns(self, make_taco):
        taco = make_taco()
        levels = [taco.tortilla.export_metadata(deep=0)]
        schema = generate_field_schema(levels, taco)

        assert "level0" in schema
        field_names = [f[0] for f in schema["level0"]]
        assert "id" in field_names
        assert "type" in field_names

    def test_includes_type_info(self, make_taco):
        taco = make_taco()
        levels = [taco.tortilla.export_metadata(deep=0)]
        schema = generate_field_schema(levels, taco)

        for field in schema["level0"]:
            assert len(field) == 3  # [name, type, description]


class TestGenerateCollectionJson:
    def test_excludes_tortilla(self, make_taco):
        taco = make_taco()
        collection = generate_collection_json(taco)

        assert "tortilla" not in collection
        assert "id" in collection
        assert "description" in collection

    def test_preserves_taco_fields(self, make_taco):
        taco = make_taco(taco_id="my-dataset", description="My description")
        collection = generate_collection_json(taco)

        assert collection["id"] == "my-dataset"
        assert collection["description"] == "My description"


class TestMetadataGenerator:
    def test_flat_taco_generates_single_level(self, make_taco):
        taco = make_taco()
        gen = MetadataGenerator(taco)
        pkg = gen.generate_all_levels()

        assert len(pkg.levels) >= 1
        assert pkg.levels[0].num_rows == 3
        assert METADATA_CURRENT_ID in pkg.levels[0].schema.names

    def test_nested_taco_generates_multiple_levels(self, make_nested_taco):
        taco = make_nested_taco(n_folders=2, n_children=3)
        gen = MetadataGenerator(taco)
        pkg = gen.generate_all_levels()

        assert len(pkg.levels) >= 2
        assert pkg.levels[0].num_rows == 2  # folders
        assert pkg.levels[1].num_rows == 6  # 2 folders * 3 children

    def test_local_metadata_for_folders(self, make_nested_taco):
        taco = make_nested_taco(n_folders=2, n_children=2)
        gen = MetadataGenerator(taco)
        pkg = gen.generate_all_levels()

        assert len(pkg.local_metadata) == 2
        assert "DATA/folder_0/" in pkg.local_metadata
        assert "DATA/folder_1/" in pkg.local_metadata

    def test_relative_path_added_to_level1(self, make_nested_taco):
        taco = make_nested_taco(n_folders=2, n_children=2)
        gen = MetadataGenerator(taco)
        pkg = gen.generate_all_levels()

        level1 = pkg.levels[1]
        assert METADATA_RELATIVE_PATH in level1.schema.names

        paths = level1.column(METADATA_RELATIVE_PATH).to_pylist()
        # Paths should be like "folder_0/child_0", "folder_0/child_1", etc.
        assert all("/" in p for p in paths)

    def test_parent_id_enables_joins(self, make_nested_taco):
        taco = make_nested_taco(n_folders=2, n_children=2)
        gen = MetadataGenerator(taco)
        pkg = gen.generate_all_levels()

        level0 = pkg.levels[0]
        level1 = pkg.levels[1]

        assert METADATA_PARENT_ID in level1.schema.names

        parent_ids = set(level1.column(METADATA_PARENT_ID).to_pylist())
        current_ids = set(level0.column(METADATA_CURRENT_ID).to_pylist())
        assert parent_ids.issubset(current_ids)

    def test_pit_schema_in_package(self, make_taco):
        taco = make_taco()
        gen = MetadataGenerator(taco)
        pkg = gen.generate_all_levels()

        assert "root" in pkg.pit_schema
        assert "hierarchy" in pkg.pit_schema

    def test_field_schema_in_package(self, make_taco):
        taco = make_taco()
        gen = MetadataGenerator(taco)
        pkg = gen.generate_all_levels()

        assert "level0" in pkg.field_schema

    def test_collection_in_package(self, make_taco):
        taco = make_taco(taco_id="test-id")
        gen = MetadataGenerator(taco)
        pkg = gen.generate_all_levels()

        assert pkg.collection["id"] == "test-id"
        assert "tortilla" not in pkg.collection
