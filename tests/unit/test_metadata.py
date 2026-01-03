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
    _compute_real_mask,
    _count_real_in_slice,
    _find_best_group,
    _process_level1,
    _process_level_n,
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

    def test_empty_array(self):
        ids = pa.chunked_array([[]], type=pa.string())
        mask = _compute_real_mask(ids)
        assert mask.to_pylist() == []


class TestCountRealInSlice:
    def test_count_in_slice(self):
        mask = pa.chunked_array([[True, True, False, True, False, False]])
        assert _count_real_in_slice(mask, start=0, length=3) == 2
        assert _count_real_in_slice(mask, start=3, length=3) == 1

    def test_count_full_array(self):
        mask = pa.chunked_array([[True, False, True, True]])
        assert _count_real_in_slice(mask, start=0, length=4) == 3




class TestFindBestGroup:
    def test_finds_group_with_most_real(self):
        table = pa.table({
            "id": ["a", f"{PADDING_PREFIX}0", "b", "c"],
            "type": ["FILE", "FILE", "FILE", "FILE"],
        })
        mask = _compute_real_mask(table.column("id"))
        ids, types, idx = _find_best_group(table, mask, group_size=2, num_groups=2)
        assert idx == 1
        assert ids == ["b", "c"]

    def test_early_exit_on_perfect_group(self):
        table = pa.table({
            "id": ["a", "b", "c", "d"],
            "type": ["FILE"] * 4,
        })
        mask = _compute_real_mask(table.column("id"))
        ids, types, idx = _find_best_group(table, mask, group_size=2, num_groups=2)
        assert idx == 0

    def test_single_group(self):
        table = pa.table({
            "id": ["x", "y"],
            "type": ["FILE", "FILE"],
        })
        mask = _compute_real_mask(table.column("id"))
        ids, types, idx = _find_best_group(table, mask, group_size=2, num_groups=1)
        assert idx == 0
        assert ids == ["x", "y"]


class TestProcessLevel1:
    def test_basic_processing(self):
        parent = pa.table({
            "id": ["folder_0", "folder_1"],
            "type": ["FOLDER", "FOLDER"],
        })
        children = pa.table({
            "id": ["a", "b", "c", "d"],
            "type": ["FILE"] * 4,
        })
        result = _process_level1(children, parent)
        assert result["n"] == 4
        assert len(result["type"]) == 2
        assert len(result["id"]) == 2

    def test_with_padding(self):
        parent = pa.table({
            "id": ["folder_0", "folder_1"],
            "type": ["FOLDER", "FOLDER"],
        })
        children = pa.table({
            "id": [f"{PADDING_PREFIX}0", "real_a", "real_b", "real_c"],
            "type": ["FILE"] * 4,
        })
        result = _process_level1(children, parent)
        assert f"{PADDING_PREFIX}0" not in result["id"]


class TestProcessLevelN:
    def test_with_folder_in_pattern(self):
        parent = pa.table({
            "id": ["sf_0", "sf_1", "sf_2", "sf_3"],
            "type": ["FOLDER", "FOLDER", "FOLDER", "FOLDER"],
            METADATA_CURRENT_ID: [0, 1, 2, 3],
        })
        parent_pattern = ["FOLDER", "FOLDER"]
        children = pa.table({
            "id": ["leaf_0", "leaf_1"] * 4,
            "type": ["FILE"] * 8,
            METADATA_PARENT_ID: [0, 0, 1, 1, 2, 2, 3, 3],
        })
        result = _process_level_n(children, parent, parent_pattern, depth=2)
        assert len(result) == 2

    def test_no_folders_returns_empty(self):
        parent = pa.table({
            "id": ["file_0", "file_1"],
            "type": ["FILE", "FILE"],
            METADATA_CURRENT_ID: [0, 1],
        })
        parent_pattern = ["FILE", "FILE"]
        children = pa.table({
            "id": [],
            "type": [],
            METADATA_PARENT_ID: [],
        })
        result = _process_level_n(children, parent, parent_pattern, depth=2)
        assert result == []


class TestGeneratePitSchema:
    def test_empty_tables_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            generate_pit_schema([])

    def test_missing_type_column_raises(self):
        table = pa.table({"id": ["a"]})
        with pytest.raises(ValueError, match="missing 'type'"):
            generate_pit_schema([table])

    def test_single_level_flat(self):
        level0 = pa.table({
            "id": ["s0", "s1", "s2"],
            "type": ["FILE", "FILE", "FILE"],
        })
        schema = generate_pit_schema([level0])
        assert schema["root"]["n"] == 3
        assert schema["root"]["type"] == "FILE"
        assert schema["hierarchy"] == {}

    def test_two_level_hierarchy(self):
        level0 = pa.table({
            "id": ["folder_0", "folder_1"],
            "type": ["FOLDER", "FOLDER"],
            METADATA_CURRENT_ID: [0, 1],
        })
        level1 = pa.table({
            "id": ["child_0", "child_1", "child_2", "child_3"],
            "type": ["FILE", "FILE", "FILE", "FILE"],
            METADATA_PARENT_ID: [0, 0, 1, 1],
        })
        schema = generate_pit_schema([level0, level1])
        assert schema["root"]["n"] == 2
        assert schema["root"]["type"] == "FOLDER"
        assert "1" in schema["hierarchy"]
        assert schema["hierarchy"]["1"][0]["n"] == 4

    def test_missing_parent_id_raises(self):
        level0 = pa.table({
            "id": ["folder_0"],
            "type": ["FOLDER"],
            METADATA_CURRENT_ID: [0],
        })
        level1 = pa.table({
            "id": ["child_0"],
            "type": ["FILE"],
        })
        with pytest.raises(ValueError, match="parent_id"):
            generate_pit_schema([level0, level1])

    def test_empty_level_skipped(self):
        level0 = pa.table({
            "id": ["folder_0"],
            "type": ["FOLDER"],
            METADATA_CURRENT_ID: [0],
        })
        level1 = pa.table({
            "id": [],
            "type": [],
            METADATA_PARENT_ID: [],
        })
        schema = generate_pit_schema([level0, level1])
        assert "1" not in schema["hierarchy"]


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
            assert len(field) == 3

    def test_multiple_levels(self, make_nested_taco):
        taco = make_nested_taco(n_folders=2, n_children=2)
        gen = MetadataGenerator(taco)
        pkg = gen.generate_all_levels()
        schema = generate_field_schema(pkg.levels, taco)
        assert "level0" in schema
        assert "level1" in schema


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
        assert pkg.levels[0].num_rows == 2
        assert pkg.levels[1].num_rows == 6

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


class TestMetadataGeneratorDeepHierarchy:
    """Tests for 3+ level hierarchies (depth > 1 in _add_relative_paths)."""

    def test_three_level_generates_all_levels(self, make_deep_nested_taco):
        taco = make_deep_nested_taco(n_folders=2, n_subfolders=2, n_leaves=2)
        gen = MetadataGenerator(taco)
        pkg = gen.generate_all_levels()
        assert len(pkg.levels) >= 3
        assert pkg.levels[0].num_rows == 2
        assert pkg.levels[1].num_rows == 4
        assert pkg.levels[2].num_rows == 8

    def test_relative_paths_depth_2(self, make_deep_nested_taco):
        taco = make_deep_nested_taco(n_folders=2, n_subfolders=2, n_leaves=2)
        gen = MetadataGenerator(taco)
        pkg = gen.generate_all_levels()
        level2 = pkg.levels[2]
        assert METADATA_RELATIVE_PATH in level2.schema.names
        paths = level2.column(METADATA_RELATIVE_PATH).to_pylist()
        assert all(p.count("/") >= 2 for p in paths)

    def test_local_metadata_nested_folders(self, make_deep_nested_taco):
        taco = make_deep_nested_taco(n_folders=2, n_subfolders=2, n_leaves=2)
        gen = MetadataGenerator(taco)
        pkg = gen.generate_all_levels()
        assert "DATA/folder_0/" in pkg.local_metadata
        assert "DATA/folder_0/subfolder_0/" in pkg.local_metadata
        assert "DATA/folder_1/subfolder_1/" in pkg.local_metadata

    def test_parent_chain_intact(self, make_deep_nested_taco):
        taco = make_deep_nested_taco(n_folders=2, n_subfolders=2, n_leaves=2)
        gen = MetadataGenerator(taco)
        pkg = gen.generate_all_levels()
        l0_ids = set(pkg.levels[0].column(METADATA_CURRENT_ID).to_pylist())
        l1_parent_ids = set(pkg.levels[1].column(METADATA_PARENT_ID).to_pylist())
        l1_ids = set(pkg.levels[1].column(METADATA_CURRENT_ID).to_pylist())
        l2_parent_ids = set(pkg.levels[2].column(METADATA_PARENT_ID).to_pylist())
        assert l1_parent_ids.issubset(l0_ids)
        assert l2_parent_ids.issubset(l1_ids)