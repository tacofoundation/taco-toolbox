"""
Create FOLDER datasets and compare against golden files in fixtures/.
"""

import json

import pyarrow.parquet as pq

from tacotoolbox import create
from tacotoolbox.datamodel import Sample, Taco, Tortilla
from tacotoolbox.taco.datamodel import Provider


def _create_flat_taco(geotiffs_dir):
    """Create Taco matching fixtures/folder/simple structure."""
    geotiffs = sorted(geotiffs_dir.glob("*.tif"))
    samples = [Sample(id=tif.stem, path=tif) for tif in geotiffs]

    return Taco(
        id="test-fixtures",
        dataset_version="1.0.0",
        description="Test fixtures for tacotoolbox",
        licenses=["CC-BY-4.0"],
        providers=[Provider(name="Test", roles=["producer"])],
        tasks=["classification"],
        tortilla=Tortilla(samples=samples),
    )


class TestFolderAgainstGoldenFile:
    """Compare created FOLDER against fixtures/folder/simple golden file."""

    def test_data_files_match_golden(self, geotiffs_dir, folder_fixture, tmp_path):
        """DATA/ must have exact same files as golden."""
        taco = _create_flat_taco(geotiffs_dir)
        output = tmp_path / "dataset"
        create(taco, output)

        golden_files = {f.name for f in (folder_fixture / "DATA").iterdir()}
        created_files = {f.name for f in (output / "DATA").iterdir()}

        assert created_files == golden_files

    def test_level0_schema_matches_golden(self, geotiffs_dir, folder_fixture, tmp_path):
        """level0.parquet schema must match golden (column names and order)."""
        taco = _create_flat_taco(geotiffs_dir)
        output = tmp_path / "dataset"
        create(taco, output)

        golden = pq.read_table(folder_fixture / "METADATA" / "level0.parquet")
        created = pq.read_table(output / "METADATA" / "level0.parquet")

        # Same columns in same order
        assert created.schema.names == golden.schema.names

    def test_level0_ids_match_golden(self, geotiffs_dir, folder_fixture, tmp_path):
        """level0.parquet sample IDs must match golden."""
        taco = _create_flat_taco(geotiffs_dir)
        output = tmp_path / "dataset"
        create(taco, output)

        golden = pq.read_table(folder_fixture / "METADATA" / "level0.parquet")
        created = pq.read_table(output / "METADATA" / "level0.parquet")

        golden_ids = set(golden.column("id").to_pylist())
        created_ids = set(created.column("id").to_pylist())

        assert created_ids == golden_ids

    def test_level0_types_match_golden(self, geotiffs_dir, folder_fixture, tmp_path):
        """level0.parquet types must all be FILE (matching golden)."""
        taco = _create_flat_taco(geotiffs_dir)
        output = tmp_path / "dataset"
        create(taco, output)

        golden = pq.read_table(folder_fixture / "METADATA" / "level0.parquet")
        created = pq.read_table(output / "METADATA" / "level0.parquet")

        golden_types = golden.column("type").to_pylist()
        created_types = created.column("type").to_pylist()

        assert created_types == golden_types

    def test_collection_pit_schema_matches_golden(
        self, geotiffs_dir, folder_fixture, tmp_path
    ):
        """COLLECTION.json pit_schema must match golden."""
        taco = _create_flat_taco(geotiffs_dir)
        output = tmp_path / "dataset"
        create(taco, output)

        with open(folder_fixture / "COLLECTION.json") as f:
            golden = json.load(f)
        with open(output / "COLLECTION.json") as f:
            created = json.load(f)

        assert created["taco:pit_schema"] == golden["taco:pit_schema"]

    def test_collection_field_schema_matches_golden(
        self, geotiffs_dir, folder_fixture, tmp_path
    ):
        """COLLECTION.json field_schema must match golden."""
        taco = _create_flat_taco(geotiffs_dir)
        output = tmp_path / "dataset"
        create(taco, output)

        with open(folder_fixture / "COLLECTION.json") as f:
            golden = json.load(f)
        with open(output / "COLLECTION.json") as f:
            created = json.load(f)

        assert created["taco:field_schema"] == golden["taco:field_schema"]


class TestFolderStructure:
    """Verify FOLDER structure requirements (not against golden)."""

    def test_creates_required_directories(self, geotiffs_dir, tmp_path):
        taco = _create_flat_taco(geotiffs_dir)
        output = tmp_path / "dataset"
        create(taco, output)

        assert (output / "DATA").is_dir()
        assert (output / "METADATA").is_dir()
        assert (output / "COLLECTION.json").is_file()
        assert (output / "METADATA" / "level0.parquet").is_file()

    def test_current_id_is_sequential(self, geotiffs_dir, tmp_path):
        """internal:current_id must be 0, 1, 2, ..."""
        taco = _create_flat_taco(geotiffs_dir)
        output = tmp_path / "dataset"
        create(taco, output)

        table = pq.read_table(output / "METADATA" / "level0.parquet")
        current_ids = table.column("internal:current_id").to_pylist()

        assert current_ids == list(range(len(current_ids)))


# --- NESTED TESTS (no golden file, verify structure only) ---


def _create_nested_taco(geotiffs_dir):
    """Create PIT-compliant nested Taco (all children have same ID)."""
    first_tif = sorted(geotiffs_dir.glob("*.tif"))[0]

    folder_samples = []
    for i in range(2):
        child = Sample(id=first_tif.stem, path=first_tif)
        folder = Sample(id=f"folder_{i:03d}", path=Tortilla(samples=[child]))
        folder_samples.append(folder)

    return Taco(
        id="test-nested",
        dataset_version="1.0.0",
        description="Nested test",
        licenses=["CC-BY-4.0"],
        providers=[Provider(name="Test", roles=["producer"])],
        tasks=["semantic-segmentation"],
        tortilla=Tortilla(samples=folder_samples),
    )


class TestNestedFolderStructure:
    """Verify nested FOLDER structure (no golden file)."""

    def test_creates_subdirectories(self, geotiffs_dir, tmp_path):
        taco = _create_nested_taco(geotiffs_dir)
        output = tmp_path / "dataset"
        create(taco, output)

        assert (output / "DATA" / "folder_000").is_dir()
        assert (output / "DATA" / "folder_001").is_dir()

    def test_each_folder_has_meta(self, geotiffs_dir, tmp_path):
        taco = _create_nested_taco(geotiffs_dir)
        output = tmp_path / "dataset"
        create(taco, output)

        assert (output / "DATA" / "folder_000" / "__meta__").exists()
        assert (output / "DATA" / "folder_001" / "__meta__").exists()

    def test_children_have_same_id_pit(self, geotiffs_dir, tmp_path):
        """PIT: all folders must have children with same ID."""
        taco = _create_nested_taco(geotiffs_dir)
        output = tmp_path / "dataset"
        create(taco, output)

        first_tif = sorted(geotiffs_dir.glob("*.tif"))[0]
        expected_child_id = first_tif.stem

        assert (output / "DATA" / "folder_000" / expected_child_id).exists()
        assert (output / "DATA" / "folder_001" / expected_child_id).exists()

    def test_creates_level1_parquet(self, geotiffs_dir, tmp_path):
        taco = _create_nested_taco(geotiffs_dir)
        output = tmp_path / "dataset"
        create(taco, output)

        level1 = output / "METADATA" / "level1.parquet"
        assert level1.exists()

        table = pq.read_table(level1)
        assert table.num_rows == 2  # 2 folders × 1 child

    def test_level0_types_are_folder(self, geotiffs_dir, tmp_path):
        taco = _create_nested_taco(geotiffs_dir)
        output = tmp_path / "dataset"
        create(taco, output)

        table = pq.read_table(output / "METADATA" / "level0.parquet")
        types = table.column("type").to_pylist()

        assert all(t == "FOLDER" for t in types)

    def test_level1_parent_ids_reference_level0(self, geotiffs_dir, tmp_path):
        taco = _create_nested_taco(geotiffs_dir)
        output = tmp_path / "dataset"
        create(taco, output)

        table = pq.read_table(output / "METADATA" / "level1.parquet")
        parent_ids = set(table.column("internal:parent_id").to_pylist())

        # Must reference level0 indices (0 and 1)
        assert parent_ids == {0, 1}

    def test_local_meta_matches_level1_children(self, geotiffs_dir, tmp_path):
        """__meta__ files must have 1 row each (1 child per folder)."""
        taco = _create_nested_taco(geotiffs_dir)
        output = tmp_path / "dataset"
        create(taco, output)

        meta0 = pq.read_table(output / "DATA" / "folder_000" / "__meta__")
        meta1 = pq.read_table(output / "DATA" / "folder_001" / "__meta__")

        assert meta0.num_rows == 1
        assert meta1.num_rows == 1
