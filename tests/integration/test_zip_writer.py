"""
Create ZIP datasets and compare against golden files in fixtures/.
"""

import json
import zipfile

import pyarrow.parquet as pq
import tacozip

from tacotoolbox import create
from tacotoolbox.datamodel import Sample, Taco, Tortilla
from tacotoolbox.taco.datamodel import Provider


def _create_flat_taco(geotiffs_dir):
    """Create Taco matching fixtures/zip/simple structure."""
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


class TestZipAgainstGoldenFile:
    """Compare created ZIP against fixtures/zip/simple golden file."""

    def test_data_entries_match_golden(self, geotiffs_dir, zip_fixture, tmp_path):
        """DATA/ entries must match golden."""
        taco = _create_flat_taco(geotiffs_dir)
        output = tmp_path / "dataset.tacozip"
        create(taco, output)

        with zipfile.ZipFile(zip_fixture) as zf:
            golden_data = {n for n in zf.namelist() if n.startswith("DATA/")}
        with zipfile.ZipFile(output) as zf:
            created_data = {n for n in zf.namelist() if n.startswith("DATA/")}

        assert created_data == golden_data

    def test_level0_schema_matches_golden(self, geotiffs_dir, zip_fixture, tmp_path):
        """level0.parquet schema must match golden."""
        taco = _create_flat_taco(geotiffs_dir)
        output = tmp_path / "dataset.tacozip"
        create(taco, output)

        with zipfile.ZipFile(zip_fixture) as zf:
            golden = pq.read_table(zf.open("METADATA/level0.parquet"))
        with zipfile.ZipFile(output) as zf:
            created = pq.read_table(zf.open("METADATA/level0.parquet"))

        assert created.schema.names == golden.schema.names

    def test_level0_ids_match_golden(self, geotiffs_dir, zip_fixture, tmp_path):
        """level0.parquet sample IDs must match golden."""
        taco = _create_flat_taco(geotiffs_dir)
        output = tmp_path / "dataset.tacozip"
        create(taco, output)

        with zipfile.ZipFile(zip_fixture) as zf:
            golden = pq.read_table(zf.open("METADATA/level0.parquet"))
        with zipfile.ZipFile(output) as zf:
            created = pq.read_table(zf.open("METADATA/level0.parquet"))

        golden_ids = set(golden.column("id").to_pylist())
        created_ids = set(created.column("id").to_pylist())

        assert created_ids == golden_ids

    def test_level0_types_match_golden(self, geotiffs_dir, zip_fixture, tmp_path):
        """level0.parquet types must match golden."""
        taco = _create_flat_taco(geotiffs_dir)
        output = tmp_path / "dataset.tacozip"
        create(taco, output)

        with zipfile.ZipFile(zip_fixture) as zf:
            golden = pq.read_table(zf.open("METADATA/level0.parquet"))
        with zipfile.ZipFile(output) as zf:
            created = pq.read_table(zf.open("METADATA/level0.parquet"))

        assert created.column("type").to_pylist() == golden.column("type").to_pylist()

    def test_collection_pit_schema_matches_golden(
        self, geotiffs_dir, zip_fixture, tmp_path
    ):
        """COLLECTION.json pit_schema must match golden."""
        taco = _create_flat_taco(geotiffs_dir)
        output = tmp_path / "dataset.tacozip"
        create(taco, output)

        with zipfile.ZipFile(zip_fixture) as zf:
            golden = json.load(zf.open("COLLECTION.json"))
        with zipfile.ZipFile(output) as zf:
            created = json.load(zf.open("COLLECTION.json"))

        assert created["taco:pit_schema"] == golden["taco:pit_schema"]

    def test_collection_field_schema_matches_golden(
        self, geotiffs_dir, zip_fixture, tmp_path
    ):
        """COLLECTION.json field_schema must match golden."""
        taco = _create_flat_taco(geotiffs_dir)
        output = tmp_path / "dataset.tacozip"
        create(taco, output)

        with zipfile.ZipFile(zip_fixture) as zf:
            golden = json.load(zf.open("COLLECTION.json"))
        with zipfile.ZipFile(output) as zf:
            created = json.load(zf.open("COLLECTION.json"))

        assert created["taco:field_schema"] == golden["taco:field_schema"]


class TestZipStructure:
    """Verify ZIP structure requirements."""

    def test_is_valid_zipfile(self, geotiffs_dir, tmp_path):
        taco = _create_flat_taco(geotiffs_dir)
        output = tmp_path / "dataset.tacozip"
        create(taco, output)

        assert zipfile.is_zipfile(output)

    def test_has_required_entries(self, geotiffs_dir, tmp_path):
        taco = _create_flat_taco(geotiffs_dir)
        output = tmp_path / "dataset.tacozip"
        create(taco, output)

        with zipfile.ZipFile(output) as zf:
            names = zf.namelist()

        assert "TACO_HEADER" in names
        assert "COLLECTION.json" in names
        assert "METADATA/level0.parquet" in names

    def test_has_taco_header_readable(self, geotiffs_dir, tmp_path):
        """TACO_HEADER must be readable by tacozip library."""
        taco = _create_flat_taco(geotiffs_dir)
        output = tmp_path / "dataset.tacozip"
        create(taco, output)

        header = tacozip.read_header(str(output))
        assert header is not None
        assert len(header) > 0


class TestZipOffsets:
    """Verify ZIP-specific offset/size columns."""

    def test_level0_has_offset_and_size(self, geotiffs_dir, tmp_path):
        taco = _create_flat_taco(geotiffs_dir)
        output = tmp_path / "dataset.tacozip"
        create(taco, output)

        with zipfile.ZipFile(output) as zf:
            table = pq.read_table(zf.open("METADATA/level0.parquet"))

        assert "internal:offset" in table.schema.names
        assert "internal:size" in table.schema.names

    def test_offsets_are_positive(self, geotiffs_dir, tmp_path):
        taco = _create_flat_taco(geotiffs_dir)
        output = tmp_path / "dataset.tacozip"
        create(taco, output)

        with zipfile.ZipFile(output) as zf:
            table = pq.read_table(zf.open("METADATA/level0.parquet"))

        offsets = table.column("internal:offset").to_pylist()
        sizes = table.column("internal:size").to_pylist()

        assert all(o > 0 for o in offsets)
        assert all(s > 0 for s in sizes)

    def test_offsets_point_to_valid_data(self, geotiffs_dir, tmp_path):
        """Offsets must point within the ZIP file bounds."""
        taco = _create_flat_taco(geotiffs_dir)
        output = tmp_path / "dataset.tacozip"
        create(taco, output)

        zip_size = output.stat().st_size

        with zipfile.ZipFile(output) as zf:
            table = pq.read_table(zf.open("METADATA/level0.parquet"))

        offsets = table.column("internal:offset").to_pylist()
        sizes = table.column("internal:size").to_pylist()

        for offset, size in zip(offsets, sizes):
            assert offset + size <= zip_size


# --- NESTED TESTS (no golden file, verify structure only) ---


def _create_nested_taco(geotiffs_dir):
    """Create PIT-compliant nested Taco."""
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


class TestNestedZipStructure:
    """Verify nested ZIP structure (no golden file)."""

    def test_has_folder_entries(self, geotiffs_dir, tmp_path):
        taco = _create_nested_taco(geotiffs_dir)
        output = tmp_path / "dataset.tacozip"
        create(taco, output)

        with zipfile.ZipFile(output) as zf:
            names = zf.namelist()

        assert any("folder_000/" in n for n in names)
        assert any("folder_001/" in n for n in names)

    def test_has_local_meta_entries(self, geotiffs_dir, tmp_path):
        taco = _create_nested_taco(geotiffs_dir)
        output = tmp_path / "dataset.tacozip"
        create(taco, output)

        with zipfile.ZipFile(output) as zf:
            names = zf.namelist()

        assert "DATA/folder_000/__meta__" in names
        assert "DATA/folder_001/__meta__" in names

    def test_has_level1_parquet(self, geotiffs_dir, tmp_path):
        taco = _create_nested_taco(geotiffs_dir)
        output = tmp_path / "dataset.tacozip"
        create(taco, output)

        with zipfile.ZipFile(output) as zf:
            names = zf.namelist()
            assert "METADATA/level1.parquet" in names

            table = pq.read_table(zf.open("METADATA/level1.parquet"))
            assert table.num_rows == 2

    def test_level1_has_offset_and_size(self, geotiffs_dir, tmp_path):
        """Nested children must also have offset/size in ZIP."""
        taco = _create_nested_taco(geotiffs_dir)
        output = tmp_path / "dataset.tacozip"
        create(taco, output)

        with zipfile.ZipFile(output) as zf:
            table = pq.read_table(zf.open("METADATA/level1.parquet"))

        assert "internal:offset" in table.schema.names
        assert "internal:size" in table.schema.names
