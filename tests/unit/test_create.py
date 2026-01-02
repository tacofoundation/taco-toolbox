"""Integration tests for create.py"""

import io
import json
import zipfile
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from tacotoolbox import create
from tacotoolbox._exceptions import TacoCreationError, TacoValidationError
from tacotoolbox.create import _group_samples_by_size, _sanitize_filename
from tacotoolbox.datamodel import Sample, Taco, Tortilla
from tacotoolbox.taco.datamodel import Provider

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
GEOTIFFS_DIR = FIXTURES_DIR / "geotiffs"
GOLDEN_ZIP = FIXTURES_DIR / "zip" / "simple" / "simple.tacozip"
GOLDEN_FOLDER = FIXTURES_DIR / "folder" / "simple"


@pytest.fixture
def geotiff_samples():
    """Samples from fixture GeoTIFFs, same as golden files."""
    geotiffs = sorted(GEOTIFFS_DIR.glob("*.tif"))
    return [Sample(id=tif.stem, path=tif) for tif in geotiffs]


@pytest.fixture
def test_taco(geotiff_samples):
    """Taco matching golden file structure."""
    return Taco(
        id="test-fixtures",
        dataset_version="1.0.0",
        description="Test fixtures for tacotoolbox",
        licenses=["CC-BY-4.0"],
        providers=[Provider(name="Test", roles=["producer"])],
        tasks=["classification"],
        tortilla=Tortilla(samples=geotiff_samples),
    )


class TestSanitizeFilename:

    def test_replaces_slashes(self):
        assert _sanitize_filename("Ocean/Sea/Lakes") == "Ocean_Sea_Lakes"

    def test_replaces_colon(self):
        assert _sanitize_filename("data:2024-01-01") == "data_2024-01-01"

    def test_replaces_special_chars(self):
        assert _sanitize_filename('file<test>?"') == "file_test"

    def test_collapses_multiple_underscores(self):
        assert _sanitize_filename("a___b   c") == "a_b_c"

    def test_strips_leading_trailing(self):
        assert _sanitize_filename("__name__") == "name"


class TestGroupSamplesBySize:

    def test_single_chunk_when_under_limit(self, geotiff_samples):
        chunks = _group_samples_by_size(geotiff_samples, max_size=100 * 1024 * 1024)
        assert len(chunks) == 1
        assert len(chunks[0]) == len(geotiff_samples)

    def test_multiple_chunks_when_over_limit(self, geotiff_samples):
        # Each GeoTIFF is ~65KB-800KB, set limit low to force splits
        chunks = _group_samples_by_size(geotiff_samples, max_size=100 * 1024)
        assert len(chunks) > 1

    def test_preserves_all_samples(self, geotiff_samples):
        chunks = _group_samples_by_size(geotiff_samples, max_size=50 * 1024)
        total = sum(len(c) for c in chunks)
        assert total == len(geotiff_samples)


class TestCreateZip:

    def test_creates_valid_zip_matching_golden(self, test_taco, tmp_path):
        output = tmp_path / "test.tacozip"
        result = create(test_taco, output)

        assert len(result) == 1
        assert result[0].exists()
        assert result[0].suffix == ".tacozip"

        # Compare structure with golden
        with zipfile.ZipFile(result[0]) as created, zipfile.ZipFile(GOLDEN_ZIP) as golden:
            created_names = set(created.namelist())
            golden_names = set(golden.namelist())

            assert "COLLECTION.json" in created_names
            assert "METADATA/level0.parquet" in created_names
            # DATA files should match
            created_data = {n for n in created_names if n.startswith("DATA/")}
            golden_data = {n for n in golden_names if n.startswith("DATA/")}
            assert created_data == golden_data

    def test_collection_json_structure(self, test_taco, tmp_path):
        output = tmp_path / "test.tacozip"
        create(test_taco, output)

        with zipfile.ZipFile(output) as zf:
            collection = json.loads(zf.read("COLLECTION.json"))

        assert collection["id"] == "test-fixtures"
        assert collection["dataset_version"] == "1.0.0"
        assert "taco:pit_schema" in collection
        assert "taco:field_schema" in collection

    def test_metadata_parquet_row_count(self, test_taco, tmp_path):
        output = tmp_path / "test.tacozip"
        create(test_taco, output)

        with zipfile.ZipFile(output) as zf:
            table = pq.read_table(io.BytesIO(zf.read("METADATA/level0.parquet")))
        assert len(table) == len(test_taco.tortilla.samples)


class TestCreateFolder:

    def test_creates_valid_folder_matching_golden(self, test_taco, tmp_path):
        output = tmp_path / "test_folder"
        result = create(test_taco, output)

        assert len(result) == 1
        assert result[0].is_dir()

        # Compare structure with golden
        created_files = {p.relative_to(result[0]) for p in result[0].rglob("*") if p.is_file()}
        golden_files = {p.relative_to(GOLDEN_FOLDER) for p in GOLDEN_FOLDER.rglob("*") if p.is_file()}

        assert created_files == golden_files

    def test_collection_json_structure(self, test_taco, tmp_path):
        output = tmp_path / "test_folder"
        create(test_taco, output)

        collection = json.loads((output / "COLLECTION.json").read_text())
        assert collection["id"] == "test-fixtures"
        assert "taco:pit_schema" in collection

    def test_metadata_parquet_row_count(self, test_taco, tmp_path):
        output = tmp_path / "test_folder"
        create(test_taco, output)

        table = pq.read_table(output / "METADATA" / "level0.parquet")
        assert len(table) == len(test_taco.tortilla.samples)


class TestAutoDetectFormat:

    def test_tacozip_extension_creates_zip(self, test_taco, tmp_path):
        output = tmp_path / "auto.tacozip"
        result = create(test_taco, output, output_format="auto")
        assert result[0].suffix == ".tacozip"

    def test_zip_extension_creates_zip(self, test_taco, tmp_path):
        output = tmp_path / "auto.zip"
        result = create(test_taco, output, output_format="auto")
        assert result[0].suffix == ".zip"

    def test_no_extension_creates_folder(self, test_taco, tmp_path):
        output = tmp_path / "auto_folder"
        result = create(test_taco, output, output_format="auto")
        assert result[0].is_dir()


class TestSplitBySize:

    def test_creates_multiple_parts(self, test_taco, tmp_path):
        output = tmp_path / "split.tacozip"
        # Low limit to force multiple files
        result = create(test_taco, output, split_size="100KB")

        assert len(result) > 1
        for path in result:
            assert "_part" in path.stem
            assert path.exists()

    def test_parts_are_valid_zips(self, test_taco, tmp_path):
        output = tmp_path / "split.tacozip"
        result = create(test_taco, output, split_size="100KB")

        for path in result:
            with zipfile.ZipFile(path) as zf:
                assert "COLLECTION.json" in zf.namelist()

    def test_auto_consolidates_tacocat(self, test_taco, tmp_path):
        output = tmp_path / "split.tacozip"
        create(test_taco, output, split_size="100KB", consolidate=True)

        tacocat_dir = tmp_path / ".tacocat"
        assert tacocat_dir.exists()
        assert (tacocat_dir / "COLLECTION.json").exists()


class TestGroupBy:

    @pytest.fixture
    def taco_with_split(self, geotiff_samples):
        """Taco with split extension on samples."""
        from tacotoolbox.sample.extensions.split import Split
        
        for i, sample in enumerate(geotiff_samples):
            split_val = "train" if i < 2 else "test"
            sample.extend_with(Split(split=split_val))

        return Taco(
            id="grouped-test",
            dataset_version="1.0.0",
            description="Test with split groups",
            licenses=["CC-BY-4.0"],
            providers=[Provider(name="Test", roles=["producer"])],
            tasks=["classification"],
            tortilla=Tortilla(samples=geotiff_samples),
        )

    def test_creates_one_zip_per_group(self, taco_with_split, tmp_path):
        output = tmp_path / "grouped.tacozip"
        result = create(taco_with_split, output, group_by="split", split_size=None)

        assert len(result) == 2
        names = {p.stem for p in result}
        assert "grouped_train" in names
        assert "grouped_test" in names

    def test_each_group_has_correct_samples(self, taco_with_split, tmp_path):
        output = tmp_path / "grouped.tacozip"
        result = create(taco_with_split, output, group_by="split", split_size=None)

        for path in result:
            with zipfile.ZipFile(path) as zf:
                table = pq.read_table(io.BytesIO(zf.read("METADATA/level0.parquet")))
            
            if "train" in path.stem:
                assert len(table) == 2
            else:
                assert len(table) == 2


class TestValidationErrors:

    def test_output_exists_raises(self, test_taco, tmp_path):
        output = tmp_path / "exists.tacozip"
        output.touch()

        with pytest.raises(TacoValidationError, match="already exists"):
            create(test_taco, output)

    def test_reserved_folder_name_raises(self, test_taco, tmp_path):
        with pytest.raises(TacoValidationError, match="reserved"):
            create(test_taco, tmp_path / "DATA", output_format="folder")

    def test_invalid_group_column_raises(self, test_taco, tmp_path):
        with pytest.raises(TacoCreationError, match="not found"):
            create(test_taco, tmp_path / "bad.tacozip", group_by="nonexistent", split_size=None)

    def test_tacocat_exists_raises(self, test_taco, tmp_path):
        (tmp_path / ".tacocat").mkdir()

        with pytest.raises(TacoValidationError, match=".tacocat"):
            create(test_taco, tmp_path / "split.tacozip", split_size="100KB", consolidate=True)