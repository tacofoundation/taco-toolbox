"""
Test tacocat.py - consolidate multiple tacozips into .tacocat/ folder.
"""

import json

import pyarrow.parquet as pq
import pytest

from tacotoolbox.tacocat import create_tacocat, TacoCatWriter
from tacotoolbox._exceptions import TacoConsolidationError, TacoSchemaError


class TestTacoCatStructure:
    """Test .tacocat/ folder structure."""

    def test_creates_tacocat_folder(self, nested_zips, tmp_path):
        """Must create .tacocat/ folder."""
        create_tacocat(nested_zips, output=tmp_path)

        tacocat = tmp_path / ".tacocat"
        assert tacocat.is_dir()

    def test_creates_level_parquets(self, nested_zips, tmp_path):
        """Must create level0.parquet and level1.parquet."""
        create_tacocat(nested_zips, output=tmp_path)

        tacocat = tmp_path / ".tacocat"
        assert (tacocat / "level0.parquet").exists()
        assert (tacocat / "level1.parquet").exists()

    def test_creates_collection_json(self, nested_zips, tmp_path):
        """Must create COLLECTION.json."""
        create_tacocat(nested_zips, output=tmp_path)

        tacocat = tmp_path / ".tacocat"
        assert (tacocat / "COLLECTION.json").exists()


class TestTacoCatMetadata:
    """Test consolidated metadata content."""

    def test_level0_has_all_folders(self, nested_zips, tmp_path):
        """level0.parquet must have all FOLDERs from both sources."""
        create_tacocat(nested_zips, output=tmp_path)

        table = pq.read_table(tmp_path / ".tacocat" / "level0.parquet")

        # 2 + 2 = 4 FOLDERs
        assert table.num_rows == 4

        ids = set(table.column("id").to_pylist())
        assert ids == {"sample_000", "sample_001", "sample_002", "sample_003"}

    def test_level1_has_all_children(self, nested_zips, tmp_path):
        """level1.parquet must have all children from both sources."""
        create_tacocat(nested_zips, output=tmp_path)

        table = pq.read_table(tmp_path / ".tacocat" / "level1.parquet")

        # 4 FOLDERs × 1 child = 4 rows
        assert table.num_rows == 4

    def test_has_source_file_column(self, nested_zips, tmp_path):
        """Consolidated parquets must have internal:source_file column."""
        create_tacocat(nested_zips, output=tmp_path)

        table = pq.read_table(tmp_path / ".tacocat" / "level0.parquet")

        assert "internal:source_file" in table.schema.names

        sources = set(table.column("internal:source_file").to_pylist())
        assert "nested_a.tacozip" in sources
        assert "nested_b.tacozip" in sources

    def test_collection_has_summed_n(self, nested_zips, tmp_path):
        """COLLECTION.json must have summed n values."""
        create_tacocat(nested_zips, output=tmp_path)

        with open(tmp_path / ".tacocat" / "COLLECTION.json") as f:
            collection = json.load(f)

        assert collection["taco:pit_schema"]["root"]["n"] == 4


class TestTacoCatWriter:
    """Test TacoCatWriter class."""

    def test_add_dataset_validates_file(self, tmp_path):
        """Must raise on invalid tacozip."""
        writer = TacoCatWriter(output_path=tmp_path / ".tacocat")

        with pytest.raises(FileNotFoundError):
            writer.add_dataset(tmp_path / "nonexistent.tacozip")

    def test_write_raises_without_datasets(self, tmp_path):
        """Must raise if no datasets added."""
        writer = TacoCatWriter(output_path=tmp_path / ".tacocat")

        with pytest.raises(TacoConsolidationError, match="No datasets added"):
            writer.write()

    def test_writer_accepts_parquet_kwargs(self, nested_zips, tmp_path):
        """Must accept custom parquet configuration."""
        tacocat_path = tmp_path / ".tacocat"
        tacocat_path.mkdir()

        writer = TacoCatWriter(
            output_path=tacocat_path,
            parquet_kwargs={"compression": "snappy", "compression_level": None},
        )

        for z in nested_zips:
            writer.add_dataset(z)

        writer.write()

        assert (tacocat_path / "level0.parquet").exists()


class TestTacoCatValidation:
    """Test schema validation during consolidation."""

    def test_rejects_incompatible_schemas(self, zip_fixture, nested_zip_a, tmp_path):
        """Must reject tacozips with different pit_schemas."""
        with pytest.raises(TacoSchemaError):
            create_tacocat([zip_fixture, nested_zip_a], output=tmp_path)

    def test_allows_skip_validation(self, zip_fixture, nested_zip_a, tmp_path):
        """Must allow skipping schema validation."""
        # This will likely fail during merge but not during validation
        # Just verify it doesn't raise TacoSchemaError
        try:
            create_tacocat(
                [zip_fixture, nested_zip_a],
                output=tmp_path,
                validate_schema=False,
            )
        except TacoSchemaError:
            pytest.fail("Should not raise TacoSchemaError with validate_schema=False")
        except Exception:
            # Other errors are acceptable (merge failures, etc.)
            pass


class TestTacoCatErrors:
    """Test error handling."""

    def test_raises_on_existing_tacocat(self, nested_zips, tmp_path):
        """Must raise if .tacocat/ already exists."""
        (tmp_path / ".tacocat").mkdir()

        with pytest.raises(TacoConsolidationError, match="already exists"):
            create_tacocat(nested_zips, output=tmp_path)

    def test_raises_on_file_as_output(self, nested_zips, tmp_path):
        """Must raise if output is a file, not directory."""
        output_file = tmp_path / "output.txt"
        output_file.touch()

        with pytest.raises(TacoConsolidationError, match="must be a directory"):
            create_tacocat(nested_zips, output=output_file)
