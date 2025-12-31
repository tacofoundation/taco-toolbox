"""
Unit tests for Sample datamodel.

Tests cover:
- ID validation (forbidden characters, reserved prefixes)
- Path validation (existence, type checking)
- Type inference (auto, explicit FILE/FOLDER)
- Bytes handling (temp file creation, cleanup)
- Extension system (extend_with, schema tracking)
- Pop method (remove extension fields)
- Export metadata (Arrow Table generation)
"""

import pathlib

import pyarrow as pa
import pytest

from tacotoolbox.sample.datamodel import Sample
from tacotoolbox.tortilla.datamodel import Tortilla


# Test data for parametrized tests

VALID_SAMPLE_IDS = [
    "simple",
    "sample_001",
    "sample-with-dashes",
    "MixedCase123",
    "a",
    "123numeric",
    "with_underscore",
]

INVALID_SAMPLE_IDS = [
    ("path/to/file", "forward slash"),
    ("path\\to\\file", "backslash"),
    ("file:name", "colon"),
    ("__reserved", "double underscore prefix"),
    ("__TACOPAD__0", "reserved padding prefix"),
]

INVALID_EXTENSION_KEYS = [
    "has spaces",
    "has-dashes",
    "special!chars",
    "dot.notation",
]


class TestSampleIdValidation:
    """Sample ID must follow strict naming rules."""
    
    @pytest.mark.parametrize("valid_id", VALID_SAMPLE_IDS)
    def test_accepts_valid_ids(self, make_file, valid_id):
        sample = Sample(id=valid_id, path=make_file())
        assert sample.id == valid_id
    
    @pytest.mark.parametrize("invalid_id,reason", INVALID_SAMPLE_IDS)
    def test_rejects_invalid_ids(self, make_file, invalid_id, reason):
        with pytest.raises(ValueError) as exc_info:
            Sample(id=invalid_id, path=make_file())
        
        # Verify error message mentions the issue
        assert invalid_id in str(exc_info.value) or reason in str(exc_info.value).lower()
    
    def test_accepts_empty_id(self, make_file):
        # NOTE: Current implementation allows empty IDs
        # TODO: Consider if this should be rejected
        sample = Sample(id="", path=make_file())
        assert sample.id == ""


class TestSamplePathValidation:
    """Path must exist or be valid Tortilla."""
    
    def test_accepts_existing_file(self, tmp_file):
        sample = Sample(id="test", path=tmp_file)
        assert sample.path == tmp_file.absolute()
    
    def test_rejects_nonexistent_path(self, tmp_path):
        nonexistent = tmp_path / "does_not_exist.bin"
        with pytest.raises(ValueError, match="does not exist"):
            Sample(id="test", path=nonexistent)
    
    def test_accepts_tortilla(self, make_sample):
        # Create a simple tortilla
        samples = [make_sample() for _ in range(2)]
        tortilla = Tortilla(samples=samples)
        
        sample = Sample(id="folder_sample", path=tortilla)
        assert isinstance(sample.path, Tortilla)
    
    def test_normalizes_to_absolute_path(self, tmp_path, monkeypatch):
        # Create file with relative path
        f = tmp_path / "relative.bin"
        f.write_bytes(b"x")
        
        monkeypatch.chdir(tmp_path)
        sample = Sample(id="test", path=pathlib.Path("relative.bin"))
        
        assert sample.path.is_absolute()


class TestSampleTypeInference:
    """Type auto-inference and explicit validation."""
    
    def test_infers_file_from_path(self, tmp_file):
        sample = Sample(id="test", path=tmp_file)
        assert sample.type == "FILE"
    
    def test_infers_file_from_bytes(self):
        sample = Sample(id="test", path=b"some bytes")
        assert sample.type == "FILE"
    
    def test_infers_folder_from_tortilla(self, make_tortilla):
        tortilla = make_tortilla(n_samples=2)
        sample = Sample(id="test", path=tortilla)
        assert sample.type == "FOLDER"
    
    def test_accepts_explicit_file_with_path(self, tmp_file):
        sample = Sample(id="test", path=tmp_file, type="FILE")
        assert sample.type == "FILE"
    
    def test_accepts_explicit_folder_with_tortilla(self, make_tortilla):
        tortilla = make_tortilla(n_samples=2)
        sample = Sample(id="test", path=tortilla, type="FOLDER")
        assert sample.type == "FOLDER"
    
    def test_rejects_file_type_with_tortilla(self, make_tortilla):
        tortilla = make_tortilla(n_samples=2)
        with pytest.raises(ValueError, match="Type mismatch"):
            Sample(id="test", path=tortilla, type="FILE")
    
    def test_rejects_folder_type_with_path(self, tmp_file):
        with pytest.raises(ValueError, match="Type mismatch"):
            Sample(id="test", path=tmp_file, type="FOLDER")
    
    def test_type_never_auto_after_validation(self, tmp_file):
        sample = Sample(id="test", path=tmp_file, type="auto")
        assert sample.type in ("FILE", "FOLDER")
        assert sample.type != "auto"


class TestSampleBytesHandling:
    """Bytes conversion to temp files."""
    
    def test_creates_temp_file_from_bytes(self):
        sample = Sample(id="test", path=b"hello world")
        
        assert isinstance(sample.path, pathlib.Path)
        assert sample.path.exists()
        assert sample.path.read_bytes() == b"hello world"
    
    def test_creates_zero_byte_file_from_empty_bytes(self):
        sample = Sample(id="test", path=b"")
        
        assert sample.path.exists()
        assert sample.path.stat().st_size == 0
    
    def test_respects_custom_temp_dir(self, tmp_path):
        custom_dir = tmp_path / "custom_temp"
        custom_dir.mkdir()
        
        sample = Sample(id="test", path=b"data", temp_dir=custom_dir)
        
        assert custom_dir in sample.path.parents or sample.path.parent == custom_dir
    
    def test_tracks_temp_files(self):
        sample = Sample(id="test", path=b"temporary")
        assert len(sample._temp_files) == 1
        assert sample._temp_files[0] == sample.path
    
    def test_cleanup_removes_temp_file(self):
        sample = Sample(id="test", path=b"temporary")
        temp_path = sample.path
        
        assert temp_path.exists()
        sample.cleanup()
        assert not temp_path.exists()
    
    def test_cleanup_clears_temp_files_list(self):
        sample = Sample(id="test", path=b"temporary")
        assert len(sample._temp_files) == 1
        
        sample.cleanup()
        assert len(sample._temp_files) == 0
    
    def test_cleanup_is_idempotent(self):
        sample = Sample(id="test", path=b"temporary")
        sample.cleanup()
        sample.cleanup()  # Should not raise


class TestSampleExtendWith:
    """Extension system for adding metadata."""
    
    def test_extends_with_dict(self, make_sample):
        sample = make_sample()
        sample.extend_with({"custom_field": 42})
        
        assert hasattr(sample, "custom_field")
        assert sample.custom_field == 42
    
    def test_extends_with_namespaced_dict(self, make_sample):
        sample = make_sample()
        sample.extend_with({"stac:crs": "EPSG:4326"})
        
        assert getattr(sample, "stac:crs") == "EPSG:4326"
    
    def test_extends_with_single_row_arrow_table(self, make_sample, single_row_arrow_table):
        sample = make_sample()
        sample.extend_with(single_row_arrow_table)
        
        assert hasattr(sample, "arrow_field")
        assert sample.arrow_field == 123
    
    def test_rejects_multi_row_arrow_table(self, make_sample, multi_row_arrow_table):
        sample = make_sample()
        with pytest.raises(ValueError, match="exactly one row"):
            sample.extend_with(multi_row_arrow_table)
    
    def test_extends_with_sample_extension(self, make_sample, mock_extension):
        sample = make_sample()
        sample.extend_with(mock_extension)
        
        assert mock_extension._called
        assert getattr(sample, "mock:value") == 42
        assert getattr(sample, "mock:name") == "test"
    
    def test_tracks_extension_schemas(self, make_sample):
        sample = make_sample()
        sample.extend_with({"int_field": 123, "str_field": "hello"})
        
        assert "int_field" in sample._extension_schemas
        assert "str_field" in sample._extension_schemas
    
    def test_rejects_override_core_field(self, make_sample):
        sample = make_sample()
        with pytest.raises(ValueError, match="Cannot override core field"):
            sample.extend_with({"id": "new_id"})
    
    @pytest.mark.parametrize("invalid_key", INVALID_EXTENSION_KEYS)
    def test_rejects_invalid_keys(self, make_sample, invalid_key):
        sample = make_sample()
        with pytest.raises(ValueError, match="Invalid key format"):
            sample.extend_with({invalid_key: "value"})
    
    def test_multiple_extend_calls_accumulate(self, make_sample):
        sample = make_sample()
        sample.extend_with({"first": 1})
        sample.extend_with({"second": 2})
        
        assert sample.first == 1
        assert sample.second == 2
        assert "first" in sample._extension_schemas
        assert "second" in sample._extension_schemas


class TestSamplePop:
    """Pop method for removing extension fields."""
    
    def test_pops_extension_field(self, sample_with_extension):
        result = sample_with_extension.pop("custom_field")
        
        assert isinstance(result, pa.Table)
        assert result.num_rows == 1
        assert "custom_field" in result.column_names
        assert not hasattr(sample_with_extension, "custom_field")
    
    def test_pop_returns_arrow_table_with_value(self, sample_with_extension):
        result = sample_with_extension.pop("custom_field")
        
        value = result.column("custom_field")[0].as_py()
        assert value == 42
    
    def test_pop_removes_from_extension_schemas(self, sample_with_extension):
        assert "custom_field" in sample_with_extension._extension_schemas
        sample_with_extension.pop("custom_field")
        assert "custom_field" not in sample_with_extension._extension_schemas
    
    def test_rejects_pop_core_field_id(self, make_sample):
        sample = make_sample()
        with pytest.raises(ValueError, match="Cannot pop core field"):
            sample.pop("id")
    
    def test_rejects_pop_core_field_type(self, make_sample):
        sample = make_sample()
        with pytest.raises(ValueError, match="Cannot pop core field"):
            sample.pop("type")
    
    def test_rejects_pop_core_field_path(self, make_sample):
        sample = make_sample()
        with pytest.raises(ValueError, match="Cannot pop core field"):
            sample.pop("path")
    
    def test_rejects_pop_nonexistent_field(self, make_sample):
        sample = make_sample()
        with pytest.raises(KeyError):
            sample.pop("does_not_exist")


class TestSampleExportMetadata:
    """Metadata export to Arrow Table."""
    
    def test_returns_arrow_table(self, make_sample):
        sample = make_sample()
        result = sample.export_metadata()
        
        assert isinstance(result, pa.Table)
    
    def test_has_single_row(self, make_sample):
        sample = make_sample()
        result = sample.export_metadata()
        
        assert result.num_rows == 1
    
    def test_contains_core_fields(self, make_sample):
        sample = make_sample()
        result = sample.export_metadata()
        
        assert "id" in result.column_names
        assert "type" in result.column_names
        assert "path" in result.column_names
    
    def test_contains_extension_fields(self, sample_with_extension):
        result = sample_with_extension.export_metadata()
        
        assert "custom_field" in result.column_names
        assert "another:field" in result.column_names
    
    def test_path_is_posix_string_for_file(self, tmp_file):
        sample = Sample(id="test", path=tmp_file)
        result = sample.export_metadata()
        
        path_value = result.column("path")[0].as_py()
        assert isinstance(path_value, str)
        assert "/" in path_value  # POSIX format
    
    def test_path_is_none_for_folder(self, make_tortilla):
        tortilla = make_tortilla()
        sample = Sample(id="folder", path=tortilla)
        result = sample.export_metadata()
        
        path_value = result.column("path")[0].as_py()
        assert path_value is None
    
    def test_type_is_never_auto(self, tmp_file):
        sample = Sample(id="test", path=tmp_file, type="auto")
        result = sample.export_metadata()
        
        type_value = result.column("type")[0].as_py()
        assert type_value in ("FILE", "FOLDER")


class TestSamplePadding:
    """_create_padding factory method."""
    
    def test_creates_padding_sample(self):
        sample = Sample._create_padding(0)
        
        assert sample.id == "__TACOPAD__0"
        assert sample.type == "FILE"
    
    def test_padding_bypasses_id_validation(self):
        # Direct construction would fail
        with pytest.raises(ValueError):
            Sample(id="__TACOPAD__0", path=b"")
        
        # Factory method succeeds
        sample = Sample._create_padding(0)
        assert sample.id.startswith("__TACOPAD__")
    
    def test_padding_creates_zero_byte_file(self):
        sample = Sample._create_padding(0)
        
        assert sample.path.exists()
        assert sample.path.stat().st_size == 0
        assert sample._size_bytes == 0
    
    def test_padding_increments_index(self):
        s0 = Sample._create_padding(0)
        s1 = Sample._create_padding(1)
        s99 = Sample._create_padding(99)
        
        assert s0.id == "__TACOPAD__0"
        assert s1.id == "__TACOPAD__1"
        assert s99.id == "__TACOPAD__99"


class TestSampleSizeCalculation:
    """_size_bytes attribute calculation."""
    
    def test_file_size_from_path(self, tmp_path):
        f = tmp_path / "sized.bin"
        content = b"x" * 1024
        f.write_bytes(content)
        
        sample = Sample(id="test", path=f)
        assert sample._size_bytes == 1024
    
    def test_file_size_from_bytes(self):
        content = b"hello world"
        sample = Sample(id="test", path=content)
        
        assert sample._size_bytes == len(content)
    
    def test_folder_size_from_tortilla(self, make_sample):
        # Create samples with known sizes
        s1 = Sample(id="s1", path=b"aaa")  # 3 bytes
        s2 = Sample(id="s2", path=b"bbbbb")  # 5 bytes
        tortilla = Tortilla(samples=[s1, s2])
        
        folder = Sample(id="folder", path=tortilla)
        
        # Should be sum of children
        assert folder._size_bytes == 8