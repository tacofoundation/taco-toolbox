"""Tests for tacotoolbox._validation module."""

import pathlib
import pytest
from tacotoolbox._constants import PADDING_PREFIX
from tacotoolbox._exceptions import TacoValidationError
from tacotoolbox._validation import (
    is_padding_id,
    parse_size,
    validate_common_directory,
    validate_format_and_split,
    validate_format_value,
    validate_output_path,
    validate_split_size,
)


class TestParseSize:
    """parse_size is the core parser - test thoroughly."""

    @pytest.mark.parametrize(
        "size_str,expected",
        [
            ("4GB", 4 * 1024**3),
            ("4G", 4 * 1024**3),
            ("512MB", 512 * 1024**2),
            ("512M", 512 * 1024**2),
            ("1024KB", 1024 * 1024),
            ("1024K", 1024 * 1024),
            ("2048B", 2048),
            ("2048", 2048),  # no unit = bytes
        ],
    )
    def test_standard_units(self, size_str, expected):
        assert parse_size(size_str) == expected

    @pytest.mark.parametrize(
        "size_str,expected",
        [
            ("4.5GB", int(4.5 * 1024**3)),
            ("1.5MB", int(1.5 * 1024**2)),
            ("0.5KB", int(0.5 * 1024)),
        ],
    )
    def test_decimal_values(self, size_str, expected):
        assert parse_size(size_str) == expected

    @pytest.mark.parametrize("size_str", ["4gb", "4Gb", "4gB", "4 GB", "  4GB  "])
    def test_case_and_whitespace_insensitive(self, size_str):
        assert parse_size(size_str) == 4 * 1024**3

    @pytest.mark.parametrize(
        "invalid",
        [
            "4XB",  # invalid unit
            "GB",  # no number
            "-4GB",  # negative
            "4.5.5GB",  # malformed decimal
            "",  # empty
            "four GB",  # non-numeric
        ],
    )
    def test_invalid_formats_raise(self, invalid):
        with pytest.raises(ValueError):
            parse_size(invalid)


class TestValidateSplitSize:
    """Wraps parse_size with TacoValidationError and positivity check."""

    def test_valid_size_returns_bytes(self):
        assert validate_split_size("4GB") == 4 * 1024**3

    def test_invalid_format_raises_validation_error(self):
        with pytest.raises(TacoValidationError, match="Invalid split_size format"):
            validate_split_size("4XB")

    def test_zero_size_raises(self):
        with pytest.raises(TacoValidationError, match="must be positive"):
            validate_split_size("0GB")


class TestValidateOutputPath:
    """Path existence checks with format-specific messages."""

    def test_existing_file_raises_for_zip(self, tmp_path):
        existing = tmp_path / "test.tacozip"
        existing.touch()
        with pytest.raises(TacoValidationError, match="Output file already exists"):
            validate_output_path(existing, "zip")

    def test_existing_dir_raises_for_folder(self, tmp_path):
        existing = tmp_path / "test_folder"
        existing.mkdir()
        with pytest.raises(
            TacoValidationError, match="Output directory already exists"
        ):
            validate_output_path(existing, "folder")

    def test_nonexistent_path_passes(self, tmp_path):
        new_path = tmp_path / "new_output.tacozip"
        validate_output_path(new_path, "zip")  # should not raise


class TestValidateCommonDirectory:
    """All inputs must share parent directory."""

    def test_empty_input_raises(self):
        with pytest.raises(TacoValidationError, match="No input files"):
            validate_common_directory([])

    def test_single_file_returns_parent(self, tmp_path):
        f = tmp_path / "data.tacozip"
        f.touch()
        assert validate_common_directory([f]) == tmp_path

    def test_same_directory_returns_parent(self, tmp_path):
        f1 = tmp_path / "a.tacozip"
        f2 = tmp_path / "b.tacozip"
        f1.touch()
        f2.touch()
        assert validate_common_directory([f1, f2]) == tmp_path

    def test_different_directories_raises(self, tmp_path):
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()
        f1 = dir1 / "a.tacozip"
        f2 = dir2 / "b.tacozip"
        f1.touch()
        f2.touch()
        with pytest.raises(TacoValidationError, match="different directories"):
            validate_common_directory([f1, f2])

    def test_accepts_string_paths(self, tmp_path):
        f = tmp_path / "data.tacozip"
        f.touch()
        result = validate_common_directory([str(f)])
        assert result == tmp_path


class TestValidateFormatValue:
    """Only 'zip' and 'folder' are valid."""

    @pytest.mark.parametrize("valid", ["zip", "folder"])
    def test_valid_formats_pass(self, valid):
        validate_format_value(valid)  # should not raise

    @pytest.mark.parametrize("invalid", ["ZIP", "tar", "gz", "", "tacozip"])
    def test_invalid_formats_raise(self, invalid):
        with pytest.raises(TacoValidationError, match="Invalid format"):
            validate_format_value(invalid)


class TestValidateFormatAndSplit:
    """split_size only allowed with zip format."""

    def test_folder_with_split_raises(self):
        with pytest.raises(
            TacoValidationError, match="not supported with format='folder'"
        ):
            validate_format_and_split("folder", "4GB")

    @pytest.mark.parametrize(
        "fmt,split",
        [
            ("folder", None),
            ("zip", "4GB"),
            ("zip", None),
        ],
    )
    def test_valid_combinations_pass(self, fmt, split):
        validate_format_and_split(fmt, split)  # should not raise


class TestIsPaddingId:
    """Simple prefix check."""

    def test_padding_id_detected(self):
        assert is_padding_id(f"{PADDING_PREFIX}001") is True

    def test_normal_id_not_detected(self):
        assert is_padding_id("sample_001") is False

    def test_prefix_in_middle_not_detected(self):
        # Prefix must be at start, not in middle
        assert is_padding_id(f"prefix_{PADDING_PREFIX}") is False
