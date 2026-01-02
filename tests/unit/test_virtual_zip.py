"""Tests for tacotoolbox._virtual_zip module."""

import pytest

from tacotoolbox._constants import (
    ZIP_LFH_BASE_SIZE,
    ZIP_TACO_HEADER_TOTAL_SIZE,
    ZIP_ZIP64_EXTRA_FIELD_SIZE,
    ZIP_ZIP64_THRESHOLD,
)
from tacotoolbox._virtual_zip import VirtualFile, VirtualTACOZIP


class TestVirtualFile:
    """VirtualFile is a simple pydantic model - minimal tests."""

    def test_defaults(self):
        vf = VirtualFile(src_path=None, arc_path="test.txt")
        assert vf.file_size == 0
        assert vf.lfh_offset == 0
        assert vf.needs_zip64 is False


class TestVirtualTACOZIPAddHeader:
    def test_header_sets_initial_offset(self):
        vzip = VirtualTACOZIP()
        size = vzip.add_header()

        assert size == ZIP_TACO_HEADER_TOTAL_SIZE
        assert vzip.current_offset == ZIP_TACO_HEADER_TOTAL_SIZE
        assert vzip.header_size == ZIP_TACO_HEADER_TOTAL_SIZE


class TestVirtualTACOZIPAddFile:
    def test_add_file_with_explicit_size(self):
        vzip = VirtualTACOZIP()
        vzip.add_header()

        vf = vzip.add_file(src_path=None, arc_path="data.bin", file_size=1000)

        assert vf.file_size == 1000
        assert vf.arc_path == "data.bin"
        assert len(vzip.files) == 1

    def test_add_file_from_real_path(self, tmp_path):
        real_file = tmp_path / "test.txt"
        real_file.write_bytes(b"x" * 500)

        vzip = VirtualTACOZIP()
        vzip.add_header()
        vf = vzip.add_file(src_path=real_file, arc_path="test.txt")

        assert vf.file_size == 500

    def test_empty_arc_path_raises(self):
        vzip = VirtualTACOZIP()
        with pytest.raises(ValueError, match="arc_path cannot be empty"):
            vzip.add_file(src_path=None, arc_path="", file_size=100)

    def test_negative_file_size_raises(self):
        vzip = VirtualTACOZIP()
        with pytest.raises(ValueError, match="non-negative"):
            vzip.add_file(src_path=None, arc_path="x.bin", file_size=-1)

    def test_missing_src_path_raises(self, tmp_path):
        vzip = VirtualTACOZIP()
        with pytest.raises(FileNotFoundError):
            vzip.add_file(src_path=tmp_path / "nonexistent.txt", arc_path="x.txt")

    def test_no_size_and_no_path_raises(self):
        vzip = VirtualTACOZIP()
        with pytest.raises(ValueError, match="Either src_path or file_size"):
            vzip.add_file(src_path=None, arc_path="x.bin", file_size=None)


class TestVirtualTACOZIPCalculateOffsets:
    """Core offset calculation logic."""

    def test_single_file_offset_calculation(self):
        vzip = VirtualTACOZIP()
        vzip.add_header()
        vzip.add_file(src_path=None, arc_path="data.bin", file_size=1000)
        vzip.calculate_offsets()

        vf = vzip.files[0]
        arc_path_len = len("data.bin".encode("utf-8"))
        expected_lfh_size = ZIP_LFH_BASE_SIZE + arc_path_len

        assert vf.lfh_offset == ZIP_TACO_HEADER_TOTAL_SIZE
        assert vf.lfh_size == expected_lfh_size
        assert vf.data_offset == ZIP_TACO_HEADER_TOTAL_SIZE + expected_lfh_size

    def test_sequential_files_no_gaps(self):
        vzip = VirtualTACOZIP()
        vzip.add_header()
        vzip.add_file(src_path=None, arc_path="a.bin", file_size=100)
        vzip.add_file(src_path=None, arc_path="b.bin", file_size=200)
        vzip.calculate_offsets()

        f1, f2 = vzip.files

        # Second file starts immediately after first file's data ends
        expected_f2_lfh = f1.data_offset + f1.file_size
        assert f2.lfh_offset == expected_f2_lfh

    def test_utf8_filename_length_calculation(self):
        vzip = VirtualTACOZIP()
        vzip.add_header()
        # UTF-8 multi-byte character
        vzip.add_file(src_path=None, arc_path="日本語.bin", file_size=100)
        vzip.calculate_offsets()

        vf = vzip.files[0]
        utf8_len = len("日本語.bin".encode("utf-8"))
        expected_lfh_size = ZIP_LFH_BASE_SIZE + utf8_len

        assert vf.lfh_size == expected_lfh_size

    def test_zip64_extra_field_for_large_files(self):
        vzip = VirtualTACOZIP()
        vzip.add_header()
        # File at ZIP64 threshold
        vzip.add_file(src_path=None, arc_path="huge.bin", file_size=ZIP_ZIP64_THRESHOLD)
        vzip.calculate_offsets()

        vf = vzip.files[0]
        arc_path_len = len("huge.bin".encode("utf-8"))
        expected_lfh_size = (
            ZIP_LFH_BASE_SIZE + arc_path_len + ZIP_ZIP64_EXTRA_FIELD_SIZE
        )

        assert vf.needs_zip64 is True
        assert vf.lfh_size == expected_lfh_size


class TestVirtualTACOZIPGetOffset:
    def test_get_offset_returns_data_position(self):
        vzip = VirtualTACOZIP()
        vzip.add_header()
        vzip.add_file(src_path=None, arc_path="test.bin", file_size=500)
        vzip.calculate_offsets()

        offset, size = vzip.get_offset("test.bin")

        assert size == 500
        assert offset == vzip.files[0].data_offset

    def test_get_offset_before_calculate_raises(self):
        vzip = VirtualTACOZIP()
        vzip.add_header()
        vzip.add_file(src_path=None, arc_path="test.bin", file_size=100)

        with pytest.raises(ValueError, match="calculate_offsets"):
            vzip.get_offset("test.bin")

    def test_get_offset_unknown_file_raises(self):
        vzip = VirtualTACOZIP()
        vzip.add_header()
        vzip.add_file(src_path=None, arc_path="exists.bin", file_size=100)
        vzip.calculate_offsets()

        with pytest.raises(KeyError, match="not found"):
            vzip.get_offset("missing.bin")

    def test_get_all_offsets(self):
        vzip = VirtualTACOZIP()
        vzip.add_header()
        vzip.add_file(src_path=None, arc_path="a.bin", file_size=100)
        vzip.add_file(src_path=None, arc_path="b.bin", file_size=200)
        vzip.calculate_offsets()

        offsets = vzip.get_all_offsets()

        assert set(offsets.keys()) == {"a.bin", "b.bin"}
        assert offsets["a.bin"][1] == 100  # size
        assert offsets["b.bin"][1] == 200


class TestVirtualTACOZIPNeedsZip64:
    def test_small_archive_no_zip64(self):
        vzip = VirtualTACOZIP()
        vzip.add_header()
        vzip.add_file(src_path=None, arc_path="small.bin", file_size=1000)
        vzip.calculate_offsets()

        assert vzip.needs_zip64() is False

    def test_large_file_triggers_zip64(self):
        vzip = VirtualTACOZIP()
        vzip.add_header()
        vzip.add_file(src_path=None, arc_path="huge.bin", file_size=ZIP_ZIP64_THRESHOLD)
        vzip.calculate_offsets()

        assert vzip.needs_zip64() is True

    def test_needs_zip64_before_calculate_raises(self):
        vzip = VirtualTACOZIP()
        vzip.add_header()
        vzip.add_file(src_path=None, arc_path="x.bin", file_size=100)

        with pytest.raises(ValueError, match="calculate_offsets"):
            vzip.needs_zip64()


class TestVirtualTACOZIPIntegration:
    """End-to-end workflow tests."""

    def test_typical_taco_workflow(self):
        """Simulates real TACO ZIP creation pattern."""
        vzip = VirtualTACOZIP()
        vzip.add_header()

        # Add files in typical order
        vzip.add_file(None, "DATA/sample_0/image.tif", file_size=1024 * 1024)
        vzip.add_file(None, "DATA/sample_0/__meta__", file_size=4096)
        vzip.add_file(None, "METADATA/level0.parquet", file_size=8192)
        vzip.add_file(None, "COLLECTION.json", file_size=2048)

        vzip.calculate_offsets()

        # Verify all offsets are sequential and non-overlapping
        prev_end = ZIP_TACO_HEADER_TOTAL_SIZE
        for vf in vzip.files:
            assert vf.lfh_offset == prev_end
            assert vf.data_offset > vf.lfh_offset
            prev_end = vf.data_offset + vf.file_size

    def test_recalculate_after_adding_files(self):
        """Adding files invalidates previous calculation."""
        vzip = VirtualTACOZIP()
        vzip.add_header()
        vzip.add_file(None, "a.bin", file_size=100)
        vzip.calculate_offsets()

        # Add another file
        vzip.add_file(None, "b.bin", file_size=200)

        # Should require recalculation
        assert vzip._calculated is False
