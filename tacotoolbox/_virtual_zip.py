"""
Virtual ZIP structure simulation for offset precalculation.

This module provides VirtualTACOZIP, which simulates the complete structure
of a ZIP file without writing any data to disk. It calculates exact byte
offsets for every file, enabling precise TACO_HEADER generation before
actual ZIP creation.

Problem solved:
    TACO containers need a header at byte 0 containing offsets to all
    metadata files. To write correct offsets, we must know where each
    file will be located in the final ZIP. VirtualTACOZIP simulates
    the complete ZIP structure using real ZIP format rules to calculate
    these offsets precisely. This rules follow the libzip implementation.

ZIP format rules implemented:
    - Local File Header (LFH) = 30 bytes + filename length
    - ZIP64 extension adds 20 bytes extra field when file >= 4GB
    - TACO_HEADER is always 157 bytes (41 LFH + 116 data)
    - Data immediately follows each LFH
"""

import pathlib

import pydantic

from tacotoolbox._constants import (
    ZIP_LFH_BASE_SIZE,
    ZIP_TACO_HEADER_TOTAL_SIZE,
    ZIP_ZIP64_EXTRA_FIELD_SIZE,
    ZIP_ZIP64_THRESHOLD,
)
from tacotoolbox._logging import get_logger

logger = get_logger(__name__)


class VirtualFile(pydantic.BaseModel):
    """
    Represents a virtual file in the simulated ZIP archive.

    Stores all information needed to calculate ZIP structure without
    writing actual data. After calculate_offsets() is called, all
    offset fields are populated with exact byte positions.
    """

    src_path: str | pathlib.Path | None
    arc_path: str
    file_size: int = 0
    lfh_offset: int = 0
    lfh_size: int = 0
    data_offset: int = 0
    needs_zip64: bool = False

    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True, slots=True  # type: ignore[typeddict-unknown-key]
    )


class VirtualTACOZIP:
    """
    Simulates complete TACO ZIP structure for precise offset calculation.

    This class follows exact ZIP format rules to calculate where each file
    will be located in the final ZIP archive. Used by ZipWriter to generate
    accurate TACO_HEADER entries before writing the actual ZIP file.

    ZIP structure simulated:
        [TACO_HEADER: 157 bytes]
        [File1 LFH + data]
        [File2 LFH + data]
        [...]
        [Central Directory]
        [End of Central Directory]

    Workflow:
        1. add_header() - Initialize with TACO_HEADER
        2. add_file() - Add files in order they'll appear in ZIP
        3. calculate_offsets() - Compute all byte positions
        4. get_offset() / get_all_offsets() - Retrieve calculated offsets

    ZIP64 handling:
        Automatically detects when ZIP64 format is needed:
        - Individual file >= 4GB
        - More than 65,535 files
        - Total archive size >= 4GB
    """

    __slots__ = ["_calculated", "current_offset", "files", "header_size"]

    def __init__(self) -> None:
        """Initialize empty virtual ZIP structure."""
        self.files: list[VirtualFile] = []
        self.current_offset: int = 0
        self.header_size: int = 0
        self._calculated: bool = False

    def add_header(self) -> int:
        """
        Add TACO_HEADER entry to virtual ZIP structure.

        TACO_HEADER is always the first entry in TACO ZIP files and has
        a fixed size of 157 bytes (41 byte LFH + 116 byte data payload).
        The data payload stores offset table for metadata files.

        Note: num_entries parameter is kept for API compatibility but
        doesn't affect file size. Entries are stored in Central Directory,
        not in the TACO_HEADER data itself.
        """
        self.header_size = ZIP_TACO_HEADER_TOTAL_SIZE
        self.current_offset = self.header_size
        self._calculated = False
        return self.header_size

    def add_file(
        self,
        src_path: str | pathlib.Path | None,
        arc_path: str,
        file_size: int | None = None,
    ) -> VirtualFile:
        """
        Add a file to the virtual ZIP structure.

        Files must be added in the exact order they'll appear in the
        actual ZIP file. The method supports three modes:

        1. Real file: Provide src_path, size auto-detected
        2. Known size: Provide file_size (for pre-calculated sizes)
        3. Both: Provide both (file_size overrides auto-detection)
        """
        if not arc_path:
            raise ValueError("arc_path cannot be empty")

        if file_size is not None:
            if file_size < 0:
                raise ValueError("file_size must be non-negative")
            size = file_size
        elif src_path is not None:
            if isinstance(src_path, str):
                src_path = pathlib.Path(src_path)
            if not src_path.exists():
                raise FileNotFoundError(f"File not found: {src_path}")
            size = src_path.stat().st_size
        else:
            raise ValueError("Either src_path or file_size must be provided")

        vfile = VirtualFile(src_path=src_path, arc_path=arc_path, file_size=size)

        self.files.append(vfile)
        self._calculated = False

        return vfile

    def calculate_offsets(self) -> None:
        """
        Calculate exact byte offsets for all files in the virtual ZIP.

        Simulates ZIP file structure following these rules:
        1. Each file has: LFH (30+ bytes) + data (file_size bytes)
        2. ZIP64 extra field (20 bytes) added if file >= 4GB
        3. Offsets calculated sequentially from TACO_HEADER end

        After calling this method, all VirtualFile objects have populated
        offset fields (lfh_offset, lfh_size, data_offset).
        """
        current_offset = self.current_offset

        logger.debug(f"Starting offset calculation from {current_offset}")

        for i, vfile in enumerate(self.files):
            # Determine if ZIP64 needed for this file
            vfile.needs_zip64 = vfile.file_size >= ZIP_ZIP64_THRESHOLD

            # Calculate LFH size (base + filename + optional ZIP64 field)
            filename_len = len(vfile.arc_path.encode("utf-8"))
            lfh_base = ZIP_LFH_BASE_SIZE + filename_len

            if vfile.needs_zip64:
                vfile.lfh_size = lfh_base + ZIP_ZIP64_EXTRA_FIELD_SIZE
            else:
                vfile.lfh_size = lfh_base

            # Calculate offsets
            vfile.lfh_offset = current_offset
            vfile.data_offset = current_offset + vfile.lfh_size

            if i < 5:  # Log first 5 files
                logger.debug(
                    f"[{i}] {vfile.arc_path}: "
                    f"filename_len={filename_len}, lfh_size={vfile.lfh_size}, "
                    f"lfh_offset={vfile.lfh_offset}, data_offset={vfile.data_offset}, "
                    f"file_size={vfile.file_size}"
                )

            # Move to next file position
            current_offset = vfile.data_offset + vfile.file_size

        self._calculated = True

    def get_offset(self, arc_path: str) -> tuple[int, int]:
        """
        Get offset and size for a specific file by archive path.

        Returns the exact byte position and size of the file data
        (not including the LFH). Used to populate TACO_HEADER entries.
        """
        if not self._calculated:
            raise ValueError("Call calculate_offsets() first")

        for vfile in self.files:
            if vfile.arc_path == arc_path:
                return (vfile.data_offset, vfile.file_size)

        raise KeyError(f"File not found in virtual ZIP: {arc_path}")

    def get_all_offsets(self) -> dict[str, tuple[int, int]]:
        """
        Get offsets for all files as a dictionary.

        Returns mapping of archive path to (offset, size) for every
        file in the virtual ZIP. Useful for bulk offset retrieval.
        """
        if not self._calculated:
            raise ValueError("Call calculate_offsets() first")

        return {
            vfile.arc_path: (vfile.data_offset, vfile.file_size) for vfile in self.files
        }

    def needs_zip64(self) -> bool:
        """
        Check if ZIP64 format is required for this archive.

        ZIP64 is needed when any of these conditions are met:
        1. More than 65,535 files (ZIP32 limit)
        2. Any individual file >= 4GB (ZIP32 size limit)
        3. Total archive size >= 4GB (ZIP32 archive limit)
        """
        if not self._calculated:
            raise ValueError("Call calculate_offsets() first")

        # Check 1: Too many files
        if len(self.files) > 65535:
            return True

        # Check 2: Any file requires ZIP64
        if any(vf.needs_zip64 for vf in self.files):
            return True

        # Check 3: Total archive size exceeds ZIP32 limit
        if self.files:
            last_file = self.files[-1]
            total_size = last_file.data_offset + last_file.file_size
            if total_size > ZIP_ZIP64_THRESHOLD:
                return True

        return False
