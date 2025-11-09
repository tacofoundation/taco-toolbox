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

Usage:
    >>> # Simulate ZIP structure
    >>> virtual = VirtualTACOZIP()
    >>> virtual.add_header()
    >>> virtual.add_file("/data/level0.parquet", "METADATA/level0.parquet")
    >>> virtual.add_file("/data/level1.parquet", "METADATA/level1.parquet")
    >>> 
    >>> # Calculate all offsets
    >>> virtual.calculate_offsets()
    >>> 
    >>> # Get offsets for TACO_HEADER
    >>> offset, size = virtual.get_offset("METADATA/level0.parquet")
    >>> # Use these offsets when writing real ZIP
"""

import pathlib

import pydantic

from tacotoolbox._constants import (
    ZIP_LFH_BASE_SIZE,
    ZIP_TACO_HEADER_TOTAL_SIZE,
    ZIP_ZIP64_EXTRA_FIELD_SIZE,
    ZIP_ZIP64_THRESHOLD,
)


class VirtualFile(pydantic.BaseModel):
    """
    Represents a virtual file in the simulated ZIP archive.

    Stores all information needed to calculate ZIP structure without
    writing actual data. After calculate_offsets() is called, all
    offset fields are populated with exact byte positions.

    Attributes:
        src_path: Original file path (or None for in-memory data)
        arc_path: Archive path inside ZIP (e.g., "DATA/sample.tif")
        file_size: Size of the file in bytes
        lfh_offset: Offset where Local File Header starts
        lfh_size: Size of LFH (30 + filename_len + extra_field_len)
        data_offset: Offset where actual file data starts
        needs_zip64: Whether this file requires ZIP64 format

    Example:
        >>> vfile = VirtualFile(
        ...     src_path="/data/image.tif",
        ...     arc_path="DATA/image.tif",
        ...     file_size=10485760
        ... )
        >>> # After calculate_offsets():
        >>> vfile.data_offset
        199  # Exact byte position in ZIP
    """

    src_path: str | pathlib.Path | None
    arc_path: str
    file_size: int = 0
    lfh_offset: int = 0
    lfh_size: int = 0
    data_offset: int = 0
    needs_zip64: bool = False

    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
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

    Example:
        >>> virtual = VirtualTACOZIP()
        >>> virtual.add_header()
        >>>
        >>> # Add files in order
        >>> virtual.add_file("/data/level0.parquet", "METADATA/level0.parquet")
        >>> virtual.add_file("/data/level1.parquet", "METADATA/level1.parquet")
        >>> virtual.add_file("/data/collection.json", "COLLECTION.json")
        >>>
        >>> # Calculate all offsets
        >>> virtual.calculate_offsets()
        >>>
        >>> # Get offsets for TACO_HEADER
        >>> offsets = [
        ...     virtual.get_offset("METADATA/level0.parquet"),
        ...     virtual.get_offset("METADATA/level1.parquet"),
        ...     virtual.get_offset("COLLECTION.json")
        ... ]
        >>> # Write real ZIP using these exact offsets
    """

    def __init__(self):
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

        Args:
            num_entries: Number of metadata entries (legacy parameter)

        Returns:
            Total header size in bytes (always 157)

        Example:
            >>> virtual = VirtualTACOZIP()
            >>> size = virtual.add_header()
            >>> size
            157
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

        Args:
            src_path: Path to source file (or None if size provided)
            arc_path: Archive path inside ZIP
            file_size: File size in bytes (or None to auto-detect)

        Returns:
            VirtualFile object added to structure

        Raises:
            ValueError: If arc_path empty or neither src_path nor file_size provided
            FileNotFoundError: If src_path doesn't exist

        Example:
            >>> virtual = VirtualTACOZIP()
            >>> virtual.add_header()
            >>>
            >>> # Mode 1: Real file (auto-detect size)
            >>> virtual.add_file("/data/level0.parquet", "METADATA/level0.parquet")
            >>>
            >>> # Mode 2: Known size only
            >>> virtual.add_file(None, "METADATA/level1.parquet", file_size=1024)
            >>>
            >>> # Mode 3: Both (size override)
            >>> virtual.add_file("/data/level2.parquet", "METADATA/level2.parquet", file_size=2048)
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

    def calculate_offsets(self, debug: bool = False) -> None:
        """
        Calculate exact byte offsets for all files in the virtual ZIP.

        Simulates ZIP file structure following these rules:
        1. Each file has: LFH (30+ bytes) + data (file_size bytes)
        2. ZIP64 extra field (20 bytes) added if file >= 4GB
        3. Offsets calculated sequentially from TACO_HEADER end

        After calling this method, all VirtualFile objects have populated
        offset fields (lfh_offset, lfh_size, data_offset).

        Args:
            debug: If True, print offset calculation details for first 5 files

        Example:
            >>> virtual = VirtualTACOZIP()
            >>> virtual.add_header()
            >>> virtual.add_file(None, "DATA/file1.tif", file_size=1000)
            >>> virtual.add_file(None, "DATA/file2.tif", file_size=2000)
            >>>
            >>> virtual.calculate_offsets(debug=True)
            DEBUG VirtualTACOZIP: Starting offset calculation from 157
              [0] DATA/file1.tif
                  filename_len=14, lfh_size=44
                  lfh_offset=157, data_offset=201, file_size=1000
              [1] DATA/file2.tif
                  filename_len=14, lfh_size=44
                  lfh_offset=1201, data_offset=1245, file_size=2000

            >>> # Now offsets are available
            >>> virtual.get_offset("DATA/file1.tif")
            (201, 1000)
        """
        current_offset = self.current_offset

        if debug:
            print(
                f"DEBUG VirtualTACOZIP: Starting offset calculation from {current_offset}"
            )

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

            if debug and i < 5:
                print(f"  [{i}] {vfile.arc_path}")
                print(f"      filename_len={filename_len}, lfh_size={vfile.lfh_size}")
                print(
                    f"      lfh_offset={vfile.lfh_offset}, data_offset={vfile.data_offset}, file_size={vfile.file_size}"
                )

            # Move to next file position
            current_offset = vfile.data_offset + vfile.file_size

        self._calculated = True

    def get_offset(self, arc_path: str) -> tuple[int, int]:
        """
        Get offset and size for a specific file by archive path.

        Returns the exact byte position and size of the file data
        (not including the LFH). Used to populate TACO_HEADER entries.

        Args:
            arc_path: Archive path to look up

        Returns:
            Tuple of (data_offset, file_size)

        Raises:
            ValueError: If calculate_offsets() not called yet
            KeyError: If arc_path not found in virtual ZIP

        Example:
            >>> virtual = VirtualTACOZIP()
            >>> virtual.add_header()
            >>> virtual.add_file(None, "METADATA/level0.parquet", file_size=1024)
            >>> virtual.calculate_offsets()
            >>>
            >>> offset, size = virtual.get_offset("METADATA/level0.parquet")
            >>> offset
            199  # Exact byte position where data starts
            >>> size
            1024
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

        Returns:
            Dictionary mapping arc_path -> (data_offset, file_size)

        Raises:
            ValueError: If calculate_offsets() not called yet

        Example:
            >>> virtual = VirtualTACOZIP()
            >>> virtual.add_header()
            >>> virtual.add_file(None, "METADATA/level0.parquet", file_size=1024)
            >>> virtual.add_file(None, "METADATA/level1.parquet", file_size=2048)
            >>> virtual.calculate_offsets()
            >>>
            >>> offsets = virtual.get_all_offsets()
            >>> offsets
            {
                'METADATA/level0.parquet': (199, 1024),
                'METADATA/level1.parquet': (1267, 2048)
            }
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

        Returns:
            True if ZIP64 format required, False otherwise

        Raises:
            ValueError: If calculate_offsets() not called yet

        Example:
            >>> virtual = VirtualTACOZIP()
            >>> virtual.add_header()
            >>> virtual.add_file(None, "DATA/small.tif", file_size=1000)
            >>> virtual.calculate_offsets()
            >>> virtual.needs_zip64()
            False

            >>> # Large file example
            >>> virtual2 = VirtualTACOZIP()
            >>> virtual2.add_header()
            >>> virtual2.add_file(None, "DATA/huge.tif", file_size=5_000_000_000)  # 5GB
            >>> virtual2.calculate_offsets()
            >>> virtual2.needs_zip64()
            True
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

    def get_summary(self) -> dict:
        """
        Get summary statistics about the virtual ZIP structure.

        Provides overview of the simulated archive including sizes,
        file counts, and ZIP64 status. Useful for debugging and logging.

        Returns:
            Dictionary with keys:
                - header_size: TACO_HEADER size (157 bytes)
                - num_files: Total number of files
                - zip64_files: Number of files using ZIP64
                - total_data_size: Sum of all file sizes
                - total_lfh_size: Sum of all LFH sizes
                - total_zip_size: Estimated total ZIP size
                - needs_zip64: Whether ZIP64 format required

        Raises:
            ValueError: If calculate_offsets() not called yet

        Example:
            >>> virtual = VirtualTACOZIP()
            >>> virtual.add_header()
            >>> virtual.add_file(None, "DATA/file1.tif", file_size=1000)
            >>> virtual.add_file(None, "DATA/file2.tif", file_size=2000)
            >>> virtual.calculate_offsets()
            >>>
            >>> summary = virtual.get_summary()
            >>> summary
            {
                'header_size': 157,
                'num_files': 2,
                'zip64_files': 0,
                'total_data_size': 3000,
                'total_lfh_size': 88,
                'total_zip_size': 3245,
                'needs_zip64': False
            }
        """
        if not self._calculated:
            raise ValueError("Call calculate_offsets() first")

        total_data_size = sum(vf.file_size for vf in self.files)
        total_lfh_size = sum(vf.lfh_size for vf in self.files)

        if self.files:
            last_file = self.files[-1]
            total_zip_size = last_file.data_offset + last_file.file_size
        else:
            total_zip_size = self.header_size

        zip64_files = sum(1 for vf in self.files if vf.needs_zip64)

        return {
            "header_size": self.header_size,
            "num_files": len(self.files),
            "zip64_files": zip64_files,
            "total_data_size": total_data_size,
            "total_lfh_size": total_lfh_size,
            "total_zip_size": total_zip_size,
            "needs_zip64": self.needs_zip64(),
        }