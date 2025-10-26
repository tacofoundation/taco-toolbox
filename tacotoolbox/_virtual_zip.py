import pathlib
import pydantic


class VirtualFile(pydantic.BaseModel):
    """
    Represents a virtual file in the ZIP archive.
    
    Attributes:
        src_path: Original file path (or None for in-memory data)
        arc_path: Archive path inside ZIP (e.g., "DATA/level0/sample1.tif")
        file_size: Size of the file in bytes
        lfh_offset: Offset where Local File Header starts
        lfh_size: Size of Local File Header (30 + filename_len + extra_field_len)
        data_offset: Offset where actual file data starts
        needs_zip64: Whether this file needs ZIP64 format
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
    Simulates a complete TACO ZIP file to precalculate offsets.
    
    Follows exact ZIP format rules:
    - LFH base size = 30 bytes + len(filename)
    - LFH with ZIP64 = 30 bytes + len(filename) + 20 bytes extra field
    - ZIP64 activates when file_size >= 4GB (4,294,967,295 bytes)
    - TACO_HEADER is a fixed 157-byte entry (41 LFH + 116 data)
    
    Usage:
        >>> virtual = VirtualTACOZIP()
        >>> virtual.add_header(num_entries=5)
        >>> virtual.add_file("/data/image.tif", "DATA/level0/image.tif")
        >>> virtual.calculate_offsets()
        >>> offset, size = virtual.get_offset("DATA/level0/image.tif")
    """
    
    LFH_BASE_SIZE = 30
    ZIP64_EXTRA_FIELD_SIZE = 20
    ZIP64_THRESHOLD = 4_294_967_295
    
    TACO_HEADER_FILENAME = "TACO_HEADER"
    TACO_HEADER_FILENAME_LEN = 11
    TACO_HEADER_LFH_SIZE = 30 + TACO_HEADER_FILENAME_LEN
    TACO_HEADER_DATA_SIZE = 116
    TACO_HEADER_TOTAL_SIZE = TACO_HEADER_LFH_SIZE + TACO_HEADER_DATA_SIZE
    
    def __init__(self):
        self.files: list[VirtualFile] = []
        self.current_offset: int = 0
        self.header_size: int = 0
        self._calculated: bool = False
    
    def add_header(self, num_entries: int = 0) -> int:
        """
        Add TACO_HEADER to the virtual ZIP.
        
        TACO_HEADER is a fixed-size file (116 bytes of data) that appears first in the ZIP.
        The num_entries parameter is kept for compatibility but does not affect file size.
        Entries are stored in the Central Directory at the end of the ZIP, not in the header file.
        
        Total size in ZIP:
        - LFH: 41 bytes (30 fixed + 11 for "TACO_HEADER" filename)
        - Data: 116 bytes
        - Total: 157 bytes
        
        Args:
            num_entries: Number of metadata entries (for documentation only, does not affect size)
        
        Returns:
            Total header size in bytes (always 157)
        """
        self.header_size = self.TACO_HEADER_TOTAL_SIZE
        self.current_offset = self.header_size
        self._calculated = False
        return self.header_size
    
    def add_file(
        self, 
        src_path: str | pathlib.Path | None,
        arc_path: str,
        file_size: int | None = None
    ) -> VirtualFile:
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
        
        vfile = VirtualFile(
            src_path=src_path,
            arc_path=arc_path,
            file_size=size
        )
        
        self.files.append(vfile)
        self._calculated = False
        
        return vfile
    
    def calculate_offsets(self, debug: bool = False) -> None:
        current_offset = self.current_offset
        
        if debug:
            print(f"DEBUG VirtualTACOZIP: Starting offset calculation from {current_offset}")
        
        for i, vfile in enumerate(self.files):
            vfile.needs_zip64 = vfile.file_size >= self.ZIP64_THRESHOLD
            
            filename_len = len(vfile.arc_path.encode('utf-8'))
            lfh_base = self.LFH_BASE_SIZE + filename_len
            
            if vfile.needs_zip64:
                vfile.lfh_size = lfh_base + self.ZIP64_EXTRA_FIELD_SIZE
            else:
                vfile.lfh_size = lfh_base
            
            vfile.lfh_offset = current_offset
            vfile.data_offset = current_offset + vfile.lfh_size
            
            if debug and i < 5:
                print(f"  [{i}] {vfile.arc_path}")
                print(f"      filename_len={filename_len}, lfh_size={vfile.lfh_size}")
                print(f"      lfh_offset={vfile.lfh_offset}, data_offset={vfile.data_offset}, file_size={vfile.file_size}")
            
            current_offset = vfile.data_offset + vfile.file_size
        
        self._calculated = True
    
    def get_offset(self, arc_path: str) -> tuple[int, int]:
        if not self._calculated:
            raise ValueError("Call calculate_offsets() first")
        
        for vfile in self.files:
            if vfile.arc_path == arc_path:
                return (vfile.data_offset, vfile.file_size)
        
        raise KeyError(f"File not found in virtual ZIP: {arc_path}")
    
    def get_all_offsets(self) -> dict[str, tuple[int, int]]:
        if not self._calculated:
            raise ValueError("Call calculate_offsets() first")
        
        return {
            vfile.arc_path: (vfile.data_offset, vfile.file_size)
            for vfile in self.files
        }
    
    def needs_zip64(self) -> bool:
        if not self._calculated:
            raise ValueError("Call calculate_offsets() first")
        
        if len(self.files) > 65535:
            return True
        
        if any(vf.needs_zip64 for vf in self.files):
            return True
        
        if self.files:
            last_file = self.files[-1]
            total_size = last_file.data_offset + last_file.file_size
            if total_size > self.ZIP64_THRESHOLD:
                return True
        
        return False
    
    def get_summary(self) -> dict:
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
            "needs_zip64": self.needs_zip64()
        }