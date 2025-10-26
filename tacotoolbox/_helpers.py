import pathlib
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tacotoolbox.sample.datamodel import Sample


class FileExtractor:
    """Extract file paths from sample hierarchies."""
    
    @staticmethod
    def extract_files_recursive(samples: list["Sample"]) -> dict[str, list[str]]:
        """
        Extract all file paths from samples recursively.
        
        Builds correct archive paths for dual metadata system:
        - Level 0 files: DATA/sample_001 (no subfolder, ID as-is)
        - Level 1+ files: DATA/folder_A/nested_001 (with subfolders, ID as-is)
        
        Args:
            samples: List of Sample objects (can contain FOLDERs)
        
        Returns:
            Dictionary with:
                - "src_files": List of source file paths (absolute)
                - "arc_files": List of archive paths in ZIP
        
        Example:
            >>> samples = [file1, file2, folder_A]
            >>> extracted = FileExtractor.extract_files_recursive(samples)
            >>> extracted["src_files"]
            ["/data/file1.tif", "/data/file2.tif", "/data/nested1.tif"]
            >>> extracted["arc_files"]
            ["DATA/file1", "DATA/file2", "DATA/folder_A/nested1"]
        """
        src_files = []
        arc_files = []
        
        # Process each sample at root level
        for sample in samples:
            if sample.type == "FILE":
                # Level 0 FILE - goes directly in DATA/
                src_files.append(str(sample.path))
                
                # Build archive path: DATA/sample_id (use ID as-is)
                arc_path = f"DATA/{sample.id}"
                arc_files.append(arc_path)
            
            elif sample.type == "FOLDER":
                # Level 1+ FOLDER - recurse into it
                folder_src, folder_arc = FileExtractor._extract_from_folder(
                    sample,
                    parent_path="DATA/"
                )
                src_files.extend(folder_src)
                arc_files.extend(folder_arc)
        
        return {
            "src_files": src_files,
            "arc_files": arc_files
        }
    
    @staticmethod
    def _extract_from_folder(
        folder_sample: "Sample",
        parent_path: str
    ) -> tuple[list[str], list[str]]:
        """
        Extract files from a FOLDER sample recursively.
        
        Args:
            folder_sample: Sample of type FOLDER
            parent_path: Parent path in archive (e.g., "DATA/" or "DATA/folder_A/")
        
        Returns:
            Tuple of (src_files, arc_files)
        """
        src_files = []
        arc_files = []
        
        # Current folder path
        folder_path = f"{parent_path}{folder_sample.id}/"
        
        # Extract files from samples inside this folder
        for child_sample in folder_sample.path.samples:
            if child_sample.type == "FILE":
                # FILE inside folder
                src_files.append(str(child_sample.path))
                
                # Build archive path: DATA/folder_A/nested_001 (use ID as-is)
                arc_path = f"{folder_path}{child_sample.id}"
                arc_files.append(arc_path)
            
            elif child_sample.type == "FOLDER":
                # Nested FOLDER - recurse
                nested_src, nested_arc = FileExtractor._extract_from_folder(
                    child_sample,
                    parent_path=folder_path
                )
                src_files.extend(nested_src)
                arc_files.extend(nested_arc)
        
        return src_files, arc_files
    
    @staticmethod
    def estimate_sample_size(sample: "Sample") -> int:
        """
        Estimate total size of a sample (including nested samples).
        
        Args:
            sample: Sample to estimate
        
        Returns:
            Estimated size in bytes
        """
        if sample.type == "FILE":
            # Get file size from filesystem
            if hasattr(sample.path, 'stat'):
                return sample.path.stat().st_size
            return 0
        
        elif sample.type == "FOLDER":
            # Sum sizes of all nested samples
            total_size = 0
            for child in sample.path.samples:
                total_size += FileExtractor.estimate_sample_size(child)
            return total_size
        
        return 0


def group_samples_by_size(
    samples: list["Sample"],
    max_size: int
) -> list[list["Sample"]]:
    """
    Group samples into chunks based on size limit.
    
    Used for splitting large datasets into multiple ZIP files.
    
    Args:
        samples: List of Sample objects
        max_size: Maximum size per chunk in bytes
    
    Returns:
        List of sample chunks, each under max_size
    
    Example:
        >>> chunks = group_samples_by_size(samples, max_size=4_000_000_000)  # 4GB
        >>> len(chunks)
        3  # Split into 3 chunks
    """
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sample in samples:
        sample_size = FileExtractor.estimate_sample_size(sample)
        
        # Check if adding this sample would exceed limit
        if current_size + sample_size > max_size and current_chunk:
            # Start new chunk
            chunks.append(current_chunk)
            current_chunk = [sample]
            current_size = sample_size
        else:
            # Add to current chunk
            current_chunk.append(sample)
            current_size += sample_size
    
    # Add remaining samples
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def parse_size(size_str: str) -> int:
    """
    Parse size string to bytes.
    
    Supports formats:
    - "4GB", "4G" -> 4 * 1024^3
    - "512MB", "512M" -> 512 * 1024^2
    - "1024KB", "1024K" -> 1024 * 1024
    - "2048B", "2048" -> 2048
    
    Args:
        size_str: Size string to parse
    
    Returns:
        Size in bytes
    
    Raises:
        ValueError: If format is invalid
    
    Example:
        >>> parse_size("4GB")
        4294967296
        >>> parse_size("512MB")
        536870912
    """
    size_str = size_str.strip().upper()
    
    # Match pattern: number + optional unit
    match = re.match(r'^(\d+(?:\.\d+)?)\s*(GB?|MB?|KB?|B?)$', size_str)
    
    if not match:
        raise ValueError(
            f"Invalid size format: '{size_str}'. "
            f"Use format like '4GB', '512MB', '1024KB', or '2048B'"
        )
    
    value = float(match.group(1))
    unit = match.group(2)
    
    # Convert to bytes
    multipliers = {
        'B': 1,
        'KB': 1024,
        'K': 1024,
        'MB': 1024 ** 2,
        'M': 1024 ** 2,
        'GB': 1024 ** 3,
        'G': 1024 ** 3,
    }
    
    if not unit or unit == 'B':
        return int(value)
    
    return int(value * multipliers.get(unit, 1))


def estimate_metadata_size(num_samples: int, avg_columns: int = 10) -> int:
    """
    Estimate size of metadata parquet file.
    
    Rough estimation based on number of samples and columns.
    
    Args:
        num_samples: Number of samples in metadata
        avg_columns: Average number of columns (default: 10)
    
    Returns:
        Estimated size in bytes
    """
    # Very rough estimate: ~1KB per row
    # Parquet is columnar so actual size varies greatly
    bytes_per_row = 1000
    return num_samples * bytes_per_row


def build_archive_path(
    sample_id: str,
    file_extension: str,
    parent_path: str = "DATA/"
) -> str:
    """
    Build archive path for a sample.
    
    Args:
        sample_id: Sample ID
        file_extension: File extension (with dot, e.g., ".tif")
        parent_path: Parent path in archive
    
    Returns:
        Complete archive path
    
    Example:
        >>> build_archive_path("sample_001", ".tif")
        'DATA/sample_001.tif'
        >>> build_archive_path("nested_001", ".tif", "DATA/folder_A/")
        'DATA/folder_A/nested_001.tif'
    """
    return f"{parent_path}{sample_id}{file_extension}"


def validate_sample_paths(samples: list["Sample"]) -> list[str]:
    """
    Validate that all FILE samples have valid paths.
    
    Args:
        samples: List of samples to validate
    
    Returns:
        List of validation errors (empty if all valid)
    
    Example:
        >>> errors = validate_sample_paths(samples)
        >>> if errors:
        ...     print("Validation errors:", errors)
    """
    errors = []
    
    for sample in samples:
        if sample.type == "FILE":
            # Check if path exists
            if not sample.path.exists():
                errors.append(f"Sample '{sample.id}': File not found at {sample.path}")
            
            # Check if path is a file (not directory)
            elif not sample.path.is_file():
                errors.append(f"Sample '{sample.id}': Path is not a file: {sample.path}")
        
        elif sample.type == "FOLDER":
            # Recurse into folder
            nested_errors = validate_sample_paths(sample.path.samples)
            errors.extend(nested_errors)
    
    return errors