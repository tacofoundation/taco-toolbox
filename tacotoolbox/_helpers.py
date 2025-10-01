import pathlib
from typing import Any

from tacotoolbox._types import ExtractedFiles


def parse_size(size_str: str) -> int:
    """
    Parse human-readable size string to bytes.

    Args:
        size_str: Size string like "4GB", "500MB", "1.5TB"

    Returns:
        Size in bytes

    Raises:
        ValueError: If format is invalid

    Examples:
        >>> parse_size("4GB")
        4294967296
        >>> parse_size("500MB")
        524288000
    """
    units = {"TB": 1024**4, "GB": 1024**3, "MB": 1024**2, "KB": 1024, "B": 1}
    size_str = size_str.upper().strip()

    for unit in sorted(units.keys(), key=len, reverse=True):
        if size_str.endswith(unit):
            number_part = size_str[: -len(unit)].strip()
            if not number_part:
                raise ValueError(f"Invalid size format: {size_str}")
            return int(float(number_part) * units[unit])

    raise ValueError(f"Invalid size format: {size_str}. Use: TB, GB, MB, KB, or B")


def calculate_sample_size(sample: Any) -> int:
    """
    Calculate total size of a sample in bytes.

    Recursively sums sizes for TORTILLA types.

    Args:
        sample: Sample object with type and path attributes

    Returns:
        Total size in bytes
    """
    if sample.type == "TORTILLA":
        return sum(calculate_sample_size(s) for s in sample.path.samples)
    return sample.path.stat().st_size


def group_samples_by_size(samples: list[Any], max_size: int) -> list[list[Any]]:
    """
    Group consecutive samples into chunks not exceeding max_size.

    Args:
        samples: List of Sample objects
        max_size: Maximum size per chunk in bytes

    Returns:
        List of sample chunks

    Notes:
        - Maintains sample order
        - Single samples larger than max_size get their own chunk
        - Greedy packing algorithm
    """
    if not samples:
        return []

    chunks: list[list[Any]] = []
    current_chunk: list[Any] = []
    current_size = 0

    for sample in samples:
        sample_size = calculate_sample_size(sample)

        # Sample too large: put in own chunk
        if sample_size > max_size:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_size = 0
            chunks.append([sample])
            continue

        # Would exceed limit: start new chunk
        if current_size + sample_size > max_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = [sample]
            current_size = sample_size
        else:
            current_chunk.append(sample)
            current_size += sample_size

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


class FileExtractor:
    """Extract file paths from samples for DATA/ directory."""

    @staticmethod
    def extract_files_recursive(
        samples: list[Any],
        data_root: str = "DATA/",
        src_files: list[str] | None = None,
        arc_files: list[str] | None = None,
        path_prefix: str = "",
    ) -> ExtractedFiles:
        """
        Recursively extract data files from samples.

        Handles nested TORTILLA structures, preserving hierarchy.

        Args:
            samples: List of Sample objects
            data_root: Root directory in archive (default: "DATA/")
            src_files: Accumulator for source paths (internal use)
            arc_files: Accumulator for archive paths (internal use)
            path_prefix: Current path prefix for nested samples (internal use)

        Returns:
            ExtractedFiles with src_files and arc_files lists

        Examples:
            >>> files = FileExtractor.extract_files_recursive(samples)
            >>> files["src_files"]
            ["/home/user/data/img1.tif", "/home/user/data/img2.tif"]
            >>> files["arc_files"]
            ["DATA/img1.tif", "DATA/img2.tif"]
        """
        if src_files is None:
            src_files = []
        if arc_files is None:
            arc_files = []

        for sample in samples:
            if sample.type == "TORTILLA":
                # Descend into nested structure
                new_path_prefix = f"{path_prefix}{sample.id}/" if path_prefix else f"{sample.id}/"
                FileExtractor.extract_files_recursive(
                    sample.path.samples,
                    data_root,
                    src_files,
                    arc_files,
                    path_prefix=new_path_prefix,
                )
            else:
                # Leaf node: add file
                src_files.append(str(sample.path))
                file_suffix = pathlib.Path(sample.path).suffix
                arc_files.append(f"{data_root}{path_prefix}{sample.id}{file_suffix}")

        return {"src_files": src_files, "arc_files": arc_files}
