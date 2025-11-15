"""
TACO Dataset Export.

Export filtered subsets from TacoDataset to FOLDER or ZIP format with auto-detection.

This module provides high-level orchestration for exporting filtered TacoDataset
instances to different container formats:
- FOLDER format: Fast, editable, human-readable directory structure
- ZIP format: Portable, compact .tacozip archives

Key features:
- Auto-detects output format from file extension (.zip/.tacozip → zip, else → folder)
- Works with filtered TacoDataset (e.g., from .sql() queries)
- Automatic temp folder cleanup for ZIP mode
- Concurrent file copying with progress bars
- Auto-detects async context (works in both sync and async code)

Example:
    >>> from tacoreader import TacoDataset
    >>> from tacotoolbox import export
    >>> 
    >>> # Load and filter dataset
    >>> dataset = TacoDataset("big_dataset.tacozip")
    >>> filtered = dataset.sql("SELECT * FROM level0 WHERE country = 'ZA'")
    >>> 
    >>> # Normal Python REPL (just works!)
    >>> export(filtered, "south_africa.tacozip")
    >>> 
    >>> # Jupyter/IPython (also works!)
    >>> export(filtered, "south_africa/")
"""

import asyncio
from pathlib import Path
from typing import Literal
import shutil

from tacoreader import TacoDataset

from tacotoolbox._writers.export_writer import ExportWriter
from tacotoolbox.translate import folder2zip


def export(
    dataset: TacoDataset,
    output: str | Path,
    format: Literal["zip", "folder"] | None = None,
    concurrency: int = 100,
    quiet: bool = False,
    debug: bool = False,
    temp_dir: str | Path | None = None,
) -> Path:
    """
    Export filtered TacoDataset to FOLDER or ZIP format.

    This function automatically handles async execution:
    - In normal Python REPL: runs asyncio.run() internally
    - In Jupyter/IPython: returns coroutine (auto-awaited by Jupyter)

    Args:
        dataset: TacoDataset with applied filters
        output: Path to output file/folder
        format: Output format ("zip", "folder", or None for auto-detect)
        concurrency: Maximum concurrent async operations (default: 100)
        quiet: If True, hide progress bars (default: False)
        debug: If True, show detailed debug messages (default: False)
        temp_dir: Temporary directory for ZIP creation

    Returns:
        Path to created output

    Example:
        >>> # Normal Python REPL - just works!
        >>> export(dataset, "output.tacozip")

        >>> # Jupyter/IPython - also works!
        >>> export(dataset, "output.tacozip")
        >>> # OR explicitly:
        >>> await export(dataset, "output.tacozip")
    """
    # Check if we're already in an async context
    try:
        asyncio.get_running_loop()
        # We're in async context (Jupyter) - return coroutine
        return _export_async(
            dataset, output, format, concurrency, quiet, debug, temp_dir
        )
    except RuntimeError:
        # No running loop - we're in sync context (normal REPL)
        # Run asyncio.run() internally
        return asyncio.run(
            _export_async(dataset, output, format, concurrency, quiet, debug, temp_dir)
        )


async def _export_async(
    dataset: TacoDataset,
    output: str | Path,
    format: Literal["zip", "folder"] | None = None,
    concurrency: int = 100,
    quiet: bool = False,
    debug: bool = False,
    temp_dir: str | Path | None = None,
) -> Path:
    """
    Internal async implementation of export.

    Creates a new TACO container from a filtered TacoDataset. Auto-detects
    output format from file extension if not explicitly specified.

    Format auto-detection:
    - .zip or .tacozip extension → ZIP format
    - Any other extension or directory path → FOLDER format

    ZIP mode workflow:
    1. Create temporary FOLDER
    2. Convert FOLDER to ZIP
    3. Cleanup temporary FOLDER

    Args:
        dataset: TacoDataset with applied filters (e.g., from .sql())
        output: Path to output file/folder
        format: Output format. If None, auto-detects from extension:
            - "zip": Creates .tacozip archive
            - "folder": Creates directory structure
            - None: Auto-detect from output extension
        concurrency: Maximum concurrent async operations (default: 100)
        quiet: If True, hide progress bars (default: False - shows progress)
        debug: If True, show detailed debug messages (default: False)
        temp_dir: Temporary directory for ZIP creation. If None, uses
            output.parent / f".{output.stem}_temp"

    Returns:
        Path to created output (FOLDER or ZIP)

    Raises:
        ValueError: If dataset has level1+ joins or is empty
        FileExistsError: If output path already exists
    """
    output = Path(output)

    # Auto-detect format if not specified
    if format is None:
        format = _detect_format(output)
        if debug:
            print(f"Auto-detected format: {format}")

    if format == "folder":
        # Direct FOLDER creation using ExportWriter
        writer = ExportWriter(
            dataset=dataset,
            output=output,
            concurrency=concurrency,
            quiet=quiet,
            debug=debug,
        )
        return await writer.create_folder()

    elif format == "zip":
        # ZIP mode: FOLDER → ZIP → cleanup

        # Determine temp directory
        if temp_dir is None:
            temp_folder = output.parent / f".{output.stem}_temp"
        else:
            temp_folder = Path(temp_dir) / f"{output.stem}_temp"

        try:
            # Step 1: Create temp FOLDER using ExportWriter
            if debug:
                print(f"Creating temporary FOLDER: {temp_folder}")

            writer = ExportWriter(
                dataset=dataset,
                output=temp_folder,
                concurrency=concurrency,
                quiet=quiet,
                debug=debug,
            )
            await writer.create_folder()

            # Step 2: Convert FOLDER → ZIP (ZipWriter is sync)
            if debug:
                print(f"Converting to ZIP: {output}")

            folder2zip(
                folder_path=temp_folder,
                zip_output=output,
                quiet=quiet,
                debug=debug,
                temp_dir=temp_folder.parent,
            )

            # Step 3: Cleanup temp FOLDER
            if debug:
                print(f"Cleaning up: {temp_folder}")
            shutil.rmtree(temp_folder)

            if debug:
                print(f"Export complete: {output}")

            return output

        except Exception as e:
            # Cleanup on failure
            if temp_folder.exists():
                shutil.rmtree(temp_folder, ignore_errors=True)
            raise e

    else:
        raise ValueError(f"Invalid format: {format}. Must be 'zip' or 'folder'.")


def _detect_format(output: Path) -> Literal["zip", "folder"]:
    """
    Auto-detect format from output path extension.

    Args:
        output: Output path

    Returns:
        "zip" if extension is .zip or .tacozip, else "folder"

    Example:
        >>> _detect_format(Path("data.tacozip"))
        'zip'
        >>> _detect_format(Path("data.zip"))
        'zip'
        >>> _detect_format(Path("data/"))
        'folder'
        >>> _detect_format(Path("data"))
        'folder'
    """
    suffix = output.suffix.lower()
    if suffix in [".zip", ".tacozip"]:
        return "zip"
    return "folder"
