"""
Export filtered TacoDataset to FOLDER or ZIP format with auto-detection.

Auto-detects format from extension:
- .zip/.tacozip → ZIP format (via temp FOLDER)
- anything else → FOLDER format

Example:
    >>> dataset = TacoDataset("big.tacozip")
    >>> filtered = dataset.sql("SELECT * FROM level0 WHERE country = 'ZA'")
    >>> export(filtered, "south_africa.tacozip")  # works in REPL and Jupyter
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
    Export filtered TacoDataset to FOLDER or ZIP.
    
    Auto-handles async execution:
    - Normal Python REPL: runs asyncio.run() internally
    - Jupyter/IPython: returns coroutine (auto-awaited)
    """
    try:
        asyncio.get_running_loop()
        return _export_async(dataset, output, format, concurrency, quiet, debug, temp_dir)
    except RuntimeError:
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
    Internal async export implementation.
    
    ZIP workflow: create temp FOLDER → convert to ZIP → cleanup temp.
    If temp_dir is None, uses output.parent / f".{output.stem}_temp"
    """
    output = Path(output)

    if format is None:
        format = _detect_format(output)
        if debug:
            print(f"Auto-detected format: {format}")

    if format == "folder":
        writer = ExportWriter(
            dataset=dataset,
            output=output,
            concurrency=concurrency,
            quiet=quiet,
            debug=debug,
        )
        return await writer.create_folder()

    elif format == "zip":
        temp_folder = (
            output.parent / f".{output.stem}_temp"
            if temp_dir is None
            else Path(temp_dir) / f"{output.stem}_temp"
        )

        try:
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

            if debug:
                print(f"Converting to ZIP: {output}")

            folder2zip(
                folder_path=temp_folder,
                zip_output=output,
                quiet=quiet,
                debug=debug,
                temp_dir=temp_folder.parent,
            )

            if debug:
                print(f"Cleaning up: {temp_folder}")
            shutil.rmtree(temp_folder)

            if debug:
                print(f"Export complete: {output}")

            return output

        except Exception as e:
            if temp_folder.exists():
                shutil.rmtree(temp_folder, ignore_errors=True)
            raise e

    else:
        raise ValueError(f"Invalid format: {format}. Must be 'zip' or 'folder'.")


def _detect_format(output: Path) -> Literal["zip", "folder"]:
    """Auto-detect format from extension: .zip/.tacozip → zip, else → folder."""
    suffix = output.suffix.lower()
    return "zip" if suffix in [".zip", ".tacozip"] else "folder"