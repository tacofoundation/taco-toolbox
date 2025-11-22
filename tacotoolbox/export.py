"""
Export filtered TacoDataset to FOLDER or ZIP format with auto-detection.

Auto-detects format from extension:
- .zip/.tacozip → ZIP format (via temp FOLDER)
- anything else → FOLDER format

Example:
    >>> dataset = TacoDataset("big.tacozip")
    >>> filtered = dataset.sql("SELECT * FROM level0 WHERE country = 'ZA'")
    >>> export(filtered, "south_africa.tacozip")
"""

import asyncio
import shutil
from pathlib import Path
from typing import Literal

from tacoreader import TacoDataset

from tacotoolbox._logging import get_logger
from tacotoolbox._nest_asyncio import apply as nest_asyncio_apply
from tacotoolbox._writers.export_writer import ExportWriter
from tacotoolbox.translate import folder2zip

nest_asyncio_apply()

logger = get_logger(__name__)


def export(
    dataset: TacoDataset,
    output: str | Path,
    output_format: Literal["zip", "folder"] | None = None,
    concurrency: int = 100,
    quiet: bool = False,
    temp_dir: str | Path | None = None,
) -> Path:
    """
    Export filtered TacoDataset to FOLDER or ZIP.

    Returns:
        Path to created output

    Example:
        >>> dataset = TacoDataset("global.tacozip")
        >>> filtered = dataset.sql("SELECT * FROM level0 WHERE country='ES'")
        >>> export(filtered, "spain.tacozip")
        PosixPath('spain.tacozip')
    """
    return asyncio.run(
        _export_async(dataset, output, output_format, concurrency, quiet, temp_dir)
    )


async def _export_async(
    dataset: TacoDataset,
    output: str | Path,
    output_format: Literal["zip", "folder"] | None = None,
    concurrency: int = 100,
    quiet: bool = False,
    temp_dir: str | Path | None = None,
) -> Path:
    """
    Internal async export implementation.

    ZIP workflow: create temp FOLDER → convert to ZIP → cleanup temp.
    If temp_dir is None, uses output.parent / f".{output.stem}_temp"
    """
    output = Path(output)

    if output_format is None:
        output_format = _detect_format(output)
        logger.debug(f"Auto-detected format: {output_format}")

    logger.info(f"Exporting to {output_format.upper()}: {output}")

    if output_format == "folder":
        writer = ExportWriter(
            dataset=dataset,
            output=output,
            concurrency=concurrency,
            quiet=quiet,
        )
        result = await writer.create_folder()
        logger.info(f"Export complete: {result}")
        return result

    elif output_format == "zip":
        temp_folder = (
            output.parent / f".{output.stem}_temp"
            if temp_dir is None
            else Path(temp_dir) / f"{output.stem}_temp"
        )

        try:
            logger.debug(f"Creating temporary FOLDER: {temp_folder}")

            writer = ExportWriter(
                dataset=dataset,
                output=temp_folder,
                concurrency=concurrency,
                quiet=quiet,
            )
            await writer.create_folder()

            logger.debug(f"Converting FOLDER to ZIP: {output}")

            folder2zip(
                input=temp_folder,
                output=output,
                quiet=quiet,
                temp_dir=temp_folder.parent,
            )

            logger.debug(f"Cleaning up temporary folder: {temp_folder}")
            shutil.rmtree(temp_folder)

        except Exception:
            logger.exception("Failed to export to ZIP")
            if temp_folder.exists():
                logger.debug(f"Cleaning up failed export: {temp_folder}")
                shutil.rmtree(temp_folder, ignore_errors=True)
            raise
        else:
            logger.info(f"Export complete: {output}")
            return output

    else:
        raise ValueError(f"Invalid format: {output_format}. Must be 'zip' or 'folder'.")


def _detect_format(output: Path) -> Literal["zip", "folder"]:
    """Auto-detect format from extension: .zip/.tacozip → zip, else → folder."""
    suffix = output.suffix.lower()
    return "zip" if suffix in [".zip", ".tacozip"] else "folder"
