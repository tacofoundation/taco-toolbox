"""Progress bar utilities for tacotoolbox.

Progress bars are automatically controlled by logging level:
- INFO or DEBUG: Show progress bars
- WARNING or higher: Hide progress bars

Example:
    from tacotoolbox._progress import progress_bar

    for item in progress_bar(items, desc="Processing", colour="green"):
        process(item)
"""

import logging
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from typing import Any

from tqdm import tqdm

from tacotoolbox._logging import get_logger

logger = get_logger(__name__)


def _should_show_progress() -> bool:
    """Check if progress bars should be shown based on logging level."""
    if logger.level >= logging.WARNING or logger.level == logging.CRITICAL + 1:
        return False
    return True


def progress_bar(
    iterable: Iterable,
    desc: str | None = None,
    total: int | None = None,
    unit: str = "it",
    colour: str | None = None,
    leave: bool = True,
    **kwargs: Any,
) -> tqdm:
    """Create progress bar with automatic suppression based on logging level."""
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        unit=unit,
        colour=colour,
        leave=leave,
        disable=not _should_show_progress(),
        **kwargs,
    )


@contextmanager
def progress_scope(
    desc: str,
    total: int,
    unit: str = "it",
    colour: str | None = None,
) -> Generator[tqdm, None, None]:
    """Context manager for manual progress bar updates."""
    pbar = tqdm(total=total, desc=desc, unit=unit, colour=colour, disable=not _should_show_progress())
    try:
        yield pbar
    finally:
        pbar.close()
