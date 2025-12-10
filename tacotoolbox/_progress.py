"""
Progress bar utilities for tacotoolbox.

Progress bars are automatically controlled by logging level:
- INFO or DEBUG: Show progress bars
- WARNING or higher: Hide progress bars

Use tacotoolbox.verbose() to control both logging and progress:
    tacotoolbox.verbose(True)   # Show logs + progress
    tacotoolbox.verbose(False)  # Hide logs + progress

Example:
    from tacotoolbox._progress import progress_bar

    # Progress automatically shown/hidden based on logging level
    for item in progress_bar(items, desc="Processing", colour="green"):
        process(item)
"""

import logging
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from typing import Any

from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_asyncio

from tacotoolbox._logging import get_logger

logger = get_logger(__name__)


def _should_show_progress() -> bool:
    """
    Check if progress bars should be shown based on logging level.

    Progress bars are shown when:
    - Logger level is INFO (20) or DEBUG (10)

    Progress bars are hidden when:
    - Logger level is WARNING (30) or higher
    - Logger is disabled (CRITICAL + 1)

    Returns:
        bool: True if progress bars should be shown
    """
    # Hide if WARNING or higher, or if completely disabled
    if logger.level >= logging.WARNING or logger.level == logging.CRITICAL + 1:
        return False

    return True


def is_progress_suppressed() -> bool:
    """
    Check if progress bars are currently suppressed.

    Returns:
        bool: True if progress bars are suppressed
    """
    return not _should_show_progress()


def progress_bar(
    iterable: Iterable,
    desc: str | None = None,
    total: int | None = None,
    unit: str = "it",
    colour: str | None = None,
    leave: bool = True,
    **kwargs: Any
) -> tqdm:
    """
    Create progress bar with automatic suppression based on logging level.

    If logging level is WARNING or higher, returns tqdm with disable=True.
    Otherwise, returns active tqdm progress bar.

    Args:
        iterable: Iterable to wrap with progress bar
        desc: Progress bar description
        total: Total number of items (if known)
        unit: Unit name for progress (default: "it")
        colour: Progress bar colour
        leave: Whether to leave progress bar after completion
        **kwargs: Additional arguments passed to tqdm

    Returns:
        tqdm progress bar (active or disabled based on logging level)

    Example:
        >>> for item in progress_bar(items, desc="Processing", colour="cyan"):
        ...     process(item)
    """
    disable = not _should_show_progress()

    return tqdm(
        iterable,
        desc=desc,
        total=total,
        unit=unit,
        colour=colour,
        leave=leave,
        disable=disable,
        **kwargs
    )


async def progress_gather(
    *tasks,
    desc: str | None = None,
    unit: str = "task",
    colour: str | None = None,
    **kwargs: Any
):
    """
    Async gather with progress bar (respects logging level).

    Returns tqdm.asyncio.gather with disable=True/False based on logging level.

    Args:
        *tasks: Async tasks to gather
        desc: Progress bar description
        unit: Unit name for progress (default: "task")
        colour: Progress bar colour
        **kwargs: Additional arguments passed to tqdm

    Returns:
        Results from gathered tasks

    Example:
        >>> results = await progress_gather(
        ...     *tasks, desc="Downloading", unit="file", colour="green"
        ... )
    """
    disable = not _should_show_progress()

    return await tqdm_asyncio.gather(
        *tasks, desc=desc, unit=unit, colour=colour, disable=disable, **kwargs
    )


@contextmanager
def progress_scope(
    desc: str,
    total: int,
    unit: str = "it",
    colour: str | None = None,
) -> Generator[tqdm, None, None]:
    """
    Context manager for manual progress bar updates (respects logging level).

    Returns tqdm with disable=True/False based on logging level.

    Args:
        desc: Progress bar description
        total: Total number of items
        unit: Unit name for progress (default: "it")
        colour: Progress bar colour

    Yields:
        tqdm progress bar (active or disabled based on logging level)

    Example:
        >>> with progress_scope("Processing", total=100, unit="file") as pbar:
        ...     for item in items:
        ...         process(item)
        ...         pbar.update(1)
    """
    disable = not _should_show_progress()
    pbar = tqdm(total=total, desc=desc, unit=unit, colour=colour, disable=disable)
    try:
        yield pbar
    finally:
        pbar.close()


async def progress_map_async(
    func,
    items: Iterable,
    desc: str | None = None,
    colour: str | None = None,
    concurrency: int = 100,
):
    """
    Map async function over items with progress bar and concurrency limit.

    Respects logging level for progress bar visibility.

    Args:
        func: Async function to map
        items: Items to process
        desc: Progress bar description
        colour: Progress bar colour
        concurrency: Maximum concurrent operations (default: 100)

    Returns:
        Results from mapped function

    Example:
        >>> results = await progress_map_async(
        ...     download_file, urls, desc="Downloading", colour="blue"
        ... )
    """
    import asyncio

    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_func(item):
        async with semaphore:
            return await func(item)

    tasks = [bounded_func(item) for item in items]

    return await progress_gather(*tasks, desc=desc, unit="item", colour=colour)
