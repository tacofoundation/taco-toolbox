"""
Progress bar utilities for tacotoolbox.

Provides centralized progress bar management with consistent styling
and easy suppression via context manager.

Usage:
    from tacotoolbox._progress import ProgressContext, progress_bar
    
    # Suppress all progress bars in a block
    with ProgressContext(quiet=True):
        # No progress bars shown here
        for item in progress_bar(items, desc="Processing"):
            process(item)
    
    # Normal usage with progress
    for item in progress_bar(items, desc="Processing", colour="green"):
        process(item)
"""

from contextlib import contextmanager
from typing import Any, Generator, Iterable, Optional

from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_asyncio


# Global state for progress suppression
_SUPPRESS_PROGRESS = False


class ProgressContext:
    """
    Context manager for controlling progress bar visibility.

    Allows temporary suppression of all progress bars within a code block.
    Useful for quiet mode or when progress bars would clutter output.
    """

    def __init__(self, quiet: bool = False):
        """Initialize progress context."""
        self.quiet = quiet
        self.previous_state = None

    def __enter__(self):
        global _SUPPRESS_PROGRESS
        self.previous_state = _SUPPRESS_PROGRESS
        _SUPPRESS_PROGRESS = self.quiet
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _SUPPRESS_PROGRESS
        _SUPPRESS_PROGRESS = self.previous_state
        return False


def is_progress_suppressed() -> bool:
    """Check if progress bars are currently suppressed."""
    return _SUPPRESS_PROGRESS


def progress_bar(
    iterable: Iterable,
    desc: Optional[str] = None,
    total: Optional[int] = None,
    unit: str = "it",
    colour: Optional[str] = None,
    leave: bool = True,
    **kwargs: Any
) -> tqdm:
    """
    Create progress bar with automatic suppression support.

    Wrapper around tqdm that respects ProgressContext state.
    If progress is suppressed, returns plain iterator without overhead.
    """
    if _SUPPRESS_PROGRESS:
        # Return plain iterable without tqdm overhead
        return iterable

    return tqdm(
        iterable,
        desc=desc,
        total=total,
        unit=unit,
        colour=colour,
        leave=leave,
        disable=False,
        **kwargs
    )


async def progress_gather(
    *tasks,
    desc: Optional[str] = None,
    unit: str = "task",
    colour: Optional[str] = None,
    **kwargs: Any
):
    """
    Async gather with progress bar.

    Wrapper around tqdm.asyncio.gather that respects ProgressContext.
    """
    if _SUPPRESS_PROGRESS:
        # Use plain asyncio.gather
        import asyncio

        return await asyncio.gather(*tasks)

    return await tqdm_asyncio.gather(
        *tasks, desc=desc, unit=unit, colour=colour, disable=False, **kwargs
    )


@contextmanager
def progress_scope(
    desc: str,
    total: int,
    unit: str = "it",
    colour: Optional[str] = None,
) -> Generator[tqdm, None, None]:
    """
    Context manager for manual progress bar updates.

    Useful when you need fine-grained control over progress updates.
    """
    if _SUPPRESS_PROGRESS:
        # Yield dummy object with no-op update
        class DummyProgress:
            def update(self, n=1):
                pass

            def set_description(self, desc):
                pass

            def close(self):
                pass

        yield DummyProgress()
    else:
        pbar = tqdm(total=total, desc=desc, unit=unit, colour=colour)
        try:
            yield pbar
        finally:
            pbar.close()


async def progress_map_async(
    func,
    items: Iterable,
    desc: Optional[str] = None,
    colour: Optional[str] = None,
    concurrency: int = 100,
):
    """Map async function over items with progress bar and concurrency limit."""
    import asyncio

    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_func(item):
        async with semaphore:
            return await func(item)

    tasks = [bounded_func(item) for item in items]

    return await progress_gather(*tasks, desc=desc, unit="item", colour=colour)
