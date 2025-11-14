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
    
    Example:
        >>> with ProgressContext(quiet=True):
        ...     # All progress bars suppressed here
        ...     for item in progress_bar(items):
        ...         process(item)
    """
    
    def __init__(self, quiet: bool = False):
        """
        Initialize progress context.
        
        Args:
            quiet: If True, suppress all progress bars in this context
        """
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
    """
    Check if progress bars are currently suppressed.
    
    Returns:
        True if progress bars should be hidden
    
    Example:
        >>> with ProgressContext(quiet=True):
        ...     print(is_progress_suppressed())
        True
    """
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
    
    Args:
        iterable: Iterable to wrap with progress bar
        desc: Description text (e.g., "Processing files")
        total: Total iterations (auto-detected if None)
        unit: Unit name (e.g., "file", "chunk")
        colour: Bar colour ("green", "blue", "cyan", "red")
        leave: Keep bar after completion
        **kwargs: Additional tqdm arguments
    
    Returns:
        tqdm progress bar or plain iterable if suppressed
    
    Example:
        >>> for item in progress_bar(items, desc="Loading", colour="green"):
        ...     process(item)
        Loading: 100%|██████████| 100/100 [00:01<00:00, 80.5it/s]
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
    
    Args:
        *tasks: Async tasks to gather
        desc: Description text
        unit: Unit name
        colour: Bar colour
        **kwargs: Additional tqdm arguments
    
    Returns:
        Results from gathered tasks
    
    Example:
        >>> results = await progress_gather(
        ...     *tasks,
        ...     desc="Downloading files",
        ...     colour="cyan"
        ... )
    """
    if _SUPPRESS_PROGRESS:
        # Use plain asyncio.gather
        import asyncio
        return await asyncio.gather(*tasks)
    
    return await tqdm_asyncio.gather(
        *tasks,
        desc=desc,
        unit=unit,
        colour=colour,
        disable=False,
        **kwargs
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
    
    Args:
        desc: Description text
        total: Total iterations
        unit: Unit name
        colour: Bar colour
    
    Yields:
        tqdm progress bar (or dummy object if suppressed)
    
    Example:
        >>> with progress_scope("Processing", total=100, colour="blue") as pbar:
        ...     for i in range(100):
        ...         process(i)
        ...         pbar.update(1)
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


# Convenience function for common async pattern
async def progress_map_async(
    func,
    items: Iterable,
    desc: Optional[str] = None,
    colour: Optional[str] = None,
    concurrency: int = 100,
):
    """
    Map async function over items with progress bar and concurrency limit.
    
    Args:
        func: Async function to apply
        items: Items to process
        desc: Progress description
        colour: Bar colour
        concurrency: Maximum concurrent tasks
    
    Returns:
        List of results
    
    Example:
        >>> results = await progress_map_async(
        ...     download_file,
        ...     urls,
        ...     desc="Downloading",
        ...     colour="green",
        ...     concurrency=50
        ... )
    """
    import asyncio
    
    semaphore = asyncio.Semaphore(concurrency)
    
    async def bounded_func(item):
        async with semaphore:
            return await func(item)
    
    tasks = [bounded_func(item) for item in items]
    
    return await progress_gather(
        *tasks,
        desc=desc,
        unit="item",
        colour=colour
    )