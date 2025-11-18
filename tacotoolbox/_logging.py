"""
Logging configuration for tacotoolbox.

Provides centralized logging setup with appropriate levels and formats.
Users can configure logging behavior externally via standard logging config.

Usage:
    from tacotoolbox._logging import get_logger

    logger = get_logger(__name__)
    logger.debug("Detailed info for debugging")
    logger.info("General information")
    logger.warning("Warning message")
    logger.error("Error occurred")

Configuration (by user):
    import logging

    # Set level for all tacotoolbox
    logging.getLogger("tacotoolbox").setLevel(logging.DEBUG)

    # Set level for specific module
    logging.getLogger("tacotoolbox.create").setLevel(logging.INFO)

    # Add custom handler
    handler = logging.FileHandler("taco.log")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.getLogger("tacotoolbox").addHandler(handler)
"""

import logging

# Default format for tacotoolbox logs
DEFAULT_FORMAT = "%(levelname)s [%(name)s] %(message)s"


def get_logger(name: str) -> logging.Logger:
    """
    Get logger for tacotoolbox module.

    Creates logger with tacotoolbox namespace for easy filtering.
    By default, loggers inherit from root logger configuration.

    Args:
        name: Module name (typically __name__)

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing dataset")
        INFO [tacotoolbox.create] Processing dataset
    """
    # Ensure name is within tacotoolbox namespace
    if not name.startswith("tacotoolbox"):
        name = "tacotoolbox" if name == "__main__" else f"tacotoolbox.{name}"

    return logging.getLogger(name)


def setup_basic_logging(level: int = logging.INFO, fmt: str | None = None) -> None:
    """
    Setup basic logging configuration for tacotoolbox.

    This is a convenience function for quick setup. Advanced users
    should configure logging directly via logging.basicConfig() or
    logging configuration files.

    Args:
        level: Logging level (default: INFO)
        fmt: Log message format (default: DEFAULT_FORMAT)

    Example:
        >>> from tacotoolbox._logging import setup_basic_logging
        >>> import logging
        >>>
        >>> # Enable debug logging
        >>> setup_basic_logging(level=logging.DEBUG)
        >>>
        >>> # Custom format
        >>> setup_basic_logging(fmt="%(asctime)s - %(message)s")
    """
    if fmt is None:
        fmt = DEFAULT_FORMAT

    # Configure root tacotoolbox logger
    logger = logging.getLogger("tacotoolbox")
    logger.setLevel(level)

    # Add console handler if none exists
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)

    # Prevent propagation to root logger (avoid duplicate logs)
    logger.propagate = False


def enable_debug_logging() -> None:
    """
    Enable debug logging for all tacotoolbox modules.

    Convenience function equivalent to:
        logging.getLogger("tacotoolbox").setLevel(logging.DEBUG)

    Example:
        >>> from tacotoolbox._logging import enable_debug_logging
        >>> enable_debug_logging()
        >>> # Now all tacotoolbox modules log debug messages
    """
    setup_basic_logging(level=logging.DEBUG)


def disable_logging() -> None:
    """
    Disable all tacotoolbox logging.

    Sets level to CRITICAL+1, effectively silencing all logs.

    Example:
        >>> from tacotoolbox._logging import disable_logging
        >>> disable_logging()
        >>> # No tacotoolbox logs will be shown
    """
    logger = logging.getLogger("tacotoolbox")
    logger.setLevel(logging.CRITICAL + 1)
