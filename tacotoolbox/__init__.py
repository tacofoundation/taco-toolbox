import logging

from tacotoolbox import datamodel, validator
from tacotoolbox._logging import disable_logging, setup_basic_logging
from tacotoolbox.create import create
from tacotoolbox.docs import generate_html, generate_markdown
from tacotoolbox.export import export
from tacotoolbox.tacocat import create_tacocat
from tacotoolbox.translate import folder2zip, zip2folder


def _get_version() -> str:
    """Get package version."""
    try:
        from importlib import metadata

        return metadata.version("tacotoolbox")
    except (ImportError, ModuleNotFoundError):
        return "0.0.0"


def verbose(level=True):
    """
    Enable/disable verbose logging for tacotoolbox operations.

    Args:
        level: Logging level to enable:
            - True or "info": Show INFO and above (default)
            - "debug": Show DEBUG and above (very detailed)
            - False: Disable all logging

    Example:
        >>> import tacotoolbox
        >>>
        >>> # Enable standard logging
        >>> tacotoolbox.verbose()
        >>>
        >>> # Enable debug logging (very detailed)
        >>> tacotoolbox.verbose("debug")
        >>>
        >>> # Disable logging
        >>> tacotoolbox.verbose(False)
    """
    if level is False:
        disable_logging()
    elif level is True or level == "info":
        setup_basic_logging(level=logging.INFO)
    elif level == "debug":
        setup_basic_logging(level=logging.DEBUG)
    else:
        raise ValueError(
            f"Invalid verbose level: {level}. " "Use True, 'info', 'debug', or False."
        )


__version__ = _get_version()


__all__ = [
    "create",
    "create_tacocat",
    "datamodel",
    "export",
    "folder2zip",
    "generate_html",
    "generate_markdown",
    "validator",
    "verbose",
    "zip2folder",
]
