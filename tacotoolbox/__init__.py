from tacotoolbox import datamodel, validator
from tacotoolbox.create import create
from tacotoolbox.tacocat import create_tacocat
from tacotoolbox.tacollection import create_tacollection
from tacotoolbox.export import export
from tacotoolbox.translate import folder2zip, zip2folder
from tacotoolbox._logging import setup_basic_logging, disable_logging
import logging


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
    "create_tacollection",
    "datamodel",
    "validator",
    "export",
    "zip2folder",
    "folder2zip",
    "verbose",
]
