from tacotoolbox import datamodel
from tacotoolbox.create import create


def _get_version() -> str:
    """Get package version."""
    try:
        from importlib import metadata

        return metadata.version("tacotoolbox")
    except (ImportError, ModuleNotFoundError):
        return "0.0.0"


__version__ = _get_version()


__all__ = [
    "create",
    "datamodel",
    "edit",
]
