from tacotoolbox import datamodel, validator
from tacotoolbox.create import create
from tacotoolbox.tacocat import create_tacocat
from tacotoolbox.tacollection import create_tacollection
from tacotoolbox.export import export
from tacotoolbox.translate import folder2zip, zip2folder


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
    "create_tacocat",
    "create_tacollection",
    "datamodel",
    "validator",
    "export",
    "zip2folder",
    "folder2zip",
]