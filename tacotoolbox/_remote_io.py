"""
Remote I/O operations for TACO creation.

Centralized remote file ops (HTTP, S3, GCS, Azure).
Uses obstore backend but designed for easy replacement.
The use of obstore is temporary until I have time to properly review OpenDAL.

Currently only used by export_writer.py for downloading remote ZIP entries.

TODO: The use of obstore is temporary until get time to review OpenDAL properly.
"""

import obstore as obs
from obstore.store import ObjectStore

from tacotoolbox._constants import PROTOCOL_MAPPINGS


def _create_store(url: str):
    """
    Create obstore ObjectStore from URL.

    Supports all protocols defined in PROTOCOL_MAPPINGS:
    - s3:// → S3Store
    - gs:// → GCSStore
    - az://, azure:// → AzureStore
    - http://, https:// → HTTPStore
    """
    # Build mapping from standard protocol to store class
    protocol_handlers: dict[str, type[ObjectStore]] = {
        PROTOCOL_MAPPINGS["s3"]["standard"]: obs.store.S3Store,
        PROTOCOL_MAPPINGS["gcs"]["standard"]: obs.store.GCSStore,
        PROTOCOL_MAPPINGS["azure"]["standard"]: obs.store.AzureStore,
        PROTOCOL_MAPPINGS["azure"]["alt"]: obs.store.AzureStore,  # azure:// alias
        PROTOCOL_MAPPINGS["http"]["standard"]: obs.store.HTTPStore,
        PROTOCOL_MAPPINGS["https"]["standard"]: obs.store.HTTPStore,
    }

    # Find matching protocol
    for protocol, store_class in protocol_handlers.items():
        if url.startswith(protocol):
            return store_class.from_url(url)  # type: ignore[union-attr, attr-defined]

    # Build error message with all supported protocols
    supported = sorted({PROTOCOL_MAPPINGS[p]["standard"] for p in PROTOCOL_MAPPINGS})
    raise ValueError(
        f"Unsupported URL scheme: {url}\n" f"Supported: {', '.join(supported)}"
    )


def download_range(url: str, offset: int, size: int, subpath: str = "") -> bytes:
    """
    Download byte range from remote file.

    Efficient for reading portions of large files without full download.
    Uses HTTP Range requests or cloud storage equivalent.

    Used by export_writer.py to download ZIP entries from remote TACOZIPs.

    Args:
        url: Base URL (e.g., "s3://bucket/file.tacozip")
        offset: Starting byte position
        size: Number of bytes to read
        subpath: Optional subpath within URL (unused for tacotoolbox, kept for API compat)

    Returns:
        Requested byte range
    """
    try:
        store = _create_store(url)
        result = obs.get_range(store, subpath, start=offset, length=size)
        return bytes(result)
    except Exception as e:
        raise OSError(
            f"Failed to download range [{offset}:{offset+size}] from {url}{subpath}: {e}"
        ) from e
