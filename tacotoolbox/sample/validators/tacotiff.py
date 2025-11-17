import functools
import pathlib
from typing import TYPE_CHECKING

from tacotoolbox.sample.validators._base import SampleValidator, ValidationError

if TYPE_CHECKING:
    from tacotoolbox.sample.datamodel import Sample


def requires_gdal(min_version="3.11"):
    """
    Decorator to ensure GDAL is available with minimum version.

    Caches the GDAL module check to make subsequent calls fast.
    """

    def decorator(func):
        # Cache the check result to make subsequent calls fast
        _gdal_checked = False
        _gdal_module = None

        def _raise_gdal_version_error(current_version, min_version):
            """Raise ImportError for GDAL version mismatch."""
            raise ImportError(
                f"GDAL {min_version}+ required. Current: {current_version}"
            )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal _gdal_checked, _gdal_module

            if not _gdal_checked:
                try:
                    from osgeo import gdal  # type: ignore[import-untyped]

                    _gdal_module = gdal

                    # Simple version comparison using tuple comparison
                    current = tuple(map(int, gdal.__version__.split(".")[:2]))
                    required = tuple(map(int, min_version.split(".")[:2]))

                    if current < required:
                        _raise_gdal_version_error(gdal.__version__, min_version)

                except ImportError as e:
                    if "GDAL" not in str(e):
                        raise ImportError(
                            f"GDAL {min_version}+ required for TACOTIFF validation. "
                            f"Install: conda install gdal>={min_version}"
                        ) from e
                    raise

                _gdal_checked = True

            return func(*args, **kwargs)

        return wrapper

    return decorator


class TacotiffValidator(SampleValidator):
    """
    Validator for TACOTIFF format using GDAL to enforce strict requirements.

    TACOTIFF format requirements:
    - Driver: GDAL generated COG (Cloud Optimized GeoTIFF)
    - Compression: ZSTD (for efficient storage and access)
    - Interleave: TILE (for efficient access patterns)
    - Overviews: None (to avoid redundant data storage)
    - BIGTIFF: YES (to standardize between large and small files)
    - GEOTIFF version: 1.1 (for standard compliance)

    Raises:
        ValidationError: If the file does not meet TACOTIFF requirements
        ImportError: If GDAL is not available or version < 3.11
    """

    @requires_gdal(min_version="3.11")
    def validate(self, sample: "Sample") -> None:
        """Validate sample as TACOTIFF format."""
        from osgeo import gdal

        # Check sample type is FILE
        if sample.type != "FILE":
            raise ValidationError(
                f"TACOTIFF requires type='FILE', got type='{sample.type}'"
            )

        # Check path is a Path object (not Tortilla)
        if not isinstance(sample.path, pathlib.Path):
            raise ValidationError(
                f"TACOTIFF requires path to be pathlib.Path, got {type(sample.path)}"
            )

        # Validate the actual file format
        self._validate_format(sample.path)

    def _validate_format(self, path: pathlib.Path) -> None:
        """Validate TACOTIFF file format using GDAL."""
        from osgeo import gdal

        # Open the dataset using GDAL
        ds = gdal.Open(str(path))

        # Check if GDAL can open the file
        if not ds:
            raise ValidationError(f"Cannot open {path} with GDAL")

        try:
            # Get image structure metadata from GDAL
            # This contains compression, interleave, and other format info
            ds_args = ds.GetMetadata("IMAGE_STRUCTURE")

            # Validate ZSTD compression
            compression = ds_args.get("COMPRESSION", "").upper()
            if compression != "ZSTD":
                raise ValidationError(
                    f"TACOTIFF assets must use ZSTD compression, found: {compression or 'NONE'}"
                )

            # Validate TILE interleave
            interleave = ds_args.get("INTERLEAVE", "").upper()
            if interleave != "TILE":
                raise ValidationError(
                    f"TACOTIFF assets must use TILE interleave, found: {interleave or 'PIXEL'}"
                )

            # Validate no overviews present
            band = ds.GetRasterBand(1)
            overview_count = band.GetOverviewCount()
            if overview_count != 0:
                raise ValidationError(
                    f"TACOTIFF assets must not have overviews, found: {overview_count} overview levels"
                )

        finally:
            # Always clean up GDAL dataset to free memory
            ds = None

    def get_supported_extensions(self) -> list[str]:
        """Get list of file extensions for TACOTIFF files."""
        return [".tif", ".tiff"]
