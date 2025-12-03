"""
Optical data extension for remote sensing datasets.

Defines spectral band characteristics for multispectral/hyperspectral imagery.
Supports automatic band lookup for common sensors or manual band definitions.

Supported sensors for auto-lookup:
- Landsat: landsat{1-5}mss, landsat{4,5}tm, landsat7etm, landsat{8,9}oli
- Sentinel: sentinel2msi
- Other: eo1ali, aster, modis

Dataset-level metadata:
- optical:sensor: String (sensor identifier)
- optical:bands: List[Struct] with spectral band definitions
- optical:num_bands: Int32 (number of bands)
"""

import pyarrow as pa
import pydantic

from tacotoolbox.taco.datamodel import TacoExtension
from tacotoolbox.taco.extensions.opticaldata_utils import get_sensor_bands

# Supported sensors for automatic band lookup
SUPPORTED_SENSORS = [
    "landsat1mss",
    "landsat2mss",
    "landsat3mss",
    "landsat4mss",
    "landsat5mss",
    "landsat4tm",
    "landsat5tm",
    "landsat7etm",
    "landsat8oli",
    "landsat9oli",
    "sentinel2msi",
    "eo1ali",
    "aster",
    "modis",
]


class SpectralBand(pydantic.BaseModel):
    """Individual spectral band definition for remote sensing data."""

    name: str
    index: int | None = None
    common_name: str | None = None
    description: str | None = None
    unit: str | None = None
    center_wavelength: float | None = None
    full_width_half_max: float | None = None


class OpticalData(TacoExtension):
    """
    Optical/spectral band information for remote sensing datasets.

    Provides spectral band definitions either explicitly or automatically
    from supported sensor types. Supports band subset selection.

    Usage:
        # Automatic band lookup
        optical = OpticalData(sensor="sentinel2msi")

        # Subset of bands
        optical = OpticalData(sensor="landsat8oli[B02,B03,B04,B05]")

        # Manual band definition
        bands = [SpectralBand(name="B04", common_name="Red", center_wavelength=665)]
        optical = OpticalData(sensor="custom", bands=bands)
    """

    sensor: str | None = None
    bands: list[SpectralBand] | None = None

    @pydantic.model_validator(mode="after")
    def populate_bands(self):
        """Auto-populate bands from sensor if not provided."""
        # Both can't be None
        if self.bands is None and self.sensor is None:
            raise ValueError("Either bands or sensor must be provided")

        # If bands provided, sensor must be provided too
        if self.bands is not None and self.sensor is None:
            raise ValueError("Sensor must be specified when bands are provided")

        # Auto-populate bands from sensor if not explicitly provided
        if (
            self.bands is None
            and self.sensor is not None
            and any(sensor in self.sensor for sensor in SUPPORTED_SENSORS)
        ):
            # Handle band subset notation: "landsat8oli[B02,B03,B04]"
            if "[" in self.sensor and "]" in self.sensor:
                # Clean up spaces
                self.sensor = self.sensor.replace(" ", "")

                # Extract band list
                start_idx = self.sensor.index("[") + 1
                end_idx = self.sensor.index("]")
                band_list = self.sensor[start_idx:end_idx].split(",")

                # Clean sensor name
                self.sensor = self.sensor[: self.sensor.index("[")]

                # Get specific bands
                bands_dict = get_sensor_bands(self.sensor, band_list)
            else:
                # Get all bands for sensor
                bands_dict = get_sensor_bands(self.sensor)

            # Convert to SpectralBand objects
            self.bands = [
                SpectralBand(name=name, **band_data)
                for name, band_data in bands_dict.items()
            ]

        return self

    def get_schema(self) -> pa.Schema:
        return pa.schema(
            [
                pa.field("optical:sensor", pa.string()),
                pa.field(
                    "optical:bands",
                    pa.list_(
                        pa.struct(
                            [
                                pa.field("name", pa.string()),
                                pa.field("index", pa.int32()),
                                pa.field("common_name", pa.string()),
                                pa.field("description", pa.string()),
                                pa.field("unit", pa.string()),
                                pa.field("center_wavelength", pa.float64()),
                                pa.field("full_width_half_max", pa.float64()),
                            ]
                        )
                    ),
                ),
                pa.field("optical:num_bands", pa.int32()),
            ]
        )

    def get_field_descriptions(self) -> dict[str, str]:
        return {
            "optical:sensor": "Sensor identifier (e.g., sentinel2msi, landsat8oli) for spectral characteristics",
            "optical:bands": "List of spectral band definitions with wavelength, name, and metadata",
            "optical:num_bands": "Total number of spectral bands in the dataset",
        }

    def _compute(self, taco) -> pa.Table:
        """Convert optical data to Table format."""
        # Convert SpectralBand objects to dictionaries
        bands_data = []
        if self.bands:
            for band in self.bands:
                bands_data.append(
                    {
                        "name": band.name,
                        "index": band.index,
                        "common_name": band.common_name,
                        "description": band.description,
                        "unit": band.unit,
                        "center_wavelength": band.center_wavelength,
                        "full_width_half_max": band.full_width_half_max,
                    }
                )

        return pa.Table.from_pylist(
            [
                {
                    "optical:sensor": self.sensor,
                    "optical:bands": bands_data,
                    "optical:num_bands": len(bands_data) if bands_data else 0,
                }
            ]
        )
