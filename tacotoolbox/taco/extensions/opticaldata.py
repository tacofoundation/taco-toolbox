import pydantic
import polars as pl

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
    """
    Individual spectral band definition for remote sensing data.
    
    Attributes:
        name: Band identifier (e.g., "B04", "Red")
        index: Band index in the data array
        common_name: Standard name (e.g., "Red", "NIR", "SWIR1")
        description: Human-readable description
        unit: Measurement unit (e.g., "nm", "μm")
        center_wavelength: Central wavelength in nm
        full_width_half_max: Bandwidth at half maximum intensity
    """
    
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
    
    Attributes:
        sensor: Sensor name (e.g., "sentinel2msi", "landsat8oli[B02,B03,B04]")
        bands: List of spectral band definitions
        
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
        if self.bands is None and self.sensor is not None:
            # Check if sensor is supported
            if any(sensor in self.sensor for sensor in SUPPORTED_SENSORS):
                # Handle band subset notation: "landsat8oli[B02,B03,B04]"
                if "[" in self.sensor and "]" in self.sensor:
                    # Clean up spaces
                    self.sensor = self.sensor.replace(" ", "")
                    
                    # Extract band list
                    start_idx = self.sensor.index("[") + 1
                    end_idx = self.sensor.index("]")
                    band_list = self.sensor[start_idx:end_idx].split(",")
                    
                    # Clean sensor name
                    self.sensor = self.sensor[:self.sensor.index("[")]
                    
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

    def get_schema(self) -> dict[str, pl.DataType]:
        return {
            "optical:sensor": pl.Utf8,
            "optical:bands": pl.List(pl.Struct([
                ("name", pl.Utf8),
                ("index", pl.Int32),
                ("common_name", pl.Utf8),
                ("description", pl.Utf8),
                ("unit", pl.Utf8),
                ("center_wavelength", pl.Float64),
                ("full_width_half_max", pl.Float64)
            ])),
            "optical:num_bands": pl.Int32
        }

    def _compute(self, taco) -> pl.DataFrame:
        """Convert optical data to DataFrame format."""
        # Convert SpectralBand objects to dictionaries
        bands_data = []
        if self.bands:
            for band in self.bands:
                bands_data.append({
                    "name": band.name,
                    "index": band.index,
                    "common_name": band.common_name,
                    "description": band.description,
                    "unit": band.unit,
                    "center_wavelength": band.center_wavelength,
                    "full_width_half_max": band.full_width_half_max
                })

        return pl.DataFrame([{
            "optical:sensor": self.sensor,
            "optical:bands": bands_data,
            "optical:num_bands": len(bands_data) if bands_data else 0
        }])


if __name__ == "__main__":
    # Auto-populate from sensor
    optical1 = OpticalData(sensor="sentinel2msi")
    
    # Subset of bands
    optical2 = OpticalData(sensor="landsat8oli[B02,B03,B04,B05]")
    
    # Manual bands
    custom_bands = [
        SpectralBand(name="RED", common_name="Red", center_wavelength=665),
        SpectralBand(name="NIR", common_name="Near Infrared", center_wavelength=842)
    ]
    optical3 = OpticalData(sensor="custom_sensor", bands=custom_bands)
