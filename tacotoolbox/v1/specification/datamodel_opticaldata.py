from typing import List, Optional

import pydantic
from typing_extensions import Self

from tacotoolbox.specification.datamodel_opticaldata_utils import get_sensor_bands

soportedsensor = [
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
    """This extension provides a way to define the spectral bands of a
    dataset. Useful for Remote Sensing datasets.

    fields:
        band (str): The name of the band.
        index (Optional[int]): The index of the band.
        description (Optional[str]): A description of the band.
        unit (Optional[str]): The unit of the band.
        wavelengths (Optional[List[float]]): The wavelengths of the band.
            It must be a list of two floats. The first float is the minimum
            wavelength and the second float is the maximum wavelength.
    """

    name: str
    index: Optional[int]
    common_name: Optional[str] = None
    description: Optional[str] = None
    unit: Optional[str] = None
    center_wavelength: Optional[float] = None
    full_width_half_max: Optional[float] = None


class OpticalData(pydantic.BaseModel):
    """This extension provides a way to define the optical data of a
    dataset. Useful for Remote Sensing datasets.

    fields:
        bands (Optional[List[SpectralBand]]): The spectral bands of the dataset.
        sensor (Optional[str]): The sensor of the dataset.
    """

    sensor: Optional[str] = None
    bands: Optional[List[SpectralBand]] = None

    @pydantic.model_validator(mode="after")
    def check_bands(self) -> Self:
        # both can't be None
        if self.bands is None and self.sensor is None:
            raise ValueError("Either bands or sensor must be present")

        # if bands is present, sensor must be present
        if self.bands is not None and self.sensor is None:
            raise ValueError("Sensor must be present if bands is present")

        # if sensor is present, and bands is not present, try to get the bands
        if self.bands is None and self.sensor is not None:

            # Check if the sensor is supported
            if any(sensor in self.sensor for sensor in soportedsensor):
                # We support selected some bands by adding the bands
                # in brackets e.g. "landsat8oli[B01,B02,B03,B04,B05]"
                if "[" in self.sensor and "]" in self.sensor:
                    # Remove blank spaces
                    self.sensor = self.sensor.replace(" ", "")

                    # Get the bands
                    bands = self.sensor[
                        self.sensor.index("[") + 1 : self.sensor.index("]")
                    ].split(",")

                    # Remove the bands from the sensor
                    self.sensor = self.sensor[: self.sensor.index("[")]

                    # Get the bands
                    bands_dict: dict = get_sensor_bands(self.sensor, bands)
                else:
                    bands_dict: dict = get_sensor_bands(self.sensor)

                self.bands = [SpectralBand(name=k, **v) for k, v in bands_dict.items()]

        return self
