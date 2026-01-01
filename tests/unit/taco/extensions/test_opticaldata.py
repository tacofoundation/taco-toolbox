"""OpticalData extension tests."""

import pytest

from tacotoolbox.taco.extensions.opticaldata import OpticalData, SpectralBand, SUPPORTED_SENSORS


class TestOpticalDataValidation:
    def test_requires_sensor_or_bands(self):
        with pytest.raises(ValueError):
            OpticalData()

    def test_bands_require_sensor(self):
        bands = [SpectralBand(name="B01")]
        with pytest.raises(ValueError, match="Sensor must be specified"):
            OpticalData(bands=bands)


class TestSensorAutoLookup:
    def test_sentinel2_all_bands(self):
        opt = OpticalData(sensor="sentinel2msi")
        assert len(opt.bands) == 13
        band_names = {b.name for b in opt.bands}
        assert {"B01", "B02", "B03", "B04", "B8A", "B12"}.issubset(band_names)

    def test_landsat8_all_bands(self):
        opt = OpticalData(sensor="landsat8oli")
        assert len(opt.bands) == 11

    @pytest.mark.parametrize("sensor", SUPPORTED_SENSORS)
    def test_all_supported_sensors_load(self, sensor):
        opt = OpticalData(sensor=sensor)
        assert len(opt.bands) > 0


class TestBandSubsetNotation:
    def test_bracket_notation_selects_bands(self):
        opt = OpticalData(sensor="sentinel2msi[B02,B03,B04]")
        assert len(opt.bands) == 3
        assert {b.name for b in opt.bands} == {"B02", "B03", "B04"}

    def test_bracket_notation_strips_spaces(self):
        opt = OpticalData(sensor="sentinel2msi[ B02 , B03 ]")
        assert len(opt.bands) == 2

    def test_sensor_name_cleaned_after_bracket(self):
        opt = OpticalData(sensor="landsat8oli[B02,B03]")
        assert opt.sensor == "landsat8oli"


class TestOpticalDataCompute:
    def test_schema_fields(self):
        opt = OpticalData(sensor="sentinel2msi[B04]")
        table = opt._compute(None)
        
        assert "optical:sensor" in table.schema.names
        assert "optical:bands" in table.schema.names
        assert "optical:num_bands" in table.schema.names
        assert table["optical:num_bands"][0].as_py() == 1

    def test_band_metadata_preserved(self):
        opt = OpticalData(sensor="sentinel2msi[B04]")
        b04 = opt.bands[0]
        
        assert b04.common_name == "red"
        assert b04.center_wavelength == 664.5
        assert b04.full_width_half_max == 29.0