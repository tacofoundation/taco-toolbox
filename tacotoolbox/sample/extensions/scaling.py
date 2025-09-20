import pydantic
import polars as pl

# Import the ABC interface  
from tacotoolbox.sample.datamodel import SampleExtension

class Scaling(SampleExtension):
    """
    Scaling metadata for data packing/unpacking following CF conventions.
    
    Fields
    ------
    scaling_factor : float
        Multiplicative factor for scaling data. Default is 1.0 (no scaling).
    scaling_offset : float  
        Additive offset applied after scaling. Default is 0.0 (no offset).
        
    Notes
    -----
    - Transformation: unpacked_value = packed_value * scaling_factor + scaling_offset
    - Both attributes must be same type as unpacked data (float or double)
    - Used for data compression and numerical precision control
    """
    
    scaling_factor: float = 1.0
    scaling_offset: float = 0.0

    @pydantic.field_validator("scaling_factor")
    def validate_scaling_factor(cls, v):
        """Ensure scaling_factor is not zero."""
        if v == 0.0:
            raise ValueError("scaling_factor must be non-zero")
        return v

    def get_schema(self) -> dict[str, pl.DataType]:
        """Return the expected schema for this extension."""
        return {
            "scaling:factor": pl.Float32,
            "scaling:offset": pl.Float32
        }

    def _compute(self, sample) -> pl.DataFrame:
        """Actual computation logic - only called when return_none=False."""
        return pl.DataFrame({
            "scaling:factor": [self.scaling_factor],
            "scaling:offset": [self.scaling_offset]
        }, schema=self.get_schema())
    

if __name__ == "__main__":
    import pathlib
    import tempfile
    import numpy as np
    from osgeo import gdal    
    from tacotoolbox.sample.datamodel import Sample
    
    # Demo Scaling extension usage
    scaling_example = Scaling(
        scaling_factor=0.1,
        scaling_offset=273.15
    )
    
    # Create a small GeoTIFF with random values
    temp_file = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)
    temp_file.close()
    
    # Create 3-band GeoTIFF (50x50 pixels)
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(temp_file.name, 50, 50, 3, gdal.GDT_UInt16)
    
    # Fill bands with random values
    band1_data = np.random.randint(0, 1000, (50, 50), dtype=np.uint16)
    band2_data = np.random.randint(500, 1500, (50, 50), dtype=np.uint16)  
    band3_data = np.random.randint(0, 2000, (50, 50), dtype=np.uint16)
    
    dataset.GetRasterBand(1).WriteArray(band1_data)
    dataset.GetRasterBand(2).WriteArray(band2_data)
    dataset.GetRasterBand(3).WriteArray(band3_data)
    dataset.FlushCache()
    dataset = None
    
    # Create Sample
    sample = Sample(id="test_sample", path=pathlib.Path(temp_file.name), type="OTHER")
    
    # Apply extension
    result = scaling_example(sample)
    
    # Extend sample with scaling metadata
    sample.extend_with(scaling_example)
    sample.extend_with(result)
    sample.export_metadata()
    
    # Cleanup
    pathlib.Path(temp_file.name).unlink()
