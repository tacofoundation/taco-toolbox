"""
Test fixtures generator for tacotoolbox.

Creates small GeoTIFFs and packages them into TACO datasets.
NO PROJ dependency - uses hardcoded WKT for CRS.

Output structure:
    tests/fixtures/
    ├── geotiffs/
    │   ├── rgb_256x256.tif
    │   ├── single_band_128x128.tif
    │   └── multitemporal_64x64.tif
    ├── zip/
    │   └── simple/simple.tacozip
    └── folder/
        └── simple/

Usage:
    python regenerate.py
"""

import os
import pathlib
import shutil

import numpy as np

# Fix PROJ database path if broken
if "PROJ_DATA" not in os.environ:
    # Try common locations
    for proj_path in [
        "/usr/share/proj",
        "/usr/local/share/proj",
        os.path.expanduser("~/miniconda3/share/proj"),
        os.path.expanduser("~/miniconda3/envs/majortom/share/proj"),
        os.path.expanduser("~/anaconda3/share/proj"),
    ]:
        if os.path.exists(proj_path) and os.path.isfile(os.path.join(proj_path, "proj.db")):
            os.environ["PROJ_DATA"] = proj_path
            break

try:
    from osgeo import gdal
    gdal.UseExceptions()
    HAS_GDAL = True
except ImportError:
    HAS_GDAL = False


FIXTURES_DIR = pathlib.Path(__file__).parent
GEOTIFFS_DIR = FIXTURES_DIR / "geotiffs"
ZIP_DIR = FIXTURES_DIR / "zip"
FOLDER_DIR = FIXTURES_DIR / "folder"

# COG creation options - TACOTIFF compliant
COG_OPTIONS = [
    "COMPRESS=ZSTD",
    "LEVEL=9",
    "PREDICTOR=2",
    "BIGTIFF=YES",
    "OVERVIEWS=NONE",
    "BLOCKSIZE=128",
    "INTERLEAVE=TILE",  # GDAL 3.11+
]


def create_geotiff(
    output_path: pathlib.Path,
    data: np.ndarray,
    origin: tuple[float, float] = (-0.5, 39.5),
    pixel_size: float = 0.001,
) -> pathlib.Path:
    """
    Create a simple COG GeoTIFF.
    """
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    
    bands, height, width = data.shape
    
    dtype_map = {
        np.float32: gdal.GDT_Float32,
        np.float64: gdal.GDT_Float64,
        np.int16: gdal.GDT_Int16,
        np.uint8: gdal.GDT_Byte,
        np.uint16: gdal.GDT_UInt16,
    }
    gdal_dtype = dtype_map.get(data.dtype.type, gdal.GDT_Float32)
    
    mem_drv = gdal.GetDriverByName("MEM")
    mem_ds = mem_drv.Create("", width, height, bands, gdal_dtype)
    mem_ds.SetGeoTransform((origin[0], pixel_size, 0, origin[1], 0, -pixel_size))
    
    # Skip projection to avoid PROJ dependency
    # For tests, geotransform is enough
    
    for i in range(bands):
        mem_ds.GetRasterBand(i + 1).WriteArray(data[i])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cog_drv = gdal.GetDriverByName("COG")
    cog_drv.CreateCopy(str(output_path), mem_ds, options=COG_OPTIONS)
    
    mem_ds = None
    return output_path


def create_fixtures():
    """Generate all GeoTIFF fixtures."""
    np.random.seed(42)
    
    # RGB 256x256
    rgb = np.random.rand(3, 256, 256).astype(np.float32)
    create_geotiff(GEOTIFFS_DIR / "rgb_256x256.tif", rgb, origin=(-0.5, 39.5))
    print("  Created: rgb_256x256.tif")
    
    # Single band 128x128
    single = np.random.rand(128, 128).astype(np.float32) * 100
    create_geotiff(GEOTIFFS_DIR / "single_band_128x128.tif", single, origin=(2.0, 48.9))
    print("  Created: single_band_128x128.tif")
    
    # Multitemporal 64x64 (4 bands)
    multi = np.random.rand(4, 64, 64).astype(np.float32)
    create_geotiff(GEOTIFFS_DIR / "multitemporal_64x64.tif", multi, origin=(139.5, 35.9))
    print("  Created: multitemporal_64x64.tif")
    
    # Categorical 128x128 (uint8 land cover)
    categorical = np.random.randint(0, 5, (128, 128), dtype=np.uint8)
    create_geotiff(GEOTIFFS_DIR / "categorical_128x128.tif", categorical, origin=(-74.0, 40.8))
    print("  Created: categorical_128x128.tif")


def create_taco_fixtures():
    """Create TACO datasets from GeoTIFFs."""
    try:
        import tacotoolbox
        from tacotoolbox.datamodel import Sample, Tortilla, Taco
    except ImportError:
        print("tacotoolbox not installed. Skipping TACO fixtures.")
        return
    
    geotiffs = sorted(GEOTIFFS_DIR.glob("*.tif"))
    if not geotiffs:
        print("No GeoTIFFs found.")
        return
    
    samples = []
    for tif in geotiffs:
        sample = Sample(id=tif.stem, path=tif)
        samples.append(sample)
    
    collection = {
        "id": "test-fixtures",
        "dataset_version": "1.0.0",
        "description": "Test fixtures for tacotoolbox",
        "licenses": ["CC-BY-4.0"],
        "providers": [{"name": "Test", "roles": ["producer"]}],
        "tasks": ["classification"],
    }
    
    tortilla = Tortilla(samples=samples)
    taco = Taco(tortilla=tortilla, **collection)
    
    # ZIP
    zip_out = ZIP_DIR / "simple" / "simple.tacozip"
    zip_out.parent.mkdir(parents=True, exist_ok=True)
    tacotoolbox.create(taco, zip_out, output_format="zip")
    print(f"  Created: {zip_out}")
    
    # FOLDER
    folder_out = FOLDER_DIR / "simple"
    if folder_out.exists():
        shutil.rmtree(folder_out)
    FOLDER_DIR.mkdir(parents=True, exist_ok=True)  # Parent only
    tacotoolbox.create(taco, folder_out, output_format="folder")
    print(f"  Created: {folder_out}")


def main():
    if not HAS_GDAL:
        print("GDAL not available. Install: conda install gdal")
        return
    
    # Clean
    for d in [GEOTIFFS_DIR, ZIP_DIR, FOLDER_DIR]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    
    print(f"GDAL {gdal.__version__}")
    print("\nGenerating GeoTIFFs...")
    create_fixtures()
    
    print("\nGenerating TACO datasets...")
    create_taco_fixtures()
    
    print("\nDone!")


if __name__ == "__main__":
    main()