"""
Test fixtures generator for tacotoolbox.

Creates GeoTIFFs and packages them into TACO datasets.

Output structure:
    tests/fixtures/
    ├── geotiffs/
    │   ├── rgb_256x256.tif
    │   ├── single_band_128x128.tif
    │   ├── multitemporal_64x64.tif
    │   └── categorical_128x128.tif
    ├── zip/
    │   ├── simple/simple.tacozip          # Flat (4 FILEs)
    │   ├── nested_a/nested_a.tacozip      # Nested (2 FOLDERs)
    │   └── nested_b/nested_b.tacozip      # Nested (2 FOLDERs, same schema)
    └── folder/
        └── simple/                         # Flat (4 FILEs)

Usage:
    python regenerate.py
"""

import os
import pathlib
import shutil

import numpy as np

# Fix PROJ database path if broken
if "PROJ_DATA" not in os.environ:
    for proj_path in [
        "/usr/share/proj",
        "/usr/local/share/proj",
        os.path.expanduser("~/miniconda3/share/proj"),
        os.path.expanduser("~/miniconda3/envs/majortom/share/proj"),
        os.path.expanduser("~/anaconda3/share/proj"),
    ]:
        if os.path.exists(proj_path) and os.path.isfile(
            os.path.join(proj_path, "proj.db")
        ):
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

COG_OPTIONS = [
    "COMPRESS=ZSTD",
    "LEVEL=9",
    "PREDICTOR=2",
    "BIGTIFF=YES",
    "OVERVIEWS=NONE",
    "BLOCKSIZE=128",
    "INTERLEAVE=TILE",
]


def create_geotiff(
    output_path: pathlib.Path,
    data: np.ndarray,
    origin: tuple[float, float] = (-0.5, 39.5),
    pixel_size: float = 0.001,
) -> pathlib.Path:
    """Create a simple COG GeoTIFF."""
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

    for i in range(bands):
        mem_ds.GetRasterBand(i + 1).WriteArray(data[i])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cog_drv = gdal.GetDriverByName("COG")
    cog_drv.CreateCopy(str(output_path), mem_ds, options=COG_OPTIONS)

    mem_ds = None
    return output_path


def create_geotiff_fixtures():
    """Generate all GeoTIFF fixtures."""
    np.random.seed(42)

    rgb = np.random.rand(3, 256, 256).astype(np.float32)
    create_geotiff(GEOTIFFS_DIR / "rgb_256x256.tif", rgb, origin=(-0.5, 39.5))
    print("  Created: rgb_256x256.tif")

    single = np.random.rand(128, 128).astype(np.float32) * 100
    create_geotiff(GEOTIFFS_DIR / "single_band_128x128.tif", single, origin=(2.0, 48.9))
    print("  Created: single_band_128x128.tif")

    multi = np.random.rand(4, 64, 64).astype(np.float32)
    create_geotiff(
        GEOTIFFS_DIR / "multitemporal_64x64.tif", multi, origin=(139.5, 35.9)
    )
    print("  Created: multitemporal_64x64.tif")

    categorical = np.random.randint(0, 5, (128, 128), dtype=np.uint8)
    create_geotiff(
        GEOTIFFS_DIR / "categorical_128x128.tif", categorical, origin=(-74.0, 40.8)
    )
    print("  Created: categorical_128x128.tif")


def create_flat_taco_fixtures():
    """Create flat TACO datasets (ZIP and FOLDER) from GeoTIFFs."""
    try:
        import tacotoolbox
        from tacotoolbox.datamodel import Sample, Tortilla, Taco
        from tacotoolbox.taco.datamodel import Provider
    except ImportError:
        print("tacotoolbox not installed. Skipping TACO fixtures.")
        return

    geotiffs = sorted(GEOTIFFS_DIR.glob("*.tif"))
    if not geotiffs:
        print("No GeoTIFFs found.")
        return

    samples = [Sample(id=tif.stem, path=tif) for tif in geotiffs]

    taco = Taco(
        id="test-fixtures",
        dataset_version="1.0.0",
        description="Test fixtures for tacotoolbox",
        licenses=["CC-BY-4.0"],
        providers=[Provider(name="Test", roles=["producer"])],
        tasks=["classification"],
        tortilla=Tortilla(samples=samples),
    )

    # ZIP
    zip_out = ZIP_DIR / "simple" / "simple.tacozip"
    zip_out.parent.mkdir(parents=True, exist_ok=True)
    tacotoolbox.create(taco, zip_out)
    print(f"  Created: {zip_out}")

    # FOLDER
    folder_out = FOLDER_DIR / "simple"
    if folder_out.exists():
        shutil.rmtree(folder_out)
    FOLDER_DIR.mkdir(parents=True, exist_ok=True)
    tacotoolbox.create(taco, folder_out)
    print(f"  Created: {folder_out}")


def create_nested_taco_fixtures():
    """
    Create nested TACO datasets for consolidation tests.

    Creates two PIT-compliant nested tacozips:
    - nested_a: 2 FOLDERs (sample_000, sample_001), each with 1 child (categorical_128x128)
    - nested_b: 2 FOLDERs (sample_002, sample_003), each with 1 child (categorical_128x128)

    Both have identical pit_schema and field_schema, allowing consolidation.
    """
    try:
        import tacotoolbox
        from tacotoolbox.datamodel import Sample, Tortilla, Taco
        from tacotoolbox.taco.datamodel import Provider
    except ImportError:
        print("tacotoolbox not installed. Skipping nested TACO fixtures.")
        return

    # Use categorical_128x128.tif as the child for all FOLDERs (PIT requirement)
    child_tif = GEOTIFFS_DIR / "categorical_128x128.tif"
    if not child_tif.exists():
        print(f"Child GeoTIFF not found: {child_tif}")
        return

    # --- nested_a: sample_000, sample_001 ---
    folder_samples_a = []
    for i in range(2):
        child = Sample(id=child_tif.stem, path=child_tif)
        folder = Sample(id=f"sample_{i:03d}", path=Tortilla(samples=[child]))
        folder_samples_a.append(folder)

    taco_a = Taco(
        id="nested-a",
        dataset_version="1.0.0",
        description="Nested fixture A for consolidation tests",
        licenses=["CC-BY-4.0"],
        providers=[Provider(name="Test", roles=["producer"])],
        tasks=["semantic-segmentation"],
        tortilla=Tortilla(samples=folder_samples_a),
    )

    zip_a = ZIP_DIR / "nested_a" / "nested_a.tacozip"
    zip_a.parent.mkdir(parents=True, exist_ok=True)
    tacotoolbox.create(taco_a, zip_a)
    print(f"  Created: {zip_a}")

    # --- nested_b: sample_002, sample_003 ---
    folder_samples_b = []
    for i in range(2, 4):
        child = Sample(id=child_tif.stem, path=child_tif)
        folder = Sample(id=f"sample_{i:03d}", path=Tortilla(samples=[child]))
        folder_samples_b.append(folder)

    taco_b = Taco(
        id="nested-b",
        dataset_version="1.0.0",
        description="Nested fixture B for consolidation tests",
        licenses=["CC-BY-4.0"],
        providers=[Provider(name="Test", roles=["producer"])],
        tasks=["semantic-segmentation"],
        tortilla=Tortilla(samples=folder_samples_b),
    )

    zip_b = ZIP_DIR / "nested_b" / "nested_b.tacozip"
    zip_b.parent.mkdir(parents=True, exist_ok=True)
    tacotoolbox.create(taco_b, zip_b)
    print(f"  Created: {zip_b}")


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
    create_geotiff_fixtures()

    print("\nGenerating flat TACO datasets...")
    create_flat_taco_fixtures()

    print("\nGenerating nested TACO datasets...")
    create_nested_taco_fixtures()

    print("\nDone!")


if __name__ == "__main__":
    main()
