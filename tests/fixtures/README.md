# Test Fixtures Generator

Generates synthetic GeoTIFFs and TACO datasets for tacotoolbox tests.

## Why

Tests need real geospatial data without external dependencies. This script creates deterministic fixtures from scratch—no downloaded files, no network, no PROJ database required.

## Output

```
tests/fixtures/
├── geotiffs/
│   ├── rgb_256x256.tif              # 3-band float32, Valencia coords
│   ├── single_band_128x128.tif      # 1-band float32, Paris coords
│   ├── multitemporal_64x64.tif      # 4-band float32, Tokyo coords
│   └── categorical_128x128.tif      # uint8 land cover, NYC coords
├── zip/
│   ├── simple/simple.tacozip        # Flat: 4 FILEs (golden file)
│   ├── nested_a/nested_a.tacozip    # Nested: 2 FOLDERs (sample_000, sample_001)
│   └── nested_b/nested_b.tacozip    # Nested: 2 FOLDERs (sample_002, sample_003)
└── folder/
    └── simple/                       # Flat: 4 FILEs (golden file)
```

## TACO Structure

Both ZIP and FOLDER formats share the same internal layout:

```
dataset/                         # or dataset.tacozip
├── COLLECTION.json              # Dataset metadata (id, version, licenses, etc.)
├── METADATA/
│   └── level0.parquet           # Root tortilla index (all level-0 samples)
└── DATA/
    ├── rgb_256x256.tif
    ├── single_band_128x128.tif
    ├── multitemporal_64x64.tif
    └── categorical_128x128.tif
```

For hierarchical datasets (samples are FOLDERs, not FILEs):

```
dataset/
├── COLLECTION.json
├── METADATA/
│   ├── level0.parquet           # Level-0 index (FOLDER samples)
│   └── level1.parquet           # Level-1 index (children)
└── DATA/
    ├── sample_000/
    │   ├── __meta__             # Local metadata (parquet)
    │   └── categorical_128x128  # Child file (same ID in all FOLDERs - PIT)
    └── sample_001/
        ├── __meta__
        └── categorical_128x128
```

ZIP format adds a 157-byte `TACO_HEADER` with byte offsets for cloud-optimized range requests.

## Fixture Details

### Flat (`simple`)
- 4 FILE samples at level0
- No hierarchy (single level)
- Used as golden file for folder_writer and zip_writer tests

### Nested (`nested_a`, `nested_b`)
- 2 FOLDER samples each at level0
- Each FOLDER contains 1 FILE child (`categorical_128x128`)
- **PIT-compliant**: same child ID at same position across all FOLDERs
- Same `pit_schema` and `field_schema` (can be consolidated)
- Used for `tacocat` and `_tacollection` consolidation tests

When consolidated (`nested_a + nested_b`):
- level0: 4 FOLDERs total (sample_000, sample_001, sample_002, sample_003)
- level1: 4 children total

## Usage

```bash
cd tests/fixtures
python regenerate.py
```

## Requirements

- **GDAL** (with COG driver): `conda install gdal`
- **tacotoolbox**: required for TACO packaging step

## Technical Notes

- **Seed fixed** (`np.random.seed(42)`): reproducible across runs
- **No CRS projection**: skips PROJ dependency, geotransform is enough for tests
- **COG-compliant**: ZSTD compression, tiled interleave, 128px blocks
- **Coordinates**: each fixture has distinct origin for spatial query tests

## When to Regenerate

- After changing tacotoolbox datamodel
- After modifying TACO spec requirements
- When adding new test scenarios that need different fixture types

## Adding New Fixtures

Edit `regenerate.py`:

```python
# Example: add int16 elevation
elevation = np.random.randint(-100, 4000, (256, 256), dtype=np.int16)
create_geotiff(GEOTIFFS_DIR / "elevation_256x256.tif", elevation, origin=(10.0, 46.0))
```

Then run the script. New GeoTIFFs auto-include in flat TACO containers.