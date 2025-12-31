# Test Fixtures

Pre-generated TACOTIFF GeoTIFFs and TACO datasets for tacotoolbox tests.

## Structure

```
fixtures/
├── README.md
├── regenerate.py
├── geotiffs/
│   ├── rgb_256x256.tif
│   ├── single_band_128x128.tif
│   ├── multitemporal_64x64.tif
│   └── categorical_128x128.tif
├── zip/
│   └── simple/simple.tacozip
└── folder/
    └── simple/
```

## GeoTIFF Fixtures

All GeoTIFFs are TACOTIFF-compliant COGs created with GDAL 3.11+:

| File | Bands | Size | Dtype | Location |
|------|-------|------|-------|----------|
| rgb_256x256.tif | 3 | 256×256 | float32 | Valencia (-0.5, 39.5) |
| single_band_128x128.tif | 1 | 128×128 | float32 | Paris (2.0, 48.9) |
| multitemporal_64x64.tif | 4 | 64×64 | float32 | Tokyo (139.5, 35.9) |
| categorical_128x128.tif | 1 | 128×128 | uint8 | NYC (-74.0, 40.8) |

### TACOTIFF Format

```
COMPRESS=ZSTD
LEVEL=9
PREDICTOR=2
BIGTIFF=YES
OVERVIEWS=NONE
BLOCKSIZE=128
INTERLEAVE=TILE
```

## TACO Datasets

| Name | Format | Samples | Purpose |
|------|--------|---------|---------|
| simple | ZIP | 4 | Basic create/read test |
| simple | FOLDER | 4 | Folder format test |

## Regenerating

```bash
# Requires GDAL >= 3.11 and tacotoolbox
python tests/fixtures/regenerate.py
```

Run when:
- TACOTIFF format requirements change
- Adding new test fixtures
- tacotoolbox updates the container format

## Usage in Tests

```python
import pytest
from pathlib import Path

FIXTURES = Path(__file__).parent.parent / "fixtures"

@pytest.fixture
def sample_geotiff():
    return FIXTURES / "geotiffs" / "rgb_256x256.tif"

@pytest.fixture
def sample_tacozip():
    return FIXTURES / "zip" / "simple" / "simple.tacozip"
```

## Requirements

- GDAL >= 3.11 (for INTERLEAVE=TILE)
- numpy
- tacotoolbox (for TACO dataset generation)