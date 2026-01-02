"""
Fixtures for Sample tests (datamodel + extensions).
"""

import pathlib
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
from osgeo import gdal


# Base fixtures


@pytest.fixture
def tmp_file(tmp_path: pathlib.Path) -> pathlib.Path:
    """Single temp file with default content."""
    f = tmp_path / "test_file.bin"
    f.write_bytes(b"test content")
    return f


@pytest.fixture
def make_file(tmp_path: pathlib.Path):
    """Factory for creating temp files."""
    counter = 0

    def _make(content: bytes = b"x", suffix: str = ".bin") -> pathlib.Path:
        nonlocal counter
        counter += 1
        f = tmp_path / f"file_{counter:03d}{suffix}"
        f.write_bytes(content)
        return f

    return _make


@pytest.fixture
def make_sample(make_file):
    """Factory for creating Sample instances."""
    from tacotoolbox.sample.datamodel import Sample

    counter = 0

    def _make(id: str | None = None, path: Any = None, **kwargs) -> Sample:
        nonlocal counter
        counter += 1
        return Sample(
            id=id or f"sample_{counter:03d}",
            path=path if path is not None else make_file(),
            **kwargs,
        )

    return _make


@pytest.fixture
def make_tortilla(make_sample):
    """Factory for creating minimal Tortilla instances."""
    from tacotoolbox.tortilla.datamodel import Tortilla

    def _make(n_samples: int = 2, **kwargs) -> Tortilla:
        samples = [make_sample() for _ in range(n_samples)]
        return Tortilla(samples=samples, **kwargs)

    return _make


@pytest.fixture
def sample_with_extension(make_sample):
    """Sample with a simple dict extension already applied."""
    s = make_sample()
    s.extend_with({"custom_field": 42, "another:field": "value"})
    return s


@pytest.fixture
def single_row_arrow_table() -> pa.Table:
    """Valid single-row Arrow Table for extend_with tests."""
    return pa.Table.from_pydict(
        {
            "arrow_field": [123],
            "arrow:namespaced": ["value"],
        }
    )


@pytest.fixture
def multi_row_arrow_table() -> pa.Table:
    """Invalid multi-row Arrow Table (should be rejected)."""
    return pa.Table.from_pydict({"field": [1, 2, 3]})


class MockSampleExtension:
    """Mock SampleExtension for testing extend_with."""

    def __init__(self, schema_only: bool = False):
        self.schema_only = schema_only
        self._called = False

    def get_schema(self) -> pa.Schema:
        return pa.schema(
            [
                pa.field("mock:value", pa.int64()),
                pa.field("mock:name", pa.string()),
            ]
        )

    def get_field_descriptions(self) -> dict[str, str]:
        return {"mock:value": "A mock integer", "mock:name": "A mock string"}

    def model_dump(self):
        return {"value": 42, "name": "test"}

    def __call__(self, sample) -> pa.Table:
        self._called = True
        if self.schema_only:
            return pa.Table.from_pydict(
                {"mock:value": [None], "mock:name": [None]},
                schema=self.get_schema(),
            )
        return pa.Table.from_pydict(
            {"mock:value": [42], "mock:name": ["test"]},
            schema=self.get_schema(),
        )


@pytest.fixture
def mock_extension() -> MockSampleExtension:
    return MockSampleExtension()


# Timestamps (microseconds since Unix epoch, UTC)


@pytest.fixture
def timestamp_2024() -> int:
    dt = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    return int(dt.timestamp() * 1_000_000)


@pytest.fixture
def timestamp_2024_end() -> int:
    dt = datetime(2024, 1, 15, 18, 0, 0, tzinfo=timezone.utc)
    return int(dt.timestamp() * 1_000_000)


# STAC fixtures


@pytest.fixture
def utm_crs() -> str:
    return "EPSG:32630"


@pytest.fixture
def wgs84_crs() -> str:
    return "EPSG:4326"


@pytest.fixture
def utm_geotransform() -> tuple[float, ...]:
    return (600000.0, 10.0, 0.0, 4500000.0, 0.0, -10.0)


@pytest.fixture
def tensor_shape_3band() -> tuple[int, ...]:
    return (3, 256, 256)


@pytest.fixture
def tensor_shape_1d() -> tuple[int, ...]:
    return (256,)


# ISTAC fixtures


@pytest.fixture
def wkb_polygon_utm() -> bytes:
    from shapely.geometry import box
    from shapely.wkb import dumps

    return dumps(box(600000, 4499000, 601000, 4500000))


@pytest.fixture
def wkb_point_wgs84() -> bytes:
    from shapely.geometry import Point
    from shapely.wkb import dumps

    return dumps(Point(-0.5, 39.5))


@pytest.fixture
def wkb_empty_polygon() -> bytes:
    from shapely.geometry import Polygon
    from shapely.wkb import dumps

    return dumps(Polygon())


# GeoTIFF fixtures


@pytest.fixture
def simple_geotiff(tmp_path) -> pathlib.Path:
    path = tmp_path / "simple.tif"
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(str(path), 64, 64, 1, gdal.GDT_Float32)
    ds.SetGeoTransform((0, 1, 0, 64, 0, -1))
    data = np.random.rand(64, 64).astype(np.float32) * 100
    band = ds.GetRasterBand(1)
    band.WriteArray(data)
    band.ComputeStatistics(False)
    ds = None
    return path


@pytest.fixture
def multiband_geotiff(tmp_path) -> pathlib.Path:
    path = tmp_path / "multiband.tif"
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(str(path), 64, 64, 3, gdal.GDT_Float32)
    ds.SetGeoTransform((0, 1, 0, 64, 0, -1))
    for i in range(1, 4):
        data = np.random.rand(64, 64).astype(np.float32) * (i * 50)
        band = ds.GetRasterBand(i)
        band.WriteArray(data)
        band.ComputeStatistics(False)
    ds = None
    return path


@pytest.fixture
def categorical_geotiff(tmp_path) -> pathlib.Path:
    path = tmp_path / "categorical.tif"
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(str(path), 64, 64, 1, gdal.GDT_Byte)
    ds.SetGeoTransform((0, 1, 0, 64, 0, -1))
    data = np.random.choice([0, 1, 2], size=(64, 64)).astype(np.uint8)
    band = ds.GetRasterBand(1)
    band.WriteArray(data)
    band.ComputeStatistics(False)
    ds = None
    return path


@pytest.fixture
def uniform_geotiff(tmp_path) -> pathlib.Path:
    path = tmp_path / "uniform.tif"
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(str(path), 64, 64, 1, gdal.GDT_Float32)
    ds.SetGeoTransform((0, 1, 0, 64, 0, -1))
    data = np.full((64, 64), 42.0, dtype=np.float32)
    band = ds.GetRasterBand(1)
    band.WriteArray(data)
    band.ComputeStatistics(False)
    ds = None
    return path
