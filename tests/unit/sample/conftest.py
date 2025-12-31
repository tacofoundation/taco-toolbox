"""
Fixtures for Sample datamodel tests.

Provides factories for creating Sample instances and related test data.
Uses real TACOTIFF fixtures when available, falls back to simple temp files.
"""

import pathlib
from typing import Any

import pyarrow as pa
import pytest

FIXTURES_DIR = pathlib.Path(__file__).parent.parent.parent / "fixtures"
GEOTIFFS_DIR = FIXTURES_DIR / "geotiffs"


@pytest.fixture
def fixtures_available() -> bool:
    """Check if pre-generated fixtures exist."""
    return GEOTIFFS_DIR.exists() and any(GEOTIFFS_DIR.glob("*.tif"))


@pytest.fixture
def real_geotiff() -> pathlib.Path | None:
    """
    Return path to a real TACOTIFF fixture if available.
    
    Use for integration-style tests that need valid GeoTIFF metadata.
    Returns None if fixtures haven't been generated.
    """
    if not GEOTIFFS_DIR.exists():
        return None
    tifs = list(GEOTIFFS_DIR.glob("*.tif"))
    return tifs[0] if tifs else None


@pytest.fixture
def all_geotiffs() -> list[pathlib.Path]:
    """Return all available TACOTIFF fixtures."""
    if not GEOTIFFS_DIR.exists():
        return []
    return sorted(GEOTIFFS_DIR.glob("*.tif"))


@pytest.fixture
def tmp_file(tmp_path: pathlib.Path) -> pathlib.Path:
    """Single temp file with default content."""
    f = tmp_path / "test_file.bin"
    f.write_bytes(b"test content")
    return f


@pytest.fixture
def make_file(tmp_path: pathlib.Path):
    """
    Factory for creating temp files.
    
    Usage:
        f = make_file()                    # default content
        f = make_file(b"custom")           # custom bytes
        f = make_file(suffix=".tif")       # custom extension
    """
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
    """
    Factory for creating Sample instances.
    
    Usage:
        s = make_sample()                      # auto id + temp file
        s = make_sample(id="custom")           # custom id
        s = make_sample(path=existing_path)    # existing path
        s = make_sample(path=b"bytes")         # bytes path
    """
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
    """
    Factory for creating minimal Tortilla instances.
    
    For testing Sample with type=FOLDER.
    """
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
    return pa.Table.from_pydict({
        "arrow_field": [123],
        "arrow:namespaced": ["value"],
    })


@pytest.fixture
def multi_row_arrow_table() -> pa.Table:
    """Invalid multi-row Arrow Table (should be rejected)."""
    return pa.Table.from_pydict({
        "field": [1, 2, 3],
    })


class MockSampleExtension:
    """
    Mock SampleExtension for testing extend_with.
    
    Mimics the interface without requiring actual computation.
    """
    
    def __init__(self, schema_only: bool = False, raise_error: bool = False):
        self.schema_only = schema_only
        self.raise_error = raise_error
        self._called = False
    
    def get_schema(self) -> pa.Schema:
        return pa.schema([
            pa.field("mock:value", pa.int64()),
            pa.field("mock:name", pa.string()),
        ])
    
    def get_field_descriptions(self) -> dict[str, str]:
        return {
            "mock:value": "A mock integer value",
            "mock:name": "A mock string name",
        }
    
    def model_dump(self):
        """Pydantic-like interface for testing."""
        return {"value": 42, "name": "test"}
    
    def __call__(self, sample) -> pa.Table:
        self._called = True
        
        if self.raise_error:
            raise ValueError("Mock extension error")
        
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
    """Mock SampleExtension instance."""
    return MockSampleExtension()


@pytest.fixture
def mock_extension_schema_only() -> MockSampleExtension:
    """Mock SampleExtension with schema_only=True."""
    return MockSampleExtension(schema_only=True)