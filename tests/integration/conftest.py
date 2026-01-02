"""
Test fixtures for integration tests.
"""

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
GEOTIFFS_DIR = FIXTURES_DIR / "geotiffs"
ZIP_DIR = FIXTURES_DIR / "zip"
FOLDER_DIR = FIXTURES_DIR / "folder"


@pytest.fixture
def geotiffs_dir():
    """Path to GeoTIFF fixtures."""
    assert GEOTIFFS_DIR.exists(), f"Missing: {GEOTIFFS_DIR}"
    return GEOTIFFS_DIR


@pytest.fixture
def folder_fixture():
    """Path to flat FOLDER fixture (golden file)."""
    path = FOLDER_DIR / "simple"
    assert path.exists(), f"Missing: {path}"
    return path


@pytest.fixture
def zip_fixture():
    """Path to flat ZIP fixture (golden file)."""
    path = ZIP_DIR / "simple" / "simple.tacozip"
    assert path.exists(), f"Missing: {path}"
    return path


@pytest.fixture
def nested_zip_a():
    """Path to nested ZIP fixture A (for consolidation tests)."""
    path = ZIP_DIR / "nested_a" / "nested_a.tacozip"
    assert path.exists(), f"Missing: {path}. Run: python tests/fixtures/regenerate.py"
    return path


@pytest.fixture
def nested_zip_b():
    """Path to nested ZIP fixture B (for consolidation tests)."""
    path = ZIP_DIR / "nested_b" / "nested_b.tacozip"
    assert path.exists(), f"Missing: {path}. Run: python tests/fixtures/regenerate.py"
    return path


@pytest.fixture
def nested_zips(nested_zip_a, nested_zip_b):
    """Both nested ZIP fixtures as a list (for consolidation)."""
    return [nested_zip_a, nested_zip_b]
