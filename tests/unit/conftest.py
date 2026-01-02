"""
Shared fixtures for tacotoolbox unit tests.

Provides core factories for Sample, Tortilla, and Taco objects.
Module-specific fixtures remain in their respective conftest.py files.
"""

from pathlib import Path

import pyarrow as pa
import pytest
from shapely.geometry import Point
from shapely.wkb import dumps as wkb_dumps

from tacotoolbox.sample.datamodel import Sample
from tacotoolbox.taco.datamodel import Provider, Taco
from tacotoolbox.tortilla.datamodel import Tortilla


@pytest.fixture
def tmp_file(tmp_path: Path):
    """Factory for temporary files with content."""
    counter = 0

    def _make(name: str | None = None, content: bytes = b"test content") -> Path:
        nonlocal counter
        counter += 1
        fname = name or f"file_{counter:03d}.bin"
        p = tmp_path / fname
        p.write_bytes(content)
        return p

    return _make


@pytest.fixture
def make_sample(tmp_file):
    """Factory for FILE samples with real temp files."""
    counter = 0

    def _make(
        sample_id: str | None = None,
        content: bytes = b"x" * 100,
        **kwargs,
    ) -> Sample:
        nonlocal counter
        counter += 1
        sid = sample_id or f"sample_{counter:03d}"
        path = tmp_file(f"{sid}.bin", content)
        return Sample(id=sid, path=path, **kwargs)

    return _make


@pytest.fixture
def make_sample_with_stac(make_sample):
    """Factory for Sample with stac:centroid extension."""

    def _make(
        sample_id: str | None = None,
        lon: float = 0.0,
        lat: float = 0.0,
        **kwargs,
    ) -> Sample:
        sample = make_sample(sample_id, **kwargs)
        centroid_wkb = wkb_dumps(Point(lon, lat))
        sample.extend_with({"stac:centroid": centroid_wkb})
        return sample

    return _make


@pytest.fixture
def make_folder_sample(tmp_file):
    """Factory for FOLDER samples containing a Tortilla with FILE children."""

    def _make(
        folder_id: str,
        child_ids: list[str],
        child_content: bytes = b"x" * 100,
    ) -> Sample:
        children = []
        for cid in child_ids:
            path = tmp_file(f"{folder_id}_{cid}.bin", child_content)
            children.append(Sample(id=cid, path=path))
        inner_tortilla = Tortilla(samples=children)
        return Sample(id=folder_id, path=inner_tortilla)

    return _make


@pytest.fixture
def make_tortilla(make_sample):
    """Factory for flat Tortilla with N FILE samples."""

    def _make(n_samples: int = 3, prefix: str = "sample", **kwargs) -> Tortilla:
        samples = [make_sample(f"{prefix}_{i:03d}") for i in range(n_samples)]
        return Tortilla(samples=samples, **kwargs)

    return _make


@pytest.fixture
def make_tortilla_with_stac(make_sample_with_stac):
    """Factory for Tortilla with STAC centroids on each sample."""

    def _make(
        coords: list[tuple[float, float]] | None = None,
        prefix: str = "sample",
        **kwargs,
    ) -> Tortilla:
        if coords is None:
            coords = [(0.0, 0.0), (1.0, 1.0), (-1.0, -1.0)]
        samples = [
            make_sample_with_stac(f"{prefix}_{i:03d}", lon=lon, lat=lat)
            for i, (lon, lat) in enumerate(coords)
        ]
        return Tortilla(samples=samples, **kwargs)

    return _make


@pytest.fixture
def make_nested_tortilla(tmp_file):
    """Factory for nested Tortilla with PIT-compliant structure.

    PIT requires all folders at the same level to have identical child IDs
    at each position (isomorphic structure).
    """

    def _make(n_folders: int = 2, n_children: int = 2) -> Tortilla:
        folder_samples = []
        for i in range(n_folders):
            children = []
            for j in range(n_children):
                # Same child ID across all folders (PIT requirement)
                path = tmp_file(f"folder_{i}_child_{j}.bin", b"x" * 100)
                children.append(Sample(id=f"child_{j}", path=path))
            inner_tortilla = Tortilla(samples=children)
            folder_sample = Sample(id=f"folder_{i}", path=inner_tortilla)
            folder_samples.append(folder_sample)
        return Tortilla(samples=folder_samples)

    return _make


@pytest.fixture
def make_taco(make_tortilla):
    """Factory for minimal valid Taco objects."""

    def _make(
        tortilla: Tortilla | None = None, taco_id: str = "test-dataset", **kwargs
    ) -> Taco:
        defaults = {
            "tortilla": tortilla or make_tortilla(3),
            "id": taco_id,
            "dataset_version": "1.0.0",
            "description": "Test dataset",
            "licenses": ["CC-BY-4.0"],
            "providers": [Provider(name="Test Org")],
            "tasks": ["classification"],
        }
        defaults.update(kwargs)
        return Taco(**defaults)

    return _make


@pytest.fixture
def make_nested_taco(make_nested_tortilla, make_taco):
    """Factory for Taco with nested FOLDER structure."""

    def _make(n_folders: int = 2, n_children: int = 2, **kwargs) -> Taco:
        tortilla = make_nested_tortilla(n_folders, n_children)
        return make_taco(tortilla=tortilla, **kwargs)

    return _make
