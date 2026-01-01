"""Fixtures for Taco unit tests."""

from datetime import datetime, timezone

import pyarrow as pa
import pytest
from shapely.geometry import Point
from shapely.wkb import dumps as wkb_dumps

from tacotoolbox.sample.datamodel import Sample
from tacotoolbox.taco.datamodel import Provider
from tacotoolbox.tortilla.datamodel import Tortilla


@pytest.fixture
def tmp_file(tmp_path):
    """Factory for temporary files."""
    def _make(name: str, content: bytes = b"test") -> str:
        p = tmp_path / name
        p.write_bytes(content)
        return p
    return _make


@pytest.fixture
def make_sample(tmp_file):
    """Factory for FILE samples with real temp files."""
    def _make(sample_id: str, size: int = 100) -> Sample:
        path = tmp_file(f"{sample_id}.tif", b"x" * size)
        return Sample(id=sample_id, path=path)
    return _make


@pytest.fixture
def make_folder_sample(tmp_file):
    """Factory for FOLDER samples containing a Tortilla with FILE children."""
    def _make(folder_id: str, child_ids: list[str]) -> Sample:
        children = []
        for cid in child_ids:
            path = tmp_file(f"{folder_id}_{cid}.tif", b"x" * 100)
            children.append(Sample(id=cid, path=path))
        inner_tortilla = Tortilla(samples=children)
        return Sample(id=folder_id, path=inner_tortilla)
    return _make


@pytest.fixture
def make_tortilla(make_sample):
    """Factory for flat Tortilla with N FILE samples."""
    def _make(n: int = 3, prefix: str = "sample") -> Tortilla:
        samples = [make_sample(f"{prefix}_{i}") for i in range(n)]
        return Tortilla(samples=samples)
    return _make


@pytest.fixture
def make_tortilla_with_stac(tmp_file):
    """Factory for Tortilla with STAC centroid and time metadata on each sample."""
    def _make(coords_times: list[tuple[float, float, datetime]]) -> Tortilla:
        samples = []
        for i, (lon, lat, dt) in enumerate(coords_times):
            path = tmp_file(f"stac_s{i}.tif", b"x" * 100)
            s = Sample(id=f"s{i}", path=path)
            # Extend each sample with STAC metadata
            s.extend_with({
                "stac:centroid": wkb_dumps(Point(lon, lat)),
                "stac:time_start": dt,
            })
            samples.append(s)
        
        return Tortilla(samples=samples)
    return _make


@pytest.fixture
def minimal_taco_kwargs(make_tortilla):
    """Minimal valid kwargs for Taco construction."""
    return {
        "tortilla": make_tortilla(2),
        "id": "test-dataset",
        "dataset_version": "1.0.0",
        "description": "Test dataset",
        "licenses": ["CC-BY-4.0"],
        "providers": [Provider(name="Test Org")],
        "tasks": ["classification"],
    }