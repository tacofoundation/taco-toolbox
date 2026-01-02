"""Fixtures for Tortilla tests."""

from pathlib import Path

import pyarrow as pa
import pytest
from shapely.geometry import Point
from shapely.wkb import dumps as wkb_dumps

from tacotoolbox.sample.datamodel import Sample
from tacotoolbox.tortilla.datamodel import Tortilla


@pytest.fixture
def tmp_file(tmp_path: Path):
    """Factory for temporary files with content."""

    def _make(name: str = "test.bin", content: bytes = b"test content") -> Path:
        p = tmp_path / name
        p.write_bytes(content)
        return p

    return _make


@pytest.fixture
def make_sample(tmp_file):
    """Factory for basic Sample objects."""

    def _make(
        sample_id: str = "sample_001",
        content: bytes = b"x" * 100,
        **kwargs,
    ) -> Sample:
        path = tmp_file(f"{sample_id}.bin", content)
        return Sample(id=sample_id, path=path, **kwargs)

    return _make


@pytest.fixture
def make_sample_with_stac(make_sample):
    """Factory for Sample with stac:centroid extension."""

    def _make(
        sample_id: str = "sample_001",
        lon: float = 0.0,
        lat: float = 0.0,
        content: bytes = b"x" * 100,
    ) -> Sample:
        sample = make_sample(sample_id, content)

        centroid_wkb = wkb_dumps(Point(lon, lat))
        stac_table = pa.table(
            {"stac:centroid": [centroid_wkb]},
            schema=pa.schema([pa.field("stac:centroid", pa.binary())]),
        )
        sample.extend_with(stac_table)
        return sample

    return _make


@pytest.fixture
def make_tortilla(make_sample):
    """Factory for basic Tortilla objects."""

    def _make(
        n_samples: int = 3,
        prefix: str = "sample",
        **kwargs,
    ) -> Tortilla:
        samples = [make_sample(f"{prefix}_{i:03d}") for i in range(n_samples)]
        return Tortilla(samples=samples, **kwargs)

    return _make


@pytest.fixture
def make_tortilla_with_stac(make_sample_with_stac):
    """Factory for Tortilla with STAC centroids."""

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
def make_nested_tortilla(make_sample):
    """Factory for nested Tortilla (FOLDER samples containing child Tortillas)."""

    def _make(
        n_folders: int = 2,
        n_children: int = 2,
    ) -> Tortilla:
        folder_samples = []

        for i in range(n_folders):
            children = [make_sample(f"folder_{i}_child_{j}") for j in range(n_children)]
            inner_tortilla = Tortilla(samples=children)

            folder_sample = Sample(
                id=f"folder_{i}",
                path=inner_tortilla,
                type="FOLDER",
            )
            folder_samples.append(folder_sample)

        return Tortilla(samples=folder_samples)

    return _make
