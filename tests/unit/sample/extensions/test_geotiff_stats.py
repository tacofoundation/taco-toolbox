"""
Tests for GeotiffStats extension.
"""

import pyarrow as pa
import pytest

from tacotoolbox.sample.datamodel import Sample
from tacotoolbox.sample.extensions.geotiff_stats import GeotiffStats


class TestGeotiffStatsValidation:

    def test_rejects_non_path_sample(self, make_tortilla):
        tortilla = make_tortilla()
        sample = Sample(id="folder", path=tortilla)
        ext = GeotiffStats()

        with pytest.raises(TypeError, match="requires sample.path to be a Path"):
            ext(sample)

    def test_rejects_nonexistent_file(self, tmp_path):
        fake_path = tmp_path / "nonexistent.tif"
        fake_path.write_bytes(b"x")
        sample = Sample(id="test", path=fake_path)
        fake_path.unlink()

        ext = GeotiffStats()
        with pytest.raises(ValueError, match="Could not open"):
            ext(sample)

    def test_categorical_requires_class_values(self, simple_geotiff):
        sample = Sample(id="test", path=simple_geotiff)
        ext = GeotiffStats(categorical=True)

        with pytest.raises(ValueError, match="class_values required"):
            ext(sample)


class TestGeotiffStatsContinuous:

    def test_returns_stats_per_band(self, simple_geotiff):
        sample = Sample(id="test", path=simple_geotiff)
        ext = GeotiffStats()

        result = ext(sample)
        stats = result.column("geotiff:stats")[0].as_py()

        assert len(stats) == 1  # single band
        assert len(stats[0]) == 9  # min, max, mean, std, valid%, p25, p50, p75, p95

    def test_multiband_stats(self, multiband_geotiff):
        sample = Sample(id="test", path=multiband_geotiff)
        ext = GeotiffStats()

        result = ext(sample)
        stats = result.column("geotiff:stats")[0].as_py()

        assert len(stats) == 3  # 3 bands
        for band_stats in stats:
            assert len(band_stats) == 9

    def test_uniform_band_stats(self, uniform_geotiff):
        sample = Sample(id="test", path=uniform_geotiff)
        ext = GeotiffStats()

        result = ext(sample)
        stats = result.column("geotiff:stats")[0].as_py()[0]

        min_val, max_val, mean_val, std_val = stats[:4]
        assert min_val == pytest.approx(42.0)
        assert max_val == pytest.approx(42.0)
        assert mean_val == pytest.approx(42.0)
        assert std_val == pytest.approx(0.0)


class TestGeotiffStatsCategorical:

    def test_returns_probabilities(self, categorical_geotiff):
        sample = Sample(id="test", path=categorical_geotiff)
        ext = GeotiffStats(categorical=True, class_values=[0, 1, 2])

        result = ext(sample)
        stats = result.column("geotiff:stats")[0].as_py()

        assert len(stats) == 1
        probs = stats[0]
        assert len(probs) == 3
        assert sum(probs) == pytest.approx(1.0)

    def test_probabilities_match_class_order(self, categorical_geotiff):
        sample = Sample(id="test", path=categorical_geotiff)

        ext1 = GeotiffStats(categorical=True, class_values=[0, 1, 2])
        ext2 = GeotiffStats(categorical=True, class_values=[2, 1, 0])

        result1 = ext1(sample)
        result2 = ext2(sample)

        probs1 = result1.column("geotiff:stats")[0].as_py()[0]
        probs2 = result2.column("geotiff:stats")[0].as_py()[0]

        assert probs1[0] == pytest.approx(probs2[2])
        assert probs1[2] == pytest.approx(probs2[0])


class TestGeotiffStatsScaling:

    def test_applies_scaling_to_continuous(self, simple_geotiff):
        sample = Sample(id="test", path=simple_geotiff)
        sample.extend_with({"scaling:scale_factor": 2.0, "scaling:scale_offset": 10.0})

        ext_unscaled = GeotiffStats()
        ext_scaled = GeotiffStats()

        result_unscaled = ext_unscaled(Sample(id="raw", path=simple_geotiff))
        result_scaled = ext_scaled(sample)

        raw_stats = result_unscaled.column("geotiff:stats")[0].as_py()[0]
        scaled_stats = result_scaled.column("geotiff:stats")[0].as_py()[0]

        # min, max, mean should be scaled
        assert scaled_stats[0] == pytest.approx(raw_stats[0] * 2.0 + 10.0, rel=0.01)
        # std only gets factor, no offset
        assert scaled_stats[3] == pytest.approx(raw_stats[3] * 2.0, rel=0.01)
        # valid% unchanged
        assert scaled_stats[4] == pytest.approx(raw_stats[4])


class TestGeotiffStatsSchema:

    def test_schema_type(self):
        ext = GeotiffStats()
        schema = ext.get_schema()

        assert schema.field("geotiff:stats").type == pa.list_(pa.list_(pa.float32()))


class TestGeotiffStatsIntegration:

    def test_extend_sample(self, simple_geotiff):
        sample = Sample(id="test", path=simple_geotiff)
        sample.extend_with(GeotiffStats())

        assert hasattr(sample, "geotiff:stats")

        metadata = sample.export_metadata()
        assert "geotiff:stats" in metadata.column_names
