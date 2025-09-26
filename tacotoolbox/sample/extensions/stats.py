import functools
import pathlib

import numpy as np
import polars as pl
import pydantic

from tacotoolbox.sample.datamodel import SampleExtension


def requires_gdal(func):
    """Decorator to ensure GDAL is available."""
    _gdal_checked = False

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal _gdal_checked
        if not _gdal_checked:
            try:
                from osgeo import gdal
            except ImportError as err:
                raise ImportError("GDAL is required for GeoTIFF operations. Install: conda install gdal") from err
            _gdal_checked = True
        return func(*args, **kwargs)

    return wrapper


class GeotiffStats(SampleExtension):
    """Extract statistics from GeoTIFF files using GDAL.

    Returns Parquet-compatible list of lists structure:
    - Categorical: list[list[float32]] where values are probabilities x 10,000
    - Continuous: list[list[float32]] with [min, max, mean, std, valid%, p25, p50, p75, p95]

    Automatically applies scaling transformation if scaling metadata exists in sample.
    """

    categorical: bool = False
    class_values: list[int] | None = None

    _percentiles: list[int] = pydantic.PrivateAttr(default=[25, 50, 75, 95])
    _histogram_buckets: int = pydantic.PrivateAttr(default=100)

    def get_schema(self) -> dict[str, pl.DataType]:
        """Return the expected schema for this extension."""
        return {"internal:stats": pl.List(pl.List(pl.Float32))}

    @requires_gdal
    def _compute(self, sample) -> pl.DataFrame:
        """Extract statistics and apply scaling if present."""
        stats = self._extract_stats(sample.path)

        # Apply scaling transformation for continuous stats if scaling metadata exists
        scaling_factor = getattr(sample, "scaling:factor", 1.0)
        scaling_offset = getattr(sample, "scaling:offset", 0.0)

        if not self.categorical and (scaling_factor != 1.0 or scaling_offset != 0.0):
            stats = self._apply_scaling(stats, scaling_factor, scaling_offset)

        return pl.DataFrame({"internal:stats": [stats]}, schema=self.get_schema())

    @requires_gdal
    def _extract_stats(self, filepath: pathlib.Path) -> list:
        """Extract statistics using GDAL stats and histograms."""
        from osgeo import gdal

        ds = gdal.Open(str(filepath))
        if ds is None:
            raise ValueError(f"Could not open file: {filepath}")

        if ds.GetDriver().GetDescription() != "GTiff":
            ds = None
            raise ValueError(f"File {filepath} is not a GeoTIFF")

        try:
            if self.categorical:
                return self._categorical_stats(ds)
            else:
                return self._continuous_stats(ds)
        finally:
            ds = None

    def _categorical_stats(self, ds: "gdal.Dataset") -> list[list[float]]:
        """Extract categorical statistics as float32 (probabilities 0-1)."""
        if not self.class_values:
            raise ValueError("class_values required for categorical=True")

        result = []

        for band_idx in range(1, ds.RasterCount + 1):
            band = ds.GetRasterBand(band_idx)

            # Get basic stats to check for uniform values
            stats = band.GetStatistics(True, True)
            min_val, max_val, _, _ = stats

            # Handle uniform bands (all pixels have same value)
            if min_val == max_val:
                single_value = int(min_val)
                if single_value not in self.class_values:
                    raise ValueError(
                        f"Band {band_idx}: all pixels have value {single_value}, not in class_values {self.class_values}"
                    )

                # Create probabilities: 1.0 for the single value, 0.0 for others
                band_probs = []
                for class_val in self.class_values:
                    prob_float = 1.0 if class_val == single_value else 0.0  # FIXED: was 10000.0
                    band_probs.append(prob_float)

                result.append(band_probs)
                continue

            # Normal case: use histogram
            min_class, max_class = min(self.class_values), max(self.class_values)
            n_bins = max_class - min_class + 1
            histogram = band.GetHistogram(min_class, max_class, n_bins)

            total_pixels = sum(histogram)
            if total_pixels == 0:
                raise ValueError(f"Band {band_idx} has no pixels in class range")

            band_probs = []
            for class_val in self.class_values:
                bin_idx = class_val - min_class
                count = histogram[bin_idx] if 0 <= bin_idx < len(histogram) else 0
                probability = float(count) / float(total_pixels)  # This stays 0-1
                band_probs.append(probability)

            # Ensure probabilities sum to exactly 1.0 (within rounding)
            total_sum = sum(band_probs)
            if total_sum > 0:
                band_probs = [p / total_sum for p in band_probs]
            result.append(band_probs)

        return result

    def _continuous_stats(self, ds: "gdal.Dataset") -> list[list[float]]:
        """Extract continuous statistics as float32."""
        result = []

        for band_idx in range(1, ds.RasterCount + 1):
            band = ds.GetRasterBand(band_idx)

            # Get basic stats from metadata or compute
            metadata = band.GetMetadata()
            min_val = metadata.get("STATISTICS_MINIMUM")
            max_val = metadata.get("STATISTICS_MAXIMUM")
            mean_val = metadata.get("STATISTICS_MEAN")
            std_val = metadata.get("STATISTICS_STDDEV")
            valid_pct = metadata.get("STATISTICS_VALID_PERCENT", "100.0")

            if not all([min_val, max_val, mean_val, std_val]):
                stats = band.GetStatistics(True, True)
                min_val, max_val, mean_val, std_val = stats
                metadata = band.GetMetadata()
                valid_pct = metadata.get("STATISTICS_VALID_PERCENT", "100.0")

            min_val = float(min_val)
            max_val = float(max_val)
            mean_val = float(mean_val)
            std_val = float(std_val)
            valid_pct = float(valid_pct)

            # Handle uniform bands (all pixels have same value)
            if min_val == max_val:
                # All percentiles equal the single value, std = 0
                uniform_value = min_val
                percentiles = [uniform_value] * len(self._percentiles)
                band_stats = [min_val, max_val, mean_val, 0.0, valid_pct, *percentiles]
                result.append(band_stats)
                continue

            # Normal case: calculate percentiles from histogram
            histogram = band.GetHistogram(min_val, max_val, self._histogram_buckets)
            cumulative = np.cumsum(histogram)
            total_pixels = cumulative[-1]

            percentiles = []
            for p in self._percentiles:
                target = (p / 100.0) * total_pixels
                bin_idx = np.searchsorted(cumulative, target)
                bin_idx = min(bin_idx, self._histogram_buckets - 1)

                bin_width = (max_val - min_val) / self._histogram_buckets
                pct_val = min_val + (bin_idx + 0.5) * bin_width
                percentiles.append(float(pct_val))

            band_stats = [min_val, max_val, mean_val, std_val, valid_pct, *percentiles]
            result.append(band_stats)

        return result

    def _apply_scaling(
        self, stats: list[list[float]], scaling_factor: float, scaling_offset: float
    ) -> list[list[float]]:
        """Apply scaling transformation: real_value = packed_value * factor + offset."""
        result = []

        for band_stats in stats:
            # Apply scaling to all stats except valid_pct and std (std only gets factor)
            scaled = [
                band_stats[0] * scaling_factor + scaling_offset,  # min
                band_stats[1] * scaling_factor + scaling_offset,  # max
                band_stats[2] * scaling_factor + scaling_offset,  # mean
                band_stats[3] * scaling_factor,  # std (no offset)
                band_stats[4],  # valid_pct (unchanged)
            ]

            # Apply scaling to percentiles
            for p in band_stats[5:]:
                scaled.append(p * scaling_factor + scaling_offset)

            result.append(scaled)

        return result
