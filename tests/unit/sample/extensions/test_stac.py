"""
Tests for STAC extension.
"""

import pyarrow as pa
import pytest
from pydantic import ValidationError
from shapely.wkb import loads as wkb_loads

from tacotoolbox.sample.extensions.stac import STAC, raster_centroid


class TestSTACValidation:

    def test_accepts_valid_params(
        self, utm_crs, utm_geotransform, tensor_shape_3band, timestamp_2024
    ):
        ext = STAC(
            crs=utm_crs,
            geotransform=utm_geotransform,
            tensor_shape=tensor_shape_3band,
            time_start=timestamp_2024,
        )
        assert ext.crs == utm_crs

    def test_rejects_time_start_after_time_end(
        self,
        utm_crs,
        utm_geotransform,
        tensor_shape_3band,
        timestamp_2024,
        timestamp_2024_end,
    ):
        with pytest.raises(ValidationError, match="time_start"):
            STAC(
                crs=utm_crs,
                geotransform=utm_geotransform,
                tensor_shape=tensor_shape_3band,
                time_start=timestamp_2024_end,
                time_end=timestamp_2024,
            )

    def test_rejects_tensor_shape_1d(
        self, utm_crs, utm_geotransform, tensor_shape_1d, timestamp_2024
    ):
        with pytest.raises(ValidationError, match="2 dimensions"):
            STAC(
                crs=utm_crs,
                geotransform=utm_geotransform,
                tensor_shape=tensor_shape_1d,
                time_start=timestamp_2024,
            )


class TestSTACAutoPopulation:

    def test_computes_centroid(
        self, utm_crs, utm_geotransform, tensor_shape_3band, timestamp_2024
    ):
        ext = STAC(
            crs=utm_crs,
            geotransform=utm_geotransform,
            tensor_shape=tensor_shape_3band,
            time_start=timestamp_2024,
        )

        assert ext.centroid is not None
        point = wkb_loads(ext.centroid)
        assert -180 <= point.x <= 180
        assert -90 <= point.y <= 90

    def test_computes_time_middle(
        self,
        utm_crs,
        utm_geotransform,
        tensor_shape_3band,
        timestamp_2024,
        timestamp_2024_end,
    ):
        ext = STAC(
            crs=utm_crs,
            geotransform=utm_geotransform,
            tensor_shape=tensor_shape_3band,
            time_start=timestamp_2024,
            time_end=timestamp_2024_end,
        )

        expected = (timestamp_2024 + timestamp_2024_end) // 2
        assert ext.time_middle == expected

    def test_no_time_middle_without_time_end(
        self, utm_crs, utm_geotransform, tensor_shape_3band, timestamp_2024
    ):
        ext = STAC(
            crs=utm_crs,
            geotransform=utm_geotransform,
            tensor_shape=tensor_shape_3band,
            time_start=timestamp_2024,
        )

        assert ext.time_middle is None

    def test_preserves_explicit_centroid(
        self,
        utm_crs,
        utm_geotransform,
        tensor_shape_3band,
        timestamp_2024,
        wkb_point_wgs84,
    ):
        ext = STAC(
            crs=utm_crs,
            geotransform=utm_geotransform,
            tensor_shape=tensor_shape_3band,
            time_start=timestamp_2024,
            centroid=wkb_point_wgs84,
        )

        assert ext.centroid == wkb_point_wgs84


class TestRasterCentroid:

    def test_utm_to_wgs84(self, utm_crs, utm_geotransform):
        centroid = raster_centroid(
            crs=utm_crs,
            geotransform=utm_geotransform,
            raster_shape=(256, 256),
        )

        point = wkb_loads(centroid)
        assert -180 <= point.x <= 180
        assert -90 <= point.y <= 90

    def test_invalid_crs_raises(self, utm_geotransform):
        with pytest.raises(ValueError, match="Invalid CRS"):
            raster_centroid(
                crs="NOT_A_CRS",
                geotransform=utm_geotransform,
                raster_shape=(256, 256),
            )

    def test_includes_sample_id_in_error(self, utm_geotransform):
        with pytest.raises(ValueError, match="my_sample"):
            raster_centroid(
                crs="INVALID",
                geotransform=utm_geotransform,
                raster_shape=(256, 256),
                sample_id="my_sample",
            )


class TestSTACSchema:

    def test_schema_fields(
        self, utm_crs, utm_geotransform, tensor_shape_3band, timestamp_2024
    ):
        ext = STAC(
            crs=utm_crs,
            geotransform=utm_geotransform,
            tensor_shape=tensor_shape_3band,
            time_start=timestamp_2024,
        )
        schema = ext.get_schema()

        assert schema.field("stac:crs").type == pa.string()
        assert schema.field("stac:tensor_shape").type == pa.list_(pa.int64())
        assert schema.field("stac:geotransform").type == pa.list_(pa.float64())
        assert schema.field("stac:centroid").type == pa.binary()


class TestSTACCompute:

    def test_returns_single_row_table(
        self, make_sample, utm_crs, utm_geotransform, tensor_shape_3band, timestamp_2024
    ):
        sample = make_sample()
        ext = STAC(
            crs=utm_crs,
            geotransform=utm_geotransform,
            tensor_shape=tensor_shape_3band,
            time_start=timestamp_2024,
        )

        result = ext(sample)

        assert result.num_rows == 1
        assert result.column("stac:crs")[0].as_py() == utm_crs


class TestSTACIntegration:

    def test_extend_sample(
        self, make_sample, utm_crs, utm_geotransform, tensor_shape_3band, timestamp_2024
    ):
        sample = make_sample()
        sample.extend_with(
            STAC(
                crs=utm_crs,
                geotransform=utm_geotransform,
                tensor_shape=tensor_shape_3band,
                time_start=timestamp_2024,
            )
        )

        assert getattr(sample, "stac:crs") == utm_crs

        metadata = sample.export_metadata()
        assert "stac:crs" in metadata.column_names
