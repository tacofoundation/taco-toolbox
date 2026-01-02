"""
Tests for ISTAC extension.
"""

import pyarrow as pa
import pytest
from pydantic import ValidationError
from shapely.wkb import loads as wkb_loads

from tacotoolbox.sample.extensions.istac import ISTAC, geometry_centroid


class TestISTACValidation:

    def test_accepts_valid_params(self, utm_crs, wkb_polygon_utm, timestamp_2024):
        ext = ISTAC(
            crs=utm_crs,
            geometry=wkb_polygon_utm,
            time_start=timestamp_2024,
        )
        assert ext.crs == utm_crs

    def test_rejects_time_start_after_time_end(
        self, utm_crs, wkb_polygon_utm, timestamp_2024, timestamp_2024_end
    ):
        with pytest.raises(ValidationError, match="time_start"):
            ISTAC(
                crs=utm_crs,
                geometry=wkb_polygon_utm,
                time_start=timestamp_2024_end,
                time_end=timestamp_2024,
            )

    def test_rejects_empty_geometry(self, utm_crs, wkb_empty_polygon, timestamp_2024):
        with pytest.raises(ValidationError, match="empty"):
            ISTAC(
                crs=utm_crs,
                geometry=wkb_empty_polygon,
                time_start=timestamp_2024,
            )


class TestISTACAutoPopulation:

    def test_computes_centroid(self, utm_crs, wkb_polygon_utm, timestamp_2024):
        ext = ISTAC(
            crs=utm_crs,
            geometry=wkb_polygon_utm,
            time_start=timestamp_2024,
        )

        assert ext.centroid is not None
        point = wkb_loads(ext.centroid)
        assert -180 <= point.x <= 180
        assert -90 <= point.y <= 90

    def test_computes_time_middle(
        self, utm_crs, wkb_polygon_utm, timestamp_2024, timestamp_2024_end
    ):
        ext = ISTAC(
            crs=utm_crs,
            geometry=wkb_polygon_utm,
            time_start=timestamp_2024,
            time_end=timestamp_2024_end,
        )

        expected = (timestamp_2024 + timestamp_2024_end) // 2
        assert ext.time_middle == expected

    def test_no_time_middle_without_time_end(
        self, utm_crs, wkb_polygon_utm, timestamp_2024
    ):
        ext = ISTAC(
            crs=utm_crs,
            geometry=wkb_polygon_utm,
            time_start=timestamp_2024,
        )

        assert ext.time_middle is None

    def test_preserves_explicit_centroid(
        self, utm_crs, wkb_polygon_utm, timestamp_2024, wkb_point_wgs84
    ):
        ext = ISTAC(
            crs=utm_crs,
            geometry=wkb_polygon_utm,
            time_start=timestamp_2024,
            centroid=wkb_point_wgs84,
        )

        assert ext.centroid == wkb_point_wgs84


class TestGeometryCentroid:

    def test_utm_polygon_to_wgs84(self, utm_crs, wkb_polygon_utm):
        centroid = geometry_centroid(
            crs=utm_crs,
            geometry=wkb_polygon_utm,
        )

        point = wkb_loads(centroid)
        assert -180 <= point.x <= 180
        assert -90 <= point.y <= 90

    def test_invalid_crs_raises(self, wkb_polygon_utm):
        with pytest.raises(ValueError, match="Invalid CRS"):
            geometry_centroid(
                crs="NOT_A_CRS",
                geometry=wkb_polygon_utm,
            )

    def test_invalid_wkb_raises(self, utm_crs):
        with pytest.raises(ValueError, match="Invalid WKB"):
            geometry_centroid(
                crs=utm_crs,
                geometry=b"not valid wkb",
            )

    def test_empty_geometry_raises(self, utm_crs, wkb_empty_polygon):
        with pytest.raises(ValueError, match="empty"):
            geometry_centroid(
                crs=utm_crs,
                geometry=wkb_empty_polygon,
            )

    def test_includes_sample_id_in_error(self, wkb_polygon_utm):
        with pytest.raises(ValueError, match="my_sample"):
            geometry_centroid(
                crs="INVALID",
                geometry=wkb_polygon_utm,
                sample_id="my_sample",
            )


class TestISTACSchema:

    def test_schema_fields(self, utm_crs, wkb_polygon_utm, timestamp_2024):
        ext = ISTAC(
            crs=utm_crs,
            geometry=wkb_polygon_utm,
            time_start=timestamp_2024,
        )
        schema = ext.get_schema()

        assert schema.field("istac:crs").type == pa.string()
        assert schema.field("istac:geometry").type == pa.binary()
        assert schema.field("istac:centroid").type == pa.binary()


class TestISTACCompute:

    def test_returns_single_row_table(
        self, make_sample, utm_crs, wkb_polygon_utm, timestamp_2024
    ):
        sample = make_sample()
        ext = ISTAC(
            crs=utm_crs,
            geometry=wkb_polygon_utm,
            time_start=timestamp_2024,
        )

        result = ext(sample)

        assert result.num_rows == 1
        assert result.column("istac:crs")[0].as_py() == utm_crs


class TestISTACIntegration:

    def test_extend_sample(self, make_sample, utm_crs, wkb_polygon_utm, timestamp_2024):
        sample = make_sample()
        sample.extend_with(
            ISTAC(
                crs=utm_crs,
                geometry=wkb_polygon_utm,
                time_start=timestamp_2024,
            )
        )

        assert getattr(sample, "istac:crs") == utm_crs

        metadata = sample.export_metadata()
        assert "istac:crs" in metadata.column_names
