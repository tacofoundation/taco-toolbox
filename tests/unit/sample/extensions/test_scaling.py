"""
Tests for Scaling extension.
"""

import pyarrow as pa
import pytest
from pydantic import ValidationError

from tacotoolbox.sample.extensions.scaling import Scaling


class TestScalingValidation:

    def test_accepts_all_none(self):
        ext = Scaling()
        assert ext.scale_factor is None
        assert ext.scale_offset is None
        assert ext.padding is None

    def test_accepts_scalar_values(self):
        ext = Scaling(scale_factor=0.1, scale_offset=100.0)
        assert ext.scale_factor == 0.1
        assert ext.scale_offset == 100.0

    def test_accepts_list_values(self):
        ext = Scaling(scale_factor=[0.1, 0.2], scale_offset=[10.0, 20.0])
        assert ext.scale_factor == [0.1, 0.2]

    def test_rejects_zero_scale_factor(self):
        with pytest.raises(ValidationError, match="cannot be zero"):
            Scaling(scale_factor=0.0)

    def test_rejects_zero_in_scale_factor_list(self):
        with pytest.raises(ValidationError, match="cannot contain zero"):
            Scaling(scale_factor=[1.0, 0.0, 2.0])

    def test_accepts_negative_scale_factor(self):
        ext = Scaling(scale_factor=-1.0)
        assert ext.scale_factor == -1.0

    def test_rejects_padding_wrong_length(self):
        with pytest.raises(ValidationError, match="4 integers"):
            Scaling(padding=[10, 20, 10])

    def test_padding_rejects_floats(self):
        with pytest.raises(ValidationError):
            Scaling(padding=[1.9, 2.1, 3.5, 4.0])


class TestScalingSchema:

    def test_scalar_schema_types(self):
        ext = Scaling(scale_factor=1.0, scale_offset=0.0)
        schema = ext.get_schema()

        assert schema.field("scaling:scale_factor").type == pa.float32()
        assert schema.field("scaling:scale_offset").type == pa.float32()
        assert schema.field("scaling:padding").type == pa.list_(pa.int32())

    def test_list_schema_types(self):
        ext = Scaling(scale_factor=[1.0, 2.0])
        schema = ext.get_schema()

        assert schema.field("scaling:scale_factor").type == pa.list_(pa.float32())
        assert schema.field("scaling:scale_offset").type == pa.list_(pa.float32())


class TestScalingCompute:

    def test_scalar_output(self, make_sample):
        sample = make_sample()
        ext = Scaling(scale_factor=0.5, scale_offset=10.0, padding=[1, 2, 3, 4])

        result = ext(sample)

        assert result.num_rows == 1
        assert result.column("scaling:scale_factor")[0].as_py() == pytest.approx(0.5)
        assert result.column("scaling:scale_offset")[0].as_py() == pytest.approx(10.0)
        assert result.column("scaling:padding")[0].as_py() == [1, 2, 3, 4]

    def test_list_output(self, make_sample):
        sample = make_sample()
        ext = Scaling(scale_factor=[0.1, 0.2], scale_offset=[10.0, 20.0])

        result = ext(sample)

        assert result.column("scaling:scale_factor")[0].as_py() == pytest.approx(
            [0.1, 0.2]
        )

    def test_none_values_in_output(self, make_sample):
        sample = make_sample()
        ext = Scaling()

        result = ext(sample)

        assert result.column("scaling:scale_factor")[0].as_py() is None


class TestScalingIntegration:

    def test_extend_sample(self, make_sample):
        sample = make_sample()
        sample.extend_with(Scaling(scale_factor=0.1, scale_offset=100.0))

        assert getattr(sample, "scaling:scale_factor") == pytest.approx(0.1)

        metadata = sample.export_metadata()
        assert "scaling:scale_factor" in metadata.column_names
