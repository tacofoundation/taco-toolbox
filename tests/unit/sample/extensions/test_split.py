"""
Tests for Split extension.
"""

import pyarrow as pa
import pytest
from pydantic import ValidationError

from tacotoolbox.sample.extensions.split import Split


class TestSplitValidation:

    @pytest.mark.parametrize("value", ["train", "test", "validation"])
    def test_accepts_valid_values(self, value):
        ext = Split(split=value)
        assert ext.split == value

    def test_rejects_invalid_value(self):
        with pytest.raises(ValidationError):
            Split(split="invalid")

    def test_rejects_case_variants(self):
        with pytest.raises(ValidationError):
            Split(split="Train")


class TestSplitSchema:

    def test_schema_has_single_string_field(self):
        ext = Split(split="train")
        schema = ext.get_schema()

        assert len(schema) == 1
        assert schema.field("split").type == pa.string()


class TestSplitCompute:

    def test_returns_single_row_table(self, make_sample):
        sample = make_sample()
        ext = Split(split="train")

        result = ext(sample)

        assert isinstance(result, pa.Table)
        assert result.num_rows == 1
        assert result.column("split")[0].as_py() == "train"

    def test_schema_only_returns_none(self, make_sample):
        sample = make_sample()
        ext = Split(split="train", return_none=True)

        result = ext(sample)

        assert result.column("split")[0].as_py() is None
        assert result.schema.equals(ext.get_schema())


class TestSplitIntegration:

    def test_extend_sample(self, make_sample):
        sample = make_sample()
        sample.extend_with(Split(split="test"))

        assert getattr(sample, "split") == "test"

        metadata = sample.export_metadata()
        assert metadata.column("split")[0].as_py() == "test"
