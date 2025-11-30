"""
Split extension for dataset partitioning.

Provides simple split field to indicate dataset partition membership
for training, evaluation, and validation workflows.

Exports to DataFrame:
- split: String ('train', 'test', or 'validation')
"""

from typing import Literal

import polars as pl
from pydantic import Field

from tacotoolbox.sample.datamodel import SampleExtension


class Split(SampleExtension):
    """
    Dataset partition identifier for ML workflows.

    Valid values:
    - train: Training partition
    - test: Test partition
    - validation: Validation partition
    """

    split: Literal["train", "test", "validation"] = Field(
        description="Dataset partition: 'train' for training, 'test' for evaluation, 'validation' for validation"
    )

    def get_schema(self) -> dict[str, pl.DataType]:
        """Return the expected schema for this extension."""
        return {"split": pl.Utf8()}

    def get_field_descriptions(self) -> dict[str, str]:
        """Return field descriptions for each field."""
        return {"split": "Dataset partition identifier (train, test, or validation)"}

    def _compute(self, sample) -> pl.DataFrame:
        """Actual computation logic - only called when schema_only=False."""
        return pl.DataFrame({"split": [self.split]}, schema=self.get_schema())
