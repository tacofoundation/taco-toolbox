"""
Split extension for dataset partitioning.

Provides simple split field to indicate dataset partition membership
for training, evaluation, and validation workflows.
"""

from typing import Literal

import polars as pl

from tacotoolbox.sample.datamodel import SampleExtension


class Split(SampleExtension):
    """
    Dataset partition identifier for ML workflows.

    Valid values:
    - train: Training partition
    - eval: Evaluation partition
    - validation: Validation partition
    """

    split: Literal["train", "eval", "validation"]

    def get_schema(self) -> dict[str, pl.DataType]:
        """Return the expected schema for this extension."""
        return {"split": pl.Utf8()}

    def _compute(self, sample) -> pl.DataFrame:
        """Actual computation logic - only called when schema_only=False."""
        return pl.DataFrame({"split": [self.split]}, schema=self.get_schema())