"""
Split extension for dataset partitioning.

Provides simple split field to indicate dataset partition membership
for training, evaluation, and validation workflows.

Exports to DataFrame:
- split: String ('train', 'test', or 'validation')
"""

from typing import Literal

import pyarrow as pa
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

    def get_schema(self) -> pa.Schema:
        """Return the expected Arrow schema for this extension."""
        return pa.schema(
            [
                pa.field("split", pa.string()),
            ]
        )

    def get_field_descriptions(self) -> dict[str, str]:
        """Return field descriptions for each field."""
        return {"split": "Dataset partition identifier (train, test, or validation)"}

    def _compute(self, sample) -> pa.Table:
        """Actual computation logic - returns PyArrow Table."""
        data = {"split": [self.split]}
        return pa.Table.from_pydict(data, schema=self.get_schema())
