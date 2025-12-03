"""
Split strategy extension for Taco datasets.

Documents how the dataset was partitioned into train/test/validation splits.

Split strategies:
- random: Random sampling without stratification
- stratified: Stratified sampling preserving class/label distribution
- manual: Manual assignment by domain expert
- other: Custom strategy not covered by standard categories
- none: Dataset has no splits (single partition)
- unknown: Split method not documented

Dataset-level metadata:
- split:strategy: String (one of: random, stratified, manual, other, none, unknown)
"""

from typing import Literal

import pyarrow as pa
from pydantic import Field

from tacotoolbox.taco.datamodel import TacoExtension

SplitStrategyType = Literal[
    "random", "stratified", "manual", "other", "none", "unknown"
]


class SplitStrategy(TacoExtension):
    """Dataset split strategy information."""

    strategy: SplitStrategyType = Field(
        description="Method used to partition dataset into train/test/validation splits."
    )

    def get_schema(self) -> pa.Schema:
        return pa.schema([pa.field("split:strategy", pa.string())])

    def get_field_descriptions(self) -> dict[str, str]:
        return {
            "split:strategy": "Dataset partitioning method (random, stratified, manual, other, none, or unknown)"
        }

    def _compute(self, taco) -> pa.Table:
        return pa.Table.from_pylist([{"split:strategy": self.strategy}])
