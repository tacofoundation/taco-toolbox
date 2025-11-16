from typing import Literal

import polars as pl

from tacotoolbox.taco.datamodel import TacoExtension

SplitStrategyType = Literal[
    "random", "stratified", "manual", "other", "none", "unknown"
]


class SplitStrategy(TacoExtension):
    """Dataset split strategy information."""

    strategy: SplitStrategyType

    def get_schema(self) -> dict[str, pl.DataType]:
        return {"split:strategy": pl.Utf8()}

    def _compute(self, taco) -> pl.DataFrame:
        return pl.DataFrame([{"split:strategy": self.strategy}])