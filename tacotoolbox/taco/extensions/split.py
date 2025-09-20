import polars as pl
from typing_extensions import Literal

from tacotoolbox.taco.datamodel import TacoExtension

SplitStrategyType = Literal[
    "random", "stratified", "manual", "other", "none", "unknown"
]


class SplitStrategy(TacoExtension):
    """Dataset split strategy information."""
    
    strategy: SplitStrategyType
    
    def get_schema(self) -> dict[str, pl.DataType]:
        return {
            "split:strategy": pl.Utf8
        }
    
    def _compute(self, taco) -> pl.DataFrame:
        return pl.DataFrame([{
            "split:strategy": self.strategy
        }])


# Example usage
if __name__ == "__main__":
    split = SplitStrategy(
        strategy="stratified",
        description="Stratified split ensuring balanced class distribution across train/val/test sets"
    )