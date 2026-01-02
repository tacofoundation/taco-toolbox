"""SplitStrategy extension tests."""

import pytest

from tacotoolbox.taco.extensions.split import SplitStrategy


class TestSplitStrategy:
    @pytest.mark.parametrize(
        "strategy", ["random", "stratified", "manual", "other", "none", "unknown"]
    )
    def test_valid_strategies(self, strategy):
        split = SplitStrategy(strategy=strategy)
        table = split._compute(None)
        assert table["split:strategy"][0].as_py() == strategy

    def test_invalid_strategy_rejected(self):
        with pytest.raises(ValueError):
            SplitStrategy(strategy="invalid")

    def test_schema_field(self):
        split = SplitStrategy(strategy="random")
        schema = split.get_schema()
        assert "split:strategy" in schema.names
