"""Labels extension tests."""

import pytest

from tacotoolbox.taco.extensions.labels import Labels, LabelClass


class TestLabelsCompute:
    def test_produces_correct_structure(self):
        labels = Labels(
            label_classes=[
                LabelClass(name="Forest", category=0),
                LabelClass(name="Urban", category=1, description="Built-up areas"),
            ]
        )
        table = labels._compute(None)
        
        assert table.num_rows == 1
        assert table["labels:num_classes"][0].as_py() == 2
        
        classes = table["labels:classes"][0].as_py()
        assert len(classes) == 2
        assert classes[0]["name"] == "Forest"
        assert classes[0]["category"] == "0"  # int converted to string
        assert classes[1]["description"] == "Built-up areas"

    def test_category_accepts_string_codes(self):
        labels = Labels(
            label_classes=[
                LabelClass(name="Forest", category="FOR"),
                LabelClass(name="Water", category="WAT"),
            ]
        )
        table = labels._compute(None)
        classes = table["labels:classes"][0].as_py()
        
        assert classes[0]["category"] == "FOR"
        assert classes[1]["category"] == "WAT"

    def test_optional_description(self):
        labels = Labels(
            label_classes=[LabelClass(name="Test", category=0)],
            label_description="Custom labeling methodology",
        )
        table = labels._compute(None)
        
        assert table["labels:description"][0].as_py() == "Custom labeling methodology"