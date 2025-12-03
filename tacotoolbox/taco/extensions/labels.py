"""
Labels extension for classification datasets.

Defines label/class schema for supervised learning tasks including
classification, detection, and segmentation.

Dataset-level metadata (not per-sample):
- labels:classes: List[Struct] with class definitions
  - name: String (human-readable class name)
  - category: String (category ID, stored as string even if integer)
  - description: String (optional class description)
- labels:description: String (optional overall labeling scheme description)
- labels:num_classes: Int32 (total number of classes)
"""

import pyarrow as pa
import pydantic
from pydantic import Field

from tacotoolbox.taco.datamodel import TacoExtension


class LabelClass(pydantic.BaseModel):
    """Individual label class definition."""

    name: str = Field(
        description="Human-readable class name (e.g., 'Forest', 'Urban', 'Water'). Should be unique within the dataset."
    )
    category: str | int = Field(
        description="Class identifier used in labels/masks. Can be integer (e.g., 0, 1, 2) or string code (e.g., 'FOR', 'URB')."
    )
    description: str | None = Field(
        default=None,
        description="Detailed description of class definition, edge cases, or labeling criteria (optional).",
    )


class Labels(TacoExtension):
    """Label definitions for classification datasets."""

    label_classes: list[LabelClass] = Field(
        description="Complete list of class definitions."
    )
    label_description: str | None = Field(
        default=None,
        description="Overall description of labeling scheme, methodology, or data source (optional).",
    )

    def get_schema(self) -> pa.Schema:
        return pa.schema(
            [
                pa.field(
                    "labels:classes",
                    pa.list_(
                        pa.struct(
                            [
                                pa.field("name", pa.string()),
                                pa.field("category", pa.string()),
                                pa.field("description", pa.string()),
                            ]
                        )
                    ),
                ),
                pa.field("labels:description", pa.string()),
                pa.field("labels:num_classes", pa.int32()),
            ]
        )

    def get_field_descriptions(self) -> dict[str, str]:
        return {
            "labels:classes": "List of class definitions with name, category ID, and optional description",
            "labels:description": "Overall description of labeling scheme, methodology, or data source",
            "labels:num_classes": "Total number of classes in the dataset",
        }

    def _compute(self, taco) -> pa.Table:
        """Convert label classes to Table format."""
        classes_data = []
        for label_class in self.label_classes:
            classes_data.append(
                {
                    "name": label_class.name,
                    "category": str(label_class.category),
                    "description": label_class.description,
                }
            )

        return pa.Table.from_pylist(
            [
                {
                    "labels:classes": classes_data,
                    "labels:description": self.label_description,
                    "labels:num_classes": len(self.label_classes),
                }
            ]
        )
