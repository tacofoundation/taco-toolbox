import polars as pl
import pydantic

from tacotoolbox.taco.datamodel import TacoExtension


class LabelClass(pydantic.BaseModel):
    """Individual label class definition."""

    name: str
    category: str | int
    description: str | None = None


class Labels(TacoExtension):
    """Label definitions for classification datasets."""

    label_classes: list[LabelClass]
    label_description: str | None = None

    def get_schema(self) -> dict[str, pl.DataType]:
        return {
            "labels:classes": pl.List(
                pl.Struct(
                    [
                        pl.Field("name", pl.Utf8()),
                        pl.Field("category", pl.Utf8()),
                        pl.Field("description", pl.Utf8()),
                    ]
                )
            ),
            "labels:description": pl.Utf8(),
            "labels:num_classes": pl.Int32(),
        }

    def _compute(self, taco) -> pl.DataFrame:
        """Convert label classes to DataFrame format."""
        classes_data = []
        for label_class in self.label_classes:
            classes_data.append(
                {
                    "name": label_class.name,
                    "category": str(label_class.category),
                    "description": label_class.description,
                }
            )

        return pl.DataFrame(
            [
                {
                    "labels:classes": classes_data,
                    "labels:description": self.label_description,
                    "labels:num_classes": len(self.label_classes),
                }
            ]
        )
