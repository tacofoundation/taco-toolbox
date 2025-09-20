import pydantic
import polars as pl
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
            "labels:classes": pl.List(pl.Struct([
                ("name", pl.Utf8),
                ("category", pl.Utf8), 
                ("description", pl.Utf8)
            ])),
            "labels:description": pl.Utf8,
            "labels:num_classes": pl.Int32
        }
    
    def _compute(self, taco) -> pl.DataFrame:
        """Convert label classes to DataFrame format."""
        classes_data = []
        for label_class in self.label_classes:
            classes_data.append({
                "name": label_class.name,
                "category": str(label_class.category),
                "description": label_class.description
            })
        
        return pl.DataFrame([{
            "labels:classes": classes_data,
            "labels:description": self.label_description,
            "labels:num_classes": len(self.label_classes)
        }])


if __name__ == "__main__":
    # Example usage
    labels = Labels(
        label_classes=[
            LabelClass(name="Water", category=0, description="Bodies of water"),
            LabelClass(name="Forest", category=1, description="Forested areas"),
            LabelClass(name="Urban", category=2, description="Urban areas")
        ],
        label_description="Land cover classification labels"
    )    