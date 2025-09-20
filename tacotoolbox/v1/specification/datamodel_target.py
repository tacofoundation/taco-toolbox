from typing import Dict, Optional

import pydantic


# ML-STAC - Label extension ----------------------------
class Target(pydantic.BaseModel):
    """This extension provides a way to define the labels of a
    dataset. Useful for Image Classification, Object Detection and
    Semantic Segmentation tasks.

    Fields:
        labels (Dict[str, int]): A dictionary with the labels and
            their corresponding index.
    """

    labels: Dict[str, int]
    layers: Optional[int] = 0
