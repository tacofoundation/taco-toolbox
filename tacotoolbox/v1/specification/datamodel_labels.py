from typing import List, Optional, Union

import pydantic


class LabelClass(pydantic.BaseModel):
    name: str
    category: Union[str, int]
    description: Optional[str] = None


class Labels(pydantic.BaseModel):
    label_classes: List[LabelClass]
    label_description: Optional[str] = None
