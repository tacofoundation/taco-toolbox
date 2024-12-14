import pathlib
from typing import List, Optional, Union

import pydantic
import tacotoolbox.specification


class TACOCollection(pydantic.BaseModel):
    id: str
    dataset_version: str
    description: str
    licenses: List[str]
    extent: tacotoolbox.specification.Extent
    providers: List[tacotoolbox.specification.Contact]
    task: tacotoolbox.specification.Task
    taco_version: str = "0.4.0"
    title: Optional[str] = None
    curators: Optional[List[tacotoolbox.specification.Contact]] = None
    keywords: Optional[List[str]] = None
    split_strategy: Optional[tacotoolbox.specification.SplitStrategy] = None
    discuss_link: Optional[tacotoolbox.specification.HyperLink] = None
    raw_link: Optional[tacotoolbox.specification.HyperLink] = None
    optical_data: Optional[tacotoolbox.specification.OpticalData] = None
    labels: Optional[tacotoolbox.specification.Labels] = None
    scientific: Optional[tacotoolbox.specification.Scientific] = None

    @pydantic.field_validator("id")
    def check_id(cls, value):
        if not value.islower():
            raise ValueError("ID must be lowercase")
        # only simple characters
        if not value.isalnum():
            raise ValueError("ID must be alphanumeric")
        return value

    @pydantic.field_validator("title")
    def check_title(cls, value):
        if len(value) > 250:
            raise ValueError("Title must be less than 250 characters")