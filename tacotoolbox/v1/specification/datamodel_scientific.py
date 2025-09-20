from typing import List

import pydantic


class Publication(pydantic.BaseModel):
    doi: str
    citation: str
    summary: str


class Scientific(pydantic.BaseModel):
    doi: str
    citation: str
    summary: str
    publications: List[Publication]
