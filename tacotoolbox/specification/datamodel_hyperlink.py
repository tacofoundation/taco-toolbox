from typing import Optional

import pydantic


class HyperLink(pydantic.BaseModel):
    """This object describes a relationship with another entity. Data
    providers are advised to be liberal with links. For a full discussion of the
    situations where relative and absolute links are recommended see the 'Use of links'
    section of the STAC best practices.
    https://github.com/radiantearth/stac-spec/blob/master/best-practices.md#use-of-links
    """

    href: str
    rel: Optional[str] = None
    type: Optional[str] = None
    title: Optional[str] = None
