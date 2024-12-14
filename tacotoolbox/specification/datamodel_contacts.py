import re
from typing import List, Optional

import pydantic


class Link(pydantic.BaseModel):
    """This object describes a relationship with another entity. Data
    providers are advised to be liberal with links. For a full discussion of the
    situations where relative and absolute links are recommended see the 'Use of links'
    section of the STAC best practices.
    https://github.com/radiantearth/stac-spec/blob/master/best-practices.md#use-of-links
    """

    href: str
    rel: str
    type: Optional[str] = None
    title: Optional[str] = None


class Info(pydantic.BaseModel):
    """Contact extension is part of the STAC extension.
    Info is a part of the Contact extension.
    """

    value: str
    roles: Optional[List[str]] = None

    @pydantic.field_validator("value")
    def check_value(cls, v):
        if not re.match(r"^\+[1-9]{1}[0-9]{3,14}$", v):
            if not re.match(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$", v):
                raise ValueError("value must be a valid email or phone number")
        return v


class Address(pydantic.BaseModel):
    """Contact extension is part of the STAC extension.
    Address is a part of the Contact extension.
    """

    deliveryPoint: Optional[List[str]] = None
    city: Optional[str] = None
    administrativeArea: Optional[str] = None
    postalCode: Optional[str] = None
    country: Optional[str] = None


class Contact(pydantic.BaseModel):
    """Pydantic Contact extension from the STAC extension.
    More info: https://github.com/stac-extensions/contacts
    """

    name: Optional[str] = None
    organization: Optional[str] = None
    identifier: Optional[str] = None
    position: Optional[str] = None
    logo: Optional[Link] = None
    phones: Optional[List[Info]] = None
    emails: Optional[List[Info]] = None
    addresses: Optional[List[Address]] = None
    links: Optional[List[Link]] = None
    contactInstructions: Optional[str] = None
    roles: Optional[List[str]] = None

    # at least name or organization must be present
    @pydantic.model_validator(mode="after")
    def check_name_or_organization(cls, values):
        if not values.name and not values.organization:
            raise ValueError("name or organization must be present")
        return values
