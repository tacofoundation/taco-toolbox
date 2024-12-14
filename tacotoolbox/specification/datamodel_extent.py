import re
from typing import List

import pydantic


class Extent(pydantic.BaseModel):
    """
    The `Extent` class represents the spatial and temporal extents of a dataset.
    It validates geographic boundaries and temporal intervals using Pydantic.

    Attributes:
        spatial (List[List[float]]):
            A list of spatial extents represented as bounding boxes.
            Each bounding box must contain either 4 (2D geometry) or 6 (3D geometry) floating-point values.
            - Longitude values must be within the range [-180, 180].
            - At least one spatial extent is required.

        temporal (List[List[str]]):
            A list of temporal extents represented as start and end timestamps in RFC 3339 format.
            - Each temporal extent must contain exactly 2 string values.
            - Both start and end times cannot be `None`.
            - At least one temporal extent is required.
    """

    spatial: List[List[float]]
    temporal: List[List[str]]

    @pydantic.field_validator("spatial")
    def check_spatial_extent(cls, v):
        for item in v:
            for value in item:
                if value < -180 or value > 180:
                    raise ValueError("Longitude must be between -180 and 180")
        return v

    @pydantic.field_validator("spatial")
    def check_spatial_length(cls, v):
        if len(v) < 1:
            raise ValueError("Must be at least 1 spatial extent")
        return v

    @pydantic.field_validator("spatial")
    def check_spatial_dimension(cls, v):
        for item in v:
            if not (len(item) == 4 or len(item) == 6):
                raise ValueError("Must be 2D or 3D geometries")
        return v

    @pydantic.field_validator("temporal")
    def check_temporal_extent(cls, v):
        if v[0][0] is None and v[0][1] is None:
            raise ValueError("Both start and end time cannot be None")
        return v

    @pydantic.field_validator("temporal")
    def check_temporal_length(cls, v):
        if len(v) < 1:
            raise ValueError("Must be at least 1 temporal extent")
        return v

    @pydantic.field_validator("temporal")
    def check_temporal_dimension(cls, v):
        for item in v:
            if not (len(item) == 2):
                raise ValueError("Must be a length of 2")
        return v

    @pydantic.field_validator("temporal")
    def check_temporal_regex(cls, v):
        regex_exp = re.compile("(\\+00:00|Z)$")
        for item in v:
            for value in item:
                if value is not None:
                    if not regex_exp.search(value):
                        raise ValueError("Must be a valid RFC 3339 timestamp")
        return v
