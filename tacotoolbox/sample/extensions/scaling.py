"""
Scaling extension.

Defines scaling transformations for packing/unpacking data.

Transformation: unpacked = packed * scale_factor + scale_offset

Exports to DataFrame:
- scaling:scale_factor: Float32 or List[Float32]
- scaling:scale_offset: Float32 or List[Float32]
- scaling:padding: List[Int32]
"""

import pyarrow as pa
import pydantic
from pydantic import Field

from tacotoolbox.sample.datamodel import SampleExtension


class Scaling(SampleExtension):
    """
    Data scaling and padding metadata for precision control.

    Defines transformations for scaling/unscaling data following CF conventions.
    All fields are optional - None indicates no transformation applied.

    Transformation: unpacked = packed * scale_factor + scale_offset

    Notes
    -----
    - When None, no transformation is applied
    - scale_factor and scale_offset must match unpacked data type
    - Padding format: [top, right, bottom, left]
    """

    scale_factor: float | list[float] | None = Field(
        default=None,
        description="Multiplicative scaling factor as Float32 or List[Float32]. Cannot be zero.",
    )
    scale_offset: float | list[float] | None = Field(
        default=None,
        description="Additive scaling offset as Float32 or List[Float32].",
    )
    padding: list[int] | None = Field(
        default=None,
        description="Spatial padding as List[Int32] with 4 elements: [top, right, bottom, left] in pixels.",
    )

    @pydantic.field_validator("scale_factor")
    @classmethod
    def validate_scale_factor(cls, v):
        """Ensure scale_factor is not zero when provided."""
        if v is None:
            return v

        if isinstance(v, list):
            for factor in v:
                if factor == 0.0:
                    raise ValueError("scale_factor cannot contain zero values")
            return v
        else:
            if v == 0.0:
                raise ValueError("scale_factor cannot be zero")
            return v

    @pydantic.field_validator("padding")
    @classmethod
    def validate_padding(cls, v):
        """Ensure padding is a list of 4 integers when provided."""
        if v is None:
            return v

        if len(v) != 4:
            raise ValueError(
                "padding must be a list of 4 integers [top, right, bottom, left]"
            )
        return [int(x) for x in v]

    def get_schema(self) -> pa.Schema:
        """Return the expected Arrow schema for this extension."""
        # Determine if factor/offset are lists or scalars
        has_list = isinstance(self.scale_factor, list) or isinstance(
            self.scale_offset, list
        )

        if has_list:
            return pa.schema(
                [
                    pa.field("scaling:scale_factor", pa.list_(pa.float32())),
                    pa.field("scaling:scale_offset", pa.list_(pa.float32())),
                    pa.field("scaling:padding", pa.list_(pa.int32())),
                ]
            )
        else:
            return pa.schema(
                [
                    pa.field("scaling:scale_factor", pa.float32()),
                    pa.field("scaling:scale_offset", pa.float32()),
                    pa.field("scaling:padding", pa.list_(pa.int32())),
                ]
            )

    def get_field_descriptions(self) -> dict[str, str]:
        """Return field descriptions for each field."""
        return {
            "scaling:scale_factor": "Multiplicative scaling factor for unpacking data (Float32 or List[Float32])",
            "scaling:scale_offset": "Additive offset for unpacking data (Float32 or List[Float32])",
            "scaling:padding": "Spatial padding [top, right, bottom, left] in pixels (List[Int32])",
        }

    def _compute(self, sample) -> pa.Table:
        """Actual computation logic - returns PyArrow Table."""
        # Determine output format based on inputs
        has_list = isinstance(self.scale_factor, list) or isinstance(
            self.scale_offset, list
        )

        if has_list:
            # Convert to lists for consistency
            factor = (
                self.scale_factor
                if isinstance(self.scale_factor, list)
                else ([self.scale_factor] if self.scale_factor is not None else None)
            )
            offset = (
                self.scale_offset
                if isinstance(self.scale_offset, list)
                else ([self.scale_offset] if self.scale_offset is not None else None)
            )

            data = {
                "scaling:scale_factor": [factor],
                "scaling:scale_offset": [offset],
                "scaling:padding": [self.padding],
            }
        else:
            # Scalar values
            data = {
                "scaling:scale_factor": [self.scale_factor],
                "scaling:scale_offset": [self.scale_offset],
                "scaling:padding": [self.padding],
            }

        return pa.Table.from_pydict(data, schema=self.get_schema())
