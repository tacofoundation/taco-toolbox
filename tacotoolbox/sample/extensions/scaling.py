import polars as pl
import pydantic

from tacotoolbox.sample.datamodel import SampleExtension


class Scaling(SampleExtension):
    """
    Scaling metadata for data packing/unpacking following CF conventions.

    Fields
    ------
    scaling_factor : float or list[float]
        Multiplicative factor for scaling data. Default is 1.0 (no scaling).
    scaling_offset : float or list[float]
        Additive offset applied after scaling. Default is 0.0 (no offset).
    padding : list[int]
        Padding applied as [top, right, bottom, left]. Default is [0, 0, 0, 0].

    Notes
    -----
    - Transformation: unpacked_value = packed_value * scaling_factor + scaling_offset
    - Both attributes must be same type as unpacked data (float or double)
    - Used for data compression and numerical precision control
    - Padding can be reverted using the stored values
    """

    scaling_factor: float | list[float] = 1.0
    scaling_offset: float | list[float] = 0.0
    padding: list[int] =  pydantic.Field(default_factory=lambda: [0, 0, 0, 0]) # [top, right, bottom, left]

    @pydantic.field_validator("scaling_factor")
    @classmethod
    def validate_scaling_factor(cls, v):
        """Ensure scaling_factor is not zero."""
        if isinstance(v, list):
            for factor in v:
                if factor == 0.0:
                    raise ValueError("scaling_factor must be non-zero")
            return v
        else:
            if v == 0.0:
                raise ValueError("scaling_factor must be non-zero")
            return v

    @pydantic.field_validator("padding")
    @classmethod
    def validate_padding(cls, v):
        """Ensure padding is a list of 4 integers."""
        if len(v) != 4:
            raise ValueError("padding must be a list of 4 integers [top, right, bottom, left]")
        return [int(x) for x in v]

    def get_schema(self) -> dict[str, pl.DataType]:
        """Return the expected schema for this extension."""
        schema = {"scaling:padding": pl.List(pl.Int32)}

        if isinstance(self.scaling_factor, list) or isinstance(self.scaling_offset, list):
            schema.update({"scaling:factor": pl.List(pl.Float32), "scaling:offset": pl.List(pl.Float32)})
        else:
            schema.update({"scaling:factor": pl.Float32, "scaling:offset": pl.Float32})

        return schema

    def _compute(self, sample) -> pl.DataFrame:
        """Actual computation logic - only called when return_none=False."""
        # Convert to lists if needed for consistency
        if isinstance(self.scaling_factor, list) or isinstance(self.scaling_offset, list):
            factor = self.scaling_factor if isinstance(self.scaling_factor, list) else [self.scaling_factor]
            offset = self.scaling_offset if isinstance(self.scaling_offset, list) else [self.scaling_offset]

            return pl.DataFrame(
                {"scaling:factor": [factor], "scaling:offset": [offset], "scaling:padding": [self.padding]},
                schema=self.get_schema(),
            )
        else:
            return pl.DataFrame(
                {
                    "scaling:factor": [self.scaling_factor],
                    "scaling:offset": [self.scaling_offset],
                    "scaling:padding": [self.padding],
                },
                schema=self.get_schema(),
            )
