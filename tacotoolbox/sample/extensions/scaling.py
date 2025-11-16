import polars as pl
import pydantic

from tacotoolbox.sample.datamodel import SampleExtension


class Scaling(SampleExtension):
    """
    Data scaling and padding metadata for compression and precision control.

    Defines transformations for scaling/unscaling data following CF conventions.
    All fields are optional - None indicates no transformation applied.

    Transformation: unpacked = packed * scale_factor + scale_offset

    Notes
    -----
    - When None, no transformation is applied
    - scale_factor and scale_offset must match unpacked data type
    - Padding format: [top, right, bottom, left]
    - Follows CF (Climate and Forecast) metadata conventions
    """

    scale_factor: float | list[float] | None = None
    scale_offset: float | list[float] | None = None
    padding: list[int] | None = None

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

    def get_schema(self) -> dict[str, pl.DataType]:
        """Return the expected schema for this extension."""
        schema: dict[str, pl.DataType] = {}

        # Determine if factor/offset are lists or scalars
        has_list = isinstance(self.scale_factor, list) or isinstance(
            self.scale_offset, list
        )

        if has_list:
            schema["scaling:scale_factor"] = pl.List(pl.Float32())
            schema["scaling:scale_offset"] = pl.List(pl.Float32())
        else:
            schema["scaling:scale_factor"] = pl.Float32()
            schema["scaling:scale_offset"] = pl.Float32()

        schema["scaling:padding"] = pl.List(pl.Int32())

        return schema

    def _compute(self, sample) -> pl.DataFrame:
        """Actual computation logic - only called when schema_only=False."""
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

            return pl.DataFrame(
                {
                    "scaling:scale_factor": [factor],
                    "scaling:scale_offset": [offset],
                    "scaling:padding": [self.padding],
                },
                schema=self.get_schema(),
            )
        else:
            # Scalar values
            return pl.DataFrame(
                {
                    "scaling:scale_factor": [self.scale_factor],
                    "scaling:scale_offset": [self.scale_offset],
                    "scaling:padding": [self.padding],
                },
                schema=self.get_schema(),
            )