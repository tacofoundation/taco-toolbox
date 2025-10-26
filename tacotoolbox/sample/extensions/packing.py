import polars as pl
import pydantic

from tacotoolbox.sample.datamodel import SampleExtension


class Packing(SampleExtension):
    """
    Data packing metadata for compression and numerical precision control.

    Defines transformations for packing/unpacking data following CF conventions.
    All fields are optional - None indicates no transformation applied.

    Fields
    ------
    scale_factor : float, list[float], or None
        Multiplicative factor for scaling. Default is None (no scaling).
        Transformation: unpacked = packed * scale_factor + scale_offset
    scale_offset : float, list[float], or None
        Additive offset applied after scaling. Default is None (no offset).
    padding : list[int] or None
        Padding as [top, right, bottom, left]. Default is None (no padding).

    Notes
    -----
    - When None, no transformation is applied for that parameter
    - scale_factor and scale_offset must match unpacked data type
    - Use for data compression and storage optimization
    - Padding can be reverted using stored values

    Example
    -------
    >>> # Basic scaling
    >>> packing = Packing(scale_factor=0.01, scale_offset=-273.15)
    >>> 
    >>> # Per-band scaling
    >>> packing = Packing(
    ...     scale_factor=[0.0001, 0.0001, 0.0001],
    ...     scale_offset=[0.0, 0.0, 0.0]
    ... )
    >>> 
    >>> # With padding
    >>> packing = Packing(
    ...     scale_factor=0.01,
    ...     scale_offset=0.0,
    ...     padding=[10, 10, 10, 10]
    ... )
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
        has_list = (
            isinstance(self.scale_factor, list) or 
            isinstance(self.scale_offset, list)
        )
        
        if has_list:
            schema["packing:scale_factor"] = pl.List(pl.Float32())
            schema["packing:scale_offset"] = pl.List(pl.Float32())
        else:
            schema["packing:scale_factor"] = pl.Float32()
            schema["packing:scale_offset"] = pl.Float32()
        
        schema["packing:padding"] = pl.List(pl.Int32())
        
        return schema

    def _compute(self, sample) -> pl.DataFrame:
        """Actual computation logic - only called when return_none=False."""
        # Determine output format based on inputs
        has_list = (
            isinstance(self.scale_factor, list) or 
            isinstance(self.scale_offset, list)
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
                    "packing:scale_factor": [factor],
                    "packing:scale_offset": [offset],
                    "packing:padding": [self.padding],
                },
                schema=self.get_schema(),
            )
        else:
            # Scalar values
            return pl.DataFrame(
                {
                    "packing:scale_factor": [self.scale_factor],
                    "packing:scale_offset": [self.scale_offset],
                    "packing:padding": [self.padding],
                },
                schema=self.get_schema(),
            )