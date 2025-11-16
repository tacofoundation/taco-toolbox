import polars as pl

from tacotoolbox.sample.datamodel import SampleExtension


class Header(SampleExtension):
    """
    Extract TACOTIFF binary header metadata.

    Stores complete binary header (TacoHeader format) in taco:header field.
    This header contains all metadata needed for ultra-fast TACOTIFF reading.
    """

    def get_schema(self) -> dict[str, pl.DataType]:
        return {"taco:header": pl.Binary()}

    def _compute(self, sample) -> pl.DataFrame:
        import tacotiff

        # Extract binary header using tacotiff package
        header_bytes = tacotiff.metadata_from_tiff(str(sample.path))

        return pl.DataFrame({"taco:header": [header_bytes]}, schema=self.get_schema())