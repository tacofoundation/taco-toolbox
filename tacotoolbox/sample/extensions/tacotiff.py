"""
TACOTIFF header extraction extension.

Extracts compact binary header from TACOTIFF files for fast reading.

TacoHeader format contains all metadata needed to read TACOTIFF without
parsing the IFD headers.

Exports to DataFrame:
- taco:header: Binary (TacoHeader format, 35 bytes + tile counts array)
"""

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

    def get_field_descriptions(self) -> dict[str, str]:
        return {
            "taco:header": "Binary TACOTIFF header (35 bytes + tile counts) for fast reading without IFD parsing"
        }

    def _compute(self, sample) -> pl.DataFrame:
        import tacotiff

        # Extract binary header using tacotiff package
        header_bytes = tacotiff.metadata_from_tiff(str(sample.path))

        return pl.DataFrame({"taco:header": [header_bytes]}, schema=self.get_schema())
