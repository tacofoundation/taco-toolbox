"""
TACOTIFF header extraction extension.

Extracts compact binary header from TACOTIFF files for fast reading.

TacoHeader format contains all metadata needed to read TACOTIFF without
parsing the IFD headers.

Exports to DataFrame:
- taco:header: Binary (TacoHeader format, 35 bytes + tile counts array)
"""

import pyarrow as pa

from tacotoolbox.sample.datamodel import SampleExtension


class Header(SampleExtension):
    """
    Extract TACOTIFF binary header metadata.

    Stores complete binary header (TacoHeader format) in taco:header field.
    This header contains all metadata needed for ultra-fast TACOTIFF reading.
    """

    def get_schema(self) -> pa.Schema:
        """Return the expected Arrow schema for this extension."""
        return pa.schema(
            [
                pa.field("taco:header", pa.binary()),
            ]
        )

    def get_field_descriptions(self) -> dict[str, str]:
        """Return field descriptions for each field."""
        return {
            "taco:header": "Binary TACOTIFF header (35 bytes + tile counts) for fast reading without IFD parsing"
        }

    def _compute(self, sample) -> pa.Table:
        """Actual computation logic - returns PyArrow Table."""
        import tacotiff

        # Extract binary header using tacotiff package
        header_bytes = tacotiff.metadata_from_tiff(str(sample.path))

        data = {"taco:header": [header_bytes]}
        return pa.Table.from_pydict(data, schema=self.get_schema())
