"""
Publications extension for Taco datasets.

Documents academic publications, technical reports, or preprints
associated with the dataset.

Dataset-level metadata:
- publications:list: List[Struct] with fields:
  - doi: String (Digital Object Identifier)
  - citation: String (formatted citation)
  - summary: String (optional description)
"""

import pyarrow as pa
import pydantic
from pydantic import Field

from tacotoolbox.taco.datamodel import TacoExtension


class Publication(pydantic.BaseModel):
    """Single publication reference."""

    doi: str = Field(
        description="Digital Object Identifier (e.g., '10.1038/s41586-021-03819-2'). Should be resolvable via https://doi.org/"
    )
    citation: str = Field(
        description="Formatted citation string (e.g., 'Smith et al. (2023). Dataset Name. Nature 123:456-789'). Use consistent citation style across publications."
    )
    summary: str | None = Field(
        default=None,
        description="Brief description of publication relevance to dataset (optional). E.g., 'Introduces methodology' or 'Validation study'.",
    )


class Publications(TacoExtension):
    """List of publications associated with dataset."""

    publications: list[Publication] = Field(
        description="List of academic publications, technical reports, or preprints describing or using the dataset. Include dataset paper, methodology papers, and significant derivative works."
    )

    def get_schema(self) -> pa.Schema:
        return pa.schema(
            [
                pa.field(
                    "publications:list",
                    pa.list_(
                        pa.struct(
                            [
                                pa.field("doi", pa.string()),
                                pa.field("citation", pa.string()),
                                pa.field("summary", pa.string()),
                            ]
                        )
                    ),
                )
            ]
        )

    def get_field_descriptions(self) -> dict[str, str]:
        return {
            "publications:list": "List of academic publications with DOI, citation, and optional summary describing dataset methodology or applications"
        }

    def _compute(self, taco) -> pa.Table:
        pubs_data = []
        for pub in self.publications:
            pubs_data.append(
                {"doi": pub.doi, "citation": pub.citation, "summary": pub.summary}
            )

        return pa.Table.from_pylist([{"publications:list": pubs_data}])
