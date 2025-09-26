import polars as pl
import pydantic

from tacotoolbox.taco.datamodel import TacoExtension


class Publication(pydantic.BaseModel):
    doi: str
    citation: str
    summary: str | None = None


class Publications(TacoExtension):
    publications: list[Publication]

    def get_schema(self) -> dict[str, pl.DataType]:
        return {
            "publications:list": pl.List(pl.Struct([("doi", pl.Utf8), ("citation", pl.Utf8), ("summary", pl.Utf8)]))
        }

    def _compute(self, taco) -> pl.DataFrame:
        pubs_data = []
        for pub in self.publications:
            pubs_data.append({"doi": pub.doi, "citation": pub.citation, "summary": pub.summary})

        return pl.DataFrame([{"publications:list": pubs_data}])


if __name__ == "__main__":
    # Example usage
    pubs = Publications(
        publications=[
            Publication(
                doi="10.1234/example.doi1", citation="Author A et al. (2023). Example Study 1. Journal of Examples."
            ),
            Publication(
                doi="10.5678/example.doi2",
                citation="Author B et al. (2024). Example Study 2. Examples Quarterly.",
            ),
        ]
    )
