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
            "publications:list": pl.List(
                pl.Struct(
                    [
                        pl.Field("doi", pl.Utf8()),
                        pl.Field("citation", pl.Utf8()),
                        pl.Field("summary", pl.Utf8()),
                    ]
                )
            )
        }

    def _compute(self, taco) -> pl.DataFrame:
        pubs_data = []
        for pub in self.publications:
            pubs_data.append(
                {"doi": pub.doi, "citation": pub.citation, "summary": pub.summary}
            )

        return pl.DataFrame([{"publications:list": pubs_data}])
