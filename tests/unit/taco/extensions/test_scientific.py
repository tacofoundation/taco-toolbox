"""Publications extension tests."""

from tacotoolbox.taco.extensions.scientific import Publications, Publication


class TestPublicationsCompute:
    def test_single_publication(self):
        pubs = Publications(
            publications=[
                Publication(
                    doi="10.1234/example",
                    citation="Author et al. (2023). Title. Journal.",
                    summary="Dataset paper",
                )
            ]
        )
        table = pubs._compute(None)

        assert table.num_rows == 1
        pub_list = table["publications"][0].as_py()
        assert len(pub_list) == 1
        assert pub_list[0]["doi"] == "10.1234/example"
        assert pub_list[0]["summary"] == "Dataset paper"

    def test_multiple_publications(self):
        pubs = Publications(
            publications=[
                Publication(doi="10.1234/a", citation="Citation A"),
                Publication(doi="10.1234/b", citation="Citation B"),
            ]
        )
        table = pubs._compute(None)

        pub_list = table["publications"][0].as_py()
        assert len(pub_list) == 2
        assert {p["doi"] for p in pub_list} == {"10.1234/a", "10.1234/b"}

    def test_summary_optional(self):
        pubs = Publications(
            publications=[Publication(doi="10.1234/x", citation="Citation")]
        )
        table = pubs._compute(None)

        pub_list = table["publications"][0].as_py()
        assert pub_list[0]["summary"] is None
