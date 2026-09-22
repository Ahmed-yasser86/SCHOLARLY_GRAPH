"""Live OpenAlex client tests: real API calls with real paper titles in output."""

from __future__ import annotations

import pytest

from scholarly_graph.ingestion.discovery import OpenAlexClient, decode_openalex_abstract

pytestmark = pytest.mark.live


def test_search_returns_papers(capsys):
    papers = OpenAlexClient().search("economic inequality intergenerational mobility")
    assert len(papers) >= 1
    for paper in papers:
        print(f"OpenAlex hit: {paper['title']} ({paper['year']})")
    out = capsys.readouterr().out
    assert "OpenAlex" in out


def test_open_access_search_returns_only_open_access():
    papers = OpenAlexClient().search(
        "economic inequality intergenerational mobility", open_access_only=True
    )
    assert papers
    assert all(paper["open_access"] is True for paper in papers)


def test_fetch_by_known_doi():
    paper = OpenAlexClient().fetch_by_doi("10.1257/aer.104.5.141")
    assert paper["title"]
    assert paper["year"] >= 2000


def test_year_range_filter_is_respected():
    papers = OpenAlexClient().search(
        "economic inequality", year_from=2020, year_to=2022
    )
    assert papers
    assert all(2020 <= paper["year"] <= 2022 for paper in papers)


def test_each_paper_has_title_and_year():
    papers = OpenAlexClient().search("social mobility income inequality")
    for paper in papers:
        assert paper["title"].strip()
        assert 1900 <= paper["year"] <= 2030


def test_abstract_decoder_sorts_positions():
    assert decode_openalex_abstract({"world": [1], "hello": [0]}) == "hello world"
