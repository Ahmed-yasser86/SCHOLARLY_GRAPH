"""Live Semantic Scholar tests: search, citations, references.

Skipped automatically when the public tier rate-limits (HTTP 429) since no
API key is configured in this environment.
"""

from __future__ import annotations

import httpx
import pytest

from scholarly_graph.ingestion.discovery import SemanticScholarClient

pytestmark = pytest.mark.live


def _guard_rate_limit(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 429:
            pytest.skip("Semantic Scholar public tier rate-limited (429)")
        raise


def test_search_returns_results(capsys):
    results = _guard_rate_limit(
        SemanticScholarClient().search, "Great Gatsby Curve inequality mobility"
    )
    assert len(results) >= 1
    print(f"SemanticScholar hits: {len(results)}")
    assert "SemanticScholar" in capsys.readouterr().out


def test_citations_returns_list():
    results = _guard_rate_limit(
        SemanticScholarClient().search, "Great Gatsby Curve"
    )
    paper_id = results[0].get("paperId") or results[0].get("title", "")
    citations = _guard_rate_limit(SemanticScholarClient().citations, paper_id)
    assert isinstance(citations, list)


def test_references_returns_list():
    results = _guard_rate_limit(
        SemanticScholarClient().search, "Great Gatsby Curve"
    )
    paper_id = results[0].get("paperId") or results[0].get("title", "")
    references = _guard_rate_limit(SemanticScholarClient().references, paper_id)
    assert isinstance(references, list)
