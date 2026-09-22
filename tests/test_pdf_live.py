"""Live PDF download test against a real open-access PDF URL."""

from __future__ import annotations

import pytest

from scholarly_graph.ingestion.discovery import OpenAlexClient, PdfDownloader

pytestmark = pytest.mark.live

KNOWN_OA_PDF = "https://arxiv.org/pdf/1801.03819.pdf"


def test_downloads_real_open_access_pdf(tmp_path):
    target = tmp_path / "paper.pdf"
    outcome = PdfDownloader().download(KNOWN_OA_PDF, str(target))
    assert outcome["status"] in {"success", "cached"}
    assert target.exists() and target.stat().st_size > 0
    print(f"downloaded {outcome['size']} bytes")


def test_openalex_discovers_real_pdf_url():
    papers = OpenAlexClient().collect(
        "Great Gatsby Curve intergenerational mobility", target_count=5
    )
    with_pdf = [p for p in papers if p.get("pdf_url")]
    assert with_pdf, "expected at least one discovered paper with a PDF URL"
