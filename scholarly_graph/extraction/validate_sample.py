"""Phase 4 validation script: 20 real OpenAlex papers → PDF → results-section
claim extraction → structured review report in the data directory.

Requires LLM credentials in the environment (LLM_API_KEY or ANTHROPIC_API_KEY).
Without them the script records BLOCKED instead of fabricating results.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys


def _llm_client():
    from scholarly_graph.llm.client import LlmClient, LlmConfig

    api_key = os.environ.get("LLM_API_KEY", "") or os.environ.get(
        "ANTHROPIC_API_KEY", ""
    )
    if not api_key and not os.environ.get("LLM_BASE_URL", ""):
        return None
    return LlmClient(
        LlmConfig(
            model=os.environ.get("LLM_MODEL", "claude-sonnet-4-6"),
            api_key=api_key,
            base_url=os.environ.get("LLM_BASE_URL", ""),
        )
    )


def main(count: int = 20, out: str = "data/validation_report.json") -> int:
    from scholarly_graph.extraction.claims import extract_claims_from_chunk
    from scholarly_graph.ingestion.discovery import OpenAlexClient, PdfDownloader
    from scholarly_graph.ingestion.extractors import PdfTextExtractor

    client = _llm_client()
    report: dict = {"papers": [], "status": "complete", "model": ""}
    if client is None:
        report = {
            "status": "BLOCKED: LLM credentials (LLM_API_KEY) not set; "
            "no extraction attempted",
            "papers": [],
        }
        target = pathlib.Path(out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(report["status"])
        return 2

    report["model"] = client.config.model
    papers = OpenAlexClient().collect(
        "economic inequality intergenerational mobility", target_count=count
    )
    downloader = PdfDownloader()
    extractor = PdfTextExtractor()
    for paper in papers:
        entry: dict = {
            "document_id": paper.get("document_id"),
            "title": paper.get("title"),
            "year": paper.get("year"),
            "claims": [],
            "review": "pending",
        }
        if not paper.get("pdf_url"):
            entry["status"] = "skipped_no_pdf"
            report["papers"].append(entry)
            continue
        dest = f"data/pdfs/{paper['document_id'].replace(':', '_')}.pdf"
        outcome = downloader.download(paper["pdf_url"], dest)
        if outcome.get("status") not in {"success", "cached"}:
            entry["status"] = outcome.get("status")
            report["papers"].append(entry)
            continue
        extraction = extractor.extract(dest)
        results_text = "\n".join(
            body for section, body in extraction.get("sections", []) if section == "results"
        ) or extraction.get("text", "")[:4000]
        claims = extract_claims_from_chunk(
            results_text[:4000],
            "results",
            paper_id=paper["document_id"],
            client=client,
        )
        entry["claims"] = [
            {
                "subject": c.subject,
                "relationship": c.relationship,
                "object": c.obj,
                "conditions": c.conditions,
                "countries": list(c.country_scope),
                "confidence": c.confidence,
                "evidence_span": c.evidence.text,
            }
            for c in claims
        ]
        entry["status"] = "extracted"
        report["papers"].append(entry)
        print(f"{paper['document_id']}: {len(claims)} claims")

    target = pathlib.Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"validation report written to {target}")
    print(
        "Please review each claim against its evidence span as accurate / "
        "inaccurate / partially accurate and return the accuracy rate."
    )
    return 0


if __name__ == "__main__":
    import os as _os

    _root = pathlib.Path(__file__).resolve().parents[2]
    _existing = _os.environ.get("PYTHONPATH", "")
    _os.environ["PYTHONPATH"] = (
        f"{_root}{_os.pathsep}{_existing}" if _existing else str(_root)
    )
    sys.exit(main())
