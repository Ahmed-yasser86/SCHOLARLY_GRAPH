"""Idempotency: re-ingesting the same papers must not duplicate storage."""

from __future__ import annotations

from scholarly_graph.domain.entities import Claim, DocumentId, EvidenceSpan, Paper
from scholarly_graph.storage.graph_store import FileGraphStore
from scholarly_graph.storage.vectors import LocalVectorStore, embed_text


def _paper() -> Paper:
    return Paper(document_id=DocumentId("openalex:W1"), title="T", year=2020)


def _claim() -> Claim:
    return Claim(
        subject="educational inequality",
        relationship="mediates",
        obj="intergenerational mobility",
        evidence=EvidenceSpan(text="Schools mediate outcomes.", section="results"),
    )


def test_graph_store_dedupes_reingestion(tmp_path):
    store = FileGraphStore(str(tmp_path / "graph.json"))
    paper = _paper()
    store.save_paper(paper)
    store.save_claim(paper.document_id, _claim())
    store.save_paper(paper)
    store.save_claim(paper.document_id, _claim())
    store.save_citation(paper.document_id, "openalex:W2")
    store.save_citation(paper.document_id, "openalex:W2")
    assert len(store.state["papers"]) == 1
    assert len(store.state["claims"]) == 1
    assert len(store.state["citations"]) == 1


def test_vector_store_upsert_is_idempotent(tmp_path):
    store = LocalVectorStore(str(tmp_path / "vectors.json"))
    vector = embed_text("educational inequality mobility")
    store.upsert("chunk-1", vector, {"paper": "P1"})
    store.upsert("chunk-1", vector, {"paper": "P1"})
    assert len(store.records) == 1
