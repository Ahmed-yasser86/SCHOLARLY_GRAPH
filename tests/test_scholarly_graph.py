"""Scholarly Graph tests for domain, ingestion, retrieval, and API layers."""

from scholarly_graph.domain.entities import Claim, DocumentId, EvidenceSpan, Paper


def test_document_id_rejects_bad_format():
    try:
        DocumentId("no-separator")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_paper_rejects_empty_title():
    try:
        Paper(document_id=DocumentId("openalex:W1"), title=" ", year=2020)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_evidence_span_rejects_unknown_section():
    try:
        EvidenceSpan(text="Some text.", section="appendix")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_claim_requires_valid_relationship():
    try:
        Claim(
            subject="economic inequality",
            relationship="sometimes_maybe",
            obj="intergenerational mobility",
            evidence=EvidenceSpan(text="X relates to Y.", section="results"),
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_concept_normalizer_maps_synonyms():
    from scholarly_graph.extraction.claims import ConceptNormalizer

    normalizer = ConceptNormalizer()
    assert normalizer.normalize("income inequality") in {
        "economic inequality",
        "income inequality",
    }
    assert normalizer.normalize("quantum entanglement") == "other"
    assert normalizer.normalize("income inequality") == normalizer.normalize(
        "income inequality"
    )


def test_section_detector_finds_methods_variants():
    from scholarly_graph.ingestion.chunking import detect_sections

    text = "Abstract\nSome summary.\nData and Methods\nWe used tax data.\nResults\nEffects found."
    sections = dict(detect_sections(text))
    assert "abstract" in sections
    assert "methods" in sections
    assert "results" in sections


def test_chunking_does_not_cross_sections():
    from scholarly_graph.ingestion.chunking import chunk_paper

    text = "Abstract\n" + ("Sentence one. Sentence two. " * 200) + "\nResults\n" + (
        "Finding one. Finding two. " * 200
    )
    chunks = chunk_paper("openalex:W1", text)
    sections = {chunk.section for chunk in chunks}
    assert sections <= {"abstract", "results"}


def test_openalex_abstract_decoder():
    from scholarly_graph.ingestion.discovery import decode_openalex_abstract

    assert decode_openalex_abstract({"hello": [0], "world": [1]}) == "hello world"
    assert decode_openalex_abstract(None) == ""


def test_contradiction_detection_flags_conflicts():
    from scholarly_graph.domain.services import detect_contradictions

    first = Claim(
        subject="economic inequality",
        relationship="increases",
        obj="intergenerational mobility",
        evidence=EvidenceSpan(text="Inequality raises mobility.", section="results"),
    )
    second = Claim(
        subject="economic inequality",
        relationship="decreases",
        obj="intergenerational mobility",
        evidence=EvidenceSpan(text="Inequality lowers mobility.", section="results"),
    )
    pairs = detect_contradictions([first, second])
    assert len(pairs) == 1


def test_fusion_deduplicates_and_ranks_results_first():
    from scholarly_graph.domain.services import fuse_evidence

    low = Claim(
        subject="economic inequality",
        relationship="associated_with",
        obj="intergenerational mobility",
        evidence=EvidenceSpan(text="Same span.", section="unknown"),
        confidence="low",
    )
    high = Claim(
        subject="economic inequality",
        relationship="associated_with",
        obj="intergenerational mobility",
        evidence=EvidenceSpan(text="Same span.", section="results"),
        confidence="high",
    )
    fused = fuse_evidence([low, high])
    assert len(fused.claims) == 1
    assert fused.claims[0].confidence == "high"


def test_file_graph_store_round_trip(tmp_path):
    from scholarly_graph.storage.graph_store import FileGraphStore

    store = FileGraphStore(str(tmp_path / "graph.json"))
    paper = Paper(document_id=DocumentId("openalex:W1"), title="Great Gatsby", year=2013)
    store.save_paper(paper)
    claim = Claim(
        subject="educational inequality",
        relationship="mediates",
        obj="intergenerational mobility",
        evidence=EvidenceSpan(text="Schools mediate outcomes.", section="results"),
        country_scope=("united states",),
    )
    store.save_claim(paper.document_id, claim)
    assert store.mechanisms_between("educational inequality", "intergenerational mobility")
    assert store.country_subgraph("united states")


def test_local_vector_store_search(tmp_path):
    from scholarly_graph.storage.vectors import LocalVectorStore, local_embed

    store = LocalVectorStore(str(tmp_path / "vectors.json"))
    store.upsert("a", local_embed("educational inequality mobility"), {"paper": "P1"})
    store.upsert("b", local_embed("quantum baking recipes"), {"paper": "P2"})
    hits = store.search(local_embed("educational inequality mobility"), limit=2)
    assert hits[0]["payload"]["paper"] == "P1"


def test_api_health_and_graph_endpoints():
    from fastapi.testclient import TestClient

    from scholarly_graph.api.app import create_app

    client = TestClient(create_app())
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/graph/concepts").status_code == 200
    assert client.get("/evaluation/questions").status_code == 200


def test_evaluation_question_set_shape():
    from scholarly_graph.evaluation.questions import EVALUATION_QUESTIONS

    assert len(EVALUATION_QUESTIONS) == 50
    kinds = {q["type"] for q in EVALUATION_QUESTIONS}
    assert kinds == {"mechanism", "geographic", "temporal", "contested"}


def test_evaluation_metrics_behave():
    from scholarly_graph.evaluation.questions import (
        geographic_specificity,
        mechanism_coverage,
    )

    assert mechanism_coverage(["educational inequality"], ["educational inequality"]) == 1.0
    assert geographic_specificity(["united states"], ["united states", "sweden"]) == 0.5


def test_query_decomposition_identifies_geography():
    from scholarly_graph.domain.services import decompose_query

    parsed = decompose_query(
        "How does the education mechanism differ between the United States and Sweden?"
    )
    assert parsed.question_type == "geographic"
    assert "united states" in parsed.countries
