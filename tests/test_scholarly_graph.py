"""Scholarly Graph tests for domain, ingestion, retrieval, and API layers."""

import json

import pytest

from scholarly_graph.domain.entities import Claim, DocumentId, EvidenceSpan, Paper


def _claim(**overrides):
    params = {
        "subject": "educational inequality",
        "relationship": "mediates",
        "obj": "intergenerational mobility",
        "evidence": EvidenceSpan(text="Schools mediate outcomes.", section="results"),
    }
    params.update(overrides)
    return Claim(**params)


def test_document_id_rejects_bad_format():
    with pytest.raises(ValueError):
        DocumentId("no-separator")
    with pytest.raises(ValueError):
        DocumentId("source:")
    assert str(DocumentId("openalex:W1")) == "openalex:W1"


def test_paper_validation_matrix():
    with pytest.raises(ValueError):
        Paper(document_id=DocumentId("openalex:W1"), title=" ", year=2020)
    with pytest.raises(ValueError):
        Paper(document_id=DocumentId("openalex:W1"), title="T", year=1899)
    with pytest.raises(ValueError):
        Paper(
            document_id=DocumentId("openalex:W1"),
            title="T",
            year=2020,
            study_design="anecdote",
        )
    paper = Paper(
        document_id=DocumentId("openalex:W1"),
        title="T",
        year=2020,
        data_countries=("United States", "Sweden"),
    )
    assert paper.open_access is False
    assert paper.data_countries == ("United States", "Sweden")
    for design in (
        "rct",
        "quasi_experimental",
        "observational",
        "meta_analysis",
        "simulation",
        "theoretical",
    ):
        assert (
            Paper(
                document_id=DocumentId("openalex:W1"), title="T", study_design=design
            ).study_design
            == design
        )


def test_evidence_span_validation_matrix():
    with pytest.raises(ValueError):
        EvidenceSpan(text=" ", section="results")
    with pytest.raises(ValueError):
        EvidenceSpan(text="word " * 501, section="results")
    with pytest.raises(ValueError):
        EvidenceSpan(text="Some text.", section="appendix")
    span = EvidenceSpan(text="Some text.", section="results")
    with pytest.raises(Exception):
        span.text = "mutated"


def test_claim_validation_matrix():
    with pytest.raises(ValueError):
        _claim(subject=" ")
    with pytest.raises(ValueError):
        _claim(relationship="sometimes_maybe")
    with pytest.raises(ValueError):
        _claim(confidence="certain")
    assert _claim().confidence == "medium"
    assert _claim(conditions=None).conditions is None
    assert _claim(country_scope=()).country_scope == ()


def test_vocabulary_loads_from_yaml():
    from scholarly_graph.domain.concepts import CANONICAL_CONCEPTS, CONCEPT_CATEGORIES

    assert len(CANONICAL_CONCEPTS) >= 50
    assert "other" in CANONICAL_CONCEPTS
    assert set(CONCEPT_CATEGORIES) >= {"inequality", "mobility", "mechanisms"}


def test_concept_normalizer_embedding_and_fallback():
    from scholarly_graph.extraction.normalize import ConceptNormalizer

    normalizer = ConceptNormalizer()
    assert normalizer.normalize("income inequality") in {
        "economic inequality",
        "income inequality",
    }
    assert normalizer.normalize("intergenerational mobility") == (
        "intergenerational mobility"
    )
    assert normalizer.normalize("quantum entanglement") == "other"
    assert normalizer.normalize("income inequality") == normalizer.normalize(
        "income inequality"
    )

    class BrokenModel:
        def encode(self, texts):
            raise RuntimeError("no model")

    fallback = ConceptNormalizer(model=BrokenModel())
    assert fallback.normalize("income inequality") in {
        "economic inequality",
        "income inequality",
    }
    assert fallback.normalize("quantum entanglement") == "other"


def test_embedding_model_dimension():
    from scholarly_graph.extraction.embeddings import EmbeddingModel

    assert EmbeddingModel.instance().dimension == 384


def test_section_detector_variants():
    from scholarly_graph.ingestion.sections import detect_sections

    text = (
        "ABSTRACT\nSome summary.\n2. Data and Methods\nWe used tax data.\n"
        "3. Empirical Results\nEffects found.\nConcluding Remarks\nDone."
    )
    sections = dict(detect_sections(text))
    assert sections["abstract"].strip()
    assert "methods" in sections
    assert "results" in sections
    assert "conclusion" in sections
    assert dict(detect_sections("Just body text with no headings.")) == {
        "unknown": "Just body text with no headings."
    }


def test_section_detector_rejects_body_text():
    from scholarly_graph.ingestion.sections import detect_sections

    body = (
        "This paper studies methods for mobility.\n"
        "Our results are preliminary and the discussion continues below."
    )
    assert list(dict(detect_sections(body))) == ["unknown"]


def test_chunking_uses_sentences_and_token_budget():
    from scholarly_graph.ingestion.chunking import chunk_paper, chunk_section_text

    text = "Abstract\n" + ("Sentence one. Sentence two. " * 200) + "\nResults\n" + (
        "Finding one. Finding two. " * 200
    )
    chunks = chunk_paper("openalex:W1", text)
    assert {chunk.section for chunk in chunks} <= {"abstract", "results"}
    tiny = chunk_section_text("p", "results", "One. Two. Three.", target_tokens=2)
    assert len(tiny) >= 2
    assert all(".." not in chunk.text for chunk in tiny)


def test_openalex_abstract_decoder():
    from scholarly_graph.ingestion.discovery import decode_openalex_abstract

    assert decode_openalex_abstract({"hello": [0], "world": [1]}) == "hello world"
    assert decode_openalex_abstract(None) == ""
    assert decode_openalex_abstract({}) == ""


def test_openalex_parse_work_shape():
    from scholarly_graph.ingestion.discovery import OpenAlexClient

    work = {
        "id": "https://openalex.org/W1",
        "title": "Great Gatsby",
        "publication_year": 2013,
        "authorships": [],
        "primary_location": {"source": {"display_name": "JEP"}},
        "doi": "https://doi.org/10.1/x",
        "open_access": {"is_oa": True, "oa_url": "https://x/y.pdf"},
        "abstract_inverted_index": {"great": [0]},
    }
    parsed = OpenAlexClient().parse_work(work)
    assert parsed["document_id"] == "openalex:W1"
    assert parsed["abstract"] == "great"
    assert parsed["doi"] == "10.1/x"


def test_snapshot_manifest_shape(tmp_path):
    from scholarly_graph.ingestion.discovery import SnapshotStore

    store = SnapshotStore(str(tmp_path))
    path = store.save_manifest(
        "snapshot-x",
        {
            "query": "q",
            "papers": [
                {
                    "document_id": "openalex:W1",
                    "title": "T",
                    "year": 2020,
                    "open_access": True,
                }
            ],
            "claim_count": 3,
        },
    )
    manifest = json.loads(open(path, encoding="utf-8").read())
    assert manifest["paper_count"] == 1
    assert manifest["open_access_count"] == 1
    assert manifest["claim_count"] == 3


def test_pdf_downloader_outcomes(tmp_path, monkeypatch):
    from scholarly_graph.ingestion.discovery import PdfDownloader

    downloader = PdfDownloader()
    assert downloader.download("", str(tmp_path / "a.pdf"))["status"] == (
        "skipped_no_pdf"
    )
    cached = tmp_path / "cached.pdf"
    cached.write_bytes(b"%PDF-1.4 data")
    assert downloader.download("https://x/y.pdf", str(cached))["status"] == "cached"

    class FakeResponse:
        headers = {"Content-Type": "text/html"}
        content = b"<html></html>"

        def raise_for_status(self):
            return None

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, headers=None):
            return FakeResponse()

    monkeypatch.setattr("httpx.Client", FakeClient)
    assert downloader.download("https://x/y", str(tmp_path / "b.pdf"))["status"] == (
        "skipped_not_pdf"
    )


def test_pdf_extractor_real_file(tmp_path):
    import fitz

    from scholarly_graph.ingestion.extractors import PdfTextExtractor

    path = str(tmp_path / "paper.pdf")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Abstract\nEducational inequality shapes mobility.")
    doc.save(path)
    doc.close()
    result = PdfTextExtractor().extract(path)
    assert result["text"].strip()
    assert result["quality"] > 0.4
    assert result["processable"] is True


def test_contradiction_detection_matrix():
    from scholarly_graph.domain.evidence import detect_contradictions

    first = _claim(relationship="increases", obj="intergenerational mobility")
    second = _claim(relationship="decreases", obj="intergenerational mobility")
    assert len(detect_contradictions([first, second])) == 1
    assert detect_contradictions([first, _claim(relationship="increases")]) == []
    assert detect_contradictions(
        [first, _claim(relationship="associated_with", obj="other outcome")]
    ) == []


def test_fusion_deduplicates_and_ranks_results_first():
    from scholarly_graph.domain.evidence import fuse_evidence

    low = _claim(
        relationship="associated_with",
        evidence=EvidenceSpan(text="Same span.", section="unknown"),
        confidence="low",
    )
    high = _claim(
        relationship="associated_with",
        evidence=EvidenceSpan(text="Same span.", section="results"),
        confidence="high",
    )
    fused = fuse_evidence([low, high])
    assert len(fused.claims) == 1
    assert fused.claims[0].confidence == "high"


def test_claim_extraction_json_repair_and_verbatim_gate():
    from scholarly_graph.extraction.json_util import parse_json_array

    assert parse_json_array('[{"a": 1,}]') == [{"a": 1}]
    assert parse_json_array("not json") == []
    assert parse_json_array('```json\n[{"a": 1}]\n```') == [{"a": 1}]
    assert parse_json_array('[{"a": 1}, {"b": 2}]')[1] == {"b": 2}

    from scholarly_graph.extraction.claims import extract_claims_from_chunk
    from scholarly_graph.extraction.normalize import ConceptNormalizer

    class FakeBlock:
        text = (
            '[{"subject": "school funding", "relationship": "increases", '
            '"object": "mobility", "conditions": null, "country_scope": [], '
            '"time_period": "", "confidence": "high", '
            '"evidence_span": "invented sentence not in chunk"}]'
        )

    class FakeResponse:
        content = [FakeBlock()]

    class FakeMessages:
        def create(self, **kwargs):
            return FakeResponse()

    class FakeClient:
        messages = FakeMessages()

    assert extract_claims_from_chunk(
        "Real chunk text.",
        "results",
        paper_id="openalex:W1",
        client=FakeClient(),
        normalizer=ConceptNormalizer(),
    ) == []


def test_generic_llm_client_drives_claim_extraction():
    from scholarly_graph.extraction.claims import extract_claims_from_chunk
    from scholarly_graph.extraction.normalize import ConceptNormalizer

    chunk = "School funding increases mobility in Texas districts."

    class GenericClient:
        def extract_json_array(self, system, user):
            assert "SECTION" in user
            return [
                {
                    "subject": "school funding",
                    "relationship": "increases",
                    "object": "mobility",
                    "conditions": None,
                    "country_scope": [],
                    "time_period": "",
                    "confidence": "high",
                    "evidence_span": chunk,
                }
            ]

    claims = extract_claims_from_chunk(
        chunk, "results", paper_id="openalex:W1", client=GenericClient()
    )
    assert len(claims) == 1
    assert claims[0].relationship == "increases"

    class LegacyClient:
        class messages:
            @staticmethod
            def create(**kwargs):
                class Block:
                    text = "not json"

                class Response:
                    content = [Block()]

                return Response()

    assert extract_claims_from_chunk(
        chunk,
        "results",
        paper_id="openalex:W1",
        client=LegacyClient(),
        normalizer=ConceptNormalizer(),
    ) == []


def test_llm_client_model_is_configurable(monkeypatch):
    import litellm

    from scholarly_graph.llm.client import LlmClient, LlmConfig

    seen = {}

    class FakeMessage:
        content = '[{"subject": "a"}]'

    class FakeChoice:
        message = FakeMessage()

    class FakeRaw:
        choices = [FakeChoice()]

    def fake_completion(**kwargs):
        seen.update(kwargs)
        return FakeRaw()

    monkeypatch.setattr(litellm, "completion", fake_completion)
    client = LlmClient(LlmConfig(model="gpt-4o-mini", api_key="k"))
    assert client.extract_json_array("sys", "user") == [{"subject": "a"}]
    assert seen["model"] == "gpt-4o-mini"


def test_country_detection_uses_iso_names():
    from scholarly_graph.extraction.normalize import detect_countries

    found = detect_countries("Evidence from Germany and the United States.")
    assert "Germany" in found
    assert "United States" in found
    assert detect_countries("No geography here.") == ()


def test_query_decomposition_links_concepts_and_countries():
    from scholarly_graph.domain.query import decompose_query

    parsed = decompose_query(
        "How does the education mechanism differ between the United States and Sweden?"
    )
    assert parsed.question_type == "geographic"
    assert "united states" in parsed.countries
    assert parsed.subject_concept
    assert decompose_query("What happened over time since 2000?").question_type == (
        "temporal"
    )


def test_file_graph_store_round_trip(tmp_path):
    from scholarly_graph.storage.graph_store import FileGraphStore, claim_node_id

    store = FileGraphStore(str(tmp_path / "graph.json"))
    paper = Paper(document_id=DocumentId("openalex:W1"), title="Great Gatsby", year=2013)
    store.save_paper(paper)
    claim = _claim(country_scope=("United States",))
    store.save_claim(paper.document_id, claim)
    store.save_claim(paper.document_id, claim)
    assert len(store.state["claims"]) == 1
    assert store.mechanisms_between("educational inequality", "intergenerational mobility")
    assert store.country_subgraph("united states")
    assert claim_node_id("p", "s", "r", "o", "evidence")


def test_local_vector_store_filters_and_ranking(tmp_path):
    from scholarly_graph.storage.vectors import LocalVectorStore, embed_text

    store = LocalVectorStore(str(tmp_path / "vectors.json"))
    store.upsert("a", embed_text("educational inequality mobility"), {"paper": "P1", "year": 2020})
    store.upsert("b", embed_text("quantum baking recipes"), {"paper": "P2", "year": 1999})
    hits = store.search(embed_text("educational inequality mobility"), limit=2)
    assert hits[0]["payload"]["paper"] == "P1"
    assert store.search(
        embed_text("mobility"), limit=2, filters={"year": (2019, 2021)}
    )[0]["payload"]["paper"] == "P1"


def test_evaluation_suite_shape_and_runner():
    from scholarly_graph.evaluation.questions import (
        EVALUATION_QUESTIONS,
        geographic_specificity,
        mechanism_coverage,
        run_evaluation,
    )

    assert len(EVALUATION_QUESTIONS) == 50
    assert {q["type"] for q in EVALUATION_QUESTIONS} == {
        "mechanism",
        "geographic",
        "temporal",
        "contested",
    }
    assert all(
        {"expected_mechanisms", "expected_countries", "expect_contradiction"}
        <= set(q)
        for q in EVALUATION_QUESTIONS
    )
    assert mechanism_coverage(["educational inequality"], ["educational inequality"]) == 1.0
    assert geographic_specificity(["united states"], ["united states", "sweden"]) == 0.5
    result = run_evaluation(
        [{"mechanisms": [], "countries": [], "contradiction_found": False}],
        [{"mechanisms": [], "countries": [], "contradiction_found": False}],
    )
    assert result["count"] == 1


def test_api_routes_and_ingest_lifecycle(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from scholarly_graph.api.app import Settings, create_app

    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    cfg = Settings(
        _env_file=None,
        vector_path=str(tmp_path / "vectors.json"),
        graph_path=str(tmp_path / "graph.json"),
        snapshot_dir=str(tmp_path / "snapshots"),
    )
    client = TestClient(create_app(cfg))
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/graph/concepts").status_code == 200
    assert client.get("/evaluation/questions").status_code == 200
    assert client.get("/corpus/status").status_code == 200
    response = client.post(
        "/search/standard", json={"question": "How does schooling shape mobility?"}
    )
    assert response.status_code == 200
    assert response.json()["trace"]
    assert client.post("/evaluation/run", json={}).status_code == 200


def test_circuit_breaker_opens_and_resets():
    import time

    from scholarly_graph.ingestion.discovery import (
        _API_FAILURES,
        _circuit_allows,
        _record_failure,
    )

    _API_FAILURES.pop("probe-api", None)
    for _ in range(5):
        _record_failure("probe-api")
    assert _circuit_allows("probe-api") is False
    _API_FAILURES["probe-api"]["opened_at"] = time.monotonic() - 61
    assert _circuit_allows("probe-api") is True
