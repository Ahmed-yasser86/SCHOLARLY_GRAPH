"""Query decomposition: embedding-backed concept linking and country detection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StructuredQuery:
    raw_question: str
    subject_concept: str = ""
    object_concept: str = ""
    question_type: str = "mechanism"
    countries: tuple = ()
    year_from: int | None = None
    year_to: int | None = None


_QUESTION_CUES: tuple = (
    (("where", "countr", "compar", "differ", "between", "versus", " vs "), "geographic"),
    (("over time", "evolv", "trend", "since", "decade", "history"), "temporal"),
    (("disagree", "contest", "contradic", "debate", "mixed evidence"), "contested"),
)


def classify_question_type(question: str) -> str:
    lowered = question.lower()
    for cues, qtype in _QUESTION_CUES:
        if any(cue in lowered for cue in cues):
            return qtype
    return "mechanism"


def link_concepts(question: str, threshold: float = 0.45, top_k: int = 4) -> list:
    from scholarly_graph.domain.concepts import CANONICAL_CONCEPTS
    from scholarly_graph.extraction.embeddings import EmbeddingModel

    concepts = [c for c in CANONICAL_CONCEPTS if c != "other"]
    lowered = question.lower()
    verbatim = [c for c in concepts if c in lowered]
    try:
        model = EmbeddingModel.instance()
        vectors = model.encode([question] + concepts)
        query_vector, concept_vectors = vectors[0], vectors[1:]
        scored = sorted(
            (
                (sum(a * b for a, b in zip(query_vector, vector)), concept)
                for concept, vector in zip(concepts, concept_vectors)
            ),
            reverse=True,
        )
        ranked = [concept for score, concept in scored if score >= threshold][:top_k]
    except Exception:
        ranked = []
    ordered = list(verbatim) + [c for c in ranked if c not in verbatim]
    return ordered[:top_k]


def decompose_query(question: str) -> StructuredQuery:
    from scholarly_graph.extraction.normalize import detect_countries

    linked = link_concepts(question)
    subject = linked[0] if len(linked) > 0 else ""
    obj = linked[1] if len(linked) > 1 else ""
    return StructuredQuery(
        raw_question=question,
        subject_concept=subject,
        object_concept=obj,
        question_type=classify_question_type(question),
        countries=tuple(c.lower() for c in detect_countries(question)),
    )
