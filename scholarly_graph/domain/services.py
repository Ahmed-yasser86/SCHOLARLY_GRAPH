"""Domain services: query decomposition, evidence fusion, contradiction detection."""

from __future__ import annotations

from dataclasses import dataclass, field

from scholarly_graph.domain.concepts import CANONICAL_CONCEPTS
from scholarly_graph.domain.entities import Claim

CONFLICTING_RELATIONSHIPS = {
    frozenset({"increases", "decreases"}),
    frozenset({"increases", "no_significant_effect"}),
    frozenset({"decreases", "no_significant_effect"}),
    frozenset({"causes", "no_significant_effect"}),
    frozenset({"associated_with", "contradicts"}),
}

SECTION_WEIGHTS = {
    "results": 1.0,
    "methods": 0.7,
    "abstract": 0.5,
    "discussion": 0.4,
    "introduction": 0.3,
    "conclusion": 0.4,
    "unknown": 0.25,
}

CONFIDENCE_WEIGHTS = {"high": 1.0, "medium": 0.6, "low": 0.3}


@dataclass(frozen=True)
class StructuredQuery:
    raw_question: str
    subject_concept: str = ""
    object_concept: str = ""
    question_type: str = "mechanism"
    countries: tuple = ()
    year_from: int | None = None
    year_to: int | None = None


@dataclass
class FusedEvidence:
    claims: list = field(default_factory=list)
    scores: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ContradictionPair:
    first: object
    second: object
    subject: str = ""
    obj: str = ""


def decompose_query(question: str) -> StructuredQuery:
    lowered = question.lower()
    subject = ""
    obj = ""
    for concept in CANONICAL_CONCEPTS:
        if concept.lower() in lowered:
            if not subject:
                subject = concept
            elif concept != subject and not obj:
                obj = concept
    if "where" in lowered or "countr" in lowered or "compar" in lowered:
        qtype = "geographic"
    elif "differ between" in lowered or "differ" in lowered or "between" in lowered:
        qtype = "geographic"
    elif "over time" in lowered or "evolv" in lowered or "trend" in lowered:
        qtype = "temporal"
    elif "disagree" in lowered or "contest" in lowered or "contradic" in lowered:
        qtype = "contested"
    else:
        qtype = "mechanism"
    countries: list = []
    for token in (
        "united states",
        "sweden",
        "denmark",
        "brazil",
        "egypt",
        "germany",
        "france",
        "norway",
        "finland",
    ):
        if token in lowered:
            countries.append(token)
    return StructuredQuery(
        raw_question=question,
        subject_concept=subject,
        object_concept=obj,
        question_type=qtype,
        countries=tuple(countries),
    )


def claim_rank_score(claim: Claim) -> float:
    section_weight = SECTION_WEIGHTS.get(claim.evidence.section, 0.25)
    confidence_weight = CONFIDENCE_WEIGHTS.get(claim.confidence, 0.6)
    return round(section_weight * 0.6 + confidence_weight * 0.4, 4)


def fuse_evidence(claims: list) -> FusedEvidence:
    seen: dict = {}
    for claim in claims:
        key = (
            claim.subject.strip().lower(),
            claim.relationship,
            claim.obj.strip().lower(),
            claim.evidence.text.strip()[:160],
        )
        existing = seen.get(key)
        if existing is None or claim_rank_score(claim) > claim_rank_score(existing):
            seen[key] = claim
    ranked = sorted(seen.values(), key=claim_rank_score, reverse=True)
    return FusedEvidence(
        claims=ranked, scores={id(c): claim_rank_score(c) for c in ranked}
    )


def _relationships_conflict(first: str, second: str) -> bool:
    if first == second:
        return False
    return frozenset({first, second}) in CONFLICTING_RELATIONSHIPS


def detect_contradictions(claims: list) -> list:
    pairs: list = []
    for i in range(len(claims)):
        for j in range(i + 1, len(claims)):
            first = claims[i]
            second = claims[j]
            if (
                first.subject.strip().lower() == second.subject.strip().lower()
                and first.obj.strip().lower() == second.obj.strip().lower()
                and _relationships_conflict(first.relationship, second.relationship)
            ):
                pairs.append(
                    ContradictionPair(
                        first=first,
                        second=second,
                        subject=first.subject,
                        obj=first.obj,
                    )
                )
    return pairs
