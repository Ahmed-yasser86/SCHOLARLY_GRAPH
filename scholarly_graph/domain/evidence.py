"""Evidence fusion and contradiction detection over domain claims."""

from __future__ import annotations

from dataclasses import dataclass, field

from scholarly_graph.domain.entities import Claim

CONFLICTING_RELATIONSHIPS = {
    frozenset({"increases", "decreases"}),
    frozenset({"increases", "no_significant_effect"}),
    frozenset({"decreases", "no_significant_effect"}),
    frozenset({"causes", "no_significant_effect"}),
    frozenset({"associated_with", "contradicts"}),
    frozenset({"increases", "contradicts"}),
    frozenset({"decreases", "contradicts"}),
    frozenset({"causes", "contradicts"}),
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

STUDY_DESIGN_WEIGHTS = {
    "rct": 1.0,
    "quasi_experimental": 0.9,
    "meta_analysis": 0.85,
    "observational": 0.6,
    "simulation": 0.4,
    "theoretical": 0.3,
}

CONFIDENCE_WEIGHTS = {"high": 1.0, "medium": 0.6, "low": 0.3}


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


def claim_rank_score(claim: Claim, study_design: str = "observational") -> float:
    section_weight = SECTION_WEIGHTS.get(claim.evidence.section, 0.25)
    confidence_weight = CONFIDENCE_WEIGHTS.get(claim.confidence, 0.6)
    design_weight = STUDY_DESIGN_WEIGHTS.get(study_design, 0.6)
    return round(section_weight * 0.5 + confidence_weight * 0.3 + design_weight * 0.2, 4)


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
