"""Claim extraction and concept normalization."""

from __future__ import annotations

import json
import logging
import math

from scholarly_graph.domain.concepts import CANONICAL_CONCEPTS

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = """You extract structured social-science claims.
A claim asserts a relationship between two concepts (for example, X reduces Y).
Only extract claims grounded in the passage. The evidence_span must be a verbatim
substring of the input passage. Return ONLY a JSON array. Each item has keys:
subject, relationship, object, conditions, country_scope (array), time_period,
confidence (high|medium|low), evidence_span.
Valid relationships: increases, decreases, associated_with, mediates, moderates,
causes, no_significant_effect, contradicts.
If there are no claims, return []."""

NORMALIZATION_THRESHOLD = 0.75


def _tokens(text: str) -> set:
    return {t for t in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if t}


def _similarity(first: str, second: str) -> float:
    first_tokens, second_tokens = _tokens(first), _tokens(second)
    if not first_tokens or not second_tokens:
        return 0.0
    overlap = len(first_tokens & second_tokens)
    return overlap / math.sqrt(len(first_tokens) * len(second_tokens))


class ConceptNormalizer:
    """Embedding-free canonical-concept mapper with deterministic behavior."""

    def __init__(self, threshold: float = NORMALIZATION_THRESHOLD) -> None:
        self.threshold = threshold

    def normalize(self, term: str) -> str:
        cleaned = term.strip().lower()
        if not cleaned:
            return "other"
        best = "other"
        best_score = 0.0
        for concept in CANONICAL_CONCEPTS:
            if cleaned == concept:
                logger.info("normalize %r -> %r (1.0 exact)", term, concept)
                return concept
            score = _similarity(cleaned, concept)
            if score > best_score:
                best_score = score
                best = concept
        if best_score >= self.threshold:
            logger.info("normalize %r -> %r (%.3f)", term, best, best_score)
            return best
        logger.info("normalize %r -> other (best %.3f)", term, best_score)
        return "other"


def _extract_with_client(client: object, chunk_text: str, section: str) -> list:
    prompt = (
        f"{EXTRACTION_SYSTEM_PROMPT}\n\nSECTION: {section}\nPASSAGE:\n{chunk_text}"
    )
    raw = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        system=EXTRACTION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in raw.content if hasattr(block, "text"))
    logger.info("claim extraction raw preview: %s", text[:300])
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError:
        logger.warning("claim extraction JSON parse failed; returning []")
        return []
    return parsed if isinstance(parsed, list) else []


def extract_claims_from_chunk(
    chunk_text: str,
    section: str,
    paper_id: str = "",
    client: object | None = None,
    normalizer: ConceptNormalizer | None = None,
) -> list:
    """Extract domain Claim objects from one chunk of paper text."""
    from scholarly_graph.domain.entities import Claim, DocumentId, EvidenceSpan

    if client is None:
        return []
    normalizer = normalizer or ConceptNormalizer()
    try:
        parsed = _extract_with_client(client, chunk_text, section)
    except Exception as exc:  # noqa: BLE001 - per-paper failure must not stop pipeline
        logger.warning("claim extraction failed: %s", exc)
        return []
    claims: list = []
    for item in parsed:
        try:
            evidence_text = str(item.get("evidence_span", "")).strip()
            if evidence_text and evidence_text not in chunk_text:
                logger.warning("skipping claim with non-verbatim evidence span")
                continue
            claims.append(
                Claim(
                    subject=normalizer.normalize(str(item.get("subject", ""))),
                    relationship=str(item.get("relationship", "")),
                    obj=normalizer.normalize(str(item.get("object", ""))),
                    evidence=EvidenceSpan(text=evidence_text, section=section),
                    paper_id=DocumentId(paper_id) if paper_id else None,
                    conditions=item.get("conditions"),
                    country_scope=tuple(item.get("country_scope", []) or []),
                    time_period=str(item.get("time_period", "") or ""),
                    confidence=str(item.get("confidence", "medium") or "medium"),
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("skipping invalid claim: %s", exc)
    logger.info("extracted %d claims", len(claims))
    return claims
