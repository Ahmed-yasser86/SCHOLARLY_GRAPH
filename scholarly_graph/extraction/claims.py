"""LLM claim extraction with the precise plan schema and JSON repair."""

from __future__ import annotations

import logging

from scholarly_graph.extraction.normalize import ConceptNormalizer

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = """You extract structured social-science claims from academic passages.
A claim is a specific assertion about the relationship between two social science concepts (X is associated with Y, X reduces Y, X mediates the relationship between Y and Z). Descriptive statistics, methodological explanations, literature review summaries, and acknowledgments are NOT claims.
Return ONLY a valid JSON array with no preamble, no markdown fences, and no trailing commas. Each item has exactly these keys:
- subject: canonical-style concept name from the passage
- relationship: one of increases, decreases, associated_with, mediates, moderates, causes, no_significant_effect, contradicts
- object: canonical-style concept name from the passage
- conditions: qualifying conditions as stated, or null when unconditional
- country_scope: array of country names the claim applies to, possibly empty
- time_period: time period studied, or empty string when absent
- confidence: one of high, medium, low
- evidence_span: the exact verbatim sentences from the passage supporting the claim; if no verbatim passage supports the claim, omit the claim entirely.
If there are no claims, return []."""


def _parse_json_array(text: str) -> list:
    import json

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        logger.warning("claim extraction JSON parse failed, trying json_repair")
    try:
        import json_repair

        parsed = json_repair.loads(stripped)
        return parsed if isinstance(parsed, list) else []
    except Exception as exc:
        logger.warning("claim extraction JSON repair failed: %s", exc)
        return []


def _extract_with_client(client: object, chunk_text: str, section: str) -> list:
    prompt = f"SECTION: {section}\nPASSAGE:\n{chunk_text}"
    raw = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        system=EXTRACTION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in raw.content if hasattr(block, "text"))
    logger.info(
        "claim extraction input=%d chars raw_preview=%s", len(chunk_text), text[:300]
    )
    parsed = _parse_json_array(text)
    logger.info("claim extraction parsed %d candidate claims", len(parsed))
    return parsed


def extract_claims_from_chunk(
    chunk_text: str,
    section: str,
    paper_id: str = "",
    client: object | None = None,
    normalizer: ConceptNormalizer | None = None,
) -> list:
    from scholarly_graph.domain.entities import Claim, DocumentId, EvidenceSpan

    if client is None:
        return []
    normalizer = normalizer or ConceptNormalizer()
    try:
        parsed = _extract_with_client(client, chunk_text, section)
    except Exception as exc:
        logger.warning("claim extraction failed: %s", exc)
        return []
    claims: list = []
    for item in parsed:
        try:
            evidence_text = str(item.get("evidence_span", "")).strip()
            if not evidence_text or evidence_text not in chunk_text:
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
        except Exception as exc:
            logger.warning("skipping invalid claim: %s", exc)
    logger.info("extracted %d claims", len(claims))
    return claims
