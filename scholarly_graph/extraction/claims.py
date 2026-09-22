"""LLM claim extraction with the precise plan schema over a generic client."""

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
        if hasattr(client, "extract_json_array"):
            parsed = client.extract_json_array(
                EXTRACTION_SYSTEM_PROMPT,
                f"SECTION: {section}\nPASSAGE:\n{chunk_text}",
            )
        else:
            from scholarly_graph.extraction.json_util import parse_json_array

            raw = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1500,
                system=EXTRACTION_SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": f"SECTION: {section}\nPASSAGE:\n{chunk_text}",
                    }
                ],
            )
            text = "".join(
                block.text for block in raw.content if hasattr(block, "text")
            )
            parsed = parse_json_array(text)
        logger.info("claim extraction parsed %d candidate claims", len(parsed))
    except Exception as exc:
        logger.warning("claim extraction failed: %s", exc)
        return []
    claims: list = []
    for item in parsed:
        try:
            if not isinstance(item, dict):
                continue
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
