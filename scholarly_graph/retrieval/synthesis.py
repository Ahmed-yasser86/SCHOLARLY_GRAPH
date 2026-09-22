"""Grounded answer synthesis through the Claude API."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

STANDARD_SYNTHESIS_PROMPT = """You are a research assistant synthesising social-science evidence.
Write a concise research summary grounded ONLY in the passages below.
Cite each factual statement with [paper title, year] from the passage metadata.
If the passages do not answer the question, say so explicitly.
Do not add information not present in the passages."""

NETWORK_SYNTHESIS_PROMPT = """You are a research assistant synthesising structured claim evidence.
Write a structured synthesis that names each mechanism, notes geographic
variation by country, and flags contested claims explicitly.
Ground EVERY statement in the provided claims and evidence spans.
Do not add information not present in the evidence package."""


def _call_claude(client: object, system: str, user: str) -> str:
    raw = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(block.text for block in raw.content if hasattr(block, "text"))
    logger.info("synthesis produced %d chars", len(text))
    return text


def synthesize_standard(question: str, hits: list, client: object) -> str:
    passages = "\n\n".join(
        f"[{hit['payload'].get('paper', 'unknown')}, "
        f"{hit['payload'].get('year', '')}]\n{hit['payload'].get('text', '')[:1500]}"
        for hit in hits[:25]
    )
    return _call_claude(
        client,
        STANDARD_SYNTHESIS_PROMPT,
        f"QUESTION:\n{question}\n\nPASSAGES:\n{passages}",
    )


def synthesize_network_aware(
    question: str, claims: list, contradictions: list, client: object
) -> str:
    lines = [
        f"- {c.subject} {c.relationship} {c.obj} "
        f"[confidence={c.confidence}, countries={list(c.country_scope)}]\n"
        f"  Evidence: {c.evidence.text[:600]}"
        for c in claims[:40]
    ]
    contested = "\n".join(
        f"- {p.subject}: {p.first.plain_language()} VS {p.second.plain_language()}"
        for p in contradictions
    )
    return _call_claude(
        client,
        NETWORK_SYNTHESIS_PROMPT,
        f"QUESTION:\n{question}\n\nCLAIMS:\n{chr(10).join(lines)}\n\n"
        f"CONTESTED:\n{contested or 'none'}",
    )
