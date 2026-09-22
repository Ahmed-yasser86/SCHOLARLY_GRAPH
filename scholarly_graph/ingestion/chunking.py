"""Sentence segmentation and token-aware section chunking.

Uses NLTK's Punkt sentence tokenizer and tiktoken token counts instead of
regex splitting and character approximations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from scholarly_graph.ingestion.sections import detect_sections

MIN_QUALITY_SCORE = 0.4


def quality_score(text: str) -> float:
    if not text:
        return 0.0
    alpha = sum(1 for ch in text if ch.isalpha())
    return round(alpha / max(len(text), 1), 4)


def _sentences(text: str) -> list:
    from nltk.tokenize import sent_tokenize

    return [s.strip() for s in sent_tokenize(text) if s.strip()]


def _token_len(text: str) -> int:
    import tiktoken

    return len(tiktoken.get_encoding("cl100k_base").encode(text))


@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    section: str
    text: str
    metadata: dict = field(default_factory=dict)


def chunk_section_text(
    paper_id: str,
    section: str,
    text: str,
    target_tokens: int = 500,
    overlap_sentences: int = 1,
) -> list:
    sentences = _sentences(text)
    chunks: list = []
    current: list = []
    current_tokens = 0
    index = 0
    for sentence in sentences:
        tokens = _token_len(sentence)
        if current and current_tokens + tokens > target_tokens:
            chunks.append(
                Chunk(
                    chunk_id=f"{paper_id}::{section}::{index}",
                    paper_id=paper_id,
                    section=section,
                    text=" ".join(current),
                )
            )
            index += 1
            current = current[-overlap_sentences:] if overlap_sentences else []
            current_tokens = sum(_token_len(s) for s in current)
        current.append(sentence)
        current_tokens += tokens
    if current:
        chunks.append(
            Chunk(
                chunk_id=f"{paper_id}::{section}::{index}",
                paper_id=paper_id,
                section=section,
                text=" ".join(current),
            )
        )
    return chunks


def chunk_paper(paper_id: str, full_text: str, metadata: dict | None = None) -> list:
    chunks: list = []
    for section, body in detect_sections(full_text):
        for chunk in chunk_section_text(paper_id, section, body):
            chunk.metadata.update(metadata or {})
            chunks.append(chunk)
    return chunks
