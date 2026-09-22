"""Section-aware chunking, section detection, and PDF text extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

SECTION_PATTERNS: dict = {
    "abstract": (r"^\s*abstract\s*$",),
    "introduction": (r"^\s*(1[\.\)\s])?introduction\s*$",),
    "methods": (
        r"^\s*methods?\s*$",
        r"^\s*methodology\s*$",
        r"^\s*data and methods\s*$",
        r"^\s*empirical strategy\s*$",
        r"^\s*research design\s*$",
    ),
    "results": (
        r"^\s*results?\s*$",
        r"^\s*findings?\s*$",
        r"^\s*empirical results?\s*$",
    ),
    "discussion": (r"^\s*discussion\s*$",),
    "conclusion": (r"^\s*conclusions?\s*$",),
}

MIN_QUALITY_SCORE = 0.4


def detect_sections(text: str) -> list:
    """Split *text* into ``(section, text)`` pairs using heading heuristics."""
    lines = text.splitlines()
    sections: list = []
    current = "unknown"
    buffer: list = []
    compiled = {
        name: [re.compile(p, re.IGNORECASE) for p in patterns]
        for name, patterns in SECTION_PATTERNS.items()
    }
    for line in lines:
        matched = None
        for name, patterns in compiled.items():
            if any(p.match(line.strip()) for p in patterns):
                matched = name
                break
        if matched is not None:
            if buffer:
                sections.append((current, "\n".join(buffer)))
            current = matched
            buffer = []
            continue
        buffer.append(line)
    if buffer or not sections:
        sections.append((current, "\n".join(buffer)))
    return [(name, body) for name, body in sections if body.strip()]


def quality_score(text: str) -> float:
    """Ratio of alphabetic characters to total characters."""
    if not text:
        return 0.0
    alpha = sum(1 for ch in text if ch.isalpha())
    return round(alpha / max(len(text), 1), 4)


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
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    chunks: list = []
    approx_chars = target_tokens * 4
    current: list = []
    current_len = 0
    index = 0
    for sentence in sentences:
        current.append(sentence)
        current_len += len(sentence)
        if current_len >= approx_chars:
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
            current_len = sum(len(s) for s in current)
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


class PdfTextExtractor:
    """PyMuPDF primary extractor with a pdfplumber fallback."""

    def extract(self, pdf_path: str) -> dict:
        import fitz

        with fitz.open(pdf_path) as doc:
            text = "\n".join(page.get_text() for page in doc)
        score = quality_score(text)
        method = "pymupdf"
        if score < MIN_QUALITY_SCORE:
            try:
                import pdfplumber

                fallback_parts: list = []
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        fallback_parts.append(page.extract_text() or "")
                fallback_text = "\n".join(fallback_parts)
                fallback_score = quality_score(fallback_text)
                if fallback_score > score:
                    text, score = fallback_text, fallback_score
                    method = "pdfplumber"
            except Exception:
                pass
        return {
            "text": text,
            "quality": score,
            "processable": score >= MIN_QUALITY_SCORE,
            "sections": detect_sections(text),
            "extractor": method,
        }

    def quality_score(self, text: str) -> float:
        return quality_score(text)
