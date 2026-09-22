"""PDF text extraction: PyMuPDF primary, pdfplumber fallback, quality gate."""

from __future__ import annotations

import logging

from scholarly_graph.ingestion.chunking import MIN_QUALITY_SCORE, quality_score
from scholarly_graph.ingestion.sections import detect_sections

logger = logging.getLogger(__name__)


class PdfTextExtractor:
    def extract(self, pdf_path: str) -> dict:
        import fitz

        with fitz.open(pdf_path) as doc:
            pages = [page.get_text() for page in doc]
            text = "\n".join(pages)
        score = quality_score(text)
        method = "pymupdf"
        logger.info(
            "extracted %d pages, %d chars, quality=%.3f via %s",
            len(pages),
            len(text),
            score,
            method,
        )
        if score < MIN_QUALITY_SCORE:
            try:
                import pdfplumber

                with pdfplumber.open(pdf_path) as pdf:
                    fallback_text = "\n".join(
                        page.extract_text() or "" for page in pdf.pages
                    )
                fallback_score = quality_score(fallback_text)
                logger.info("pdfplumber fallback quality=%.3f", fallback_score)
                if fallback_score > score:
                    text, score = fallback_text, fallback_score
                    method = "pdfplumber"
            except Exception as exc:
                logger.warning("pdfplumber fallback failed: %s", exc)
        return {
            "text": text,
            "quality": score,
            "processable": score >= MIN_QUALITY_SCORE,
            "sections": detect_sections(text),
            "extractor": method,
        }

    def quality_score(self, text: str) -> float:
        return quality_score(text)
