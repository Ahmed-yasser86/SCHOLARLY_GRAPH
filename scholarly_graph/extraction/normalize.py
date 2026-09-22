"""Embedding-backed concept normalization and country detection.

Concept matching uses the local BAAI/bge-small-en-v1.5 sentence-transformer
model with cosine similarity at the 0.75 threshold from the plan, with a
RapidFuzz token-set fallback when the model cannot be loaded. Country
detection resolves ISO names, official names, and common aliases through
pycountry, replacing the hard-coded country list.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

NORMALIZATION_THRESHOLD = 0.75

_COMMON_COUNTRY_ALIASES: dict = {
    "united states": "United States",
    "usa": "United States",
    "u.s.": "United States",
    "u.s.a.": "United States",
    "america": "United States",
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "england": "United Kingdom",
    "britain": "United Kingdom",
}


def _country_index() -> dict:
    import pycountry

    index: dict = {alias: canonical for alias, canonical in _COMMON_COUNTRY_ALIASES.items()}
    for country in pycountry.countries:
        for name in {country.name, getattr(country, "official_name", country.name)}:
            index.setdefault(name.lower(), country.name)
        for code in (
            getattr(country, "alpha_2", ""),
            getattr(country, "alpha_3", ""),
        ):
            if code:
                index.setdefault(code.lower(), country.name)
    return index


class ConceptNormalizer:
    def __init__(
        self,
        threshold: float = NORMALIZATION_THRESHOLD,
        model: object | None = None,
    ) -> None:
        from scholarly_graph.domain.concepts import CANONICAL_CONCEPTS

        self.threshold = threshold
        self.concepts = [c for c in CANONICAL_CONCEPTS if c != "other"]
        self._model = model
        self._concept_vectors: list | None = None

    def _vectors(self) -> list | None:
        if self._concept_vectors is not None:
            return self._concept_vectors
        try:
            from scholarly_graph.extraction.embeddings import EmbeddingModel

            model = self._model or EmbeddingModel.instance()
            self._concept_vectors = model.encode(self.concepts)
        except Exception as exc:
            logger.warning("embedding model unavailable, using fuzzy fallback: %s", exc)
            self._concept_vectors = None
        return self._concept_vectors

    def normalize(self, term: str) -> str:
        cleaned = term.strip().lower()
        if not cleaned:
            return "other"
        for concept in self.concepts:
            if cleaned == concept:
                logger.info("normalize %r -> %r (1.0 exact)", term, concept)
                return concept
        vectors = self._vectors()
        if vectors is not None:
            try:
                from scholarly_graph.extraction.embeddings import EmbeddingModel

                model = self._model or EmbeddingModel.instance()
                (query_vector,) = model.encode([term])
                best, best_score = "other", 0.0
                for concept, vector in zip(self.concepts, vectors):
                    score = sum(a * b for a, b in zip(query_vector, vector))
                    if score > best_score:
                        best, best_score = concept, float(score)
                if best_score >= self.threshold:
                    logger.info("normalize %r -> %r (%.3f)", term, best, best_score)
                    return best
                logger.info("normalize %r -> other (best %.3f)", term, best_score)
                return "other"
            except Exception as exc:
                logger.warning("embedding similarity failed, fuzzy fallback: %s", exc)
        from rapidfuzz import fuzz

        best, best_score = "other", 0.0
        for concept in self.concepts:
            score = fuzz.token_set_ratio(cleaned, concept) / 100.0
            if score > best_score:
                best, best_score = concept, score
        if best_score >= self.threshold:
            logger.info("normalize %r -> %r (fuzzy %.3f)", term, best, best_score)
            return best
        return "other"


def detect_countries(text: str) -> tuple:
    lowered = text.lower()
    found: list = []
    for alias, canonical in _country_index().items():
        if len(alias) <= 3:
            continue
        if alias in lowered and canonical not in found:
            found.append(canonical)
    return tuple(found)
