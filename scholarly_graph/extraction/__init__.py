"""Claim extraction package."""
from scholarly_graph.extraction.claims import extract_claims_from_chunk
from scholarly_graph.extraction.embeddings import EmbeddingModel
from scholarly_graph.extraction.json_util import parse_json_array
from scholarly_graph.extraction.normalize import (
    ConceptNormalizer,
    detect_countries,
)

__all__ = [
    "ConceptNormalizer",
    "EmbeddingModel",
    "detect_countries",
    "extract_claims_from_chunk",
    "parse_json_array",
]
