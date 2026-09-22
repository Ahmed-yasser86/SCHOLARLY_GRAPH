"""Embedding service: local BAAI/bge-small-en-v1.5 via sentence-transformers."""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-small-en-v1.5"
VECTOR_DIMENSION = 384


class EmbeddingModel:
    _instance: "EmbeddingModel | None" = None
    _lock = threading.Lock()

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        logger.info("loading embedding model %s (~130MB)", model_name)
        self.model = SentenceTransformer(model_name)
        self.dimension = int(self.model.get_embedding_dimension())
        logger.info("embedding model ready, dimension=%d", self.dimension)

    @classmethod
    def instance(cls) -> "EmbeddingModel":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def encode(self, texts: list) -> list:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return [list(map(float, row)) for row in vectors]

    def encode_one(self, text: str) -> list:
        return self.encode([text])[0]
