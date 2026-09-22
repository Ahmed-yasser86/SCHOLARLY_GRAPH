"""Local vector retrieval: hashing embedder, file-backed chunks, Qdrant adapter."""

from __future__ import annotations

import hashlib
import json
import math
import pathlib


def local_embed(text: str, dimension: int = 384) -> list:
    vector = [0.0] * dimension
    for token in text.lower().split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for i in range(4):
            index = int.from_bytes(digest[i * 2 : i * 2 + 2], "big") % dimension
            vector[index] += 1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def cosine_similarity(first: list, second: list) -> float:
    return sum(a * b for a, b in zip(first, second))


class LocalVectorStore:
    """Deterministic file-backed vector store used when Qdrant is unavailable."""

    def __init__(self, path: str = "data/vector_chunks.json") -> None:
        self.path = pathlib.Path(path)
        self.records: list = []
        if self.path.exists():
            self.records = json.loads(self.path.read_text(encoding="utf-8"))

    def collection_exists(self, name: str) -> bool:
        return any(r.get("collection") == name for r in self.records)

    def create_collection(self, name: str, dimension: int) -> None:
        return None

    def upsert(self, chunk_id: str, vector: list, payload: dict) -> None:
        self.records = [r for r in self.records if r.get("chunk_id") != chunk_id]
        self.records.append(
            {"chunk_id": chunk_id, "vector": vector, "payload": payload}
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.records), encoding="utf-8")

    def search(
        self, query_vector: list, limit: int = 25, filters: dict | None = None
    ) -> list:
        filters = filters or {}
        scored = []
        for record in self.records:
            payload = record.get("payload", {})
            if any(payload.get(k) != v for k, v in filters.items() if v is not None):
                continue
            score = cosine_similarity(query_vector, record.get("vector", []))
            scored.append({"score": score, "payload": payload})
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:limit]


class QdrantVectorStoreAdapter:
    """Qdrant adapter implementing the VectorStore port."""

    def __init__(self, url: str, api_key: str = "", collection: str = "scholarly_graph") -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self._Distance = Distance
        self._VectorParams = VectorParams
        self.collection = collection
        self.client = QdrantClient(url=url, api_key=api_key or None)

    def collection_exists(self, name: str) -> bool:
        return bool(self.client.collection_exists(name or self.collection))

    def create_collection(self, name: str, dimension: int) -> None:
        target = name or self.collection
        if self.client.collection_exists(target):
            return None
        self.client.create_collection(
            collection_name=target,
            vectors_config=self._VectorParams(
                size=dimension, distance=self._Distance.COSINE
            ),
        )
        return None

    def _point_id(self, chunk_id: str) -> str:
        import uuid

        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    def upsert(self, chunk_id: str, vector: list, payload: dict) -> None:
        from qdrant_client.models import PointStruct

        target = payload.get("collection", self.collection)
        self.client.upsert(
            collection_name=target,
            points=[
                PointStruct(id=self._point_id(chunk_id), vector=vector, payload=payload)
            ],
        )

    def search(
        self, query_vector: list, limit: int = 25, filters: dict | None = None
    ) -> list:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        must = [
            FieldCondition(key=key, match=MatchValue(value=value))
            for key, value in (filters or {}).items()
            if value is not None
        ]
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=limit,
            query_filter=Filter(must=must) if must else None,
            with_payload=True,
        )
        return [
            {"score": float(point.score), "payload": point.payload or {}}
            for point in results.points
        ]
