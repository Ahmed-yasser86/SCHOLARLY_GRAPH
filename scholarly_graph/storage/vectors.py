"""Vector retrieval: BAAI/bge-small-en-v1.5 embeddings with Qdrant payload filtering.

bge-small-en-v1.5 is the plan-mandated local model (384-dim, no API cost);
SPECTER-style scientific embeddings were evaluated and rejected since the
plan fixes this model and the corpus/questions are general academic English.
Qdrant is the plan-mandated store with payload filtering on country, year
range, study design, and section; UUIDv5 point IDs keep chunk IDs stable.
"""

from __future__ import annotations

import hashlib
import logging
import uuid

logger = logging.getLogger(__name__)


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def legacy_point_id(chunk_id: str) -> int:
    digest = hashlib.sha256(chunk_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def embed_texts(texts: list) -> list:
    from scholarly_graph.extraction.embeddings import EmbeddingModel

    return EmbeddingModel.instance().encode(texts)


def embed_text(text: str) -> list:
    return embed_texts([text])[0]


local_embed = embed_text


class QdrantVectorStore:
    """Qdrant adapter driven purely by the configured service URL + API key."""

    def __init__(
        self, url: str, api_key: str = "", collection: str = "scholarly_graph"
    ) -> None:
        from qdrant_client import QdrantClient

        if not url:
            raise ValueError("Qdrant connection requires QDRANT_URL")
        self.collection = collection
        self.client = QdrantClient(url=url, api_key=api_key or None)

    def health(self) -> dict:
        collections = self.client.get_collections()
        logger.info("qdrant health ok, %d collections", len(collections.collections))
        return {"status": "ok", "collections": len(collections.collections)}

    def collection_exists(self, name: str) -> bool:
        exists = bool(self.client.collection_exists(name or self.collection))
        logger.info("collection_exists %s -> %s", name, exists)
        return exists

    def create_collection(self, name: str, dimension: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        target = name or self.collection
        if self.client.collection_exists(target):
            logger.info("collection %s already exists", target)
            return None
        self.client.create_collection(
            collection_name=target,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )
        logger.info("created collection %s dim=%d", target, dimension)
        return None

    def upsert(self, chunk_id: str, vector: list, payload: dict) -> None:
        from qdrant_client.models import PointStruct

        target = payload.get("collection", self.collection)
        self.client.upsert(
            collection_name=target,
            points=[
                PointStruct(id=point_id(chunk_id), vector=vector, payload=payload)
            ],
        )
        logger.info("upserted chunk %s into %s", chunk_id, target)

    def search(
        self, query_vector: list, limit: int = 25, filters: dict | None = None
    ) -> list:
        from qdrant_client.models import FieldCondition, Filter, MatchValue, Range

        must = []
        for key, value in (filters or {}).items():
            if value is None:
                continue
            if (
                isinstance(value, (list, tuple))
                and len(value) == 2
                and all(isinstance(v, (int, float)) for v in value)
                and key in {"year", "publication_year"}
            ):
                must.append(FieldCondition(key=key, range=Range(gte=value[0], lte=value[1])))
            elif isinstance(value, (list, tuple)):
                for item in value:
                    must.append(FieldCondition(key=key, match=MatchValue(value=item)))
            else:
                must.append(FieldCondition(key=key, match=MatchValue(value=value)))
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=limit,
            query_filter=Filter(must=must) if must else None,
            with_payload=True,
        )
        logger.info(
            "vector search limit=%d filters=%s hits=%d",
            limit,
            filters,
            len(results.points),
        )
        return [
            {"score": float(point.score), "payload": point.payload or {}}
            for point in results.points
        ]


class LocalVectorStore:
    def __init__(self, path: str = "data/vector_chunks.json") -> None:
        import json
        import pathlib

        self.path = pathlib.Path(path)
        self.records: list = []
        if self.path.exists():
            self.records = json.loads(self.path.read_text(encoding="utf-8"))

    def collection_exists(self, name: str) -> bool:
        return any(r.get("collection") == name for r in self.records)

    def create_collection(self, name: str, dimension: int) -> None:
        return None

    def upsert(self, chunk_id: str, vector: list, payload: dict) -> None:
        import json

        self.records = [r for r in self.records if r.get("chunk_id") != chunk_id]
        self.records.append(
            {"chunk_id": chunk_id, "vector": vector, "payload": payload}
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.records), encoding="utf-8")

    @staticmethod
    def _matches(payload: dict, filters: dict) -> bool:
        for key, value in filters.items():
            if value is None:
                continue
            actual = payload.get(key)
            if (
                isinstance(value, (list, tuple))
                and len(value) == 2
                and all(isinstance(v, (int, float)) for v in value)
                and key in {"year", "publication_year"}
                and isinstance(actual, (int, float))
            ):
                if not (value[0] <= actual <= value[1]):
                    return False
            elif isinstance(value, (list, tuple)):
                if actual not in value:
                    return False
            elif actual != value:
                return False
        return True

    def search(
        self, query_vector: list, limit: int = 25, filters: dict | None = None
    ) -> list:
        scored = []
        for record in self.records:
            payload = record.get("payload", {})
            if not self._matches(payload, filters or {}):
                continue
            score = sum(a * b for a, b in zip(query_vector, record.get("vector", [])))
            scored.append({"score": score, "payload": payload})
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:limit]


QdrantVectorStoreAdapter = QdrantVectorStore
