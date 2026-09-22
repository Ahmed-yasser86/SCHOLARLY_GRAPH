"""Live Qdrant integration via the configured external service URL + API key.

Skips when QDRANT_URL is absent. Creates a uniquely-named collection,
verifies create-exists idempotency, upsert, filtered search, then deletes
the collection so the live service is left clean.
"""

from __future__ import annotations

import os
import uuid

import pytest

pytestmark = pytest.mark.live


def _config():
    url = os.environ.get("QDRANT_URL", "")
    if not url:
        pytest.skip("QDRANT_URL not set")
    return url, os.environ.get("QDRANT_API_KEY", "")


def test_external_qdrant_roundtrip():
    from scholarly_graph.storage.vectors import QdrantVectorStore, embed_text

    url, api_key = _config()
    collection = f"scholarly-test-{uuid.uuid4().hex[:8]}"
    store = QdrantVectorStore(url=url, api_key=api_key, collection=collection)
    assert store.health()["status"] == "ok"
    try:
        store.create_collection(collection, 384)
        store.create_collection(collection, 384)
        assert store.collection_exists(collection) is True
        vector = embed_text("educational inequality shapes mobility")
        store.upsert(
            "chunk-a",
            vector,
            {
                "text": "educational inequality shapes mobility",
                "section": "results",
                "paper": "Aura verifique",
                "year": 2020,
            },
        )
        hits = store.search(vector, limit=5)
        assert hits and hits[0]["payload"]["paper"] == "Aura verifique"
        filtered = store.search(vector, limit=5, filters={"section": "results"})
        assert filtered
    finally:
        store.client.delete_collection(collection)
