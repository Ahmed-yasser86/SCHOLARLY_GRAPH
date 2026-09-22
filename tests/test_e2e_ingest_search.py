"""End-to-end test: ingest (10 papers) then search, all against the live app."""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from scholarly_graph.api.app import Settings, create_app


def test_ingest_then_search_end_to_end(tmp_path):
    cfg = Settings(
        _env_file=None,
        vector_path=str(tmp_path / "vectors.json"),
        graph_path=str(tmp_path / "graph.json"),
        snapshot_dir=str(tmp_path / "snapshots"),
    )
    client = TestClient(create_app(cfg))
    started = client.post(
        "/ingest",
        json={
            "query": "Great Gatsby Curve intergenerational mobility",
            "target_count": 10,
            "year_from": 2000,
            "year_to": 2025,
            "open_access_only": True,
        },
    )
    assert started.status_code == 200
    snapshot_id = started.json()["snapshot_id"]

    result = {"status": "running"}
    for _ in range(600):
        result = client.get(f"/ingest/{snapshot_id}").json()
        if result.get("status") != "running":
            break
        time.sleep(1)
    assert result.get("status") == "complete", result
    assert result["result"]["discovered"] >= 1

    search = client.post(
        "/search/standard",
        json={"question": "How does inequality shape mobility?"},
    )
    assert search.status_code == 200
    body = search.json()
    assert body["sources"]
    assert body["trace"]
