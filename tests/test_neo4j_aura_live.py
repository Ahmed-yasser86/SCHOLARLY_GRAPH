"""Live Neo4j Aura integration: connect, write/read graph data, mechanism paths.

Credentials come ONLY from the local environment (NEO4J_URI, NEO4J_USER,
NEO4J_PASSWORD). They are never committed, printed, or stored. Skips when
any variable is absent. Uses a unique TEST_RUN label so live data is
isolated and cleaned up afterwards.
"""

from __future__ import annotations

import os
import uuid

import pytest

pytestmark = pytest.mark.live


def _creds():
    uri = os.environ.get("NEO4J_URI", "")
    user = os.environ.get("NEO4J_USER", "")
    password = os.environ.get("NEO4J_PASSWORD", "")
    if not (uri and user and password):
        pytest.skip("NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD not set")
    return uri, user, password


def test_aura_connectivity():
    from scholarly_graph.storage.graph_store import Neo4jGraphStore

    uri, user, password = _creds()
    store = Neo4jGraphStore(uri, user, password)
    try:
        assert store.verify_connectivity() == {"status": "ok"}
    finally:
        store.close()


def test_aura_paper_claim_country_citation_roundtrip():
    from scholarly_graph.domain.entities import Claim, DocumentId, EvidenceSpan, Paper
    from scholarly_graph.storage.graph_store import Neo4jGraphStore

    uri, user, password = _creds()
    store = Neo4jGraphStore(uri, user, password)
    run = f"test-{uuid.uuid4().hex[:8]}"
    paper_id = DocumentId(f"test:{run}-W1")
    cited_id = DocumentId(f"test:{run}-W2")
    try:
        store.save_paper(
            Paper(
                document_id=paper_id,
                title=f"Aura roundtrip {run}",
                year=2020,
                data_countries=("Sweden",),
            )
        )
        claim = Claim(
            subject="educational inequality",
            relationship="mediates",
            obj="intergenerational mobility",
            evidence=EvidenceSpan(
                text=f"Schools mediate outcomes in run {run}.", section="results"
            ),
            country_scope=("Sweden",),
        )
        store.save_claim(paper_id, claim)
        store.save_citation(paper_id, cited_id, role="neutral")
        found = store.mechanisms_between(
            "educational inequality", "intergenerational mobility"
        )
        assert found, "expected the written claim back from Aura"
        assert store.country_subgraph("Sweden"), "expected Sweden subgraph hit"
        assert store.get_mechanism_paths(
            "educational inequality", "intergenerational mobility"
        ), "expected at least one mechanism path"
    finally:
        with store.driver.session() as session:
            session.run(
                "MATCH (n) WHERE n.id STARTS WITH $prefix DETACH DELETE n",
                {"prefix": f"test:{run}"},
            )
            session.run(
                "MATCH (c:Claim) WHERE c.evidence CONTAINS $run DETACH DELETE c",
                {"run": run},
            )
        store.close()
