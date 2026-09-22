"""Graph stores: Neo4j primary with Paper/Claim/Concept/Country schema."""

from __future__ import annotations

import hashlib
import json
import logging
import pathlib

logger = logging.getLogger(__name__)


def claim_node_id(paper_id: str, subject: str, relationship: str, obj: str, evidence: str) -> str:
    digest = hashlib.sha256(evidence.encode("utf-8")).hexdigest()[:16]
    return f"{paper_id}::{subject}::{relationship}::{obj}::{digest}"


class FileGraphStore:
    def __init__(self, path: str = "data/graph.json") -> None:
        self.path = pathlib.Path(path)
        self.state = {"papers": {}, "claims": [], "citations": []}
        if self.path.exists():
            self.state = json.loads(self.path.read_text(encoding="utf-8"))

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    def save_paper(self, paper) -> None:
        self.state["papers"][str(paper.document_id)] = {
            "document_id": str(paper.document_id),
            "title": paper.title,
            "year": paper.year,
            "data_countries": list(paper.data_countries),
        }
        logger.info("saved paper node %s", paper.document_id)
        self._persist()

    def save_claim(self, paper_id, claim) -> None:
        record = {
            "paper_id": str(paper_id),
            "subject": claim.subject,
            "relationship": claim.relationship,
            "object": claim.obj,
            "countries": list(claim.country_scope),
            "confidence": claim.confidence,
            "section": claim.evidence.section,
            "evidence": claim.evidence.text,
        }
        if record not in self.state["claims"]:
            self.state["claims"].append(record)
            logger.info("saved claim %s -[%s]-> %s", claim.subject, claim.relationship, claim.obj)
            self._persist()

    def save_citation(self, citing_id, cited_id, role: str = "neutral") -> None:
        record = {"citing": str(citing_id), "cited": str(cited_id), "role": role}
        if record not in self.state["citations"]:
            self.state["citations"].append(record)
            self._persist()

    def mechanisms_between(self, subject: str, obj: str) -> list:
        subject, obj = subject.strip().lower(), obj.strip().lower()
        return [
            claim
            for claim in self.state["claims"]
            if claim["subject"].lower() == subject and claim["object"].lower() == obj
        ]

    def get_mechanism_paths(self, subject: str, obj: str) -> list:
        direct = self.mechanisms_between(subject, obj)
        paths = [{"hops": 1, "claims": [claim]} for claim in direct]
        for first_leg in self.state["claims"]:
            if first_leg["subject"].lower() != subject.strip().lower():
                continue
            via = first_leg["object"].lower()
            for second_leg in self.state["claims"]:
                if (
                    second_leg["subject"].lower() == via
                    and second_leg["object"].lower() == obj.strip().lower()
                ):
                    paths.append(
                        {
                            "hops": 2,
                            "via": first_leg["object"],
                            "claims": [first_leg, second_leg],
                        }
                    )
        return paths

    def country_subgraph(self, country: str) -> list:
        wanted = country.strip().lower()
        return [
            claim
            for claim in self.state["claims"]
            if any(c.lower() == wanted for c in claim.get("countries", []))
        ]


class Neo4jGraphStore:
    def __init__(self, uri: str, user: str, password: str) -> None:
        from neo4j import GraphDatabase

        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def save_paper(self, paper) -> None:
        with self.driver.session() as session:
            session.run(
                "MERGE (p:Paper {id: $id}) "
                "SET p.title=$title, p.year=$year "
                "WITH p UNWIND $countries AS country "
                "MERGE (c:Country {name: country}) "
                "MERGE (p)-[:FROM_COUNTRY]->(c)",
                {
                    "id": str(paper.document_id),
                    "title": paper.title,
                    "year": paper.year,
                    "countries": list(paper.data_countries),
                },
            )
        logger.info("saved paper node %s", paper.document_id)

    def save_claim(self, paper_id, claim) -> None:
        node_id = claim_node_id(
            str(paper_id), claim.subject, claim.relationship, claim.obj, claim.evidence.text
        )
        with self.driver.session() as session:
            session.run(
                "MERGE (paper:Paper {id: $paper_id}) "
                "MERGE (claim:Claim {id: $claim_id}) "
                "SET claim.subject=$subject, claim.relationship=$relationship, "
                "claim.object=$object, claim.confidence=$confidence, "
                "claim.section=$section, claim.evidence=$evidence "
                "MERGE (paper)-[:HAS_CLAIM]->(claim) "
                "MERGE (subject:Concept {name: $subject}) "
                "MERGE (object:Concept {name: $object}) "
                "MERGE (claim)-[:SUBJECT]->(subject) "
                "MERGE (claim)-[:OBJECT]->(object) "
                "WITH claim UNWIND $countries AS country "
                "MERGE (c:Country {name: country}) "
                "MERGE (claim)-[:STUDIED_IN]->(c)",
                {
                    "paper_id": str(paper_id),
                    "claim_id": node_id,
                    "subject": claim.subject,
                    "relationship": claim.relationship,
                    "object": claim.obj,
                    "confidence": claim.confidence,
                    "section": claim.evidence.section,
                    "evidence": claim.evidence.text,
                    "countries": list(claim.country_scope),
                },
            )
        logger.info("saved claim node %s", node_id)

    def save_citation(self, citing_id, cited_id, role: str = "neutral") -> None:
        with self.driver.session() as session:
            session.run(
                "MERGE (a:Paper {id: $citing}) "
                "MERGE (b:Paper {id: $cited}) "
                "MERGE (a)-[r:CITES]->(b) SET r.role=$role",
                {"citing": str(citing_id), "cited": str(cited_id), "role": role},
            )

    def mechanism_paths(self, subject: str, obj: str, max_hops: int = 2) -> list:
        with self.driver.session() as session:
            result = session.run(
                "MATCH path = (s:Concept {name: $subject})"
                "-[:SUBJECT]-(:Claim)-[:OBJECT]-"
                "(:Concept)-[:SUBJECT]-(:Claim)-[:OBJECT]-(o:Concept {name: $object}) "
                "RETURN path LIMIT 50",
                {"subject": subject, "object": obj},
            )
            return [record["path"] for record in result]

    def get_mechanism_paths(self, subject: str, obj: str) -> list:
        return [
            {"hops": 1, "claims": [claim]}
            for claim in self.mechanisms_between(subject, obj)
        ] + [
            {"hops": 2, "path": path}
            for path in self.mechanism_paths(subject, obj)
        ]

    def mechanisms_between(self, subject: str, obj: str) -> list:
        with self.driver.session() as session:
            result = session.run(
                "MATCH (claim:Claim)-[:SUBJECT]->(s:Concept {name: $subject}), "
                "(claim)-[:OBJECT]->(o:Concept {name: $object}) "
                "RETURN claim",
                {"subject": subject, "object": obj},
            )
            return [record["claim"] for record in result]

    def country_subgraph(self, country: str) -> list:
        with self.driver.session() as session:
            result = session.run(
                "MATCH (claim:Claim)-[:STUDIED_IN]->(c:Country {name: $country}) "
                "RETURN claim",
                {"country": country},
            )
            return [record["claim"] for record in result]
