"""Corpus discovery clients: OpenAlex, Semantic Scholar, CORE, PDF download."""

from __future__ import annotations

import json
import logging
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

DEFAULT_CONTACT_EMAIL = "scholarly-graph@example.org"


def _http_get_json(url: str, headers: dict | None = None, timeout: int = 30) -> dict:
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def decode_openalex_abstract(inverted_index: dict | None) -> str:
    if not inverted_index:
        return ""
    positions: list = []
    for word, indices in inverted_index.items():
        for index in indices:
            positions.append((index, word))
    positions.sort()
    return " ".join(word for _, word in positions)


@dataclass
class OpenAlexClient:
    contact_email: str = DEFAULT_CONTACT_EMAIL
    per_page: int = 25
    sleep_seconds: float = 0.2
    max_retries: int = 4

    def _headers(self) -> dict:
        return {"User-Agent": f"scholarly-graph (mailto:{self.contact_email})"}

    def search(
        self,
        query: str,
        year_from: int = 2000,
        year_to: int = 2025,
        open_access_only: bool = True,
        page: int = 1,
    ) -> list:
        params = {
            "search": query,
            "filter": f"from_publication_date:{year_from}-01-01,"
            f"to_publication_date:{year_to}-12-31",
            "per-page": str(self.per_page),
            "page": str(page),
        }
        if open_access_only:
            params["filter"] += ",is_oa:true"
        url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                logger.info("OpenAlex search query=%r page=%s", query, page)
                payload = _http_get_json(url, headers=self._headers())
                results = payload.get("results", [])
                logger.info("OpenAlex returned %d works", len(results))
                return [self.parse_work(work) for work in results]
            except Exception as exc:  # noqa: BLE001 - retry then raise
                last_error = exc
                logger.warning("OpenAlex attempt %d failed: %s", attempt + 1, exc)
                time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"OpenAlex search failed: {last_error}")

    def parse_work(self, work: dict) -> dict:
        openalex_id = str(work.get("id", "")).rsplit("/", 1)[-1]
        title = work.get("title") or work.get("display_name") or ""
        year = work.get("publication_year") or 0
        authors = [
            a.get("author", {}).get("display_name", "")
            for a in work.get("authorships", [])
        ]
        primary = work.get("primary_location") or {}
        source = primary.get("source") or {}
        doi = (work.get("doi") or "").replace("https://doi.org/", "")
        pdf_url = (work.get("open_access") or {}).get("oa_url") or ""
        countries = sorted(
            {
                (c or {}).get("country_code", "")
                for a in work.get("authorships", [])
                for inst in (a.get("institutions") or [])
                for c in (inst.get("country_code"),)
                if isinstance(inst, dict)
            }
            - {""}
        )
        return {
            "document_id": f"openalex:{openalex_id}",
            "title": title,
            "authors": [a for a in authors if a],
            "year": int(year) if year else 0,
            "journal": source.get("display_name", ""),
            "doi": doi,
            "open_access": bool((work.get("open_access") or {}).get("is_oa", False)),
            "pdf_url": pdf_url,
            "abstract": decode_openalex_abstract(work.get("abstract_inverted_index")),
            "data_countries": countries,
        }

    def collect(
        self,
        query: str,
        target_count: int = 100,
        year_from: int = 2000,
        year_to: int = 2025,
        open_access_only: bool = True,
    ) -> list:
        collected: list = []
        page = 1
        while len(collected) < target_count:
            batch = self.search(
                query,
                year_from=year_from,
                year_to=year_to,
                open_access_only=open_access_only,
                page=page,
            )
            if not batch:
                break
            collected.extend(batch)
            page += 1
            time.sleep(self.sleep_seconds)
        return collected[:target_count]

    def fetch_by_doi(self, doi: str) -> dict:
        url = f"https://api.openalex.org/works/https://doi.org/{doi}"
        payload = _http_get_json(url, headers=self._headers())
        return self.parse_work(payload)


@dataclass
class SemanticScholarClient:
    api_key: str = ""
    sleep_seconds: float = 1.0
    max_retries: int = 4

    def _headers(self) -> dict:
        headers = {"User-Agent": f"scholarly-graph (mailto:{DEFAULT_CONTACT_EMAIL})"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    def search(self, query: str, limit: int = 25) -> list:
        params = urllib.parse.urlencode(
            {"query": query, "limit": str(limit), "fields": "title,year,authors,url"}
        )
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?{params}"
        for attempt in range(self.max_retries):
            try:
                logger.info("Semantic Scholar search query=%r", query)
                payload = _http_get_json(url, headers=self._headers())
                results = payload.get("data", [])
                logger.info("Semantic Scholar returned %d papers", len(results))
                time.sleep(self.sleep_seconds if not self.api_key else 0.2)
                return results
            except Exception as exc:  # noqa: BLE001
                logger.warning("SemanticScholar attempt %d failed: %s", attempt + 1, exc)
                time.sleep(min(2**attempt, 8))
        raise RuntimeError("Semantic Scholar search failed")

    def citations(self, paper_id: str, limit: int = 100) -> list:
        url = (
            "https://api.semanticscholar.org/graph/v1/paper/"
            f"{urllib.parse.quote(paper_id, safe='')}/citations?"
            + urllib.parse.urlencode(
                {"fields": "title,year,externalIds", "limit": str(limit)}
            )
        )
        payload = _http_get_json(url, headers=self._headers())
        time.sleep(self.sleep_seconds if not self.api_key else 0.2)
        return payload.get("data", [])

    def references(self, paper_id: str, limit: int = 100) -> list:
        url = (
            "https://api.semanticscholar.org/graph/v1/paper/"
            f"{urllib.parse.quote(paper_id, safe='')}/references?"
            + urllib.parse.urlencode(
                {"fields": "title,year,externalIds", "limit": str(limit)}
            )
        )
        payload = _http_get_json(url, headers=self._headers())
        time.sleep(self.sleep_seconds if not self.api_key else 0.2)
        return payload.get("data", [])


@dataclass
class CoreClient:
    api_key: str = ""

    def search(self, query: str, limit: int = 25) -> list:
        if not self.api_key:
            raise RuntimeError("CORE API key is required; see CORE registration docs")
        params = urllib.parse.urlencode({"q": query, "limit": str(limit)})
        url = f"https://api.core.ac.uk/v3/search/works?{params}"
        payload = _http_get_json(url, headers={"Authorization": f"Bearer {self.api_key}"})
        logger.info("CORE returned %d works", len(payload.get("results", [])))
        return payload.get("results", [])


@dataclass
class PdfDownloader:
    timeout: int = 60

    def download(self, url: str, destination: str) -> dict:
        import pathlib

        target = pathlib.Path(destination)
        if target.exists() and target.stat().st_size > 0:
            logger.info("PDF already cached: %s", destination)
            return {"status": "cached", "path": destination, "size": target.stat().st_size}
        if not url:
            return {"status": "skipped_no_pdf", "path": destination, "size": 0}
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": f"scholarly-graph (mailto:{DEFAULT_CONTACT_EMAIL})"}
            )
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                content_type = response.headers.get("Content-Type", "")
                if "pdf" not in content_type.lower():
                    logger.warning("Non-PDF content type %r for %s", content_type, url)
                    return {"status": "skipped_not_pdf", "path": destination, "size": 0}
                target.parent.mkdir(parents=True, exist_ok=True)
                data = response.read()
                target.write_bytes(data)
                logger.info("Downloaded PDF %s (%d bytes)", url, len(data))
                return {"status": "success", "path": destination, "size": len(data)}
        except Exception as exc:  # noqa: BLE001
            logger.warning("PDF download failed for %s: %s", url, exc)
            return {"status": "failed", "path": destination, "size": 0, "error": str(exc)}


@dataclass
class SnapshotStore:
    directory: str = "data/snapshots"

    def new_snapshot_id(self, now: str | None = None) -> str:
        from datetime import datetime, timezone

        stamp = now or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"snapshot-{stamp}"

    def save_manifest(self, snapshot_id: str, payload: dict) -> str:
        import pathlib

        directory = pathlib.Path(self.directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{snapshot_id}.json"
        manifest = {
            "snapshot_id": snapshot_id,
            "query": payload.get("query", ""),
            "paper_count": len(payload.get("papers", [])),
            "open_access_count": sum(
                1 for p in payload.get("papers", []) if p.get("open_access")
            ),
            "papers": [
                {
                    "document_id": p.get("document_id"),
                    "title": p.get("title"),
                    "year": p.get("year"),
                    "open_access": p.get("open_access"),
                }
                for p in payload.get("papers", [])
            ],
        }
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return str(path)
