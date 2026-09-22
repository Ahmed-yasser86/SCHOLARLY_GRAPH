"""Corpus discovery clients: OpenAlex, Semantic Scholar, CORE, PDF download.

All clients use httpx with exponential-backoff retries via tenacity,
polite User-Agent contact headers, tier-aware request spacing, and
structured logging of every request, retry, outcome, and skip.
"""

from __future__ import annotations

import json
import logging
import pathlib
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_CONTACT_EMAIL = "scholarly-graph@example.org"
_API_FAILURES: dict = {}


def _record_failure(api: str, threshold: int = 5) -> None:
    entry = _API_FAILURES.setdefault(api, {"failures": 0, "opened_at": None})
    entry["failures"] += 1
    if entry["failures"] >= threshold and entry["opened_at"] is None:
        entry["opened_at"] = time.monotonic()
        logger.warning("circuit breaker opened for %s after %d failures", api, threshold)


def _circuit_allows(api: str, reset_seconds: int = 60) -> bool:
    entry = _API_FAILURES.get(api)
    if not entry or entry["opened_at"] is None:
        return True
    if time.monotonic() - entry["opened_at"] >= reset_seconds:
        _API_FAILURES[api] = {"failures": 0, "opened_at": None}
        return True
    return False


def _circuit_reset(api: str) -> None:
    _API_FAILURES[api] = {"failures": 0, "opened_at": None}


def _http_get_json(
    url: str, headers: dict | None = None, timeout: int = 30, api: str = "http"
) -> dict:
    import httpx

    if not _circuit_allows(api):
        raise RuntimeError(f"circuit breaker open for {api}; backing off 60s")
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(url, headers=headers or {})
            response.raise_for_status()
            return response.json()
    except Exception:
        _record_failure(api)
        raise
    finally:
        _circuit_reset(api)


def _get_with_retry(url: str, headers: dict, api: str) -> dict:
    from tenacity import (
        before_sleep_log,
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )

    import httpx

    @retry(
        retry=retry_if_exception_type(
            (httpx.HTTPStatusError, httpx.TransportError, RuntimeError)
        ),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _call() -> dict:
        logger.info("GET %s", url)
        return _http_get_json(url, headers=headers, api=api)

    return _call()


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
    max_attempts: int = 5

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
        import urllib.parse

        params = {
            "search": query,
            "filter": (
                f"from_publication_date:{year_from}-01-01,"
                f"to_publication_date:{year_to}-12-31"
            ),
            "per-page": str(self.per_page),
            "page": str(page),
        }
        if open_access_only:
            params["filter"] += ",is_oa:true"
        url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
        payload = _get_with_retry(url, self._headers(), "openalex")
        results = payload.get("results", [])
        parsed = [self.parse_work(work) for work in results]
        logger.info(
            "openalex query=%r page=%d responses=%d parsed=%d skipped=%d",
            query,
            page,
            len(results),
            len(parsed),
            len(results) - len(parsed),
        )
        print(
            f"OpenAlex: query={query!r} page={page} "
            f"responses={len(results)} parsed={len(parsed)}"
        )
        return parsed

    def parse_work(self, work: dict) -> dict:
        openalex_id = str(work.get("id", "")).rsplit("/", 1)[-1]
        title = work.get("title") or work.get("display_name") or ""
        year = work.get("publication_year") or 0
        authors = [
            author.get("author", {}).get("display_name", "")
            for author in work.get("authorships", [])
        ]
        primary = work.get("primary_location") or {}
        source = primary.get("source") or {}
        doi = (work.get("doi") or "").replace("https://doi.org/", "")
        pdf_url = (work.get("open_access") or {}).get("oa_url") or ""
        countries = sorted(
            {
                code
                for authorship in work.get("authorships", [])
                for institution in (authorship.get("institutions") or [])
                if isinstance(institution, dict)
                for code in [institution.get("country_code", "")]
                if code
            }
        )
        return {
            "document_id": f"openalex:{openalex_id}",
            "title": title,
            "authors": [author for author in authors if author],
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
        import urllib.parse

        url = (
            "https://api.openalex.org/works/"
            f"{urllib.parse.quote(f'https://doi.org/{doi}', safe='')}"
        )
        return self.parse_work(_get_with_retry(url, self._headers(), "openalex"))


@dataclass
class SemanticScholarClient:
    api_key: str = ""
    sleep_seconds: float = 1.0
    max_attempts: int = 5

    def _headers(self) -> dict:
        headers = {"User-Agent": f"scholarly-graph (mailto:{DEFAULT_CONTACT_EMAIL})"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    def _spacing(self) -> float:
        return 0.2 if self.api_key else self.sleep_seconds

    def search(self, query: str, limit: int = 25) -> list:
        import urllib.parse

        params = urllib.parse.urlencode(
            {"query": query, "limit": str(limit), "fields": "title,year,authors,url"}
        )
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?{params}"
        payload = _get_with_retry(url, self._headers(), "semanticscholar")
        results = payload.get("data", [])
        logger.info("semanticscholar query=%r responses=%d", query, len(results))
        print(f"SemanticScholar: query={query!r} responses={len(results)}")
        time.sleep(self._spacing())
        return results

    def citations(self, paper_id: str, limit: int = 100) -> list:
        import urllib.parse

        url = (
            "https://api.semanticscholar.org/graph/v1/paper/"
            f"{urllib.parse.quote(paper_id, safe='')}/citations?"
            + urllib.parse.urlencode(
                {"fields": "title,year,externalIds,citationCount,isInfluential",
                 "limit": str(limit)}
            )
        )
        payload = _get_with_retry(url, self._headers(), "semanticscholar")
        time.sleep(self._spacing())
        return payload.get("data", [])

    def references(self, paper_id: str, limit: int = 100) -> list:
        import urllib.parse

        url = (
            "https://api.semanticscholar.org/graph/v1/paper/"
            f"{urllib.parse.quote(paper_id, safe='')}/references?"
            + urllib.parse.urlencode(
                {"fields": "title,year,externalIds,citationCount,isInfluential",
                 "limit": str(limit)}
            )
        )
        payload = _get_with_retry(url, self._headers(), "semanticscholar")
        time.sleep(self._spacing())
        return payload.get("data", [])


@dataclass
class CoreClient:
    api_key: str = ""

    def search(self, query: str, limit: int = 25) -> list:
        if not self.api_key:
            raise RuntimeError("CORE API key is required; see CORE registration docs")
        import urllib.parse

        params = urllib.parse.urlencode({"q": query, "limit": str(limit)})
        url = f"https://api.core.ac.uk/v3/search/works?{params}"
        payload = _get_with_retry(
            url, {"Authorization": f"Bearer {self.api_key}"}, "core"
        )
        results = payload.get("results", [])
        logger.info("core query=%r responses=%d", query, len(results))
        print(f"CORE: query={query!r} responses={len(results)}")
        return results


@dataclass
class PdfDownloader:
    timeout: int = 60

    def download(self, url: str, destination: str) -> dict:
        target = pathlib.Path(destination)
        if target.exists() and target.stat().st_size > 0:
            logger.info("PDF already cached: %s", destination)
            return {
                "status": "cached",
                "path": destination,
                "size": target.stat().st_size,
            }
        if not url:
            logger.info("PDF skipped (no URL): %s", destination)
            return {"status": "skipped_no_pdf", "path": destination, "size": 0}
        import httpx

        try:
            with httpx.Client(
                timeout=self.timeout, follow_redirects=True
            ) as client:
                response = client.get(
                    url,
                    headers={
                        "User-Agent": f"scholarly-graph (mailto:{DEFAULT_CONTACT_EMAIL})"
                    },
                )
                content_type = response.headers.get("Content-Type", "")
                if "pdf" not in content_type.lower():
                    logger.warning(
                        "non-PDF content type %r for %s", content_type, url
                    )
                    return {
                        "status": "skipped_not_pdf",
                        "path": destination,
                        "size": 0,
                    }
                response.raise_for_status()
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(response.content)
                logger.info("downloaded %s (%d bytes)", url, len(response.content))
                return {
                    "status": "success",
                    "path": destination,
                    "size": len(response.content),
                }
        except Exception as exc:
            logger.warning("PDF download failed for %s: %s", url, exc)
            return {
                "status": "failed",
                "path": destination,
                "size": 0,
                "error": str(exc),
            }

    def close(self) -> None:
        return None


@dataclass
class SnapshotStore:
    directory: str = "data/snapshots"

    def new_snapshot_id(self, now: str | None = None) -> str:
        from datetime import datetime, timezone

        stamp = now or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"snapshot-{stamp}"

    def save_manifest(self, snapshot_id: str, payload: dict) -> str:
        directory = pathlib.Path(self.directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{snapshot_id}.json"
        manifest = {
            "snapshot_id": snapshot_id,
            "query": payload.get("query", ""),
            "timestamp": payload.get("timestamp", snapshot_id),
            "paper_count": len(payload.get("papers", [])),
            "open_access_count": sum(
                1 for paper in payload.get("papers", []) if paper.get("open_access")
            ),
            "claim_count": payload.get("claim_count", 0),
            "papers": [
                {
                    "document_id": paper.get("document_id"),
                    "title": paper.get("title"),
                    "year": paper.get("year"),
                    "source": paper.get("source", ""),
                    "open_access": paper.get("open_access"),
                }
                for paper in payload.get("papers", [])
            ],
        }
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return str(path)
