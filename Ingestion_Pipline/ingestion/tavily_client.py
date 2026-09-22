from __future__ import annotations

from typing import Any

try:
    from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
except Exception:  # optional dependency; scholarly ingestion uses OpenAlex/CORE
    TavilyCrawl = TavilyExtract = TavilyMap = None  # type: ignore[assignment]
from rich.console import Console
from rich.panel import Panel

from Ingestion_Pipline.config.settings import DEFAULT_SCRAPING_URL, IngestionSettings
from Ingestion_Pipline.infra.retry_policies import url_extraction_retry

console = Console()


def _require_tavily(name: str):
    if TavilyCrawl is None:
        raise RuntimeError(
            f"Tavily {name} is unavailable. Scholarly ingestion now uses OpenAlex, "
            "Semantic Scholar, and CORE; install langchain-tavily only for legacy use."
        )


def build_tavily_extract():
    _require_tavily("extract")
    return TavilyExtract()


def build_tavily_map(settings: IngestionSettings | None = None):
    _require_tavily("map")
    settings = settings or IngestionSettings()
    return TavilyMap(
        max_depth=settings.tavily_max_depth,
        max_breadth=settings.tavily_max_breadth,
        limit=settings.tavily_limit,
    )


def build_tavily_crawl():
    _require_tavily("crawl")
    return TavilyCrawl()


def docs_crawling(
    tavily_crawl,
    settings: IngestionSettings | None = None,
):
    settings = settings or IngestionSettings()
    return tavily_crawl.invoke(
        {
            "url": settings.scraping_url or DEFAULT_SCRAPING_URL,
            "max_depth": settings.crawl_max_depth,
            "extract_depth": settings.crawl_extract_depth,
        }
    )


def doc_scrolling_using_tavily_langchain(
    tavily_map,
    tavily_extract,
    settings: IngestionSettings | None = None,
):
    settings = settings or IngestionSettings()
    scraping_url = settings.scraping_url or DEFAULT_SCRAPING_URL

    tavily = tavily_map.invoke(scraping_url)
    sample_urls = tavily[:8]

    console.print(
        f"🔍 Extracting content from {len(sample_urls)} URLs...",
        style="bold blue",
    )

    extraction_result = tavily_extract.invoke({"urls": sample_urls})
    extracted_docs = extraction_result.get("results", [])

    console.print(
        f"✅ Successfully extracted {len(extracted_docs)} documents",
        style="bold green",
    )

    for i, doc in enumerate(extracted_docs, 1):
        url = doc.get("url", "Unknown")
        content = doc.get("raw_content", "")

        panel_content = (
            f"URL: {url}\n\n"
            f"Content Length: {len(content)} characters\n\n"
            f"Preview:\n{content[:500]}..."
        )

        console.print(
            Panel(panel_content, title=f"Document {i}", border_style="blue")
        )
        print()

    return extracted_docs


def get_site_urls(
    tavily_map,
    settings: IngestionSettings | None = None,
) -> list[str]:
    settings = settings or IngestionSettings()
    scraping_url = settings.scraping_url or DEFAULT_SCRAPING_URL

    console.print("[bold blue] Mapping documentation URLs...[/bold blue]")

    site_map = tavily_map.invoke(scraping_url)
    urls = site_map["results"]

    console.print(f"[green]✅ Found {len(urls)} URLs[/green]")

    return urls


@url_extraction_retry()
async def extract_urls(
    tavily_extract,
    urls: list[str],
    batch_num: int,
    settings: IngestionSettings | None = None,
) -> list[dict[str, Any]]:
    """Extract docs from a batch of URLs."""
    settings = settings or IngestionSettings()

    try:
        console.print(f"Started Extraction batch {batch_num}")

        docs = await tavily_extract.ainvoke(
            {
                "urls": urls,
                "instructions": settings.extract_instructions,
            }
        )

        results = docs.get("results", [])
        console.print(f"Extracted batch {batch_num} successfully")
        return results

    except Exception as e:
        console.print(f"❌ Batch {batch_num} failed: {e}", style="red")
        raise
