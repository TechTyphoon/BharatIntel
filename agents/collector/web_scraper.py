"""
BharatIntel — Web Scraper Source Adapter

Extracts article links and metadata from web pages that don't offer RSS/API.
Uses site-specific parsers registered in _PARSER_MAP.

Responsibilities:
  - Async fetch of HTML pages
  - Site-specific parsing (each parser is a function)
  - Normalize output to RawArticle
  - Graceful degradation if page structure changes

Dependencies: httpx, lxml (via readability-lxml for article extraction)

Adding a new scraper:
  1. Write an async function: async def parse_<site>(html, source_config) -> list[RawArticle]
  2. Register it in _PARSER_MAP with the key matching sources.yaml 'parser' field
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine
from urllib.parse import urljoin

import httpx

from agents.collector.models import RawArticle
from core.exceptions import ParseError, SourceFetchError
from core.logger import get_logger
from core.retry import with_retry

log = get_logger("collector.web")

SNIPPET_MAX_LEN = 500

# Type alias for parser functions
ParserFn = Callable[[str, dict[str, Any]], Coroutine[Any, Any, list[RawArticle]]]


@with_retry(max_attempts=2, backoff_base=2.0, exceptions=(httpx.TimeoutException, httpx.ConnectError))
async def _fetch_html(url: str, timeout: float, user_agent: str) -> str:
    """Download page HTML with retry."""
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout),
        follow_redirects=True,
        headers={"User-Agent": user_agent},
    ) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.text


# ── Site-Specific Parsers ────────────────────────────────────────────


async def _parse_hackernews(html: str, source_config: dict[str, Any]) -> list[RawArticle]:
    """
    Parse Hacker News front page HTML.

    Extracts story titles, URLs, and point counts from the HN table structure.
    """
    name = source_config["name"]
    categories = source_config.get("categories", [])
    base_url = "https://news.ycombinator.com/"

    articles: list[RawArticle] = []

    # HN uses <span class="titleline"><a href="...">Title</a></span>
    # and <span class="score">N points</span>
    title_pattern = re.compile(
        r'<span class="titleline"[^>]*><a href="([^"]+)"[^>]*>([^<]+)</a>',
        re.DOTALL,
    )
    score_pattern = re.compile(r'<span class="score"[^>]*>(\d+)\s*points?</span>')

    titles = title_pattern.findall(html)
    scores = score_pattern.findall(html)

    for idx, (href, title) in enumerate(titles):
        title = title.strip()
        if not title:
            continue

        # Resolve relative URLs
        url = href if href.startswith("http") else urljoin(base_url, href)

        points = int(scores[idx]) if idx < len(scores) else 0
        snippet = f"{points} points on Hacker News" if points else ""

        try:
            article = RawArticle(
                title=title,
                url=url,
                source_name=name,
                published_at=datetime.now(timezone.utc),
                snippet=snippet,
                categories=list(categories),
            )
            articles.append(article)
        except (ValueError, TypeError) as exc:
            log.warning("web_article_parse_error", source=name, title=title, error=str(exc))
            continue

    return articles


# ── Parser Registry ──────────────────────────────────────────────────

_PARSER_MAP: dict[str, ParserFn] = {
    "hackernews": _parse_hackernews,
}


# ── Public Interface ─────────────────────────────────────────────────


async def fetch_web(
    source_config: dict[str, Any],
    timeout: float = 15.0,
    user_agent: str = "BharatIntel/1.0",
    max_articles: int = 50,
) -> list[RawArticle]:
    """
    Fetch and parse a web source using its registered parser.

    Args:
        source_config: Dict from sources.yaml with keys: name, url, parser, categories
        timeout:       HTTP timeout in seconds
        user_agent:    Request user-agent header
        max_articles:  Cap articles per source

    Returns:
        List[RawArticle]

    Raises:
        SourceFetchError: HTTP failure after retries
        ParseError:       Unknown parser or parser failure
    """
    name = source_config["name"]
    url = source_config["url"]
    parser_key = source_config.get("parser", "").lower()

    parser = _PARSER_MAP.get(parser_key)
    if parser is None:
        raise ParseError(
            f"No web parser registered for: {parser_key}",
            context={"source": name, "parser": parser_key},
        )

    log.info("web_fetch_start", source=name, url=url, parser=parser_key)

    try:
        html = await _fetch_html(url, timeout, user_agent)
    except httpx.HTTPStatusError as exc:
        raise SourceFetchError(
            f"Web fetch failed: {name}",
            context={"source": name, "url": url, "status": exc.response.status_code},
        ) from exc
    except (httpx.TimeoutException, httpx.ConnectError) as exc:
        raise SourceFetchError(
            f"Web fetch connection failed: {name}",
            context={"source": name, "url": url, "error": str(exc)},
        ) from exc

    try:
        articles = await parser(html, source_config)
    except Exception as exc:
        raise ParseError(
            f"Web parser failed: {name}",
            context={"source": name, "parser": parser_key, "error": str(exc)},
        ) from exc

    articles = articles[:max_articles]
    log.info("web_fetch_complete", source=name, articles_collected=len(articles))
    return articles
