"""
BharatIntel — News API Source Adapter

Fetches articles from external news APIs (GNews, etc.) into RawArticle instances.

Responsibilities:
  - Async HTTP calls to news API endpoints
  - Normalize JSON responses to RawArticle
  - Handle missing/malformed API keys gracefully
  - Rate-limit aware via retry decorator

Dependencies: httpx
Environment: GNEWS_API_KEY in .env
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import httpx

from agents.collector.models import RawArticle
from core.exceptions import SourceFetchError
from core.logger import get_logger
from core.retry import with_retry

log = get_logger("collector.api")

SNIPPET_MAX_LEN = 500


# ── Provider: GNews ──────────────────────────────────────────────────

GNEWS_BASE_URL = "https://gnews.io/api/v4/top-headlines"


def _parse_iso_date(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


@with_retry(max_attempts=2, backoff_base=3.0, exceptions=(httpx.TimeoutException, httpx.ConnectError))
async def _gnews_request(
    params: dict[str, Any],
    api_key: str,
    timeout: float,
    user_agent: str,
) -> dict[str, Any]:
    """Execute a single GNews API request with retry."""
    query_params = {
        "token": api_key,
        "lang": params.get("lang", "en"),
        "country": params.get("country", "us"),
        "max": str(params.get("max_results", 10)),
    }

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout),
        headers={"User-Agent": user_agent},
    ) as client:
        resp = await client.get(GNEWS_BASE_URL, params=query_params)
        resp.raise_for_status()
        return resp.json()


async def fetch_gnews(
    source_config: dict[str, Any],
    timeout: float = 15.0,
    user_agent: str = "BharatIntel/1.0",
    max_articles: int = 50,
) -> list[RawArticle]:
    """
    Fetch articles from GNews API.

    Requires GNEWS_API_KEY environment variable.
    Returns empty list if API key is missing (logs warning, does not crash).
    """
    name = source_config["name"]
    params = source_config.get("params", {})
    categories = source_config.get("categories", [])

    api_key = os.environ.get("GNEWS_API_KEY", "").strip()
    if not api_key:
        log.warning("gnews_skip", source=name, reason="GNEWS_API_KEY not set")
        return []

    log.info("api_fetch_start", source=name, provider="gnews")

    try:
        data = await _gnews_request(params, api_key, timeout, user_agent)
    except httpx.HTTPStatusError as exc:
        raise SourceFetchError(
            f"GNews API error: {name}",
            context={"source": name, "status": exc.response.status_code, "error": str(exc)},
        ) from exc
    except (httpx.TimeoutException, httpx.ConnectError) as exc:
        raise SourceFetchError(
            f"GNews API connection failed: {name}",
            context={"source": name, "error": str(exc)},
        ) from exc

    raw_articles = data.get("articles", [])
    articles: list[RawArticle] = []

    for item in raw_articles[:max_articles]:
        title = (item.get("title") or "").strip()
        url = (item.get("url") or "").strip()
        if not title or not url:
            continue

        snippet = (item.get("description") or "")[:SNIPPET_MAX_LEN]

        try:
            article = RawArticle(
                title=title,
                url=url,
                source_name=name,
                published_at=_parse_iso_date(item.get("publishedAt")),
                snippet=snippet,
                categories=list(categories),
                author=item.get("source", {}).get("name"),
                image_url=item.get("image"),
            )
            articles.append(article)
        except (ValueError, TypeError) as exc:
            log.warning("api_article_parse_error", source=name, title=title, error=str(exc))
            continue

    log.info("api_fetch_complete", source=name, articles_collected=len(articles))
    return articles


# ── Provider Router ──────────────────────────────────────────────────

_PROVIDER_MAP = {
    "gnews": fetch_gnews,
}


async def fetch_api(
    source_config: dict[str, Any],
    timeout: float = 15.0,
    user_agent: str = "BharatIntel/1.0",
    max_articles: int = 50,
) -> list[RawArticle]:
    """
    Route to the correct API provider based on source_config['provider'].

    Raises SourceFetchError if provider is unknown.
    """
    provider = source_config.get("provider", "").lower()
    handler = _PROVIDER_MAP.get(provider)

    if handler is None:
        raise SourceFetchError(
            f"Unknown API provider: {provider}",
            context={"source": source_config.get("name"), "provider": provider},
        )

    return await handler(source_config, timeout, user_agent, max_articles)
