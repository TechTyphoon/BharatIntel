"""
BharatIntel — RSS Feed Source Adapter

Fetches and parses RSS/Atom feeds into RawArticle instances.

Responsibilities:
  - Async HTTP fetch of feed XML
  - Parse entries via feedparser
  - Normalize dates to datetime
  - Truncate snippet to configured length
  - Skip entries missing title or link

Dependencies: httpx, feedparser
"""

from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import feedparser
import httpx

from agents.collector.models import RawArticle
from core.exceptions import ParseError, SourceFetchError
from core.logger import get_logger
from core.retry import with_retry

log = get_logger("collector.rss")

SNIPPET_MAX_LEN = 500


def _parse_date(entry: dict[str, Any]) -> datetime | None:
    """Extract publication date from a feed entry, return None on failure."""
    for field in ("published", "updated"):
        raw = entry.get(field)
        if not raw:
            continue
        try:
            return parsedate_to_datetime(raw).astimezone(timezone.utc)
        except (ValueError, TypeError):
            pass
    # feedparser sometimes provides a parsed struct
    struct = entry.get("published_parsed") or entry.get("updated_parsed")
    if struct:
        try:
            return datetime(*struct[:6], tzinfo=timezone.utc)
        except (ValueError, TypeError):
            pass
    return None


def _extract_snippet(entry: dict[str, Any]) -> str:
    """Best-effort snippet from summary or content fields."""
    text = ""
    if entry.get("summary"):
        text = entry["summary"]
    elif entry.get("content"):
        # Atom feeds put body in content[0].value
        content_list = entry["content"]
        if isinstance(content_list, list) and content_list:
            text = content_list[0].get("value", "")

    # Strip HTML tags naively (good enough for snippet)
    import re

    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:SNIPPET_MAX_LEN]


def _extract_image(entry: dict[str, Any]) -> str | None:
    """Extract thumbnail or media URL if present."""
    # media:thumbnail
    thumb = entry.get("media_thumbnail")
    if thumb and isinstance(thumb, list) and thumb[0].get("url"):
        return thumb[0]["url"]
    # media:content
    media = entry.get("media_content")
    if media and isinstance(media, list) and media[0].get("url"):
        return media[0]["url"]
    # enclosure
    links = entry.get("links", [])
    for link in links:
        if link.get("type", "").startswith("image/"):
            return link.get("href")
    return None


@with_retry(max_attempts=2, backoff_base=2.0, exceptions=(httpx.TimeoutException, httpx.ConnectError))
async def _fetch_feed_xml(url: str, timeout: float, user_agent: str) -> str:
    """Download raw feed XML with retry."""
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout),
        follow_redirects=True,
        headers={"User-Agent": user_agent},
    ) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.text


async def fetch_rss(
    source_config: dict[str, Any],
    timeout: float = 15.0,
    user_agent: str = "BharatIntel/1.0",
    max_articles: int = 50,
) -> list[RawArticle]:
    """
    Fetch a single RSS source and return parsed articles.

    Args:
        source_config: Dict from sources.yaml with keys: name, url, categories
        timeout:       HTTP timeout in seconds
        user_agent:    Request user-agent header
        max_articles:  Cap entries per feed

    Returns:
        List[RawArticle] — may be empty if feed has no valid entries.

    Raises:
        SourceFetchError: If HTTP fetch fails after retries.
    """
    name = source_config["name"]
    url = source_config["url"]
    categories = source_config.get("categories", [])

    log.info("rss_fetch_start", source=name, url=url)

    try:
        xml = await _fetch_feed_xml(url, timeout, user_agent)
    except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError) as exc:
        raise SourceFetchError(
            f"Failed to fetch RSS feed: {name}",
            context={"source": name, "url": url, "error": str(exc)},
        ) from exc

    feed = feedparser.parse(xml)
    if feed.bozo and not feed.entries:
        raise ParseError(
            f"RSS feed parse failed with no entries: {name}",
            context={"source": name, "bozo_exception": str(feed.bozo_exception)},
        )

    articles: list[RawArticle] = []
    for entry in feed.entries[:max_articles]:
        title = (entry.get("title") or "").strip()
        link = (entry.get("link") or "").strip()
        if not title or not link:
            log.debug("rss_skip_entry", source=name, reason="missing title or link")
            continue

        try:
            article = RawArticle(
                title=title,
                url=link,
                source_name=name,
                published_at=_parse_date(entry),
                snippet=_extract_snippet(entry),
                categories=list(categories),
                author=entry.get("author"),
                image_url=_extract_image(entry),
            )
            articles.append(article)
        except (ValueError, TypeError) as exc:
            log.warning("rss_article_parse_error", source=name, title=title, error=str(exc))
            continue

    log.info("rss_fetch_complete", source=name, articles_collected=len(articles))
    return articles
