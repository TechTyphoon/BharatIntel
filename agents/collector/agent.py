"""
BharatIntel — Collector Agent

Orchestrates news collection across all enabled sources in parallel.
This is the single entry point for the ingestion module.

Responsibilities:
  - Load source definitions from config
  - Dispatch to correct adapter (rss, api, web) per source
  - Run all sources concurrently via asyncio.gather
  - Aggregate results, skip failed sources without crashing
  - Log collection summary

Usage:
    from agents.collector.agent import CollectorAgent
    agent = CollectorAgent("config/sources.yaml", "config/settings.yaml")
    articles = await agent.run()
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import yaml

from agents.collector.api_source import fetch_api
from agents.collector.models import RawArticle
from agents.collector.rss_source import fetch_rss
from agents.collector.web_scraper import fetch_web
from core.exceptions import BharatIntelError
from core.logger import get_logger

log = get_logger("collector")


class CollectorAgent:
    """
    Orchestrates parallel fetching from all enabled news sources.

    Args:
        sources_path: Path to sources.yaml
        settings_path: Path to settings.yaml
    """

    def __init__(self, sources_path: str = "config/sources.yaml", settings_path: str = "config/settings.yaml"):
        self._sources = self._load_sources(sources_path)
        self._settings = self._load_settings(settings_path)

    @staticmethod
    def _load_sources(path: str) -> list[dict[str, Any]]:
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Sources config not found: {path}")
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        sources = data.get("sources", [])
        enabled = [s for s in sources if s.get("enabled", False)]
        log.info("sources_loaded", total=len(sources), enabled=len(enabled))
        return enabled

    @staticmethod
    def _load_settings(path: str) -> dict[str, Any]:
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Settings config not found: {path}")
        with open(filepath, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _get_collector_setting(self, key: str, default: Any = None) -> Any:
        return self._settings.get("collector", {}).get(key, default)

    async def _fetch_source(self, source: dict[str, Any]) -> list[RawArticle]:
        """
        Dispatch a single source to the correct adapter.

        Returns empty list on failure (logs error, never crashes).
        """
        source_type = source.get("type", "").lower()
        name = source.get("name", "unknown")
        timeout = self._get_collector_setting("timeout_seconds", 15)
        user_agent = self._get_collector_setting("user_agent", "BharatIntel/1.0")
        max_articles = self._get_collector_setting("max_articles_per_source", 50)

        try:
            if source_type == "rss":
                return await fetch_rss(source, timeout, user_agent, max_articles)
            elif source_type == "api":
                return await fetch_api(source, timeout, user_agent, max_articles)
            elif source_type == "web":
                return await fetch_web(source, timeout, user_agent, max_articles)
            else:
                log.error("unknown_source_type", source=name, type=source_type)
                return []
        except BharatIntelError as exc:
            log.error(
                "source_fetch_failed",
                source=name,
                type=source_type,
                error=str(exc),
                context=exc.context,
            )
            return []
        except Exception as exc:
            log.error(
                "source_unexpected_error",
                source=name,
                type=source_type,
                error=str(exc),
            )
            return []

    async def run(self) -> list[RawArticle]:
        """
        Execute the full collection pipeline.

        1. Fan out to all enabled sources concurrently
        2. Aggregate all articles
        3. Deduplicate by URL
        4. Return combined list

        Returns:
            List[RawArticle] — all articles from all successful sources.
            May be empty if every source fails.
        """
        if not self._sources:
            log.warning("no_sources_enabled")
            return []

        concurrency = self._get_collector_setting("concurrent_sources", 10)
        semaphore = asyncio.Semaphore(concurrency)

        async def _bounded_fetch(source: dict[str, Any]) -> list[RawArticle]:
            async with semaphore:
                return await self._fetch_source(source)

        log.info("collection_start", source_count=len(self._sources))

        tasks = [_bounded_fetch(source) for source in self._sources]
        results = await asyncio.gather(*tasks)

        # Flatten and deduplicate by URL
        seen_urls: set[str] = set()
        all_articles: list[RawArticle] = []
        duplicates = 0

        for source_articles in results:
            for article in source_articles:
                if not article.url:
                    log.warning("article_missing_url", title=article.title[:60])
                    continue
                if article.url in seen_urls:
                    duplicates += 1
                    continue
                seen_urls.add(article.url)
                all_articles.append(article)

        # Sort by published_at descending (newest first), None dates go last
        from datetime import datetime, timezone
        _epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        all_articles.sort(
            key=lambda a: a.published_at if a.published_at else _epoch,
            reverse=True,
        )

        log.info(
            "collection_complete",
            total_articles=len(all_articles),
            duplicates_removed=duplicates,
            sources_attempted=len(self._sources),
        )

        return all_articles
