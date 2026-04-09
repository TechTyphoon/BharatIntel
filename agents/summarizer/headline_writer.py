"""
BharatIntel — Headline Writer

Generates the top N headlines for the briefing masthead using LLM.

Responsibilities:
  - Select the most significant articles for headline treatment
  - Call LLM to rewrite headlines as punchy, concise lines
  - Parse JSON response into Headline instances
  - Fall back to raw article titles if LLM fails

Dependencies: core.llm_client, config/prompts.yaml
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from agents.curator.models import RankedArticle
from agents.summarizer.models import Headline
from core.exceptions import LLMError, LLMResponseError
from core.llm_client import LLMClient
from core.logger import get_logger

log = get_logger("summarizer.headline")


class HeadlineWriter:
    """
    Generates the top headlines for the briefing masthead.

    Args:
        llm:          LLMClient instance (quality model recommended)
        prompts_path: Path to prompts.yaml
    """

    def __init__(
        self,
        llm: LLMClient,
        prompts_path: str = "config/prompts.yaml",
    ):
        self._llm = llm
        self._prompts = self._load_prompts(prompts_path)

    @staticmethod
    def _load_prompts(path: str) -> dict[str, Any]:
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Prompts config not found: {path}")
        with open(filepath, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _format_articles_block(self, articles: list[RankedArticle]) -> str:
        """Build a numbered article block for the headline prompt."""
        lines: list[str] = []
        for idx, article in enumerate(articles, 1):
            snippet = article.snippet[:200] if article.snippet else ""
            lines.append(
                f"[{idx}] Title: {article.title}\n"
                f"    Source: {article.source_name}\n"
                f"    URL: {article.url}\n"
                f"    Snippet: {snippet}"
            )
        return "\n\n".join(lines)

    def _build_fallback_headlines(self, articles: list[RankedArticle], count: int) -> list[Headline]:
        """
        Fallback when LLM fails: use raw article titles as headlines.
        """
        headlines: list[Headline] = []
        for article in articles[:count]:
            snippet_line = article.snippet[:120] if article.snippet else article.rank_reason
            try:
                headline = Headline(
                    title=article.title,
                    oneliner=snippet_line,
                    url=article.url,
                    source=article.source_name,
                )
                headlines.append(headline)
            except ValueError:
                continue
        return headlines

    def _parse_headlines_response(
        self,
        data: dict[str, Any],
        articles: list[RankedArticle],
    ) -> list[Headline]:
        """
        Parse LLM JSON response into Headline objects.

        Tolerates missing fields gracefully — skips malformed entries.
        """
        raw_headlines = data.get("headlines", [])
        if not isinstance(raw_headlines, list):
            log.warning("headline_bad_type", type=type(raw_headlines).__name__)
            return []

        # Build a url→article lookup for source enrichment
        url_to_article = {a.url: a for a in articles}

        result: list[Headline] = []
        for entry in raw_headlines:
            try:
                title = str(entry.get("title", "")).strip()
                oneliner = str(entry.get("oneliner", "")).strip()
                url = str(entry.get("url", "")).strip()
                source = str(entry.get("source", "")).strip()

                if not title or not url:
                    log.debug("headline_skip_entry", reason="missing title or url")
                    continue

                # Validate URL exists in our article set (prevent hallucination)
                if url not in url_to_article:
                    # Try to find the closest match by title substring
                    matched = False
                    for a in articles:
                        if a.title[:30].lower() in title.lower() or title[:30].lower() in a.title.lower():
                            url = a.url
                            source = source or a.source_name
                            matched = True
                            break
                    if not matched:
                        log.debug("headline_skip_hallucinated_url", url=url[:100])
                        continue

                if not source:
                    source = url_to_article.get(url, articles[0]).source_name if articles else ""

                headline = Headline(
                    title=title,
                    oneliner=oneliner,
                    url=url,
                    source=source,
                )
                result.append(headline)
            except (ValueError, TypeError, KeyError) as exc:
                log.warning("headline_parse_error", entry=entry, error=str(exc))
                continue

        return result

    async def write_headlines(
        self,
        articles: list[RankedArticle],
        count: int = 5,
    ) -> list[Headline]:
        """
        Generate top N headlines from the highest-ranked articles.

        Strategy:
          1. Take top articles (2x count for LLM selection pool)
          2. Call LLM to rewrite and select the best N
          3. Parse and validate
          4. Fall back to raw titles on LLM failure

        Args:
            articles: Ranked articles sorted by score descending
            count:    Number of headlines to produce

        Returns:
            List[Headline] of length <= count.
        """
        if not articles:
            log.warning("headline_empty_input")
            return []

        # Give LLM a pool of 2x to choose from
        pool_size = min(count * 2, len(articles))
        pool = articles[:pool_size]

        log.info("headline_start", pool_size=pool_size, target_count=count)

        articles_block = self._format_articles_block(pool)

        prompt_cfg = self._prompts.get("write_headlines_v1", {})
        system = prompt_cfg.get("system", "")
        user_template = prompt_cfg.get("user", "")
        user_prompt = user_template.format(
            count=count,
            articles_block=articles_block,
        )

        headlines: list[Headline] = []

        try:
            data = await self._llm.complete_json(
                prompt=user_prompt,
                system=system,
                temperature=0.4,
                max_tokens=1024,
            )
            headlines = self._parse_headlines_response(data, pool)
        except LLMResponseError as exc:
            log.warning("headline_llm_bad_response", error=str(exc))
        except LLMError as exc:
            log.error("headline_llm_error", error=str(exc))

        # Fallback if LLM produced nothing
        if not headlines:
            headlines = self._build_fallback_headlines(pool, count)
            log.warning("headline_using_fallback", count=len(headlines))

        # Ensure we don't exceed requested count
        headlines = headlines[:count]

        log.info("headline_complete", count=len(headlines))
        return headlines
