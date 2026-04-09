"""
BharatIntel — LLM-Based Article Ranker

Scores articles for relevance using an LLM via batch prompting.

Responsibilities:
  - Format articles into batch prompt from prompts.yaml template
  - Call LLM and parse structured JSON scores
  - Map LLM output back to RankedArticle instances
  - Fallback: keyword-based scoring if LLM fails completely
  - Handles partial LLM responses (some articles scored, some not)

Dependencies: core.llm_client, pyyaml
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import yaml

from agents.collector.models import RawArticle
from agents.curator.models import RankedArticle
from core.exceptions import LLMError, LLMResponseError
from core.llm_client import LLMClient
from core.logger import get_logger

log = get_logger("curator.ranker")

VALID_CATEGORIES = {"world", "politics", "business", "technology", "science", "india", "general"}

# Keyword fallback when LLM is unavailable
_KEYWORD_SCORES: dict[str, int] = {
    "breaking": 8, "war": 8, "crisis": 8, "election": 7,
    "gdp": 7, "recession": 7, "ai": 6, "climate": 6,
    "government": 6, "policy": 6, "summit": 6, "trade": 5,
    "market": 5, "startup": 4, "launch": 4, "update": 3,
}

# Max articles per LLM batch (balances token cost vs. accuracy)
BATCH_SIZE = 10


class Ranker:
    """
    Scores and categorizes articles using LLM batch prompting.

    Args:
        llm:          LLMClient instance (fast/cheap model recommended)
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

    def _format_articles_block(self, articles: list[RawArticle]) -> str:
        """Format a batch of articles for the LLM prompt."""
        lines: list[str] = []
        for idx, article in enumerate(articles):
            categories_str = ", ".join(article.categories) if article.categories else "general"
            snippet = article.snippet[:200] if article.snippet else "(no snippet)"
            lines.append(
                f"[{idx}] Title: {article.title}\n"
                f"    Source: {article.source_name}\n"
                f"    Snippet: {snippet}\n"
                f"    Categories: {categories_str}"
            )
        return "\n\n".join(lines)

    def _parse_batch_response(
        self,
        data: dict[str, Any],
        articles: list[RawArticle],
    ) -> list[RankedArticle]:
        """
        Parse LLM JSON response into RankedArticle list.

        Tolerates missing indices and invalid scores gracefully.
        """
        rankings = data.get("rankings", [])
        if not isinstance(rankings, list):
            log.warning("ranker_bad_rankings_type", type=type(rankings).__name__)
            return []

        result: list[RankedArticle] = []
        for entry in rankings:
            try:
                idx = int(entry.get("index", -1))
                if idx < 0 or idx >= len(articles):
                    log.debug("ranker_skip_bad_index", index=idx)
                    continue

                score = int(entry.get("score", 0))
                score = max(1, min(10, score))  # Clamp to 1-10

                category = str(entry.get("category", "general")).lower().strip()
                if category not in VALID_CATEGORIES:
                    category = "general"

                reason = str(entry.get("reason", ""))[:200]
                article = articles[idx]

                ranked = RankedArticle(
                    title=article.title,
                    url=article.url,
                    source_name=article.source_name,
                    relevance_score=score,
                    assigned_category=category,
                    rank_reason=reason,
                    published_at=article.published_at,
                    snippet=article.snippet,
                    author=article.author,
                    image_url=article.image_url,
                    fetched_at=article.fetched_at,
                    original_categories=list(article.categories),
                )
                result.append(ranked)
            except (ValueError, TypeError, KeyError) as exc:
                log.warning("ranker_parse_entry_error", entry=entry, error=str(exc))
                continue

        return result

    async def _rank_batch(self, articles: list[RawArticle]) -> list[RankedArticle]:
        """
        Score a single batch of articles via LLM.

        Returns whatever articles were successfully scored (may be partial).
        """
        prompt_cfg = self._prompts.get("rank_batch_v1", {})
        system = prompt_cfg.get("system", "")
        user_template = prompt_cfg.get("user", "")

        articles_block = self._format_articles_block(articles)
        user_prompt = user_template.format(articles_block=articles_block)

        try:
            data = await self._llm.complete_json(
                prompt=user_prompt,
                system=system,
                temperature=0.1,
                max_tokens=2048,
            )
        except LLMResponseError as exc:
            log.warning("ranker_llm_bad_json", error=str(exc))
            return []
        except LLMError as exc:
            log.error("ranker_llm_error", error=str(exc))
            return []

        ranked = self._parse_batch_response(data, articles)
        log.debug("ranker_batch_done", input=len(articles), scored=len(ranked))
        return ranked

    def _keyword_fallback_score(self, article: RawArticle) -> int:
        """
        Simple keyword-based relevance score as LLM fallback.

        Scans title + snippet for high-signal keywords and returns max match.
        """
        text = f"{article.title} {article.snippet}".lower()
        max_score = 3  # Default baseline
        for keyword, score in _KEYWORD_SCORES.items():
            if keyword in text:
                max_score = max(max_score, score)
        return max_score

    def _keyword_fallback(self, articles: list[RawArticle]) -> list[RankedArticle]:
        """Fallback: score all articles using keyword matching."""
        log.warning("ranker_using_keyword_fallback", article_count=len(articles))
        result: list[RankedArticle] = []
        for article in articles:
            score = self._keyword_fallback_score(article)
            category = article.categories[0] if article.categories else "general"
            if category not in VALID_CATEGORIES:
                category = "general"

            ranked = RankedArticle(
                title=article.title,
                url=article.url,
                source_name=article.source_name,
                relevance_score=score,
                assigned_category=category,
                rank_reason="Scored by keyword fallback",
                published_at=article.published_at,
                snippet=article.snippet,
                author=article.author,
                image_url=article.image_url,
                fetched_at=article.fetched_at,
                original_categories=list(article.categories),
            )
            result.append(ranked)
        return result

    async def rank(self, articles: list[RawArticle]) -> list[RankedArticle]:
        """
        Score all articles using LLM batch prompting.

        Strategy:
          1. Split into batches of BATCH_SIZE
          2. Send batches sequentially (avoids rate limit pressure)
          3. For articles the LLM failed to score, apply keyword fallback
          4. Return all scored articles

        Args:
            articles: Deduplicated articles from collector

        Returns:
            List[RankedArticle] — every input article will appear in output with a score.
        """
        if not articles:
            return []

        log.info("ranker_start", total_articles=len(articles))

        # Split into batches
        batches: list[list[RawArticle]] = []
        for i in range(0, len(articles), BATCH_SIZE):
            batches.append(articles[i : i + BATCH_SIZE])

        all_ranked: list[RankedArticle] = []
        scored_urls: set[str] = set()

        for batch_idx, batch in enumerate(batches):
            log.debug("ranker_batch_start", batch=batch_idx + 1, total_batches=len(batches))
            ranked = await self._rank_batch(batch)
            for r in ranked:
                scored_urls.add(r.url)
                all_ranked.append(r)

            # Small delay between batches to be kind to APIs
            if batch_idx < len(batches) - 1:
                await asyncio.sleep(0.5)

        # Fallback for any articles the LLM missed
        missed = [a for a in articles if a.url not in scored_urls]
        if missed:
            log.warning("ranker_missed_articles", count=len(missed))
            fallback_ranked = self._keyword_fallback(missed)
            all_ranked.extend(fallback_ranked)

        log.info(
            "ranker_complete",
            total_scored=len(all_ranked),
            llm_scored=len(scored_urls),
            fallback_scored=len(missed),
            llm_usage=self._llm.usage_summary,
        )

        return all_ranked
