"""
BharatIntel — Multi-Dimensional LLM Scorer

Scores articles across multiple quality dimensions using LLM batch prompting.

Responsibilities:
  - Format articles into batched multi-dimensional scoring prompts
  - Call LLM for structured JSON with per-dimension scores
  - Parse and validate multi-signal response
  - Keyword fallback when LLM is completely unavailable
  - Handle partial responses (some articles scored, some not)

Output per article:
  relevance  (1-10): How relevant to a daily intelligence briefing
  impact     (1-10): Breadth and depth of real-world impact
  novelty    (1-10): New information vs. routine / expected
  timeliness (1-10): Breaking/developing vs. stale

Dependencies: core.llm_client, pyyaml
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from agents.collector.models import RawArticle
from core.exceptions import LLMError, LLMResponseError
from core.llm_client import LLMClient
from core.logger import get_logger

log = get_logger("ranker.llm_scorer")

VALID_CATEGORIES = {"geopolitics", "technology", "indian_politics", "science", "civilisation", "editors_picks"}

BATCH_SIZE = 10


@dataclass(frozen=True)
class ArticleScores:
    """
    Multi-dimensional LLM scores for a single article.

    All scores are integers in [1, 10].
    """

    article_index: int
    relevance: int
    impact: int
    novelty: int
    timeliness: int
    category: str
    reason: str

    @property
    def dimensions(self) -> dict[str, int]:
        return {
            "relevance": self.relevance,
            "impact": self.impact,
            "novelty": self.novelty,
            "timeliness": self.timeliness,
        }


# ── Keyword fallback signals ────────────────────────────────────────

_KEYWORD_SIGNALS: dict[str, dict[str, int]] = {
    "breaking":   {"relevance": 8, "impact": 7, "novelty": 8, "timeliness": 9},
    "war":        {"relevance": 9, "impact": 9, "novelty": 5, "timeliness": 7},
    "crisis":     {"relevance": 8, "impact": 8, "novelty": 6, "timeliness": 7},
    "election":   {"relevance": 8, "impact": 7, "novelty": 5, "timeliness": 6},
    "recession":  {"relevance": 7, "impact": 8, "novelty": 5, "timeliness": 5},
    "gdp":        {"relevance": 6, "impact": 7, "novelty": 4, "timeliness": 5},
    "ai":         {"relevance": 6, "impact": 6, "novelty": 6, "timeliness": 5},
    "climate":    {"relevance": 6, "impact": 7, "novelty": 4, "timeliness": 4},
    "government": {"relevance": 6, "impact": 5, "novelty": 3, "timeliness": 4},
    "summit":     {"relevance": 6, "impact": 5, "novelty": 5, "timeliness": 6},
    "trade":      {"relevance": 5, "impact": 5, "novelty": 3, "timeliness": 4},
    "startup":    {"relevance": 4, "impact": 3, "novelty": 5, "timeliness": 4},
    "launch":     {"relevance": 4, "impact": 3, "novelty": 5, "timeliness": 5},
    "update":     {"relevance": 3, "impact": 2, "novelty": 2, "timeliness": 3},
}

_DEFAULT_FALLBACK = {"relevance": 3, "impact": 3, "novelty": 3, "timeliness": 3}


class LLMScorer:
    """
    Scores articles on multiple quality dimensions via LLM batch prompting.

    Args:
        llm:          LLMClient instance (fast/cheap model recommended)
        prompts_path: Path to prompts.yaml
        batch_size:   Articles per LLM call
    """

    def __init__(
        self,
        llm: LLMClient,
        prompts_path: str = "config/prompts.yaml",
        batch_size: int = BATCH_SIZE,
    ):
        self._llm = llm
        self._batch_size = batch_size
        self._prompts = self._load_prompts(prompts_path)

    @staticmethod
    def _load_prompts(path: str) -> dict[str, Any]:
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Prompts config not found: {path}")
        with open(filepath, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _format_articles_block(self, articles: list[RawArticle]) -> str:
        lines: list[str] = []
        for idx, article in enumerate(articles):
            categories_str = ", ".join(article.categories) if article.categories else "general"
            snippet = article.snippet[:250] if article.snippet else "(no snippet)"
            pub = article.published_at.isoformat() if article.published_at else "unknown"
            lines.append(
                f"[{idx}] Title: {article.title}\n"
                f"    Source: {article.source_name}\n"
                f"    Published: {pub}\n"
                f"    Snippet: {snippet}\n"
                f"    Categories: {categories_str}"
            )
        return "\n\n".join(lines)

    @staticmethod
    def _clamp(val: Any, lo: int = 1, hi: int = 10) -> int:
        try:
            return max(lo, min(hi, int(val)))
        except (ValueError, TypeError):
            return lo

    def _parse_batch_response(
        self,
        data: dict[str, Any],
        articles: list[RawArticle],
    ) -> list[ArticleScores]:
        """Parse the multi-dimensional LLM response. Tolerates partial/malformed data."""
        rankings = data.get("rankings", [])
        if not isinstance(rankings, list):
            log.warning("scorer_bad_rankings_type", type=type(rankings).__name__)
            return []

        result: list[ArticleScores] = []
        for entry in rankings:
            try:
                idx = int(entry.get("index", -1))
                if idx < 0 or idx >= len(articles):
                    continue

                category = str(entry.get("category", "general")).lower().strip()
                if category not in VALID_CATEGORIES:
                    category = "editors_picks"

                scores = ArticleScores(
                    article_index=idx,
                    relevance=self._clamp(entry.get("relevance", 3)),
                    impact=self._clamp(entry.get("impact", 3)),
                    novelty=self._clamp(entry.get("novelty", 3)),
                    timeliness=self._clamp(entry.get("timeliness", 3)),
                    category=category,
                    reason=str(entry.get("reason", ""))[:200],
                )
                result.append(scores)
            except (ValueError, TypeError, KeyError) as exc:
                log.warning("scorer_parse_entry_error", entry=entry, error=str(exc))
                continue

        return result

    async def _score_batch(self, articles: list[RawArticle]) -> list[ArticleScores]:
        """Score a single batch via LLM. Returns partial results on partial failure."""
        prompt_cfg = self._prompts.get("rank_multidim_v1", {})
        system = prompt_cfg.get("system", "")
        user_template = prompt_cfg.get("user", "")

        articles_block = self._format_articles_block(articles)
        user_prompt = user_template.format(articles_block=articles_block)

        try:
            data = await self._llm.complete_json(
                prompt=user_prompt,
                system=system,
                temperature=0.1,
                max_tokens=4096,
            )
        except LLMResponseError as exc:
            log.warning("scorer_llm_bad_json", error=str(exc), context=getattr(exc, 'context', {}))
            return []
        except LLMError as exc:
            log.error("scorer_llm_error", error=str(exc))
            return []

        scores = self._parse_batch_response(data, articles)
        log.debug("scorer_batch_done", input=len(articles), scored=len(scores))
        return scores

    def _keyword_fallback_single(self, article: RawArticle, index: int) -> ArticleScores:
        """Compute keyword-based multi-dimensional scores as fallback."""
        text = f"{article.title} {article.snippet}".lower()
        best = dict(_DEFAULT_FALLBACK)

        for keyword, signals in _KEYWORD_SIGNALS.items():
            if keyword in text:
                for dim, val in signals.items():
                    best[dim] = max(best[dim], val)

        category = article.categories[0] if article.categories else "editors_picks"
        if category not in VALID_CATEGORIES:
            category = "editors_picks"

        return ArticleScores(
            article_index=index,
            relevance=best["relevance"],
            impact=best["impact"],
            novelty=best["novelty"],
            timeliness=best["timeliness"],
            category=category,
            reason="Scored by keyword fallback",
        )

    async def score(self, articles: list[RawArticle], skip_llm: bool = False) -> dict[int, ArticleScores]:
        """
        Score all articles using LLM with keyword fallback for failures.

        Args:
            articles: Deduplicated raw articles (order = index reference)
            skip_llm: If True, use keyword fallback for all articles (saves API quota)

        Returns:
            Dict mapping article index → ArticleScores.
            Every input article is guaranteed to have an entry.
        """
        if not articles:
            return {}

        log.info("scorer_start", total_articles=len(articles), skip_llm=skip_llm)

        if skip_llm:
            scored: dict[int, ArticleScores] = {}
            for i, article in enumerate(articles):
                scored[i] = self._keyword_fallback_single(article, i)
            log.info("scorer_complete", total=len(articles), llm_scored=0, fallback_scored=len(articles))
            return scored

        # Split into batches
        batches: list[list[RawArticle]] = []
        batch_offsets: list[int] = []
        for i in range(0, len(articles), self._batch_size):
            batches.append(articles[i: i + self._batch_size])
            batch_offsets.append(i)

        scored: dict[int, ArticleScores] = {}

        for batch_idx, (batch, offset) in enumerate(zip(batches, batch_offsets)):
            log.debug("scorer_batch_start", batch=batch_idx + 1, total=len(batches))
            batch_scores = await self._score_batch(batch)

            for s in batch_scores:
                # Map batch-local index to global index
                global_idx = offset + s.article_index
                scored[global_idx] = ArticleScores(
                    article_index=global_idx,
                    relevance=s.relevance,
                    impact=s.impact,
                    novelty=s.novelty,
                    timeliness=s.timeliness,
                    category=s.category,
                    reason=s.reason,
                )

            if batch_idx < len(batches) - 1:
                await asyncio.sleep(15)

        # Fill missing with keyword fallback
        missed = 0
        for i, article in enumerate(articles):
            if i not in scored:
                scored[i] = self._keyword_fallback_single(article, i)
                missed += 1

        if missed:
            log.warning("scorer_fallback_used", count=missed)

        log.info(
            "scorer_complete",
            total=len(scored),
            llm_scored=len(scored) - missed,
            fallback_scored=missed,
        )

        return scored
