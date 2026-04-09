"""
BharatIntel — Ranker Agent

Orchestrates the full ranking pipeline:
  LLM multi-dimensional scoring → composite scoring → diversity reranking.

This replaces the basic single-score ranker in the curator pipeline.

Responsibilities:
  - Initialize LLM scorer, composite ranker with config
  - Run multi-dimensional LLM scoring
  - Combine with heuristic signals
  - Apply diversity-aware reranking
  - Return final ranked article list

Usage:
    from agents.ranker.agent import RankerAgent
    agent = RankerAgent()
    ranked = await agent.run(deduped_articles)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from agents.collector.models import RawArticle
from agents.curator.models import RankedArticle
from agents.ranker.composite import CompositeRanker
from agents.ranker.llm_scorer import LLMScorer
from core.llm_client import LLMClient
from core.logger import get_logger

log = get_logger("ranker")


class RankerAgent:
    """
    Orchestrates multi-dimensional LLM scoring → composite ranking → diversity reranking.

    Args:
        settings_path:  Path to settings.yaml
        prompts_path:   Path to prompts.yaml
        llm_model:      Override for LLM model string
    """

    def __init__(
        self,
        settings_path: str = "config/settings.yaml",
        prompts_path: str = "config/prompts.yaml",
        llm_model: str | None = None,
    ):
        settings = self._load_settings(settings_path)
        ranker_cfg = settings.get("ranker", {})
        curator_cfg = settings.get("curator", {})

        # LLM client with fallback chain
        from core.llm_client import build_fallback_chain, auto_select_model
        model = llm_model or os.environ.get("LLM_RANK_MODEL") or auto_select_model()
        fallbacks = build_fallback_chain(model)
        llm = LLMClient(model=model, fallback_models=fallbacks, temperature=0.1, max_tokens=2048, max_retries=0)

        # Sub-components
        batch_size = ranker_cfg.get("batch_size", 10)
        self._scorer = LLMScorer(llm=llm, prompts_path=prompts_path, batch_size=batch_size)

        # Composite ranker with configured weights + diversity
        weights = ranker_cfg.get("weights")
        diversity_penalty = ranker_cfg.get("diversity_penalty", 0.15)
        min_per_category = ranker_cfg.get("min_per_category", 1)
        self._composite = CompositeRanker(
            weights=weights,
            diversity_penalty=diversity_penalty,
            min_per_category=min_per_category,
        )

        # Output limits
        self._top_n = curator_cfg.get("top_n_articles", 25)
        self._min_score = curator_cfg.get("min_relevance_score", 4)

        log.info(
            "ranker_initialized",
            model=model,
            batch_size=batch_size,
            top_n=self._top_n,
            min_score=self._min_score,
            diversity_penalty=diversity_penalty,
        )

    @staticmethod
    def _load_settings(path: str) -> dict[str, Any]:
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Settings config not found: {path}")
        with open(filepath, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    async def run(self, articles: list[RawArticle]) -> list[RankedArticle]:
        """
        Execute the full ranking pipeline.

        Pipeline:
          1. Multi-dimensional LLM scoring (with keyword fallback)
          2. Composite scoring (LLM + recency + source authority + content)
          3. Diversity-aware greedy selection
          4. Filter by minimum composite score
          5. Truncate to top N

        Args:
            articles: Deduplicated raw articles from the curator dedup step

        Returns:
            List[RankedArticle] in final rank order.
        """
        if not articles:
            log.warning("ranker_empty_input")
            return []

        log.info("ranker_start", input_articles=len(articles))

        # ── Step 1: Multi-dimensional scoring ────────────────────────
        # Skip LLM scoring if SKIP_LLM_RANKING=1 (saves API quota for summarization)
        skip_llm = os.environ.get("SKIP_LLM_RANKING", "0") == "1"
        llm_scores = await self._scorer.score(articles, skip_llm=skip_llm)
        log.info("ranker_scoring_done", scored=len(llm_scores))

        # ── Steps 2-3: Composite + diversity reranking ───────────────
        ranked = self._composite.rank(articles, llm_scores, top_n=self._top_n)
        log.info("ranker_composite_done", ranked=len(ranked))

        # ── Step 4: Filter by minimum score ──────────────────────────
        filtered = [r for r in ranked if r.relevance_score >= self._min_score]

        if len(filtered) < len(ranked):
            log.info(
                "ranker_filtered",
                before=len(ranked),
                after=len(filtered),
                min_score=self._min_score,
            )

        # ── Log summary ──────────────────────────────────────────────
        category_dist: dict[str, int] = {}
        for r in filtered:
            category_dist[r.assigned_category] = category_dist.get(r.assigned_category, 0) + 1

        log.info(
            "ranker_complete",
            output_articles=len(filtered),
            category_distribution=category_dist,
            score_range=(
                f"{filtered[-1].relevance_score}-{filtered[0].relevance_score}"
                if filtered else "n/a"
            ),
            llm_usage=self._scorer._llm.usage_summary,
        )

        return filtered
