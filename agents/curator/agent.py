"""
BharatIntel — Curator Agent

Orchestrates the classification pipeline: dedup → rank → output.
Delegates ranking to the dedicated Ranker module for multi-dimensional
LLM scoring, composite ranking, and diversity-aware selection.

Usage:
    from agents.curator.agent import CuratorAgent
    agent = CuratorAgent()
    ranked = await agent.run(raw_articles)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from agents.collector.models import RawArticle
from agents.curator.deduplicator import Deduplicator
from agents.curator.models import RankedArticle
from agents.ranker.agent import RankerAgent
from core.logger import get_logger

log = get_logger("curator")


class CuratorAgent:
    """
    Orchestrates dedup → rank (via RankerAgent) for incoming articles.

    Args:
        settings_path:  Path to settings.yaml
        prompts_path:   Path to prompts.yaml
        llm_model:      litellm model string override
    """

    def __init__(
        self,
        settings_path: str = "config/settings.yaml",
        prompts_path: str = "config/prompts.yaml",
        llm_model: str | None = None,
    ):
        self._settings = self._load_settings(settings_path)
        curator_cfg = self._settings.get("curator", {})

        # Deduplicator
        similarity_threshold = curator_cfg.get("dedup_similarity_threshold", 0.85)
        self._deduplicator = Deduplicator(similarity_threshold=similarity_threshold)

        # Ranker (handles LLM scoring + composite + diversity)
        self._ranker = RankerAgent(
            settings_path=settings_path,
            prompts_path=prompts_path,
            llm_model=llm_model,
        )

        log.info(
            "curator_initialized",
            similarity_threshold=similarity_threshold,
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
        Execute the full classification pipeline.

        Pipeline:
          1. Semantic deduplication
          2. Multi-dimensional LLM ranking (via RankerAgent)
             - LLM scoring on 4 dimensions
             - Heuristic signal computation
             - Composite scoring with configurable weights
             - Diversity-aware greedy reranking
             - Min score filtering + top N truncation

        Args:
            articles: Raw articles from collector agent

        Returns:
            List[RankedArticle] in final rank order.
        """
        if not articles:
            log.warning("curator_empty_input")
            return []

        log.info("curator_start", input_articles=len(articles))

        # ── Step 1: Dedup ────────────────────────────────────────────
        deduped = self._deduplicator.deduplicate(articles)
        log.info("curator_dedup_done", before=len(articles), after=len(deduped))

        if not deduped:
            log.warning("curator_all_duplicates")
            return []

        # ── Step 1.5: Pre-filter to cap article count before LLM ranking ─
        max_to_rank = 30
        if len(deduped) > max_to_rank:
            log.info("curator_pre_filter", before=len(deduped), cap=max_to_rank)
            # Keep the first max_to_rank articles (sources already sorted by priority)
            deduped = deduped[:max_to_rank]

        # ── Step 2: Rank via dedicated ranker module ─────────────────
        ranked = await self._ranker.run(deduped)
        log.info("curator_complete", output_articles=len(ranked))

        return ranked
