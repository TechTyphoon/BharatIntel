"""
BharatIntel — Summarizer Agent

Orchestrates the full summarization pipeline:
  group → write sections → write headlines → executive summary → assemble.

This is the single entry point for the summarization module.

Responsibilities:
  - Group ranked articles by assigned_category
  - Cap sections at max_sections (drop lowest-scoring categories)
  - Dispatch section writing to SectionWriter (one LLM call per category)
  - Dispatch headline writing to HeadlineWriter (one LLM call)
  - Dispatch executive summary writing to ExecutiveWriter (one LLM call)
  - Track token budget and enforce limits
  - Quality gates on all LLM outputs
  - Assemble complete Briefing object

Usage:
    from agents.summarizer.agent import SummarizerAgent
    agent = SummarizerAgent()
    briefing = await agent.run(ranked_articles)
"""

from __future__ import annotations

import asyncio
import os
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from agents.curator.models import RankedArticle
from agents.summarizer.executive_writer import ExecutiveWriter
from agents.summarizer.headline_writer import HeadlineWriter
from agents.summarizer.models import Briefing, BriefingSection, Headline
from agents.summarizer.section_writer import SectionWriter
from core.llm_client import LLMClient
from core.logger import get_logger

log = get_logger("summarizer")

# Category display order (preferred ordering in final briefing)
# Matches the 6 required assignment categories
_CATEGORY_ORDER = [
    "geopolitics", "technology", "indian_politics",
    "science", "civilisation", "editors_picks",
]


class SummarizerAgent:
    """
    Orchestrates article grouping → sections → headlines → executive summary → briefing.

    Args:
        settings_path:  Path to settings.yaml
        prompts_path:   Path to prompts.yaml
        llm_model:      litellm model string override (quality model recommended)
    """

    def __init__(
        self,
        settings_path: str = "config/settings.yaml",
        prompts_path: str = "config/prompts.yaml",
        llm_model: str | None = None,
    ):
        self._settings = self._load_settings(settings_path)
        self._prompts_path = prompts_path
        summarizer_cfg = self._settings.get("summarizer", {})

        self._max_sections = summarizer_cfg.get("max_sections", 8)
        self._top_headlines = summarizer_cfg.get("top_headlines", 5)
        self._token_budget = summarizer_cfg.get("token_budget", 50000)
        self._min_summary_length = summarizer_cfg.get("min_summary_length", 80)
        self._takeaways_per_section = summarizer_cfg.get("takeaways_per_section", 3)

        # LLM client (quality model for summarization) with fallback chain
        from core.llm_client import build_fallback_chain, auto_select_model
        model = llm_model or os.environ.get("LLM_SUMMARY_MODEL") or auto_select_model()
        fallbacks = build_fallback_chain(model)
        self._llm = LLMClient(model=model, fallback_models=fallbacks, temperature=0.3, max_tokens=1536, max_retries=0)

        self._section_writer = SectionWriter(
            llm=self._llm,
            prompts_path=prompts_path,
            min_summary_length=self._min_summary_length,
        )
        self._headline_writer = HeadlineWriter(llm=self._llm, prompts_path=prompts_path)
        self._executive_writer = ExecutiveWriter(llm=self._llm, prompts_path=prompts_path)

        log.info(
            "summarizer_initialized",
            model=model,
            max_sections=self._max_sections,
            top_headlines=self._top_headlines,
            token_budget=self._token_budget,
        )

    @staticmethod
    def _load_settings(path: str) -> dict[str, Any]:
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Settings config not found: {path}")
        with open(filepath, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _group_by_category(
        self,
        articles: list[RankedArticle],
    ) -> dict[str, list[RankedArticle]]:
        """
        Group articles by assigned_category.

        Within each group, articles remain sorted by relevance_score descending.
        """
        groups: dict[str, list[RankedArticle]] = defaultdict(list)
        for article in articles:
            groups[article.assigned_category.lower()].append(article)
        return dict(groups)

    def _select_top_categories(
        self,
        groups: dict[str, list[RankedArticle]],
    ) -> list[str]:
        """
        Select up to max_sections categories, prioritizing by:
          1. Total relevance score in the category (sum of article scores)
          2. Breaking ties by preferred category order

        Returns category names in display order.
        """
        category_scores: dict[str, int] = {}
        for cat, articles in groups.items():
            category_scores[cat] = sum(a.relevance_score for a in articles)

        def sort_key(cat: str) -> tuple[int, int]:
            order_idx = _CATEGORY_ORDER.index(cat) if cat in _CATEGORY_ORDER else len(_CATEGORY_ORDER)
            return (-category_scores[cat], order_idx)

        sorted_cats = sorted(groups.keys(), key=sort_key)
        selected = sorted_cats[: self._max_sections]

        def display_key(cat: str) -> int:
            return _CATEGORY_ORDER.index(cat) if cat in _CATEGORY_ORDER else len(_CATEGORY_ORDER)

        selected.sort(key=display_key)
        return selected

    def _check_token_budget(self) -> bool:
        """Return True if remaining budget allows more LLM calls."""
        usage = self._llm.usage_summary
        total_used = usage["total_prompt_tokens"] + usage["total_completion_tokens"]
        remaining = self._token_budget - total_used
        if remaining <= 0:
            log.warning(
                "token_budget_exhausted",
                budget=self._token_budget,
                used=total_used,
            )
            return False
        return True

    async def run(self, articles: list[RankedArticle]) -> Briefing:
        """
        Execute the full summarization pipeline.

        Pipeline:
          1. Group articles by category
          2. Select top N categories by aggregate score
          3. Write section summaries + takeaways (sequential LLM calls)
          4. Write top headlines (single LLM call)
          5. Write executive summary from sections + headlines
          6. Assemble Briefing with token usage report

        Token budget is checked between steps; remaining steps are skipped
        if budget is exhausted (partial briefing is returned).

        Args:
            articles: Ranked articles from curator, sorted by score descending

        Returns:
            Briefing with executive_summary, headlines, and sections. Never None.
        """
        today = date.today().isoformat()

        if not articles:
            log.warning("summarizer_empty_input")
            return Briefing(
                date=today,
                headlines=[],
                sections=[
                    BriefingSection(
                        category="General",
                        summary="No significant news collected for today's briefing.",
                    )
                ],
            )

        log.info("summarizer_start", input_articles=len(articles), date=today)

        # ── Step 1: Group by category ────────────────────────────────
        groups = self._group_by_category(articles)
        log.info(
            "summarizer_grouped",
            categories=list(groups.keys()),
            distribution={k: len(v) for k, v in groups.items()},
        )

        # ── Step 2: Select top categories ────────────────────────────
        selected_categories = self._select_top_categories(groups)
        log.info("summarizer_categories_selected", categories=selected_categories)

        # ── Step 3: Write sections (sequential to avoid rate limits) ─
        sections: list[BriefingSection] = []
        for idx, category in enumerate(selected_categories):
            if not self._check_token_budget():
                log.warning(
                    "sections_truncated_by_budget",
                    completed=len(sections),
                    total=len(selected_categories),
                )
                break
            # Rate-limit delay between LLM calls (avoid provider throttling)
            if idx > 0:
                await asyncio.sleep(15)
            cat_articles = groups[category]
            section = await self._section_writer.write_section(category, cat_articles)
            sections.append(section)

        log.info("summarizer_sections_done", count=len(sections))

        # ── Step 4: Write headlines ──────────────────────────────────
        headlines: list[Headline] = []
        if self._check_token_budget():
            await asyncio.sleep(15)
            headlines = await self._headline_writer.write_headlines(
                articles, count=self._top_headlines
            )
        else:
            log.warning("headlines_skipped_budget")
        log.info("summarizer_headlines_done", count=len(headlines))

        # ── Step 5: Write executive summary ──────────────────────────
        executive_summary = None
        if self._check_token_budget() and (sections or headlines):
            await asyncio.sleep(15)
            executive_summary = await self._executive_writer.write_executive_summary(
                sections, headlines
            )
        else:
            log.warning("executive_summary_skipped",
                        reason="budget" if not self._check_token_budget() else "no_content")
        log.info("summarizer_executive_done", has_executive=executive_summary is not None)

        # ── Step 6: Assemble ─────────────────────────────────────────
        token_usage = self._llm.usage_summary
        briefing = Briefing(
            date=today,
            executive_summary=executive_summary,
            headlines=headlines,
            sections=sections,
            token_usage=token_usage,
        )

        log.info(
            "summarizer_complete",
            date=today,
            headline_count=len(headlines),
            section_count=len(sections),
            has_executive_summary=executive_summary is not None,
            llm_usage=token_usage,
        )

        return briefing
