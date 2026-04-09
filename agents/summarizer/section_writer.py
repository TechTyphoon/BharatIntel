"""
BharatIntel — Section Writer

Generates a narrative summary + key takeaways for a single briefing category.

Responsibilities:
  - Format ranked articles in a category into an LLM prompt
  - Call LLM to produce a 3-5 sentence synthesis + 2-4 key takeaways
  - Validate output quality (length, takeaway count)
  - Parse JSON response into BriefingSection
  - Fall back to concatenated snippets if LLM fails

Dependencies: core.llm_client, config/prompts.yaml
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import yaml

from agents.curator.models import RankedArticle
from agents.summarizer.models import BriefingSection
from core.exceptions import LLMError, LLMResponseError
from core.llm_client import LLMClient
from core.logger import get_logger

log = get_logger("summarizer.section")

# ── Quality thresholds ───────────────────────────────────────────────
_MIN_SUMMARY_CHARS = 80       # Reject summaries shorter than this
_MAX_SUMMARY_CHARS = 3000     # Truncate summaries longer than this
_MIN_TAKEAWAYS = 2
_MAX_TAKEAWAYS = 5


class SectionWriter:
    """
    Generates LLM-written summaries + key takeaways per briefing section.

    Args:
        llm:                LLMClient instance (quality model recommended)
        prompts_path:       Path to prompts.yaml
        min_summary_length: Override minimum summary char count for quality gate
    """

    def __init__(
        self,
        llm: LLMClient,
        prompts_path: str = "config/prompts.yaml",
        min_summary_length: int = _MIN_SUMMARY_CHARS,
    ):
        self._llm = llm
        self._prompts = self._load_prompts(prompts_path)
        self._min_summary_length = min_summary_length

    @staticmethod
    def _load_prompts(path: str) -> dict[str, Any]:
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Prompts config not found: {path}")
        with open(filepath, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _format_articles_block(self, articles: list[RankedArticle]) -> str:
        """Build a numbered article block for the prompt."""
        lines: list[str] = []
        for idx, article in enumerate(articles, 1):
            snippet = article.snippet[:300] if article.snippet else "(no snippet available)"
            lines.append(
                f"[{idx}] {article.title}\n"
                f"    Source: {article.source_name}\n"
                f"    Snippet: {snippet}"
            )
        return "\n\n".join(lines)

    def _build_fallback_summary(self, category: str, articles: list[RankedArticle]) -> str:
        """
        Fallback when LLM fails: concatenate top article titles + snippets.
        """
        parts: list[str] = []
        for article in articles[:5]:
            if article.snippet:
                parts.append(f"{article.title}. {article.snippet[:150]}")
            else:
                parts.append(article.title)
        return " ".join(parts)

    def _build_fallback_takeaways(self, articles: list[RankedArticle]) -> list[str]:
        """Fallback takeaways: first sentence from top article snippets."""
        takeaways: list[str] = []
        for article in articles[:_MAX_TAKEAWAYS]:
            if article.snippet:
                first_sentence = article.snippet.split(". ")[0].strip()
                if first_sentence and not first_sentence.endswith("."):
                    first_sentence += "."
                if first_sentence:
                    takeaways.append(first_sentence)
        return takeaways

    def _validate_summary(self, summary: str, category: str) -> str:
        """
        Quality gate for summary text.

        Checks minimum length and truncates if too long.
        Returns cleaned summary or raises LLMResponseError.
        """
        if not summary or len(summary) < self._min_summary_length:
            raise LLMResponseError(
                f"Summary too short ({len(summary)} chars, min {self._min_summary_length})",
                context={"category": category, "summary_len": len(summary)},
            )
        if len(summary) > _MAX_SUMMARY_CHARS:
            summary = summary[:_MAX_SUMMARY_CHARS].rsplit(". ", 1)[0] + "."
            log.warning(
                "section_summary_truncated",
                category=category,
                original_len=len(summary),
            )
        return summary

    def _validate_takeaways(self, takeaways: Any) -> list[str]:
        """
        Quality gate for takeaways.

        Ensures they are a list of clean strings, capped at MAX_TAKEAWAYS.
        Returns cleaned list (may be empty).
        """
        if not isinstance(takeaways, list):
            return []
        cleaned: list[str] = []
        for item in takeaways:
            text = str(item).strip()
            if text and len(text) > 10:
                cleaned.append(text)
        return cleaned[:_MAX_TAKEAWAYS]

    async def write_section(
        self,
        category: str,
        articles: list[RankedArticle],
    ) -> BriefingSection:
        """
        Generate a single briefing section with summary + key takeaways.

        Strategy:
          1. Try v2 prompt (summary + takeaways in one call)
          2. Validate quality of both outputs
          3. On failure: fall back to v1 prompt (summary only) + mechanical takeaways
          4. On total failure: concatenated snippets as summary

        Args:
            category: Section category (e.g. "technology")
            articles: Ranked articles assigned to this category

        Returns:
            BriefingSection with summary and key_takeaways populated.
        """
        if not articles:
            log.warning("section_empty", category=category)
            return BriefingSection(
                category=category.title(),
                summary=f"No significant {category} news for today's briefing.",
                article_count=0,
            )

        display_category = category.title()
        articles_block = self._format_articles_block(articles)

        summary_text: str | None = None
        takeaways: list[str] = []

        # ── Attempt 1: v2 prompt (summary + takeaways) ──────────────
        prompt_cfg = self._prompts.get("summarize_section_v2")
        if prompt_cfg:
            try:
                summary_text, takeaways = await self._try_v2(
                    prompt_cfg, display_category, articles_block, category
                )
            except (LLMResponseError, LLMError) as exc:
                log.warning("section_v2_failed", category=category, error=str(exc))
                summary_text = None
                takeaways = []

        # ── Attempt 2: v1 prompt (summary only) ─────────────────────
        if not summary_text:
            await asyncio.sleep(15)
            prompt_cfg_v1 = self._prompts.get("summarize_section_v1", {})
            if prompt_cfg_v1:
                try:
                    summary_text = await self._try_v1(
                        prompt_cfg_v1, display_category, articles_block, category
                    )
                except (LLMResponseError, LLMError) as exc:
                    log.warning("section_v1_failed", category=category, error=str(exc))

        # ── Attempt 3: Fallback ──────────────────────────────────────
        if not summary_text:
            summary_text = self._build_fallback_summary(category, articles)
            log.warning("section_using_fallback", category=category)

        if not takeaways:
            takeaways = self._build_fallback_takeaways(articles)

        section = BriefingSection(
            category=display_category,
            summary=summary_text,
            key_takeaways=takeaways,
            article_count=len(articles),
            article_urls=[a.url for a in articles],
            article_titles=[a.title for a in articles],
        )

        log.info(
            "section_written",
            category=display_category,
            article_count=len(articles),
            summary_len=len(summary_text),
            takeaway_count=len(takeaways),
        )

        return section

    async def _try_v2(
        self,
        prompt_cfg: dict,
        display_category: str,
        articles_block: str,
        category: str,
    ) -> tuple[str, list[str]]:
        """
        Attempt v2 prompt: returns both summary and takeaways.

        Raises LLMResponseError or LLMError on failure.
        """
        system = prompt_cfg.get("system", "")
        user_template = prompt_cfg.get("user", "")
        user_prompt = user_template.format(
            category=display_category,
            articles_block=articles_block,
        )

        data = await self._llm.complete_json(
            prompt=user_prompt,
            system=system,
            temperature=0.3,
            max_tokens=1024,
        )

        summary = data.get("summary", "").strip()
        summary = self._validate_summary(summary, category)
        takeaways = self._validate_takeaways(data.get("key_takeaways", []))

        return summary, takeaways

    async def _try_v1(
        self,
        prompt_cfg: dict,
        display_category: str,
        articles_block: str,
        category: str,
    ) -> str:
        """
        Attempt v1 prompt: returns summary only.

        Raises LLMResponseError or LLMError on failure.
        """
        system = prompt_cfg.get("system", "")
        user_template = prompt_cfg.get("user", "")
        user_prompt = user_template.format(
            category=display_category,
            articles_block=articles_block,
        )

        data = await self._llm.complete_json(
            prompt=user_prompt,
            system=system,
            temperature=0.3,
            max_tokens=1024,
        )

        summary = data.get("summary", "").strip()
        return self._validate_summary(summary, category)
