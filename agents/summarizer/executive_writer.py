"""
BharatIntel — Executive Summary Writer

Generates a top-level overview of the entire day's briefing:
  - 2-3 paragraph narrative of the day's most important developments
  - 3-5 key development bullet points across all categories
  - Overall tone assessment

Dependencies: core.llm_client, config/prompts.yaml
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from agents.summarizer.models import BriefingSection, ExecutiveSummary, Headline
from core.exceptions import LLMError, LLMResponseError
from core.llm_client import LLMClient
from core.logger import get_logger

log = get_logger("summarizer.executive")


class ExecutiveWriter:
    """
    Generates a holistic executive summary from completed sections and headlines.

    This runs AFTER section writing, consuming the already-written section
    summaries and headlines to produce a birds-eye overview without
    additional article-level processing.

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

    def _format_sections_block(self, sections: list[BriefingSection]) -> str:
        """Build a section digest block for the executive summary prompt."""
        lines: list[str] = []
        for section in sections:
            takeaways = ""
            if section.key_takeaways:
                bullets = "\n".join(f"  - {t}" for t in section.key_takeaways)
                takeaways = f"\n  Takeaways:\n{bullets}"
            lines.append(
                f"[{section.category}] ({section.article_count} articles)\n"
                f"  Summary: {section.summary}{takeaways}"
            )
        return "\n\n".join(lines)

    def _format_headlines_block(self, headlines: list[Headline]) -> str:
        """Build a headline digest block for the executive summary prompt."""
        lines: list[str] = []
        for idx, h in enumerate(headlines, 1):
            lines.append(f"{idx}. {h.title} — {h.oneliner}")
        return "\n".join(lines)

    def _build_fallback(
        self,
        sections: list[BriefingSection],
        headlines: list[Headline],
    ) -> ExecutiveSummary:
        """
        Fallback executive summary when LLM fails.

        Concatenates top headlines and first sentence of each section.
        """
        overview_parts: list[str] = []

        if headlines:
            top_stories = ", ".join(h.title for h in headlines[:3])
            overview_parts.append(
                f"Today's top stories include: {top_stories}."
            )

        for section in sections[:4]:
            first_sentence = section.summary.split(". ")[0] + "."
            overview_parts.append(first_sentence)

        overview = " ".join(overview_parts) if overview_parts else "No significant developments today."

        key_developments = [h.title for h in headlines[:5]]

        return ExecutiveSummary(
            overview=overview,
            key_developments=key_developments,
            tone="neutral",
        )

    async def write_executive_summary(
        self,
        sections: list[BriefingSection],
        headlines: list[Headline],
    ) -> ExecutiveSummary:
        """
        Generate the executive summary from completed briefing sections and headlines.

        Strategy:
          1. Build prompt with section summaries + headline digest
          2. Call LLM to produce overview, key developments, and tone
          3. Validate output quality
          4. Fall back to mechanical summary on failure

        Args:
            sections:  Completed BriefingSection objects
            headlines: Completed Headline objects

        Returns:
            ExecutiveSummary with LLM-written or fallback content.
        """
        if not sections and not headlines:
            log.warning("executive_empty_input")
            return ExecutiveSummary(
                overview="No significant news collected for today's briefing.",
                key_developments=[],
                tone="neutral",
            )

        sections_block = self._format_sections_block(sections)
        headlines_block = self._format_headlines_block(headlines)

        prompt_cfg = self._prompts.get("executive_summary_v1", {})
        system = prompt_cfg.get("system", "")
        user_template = prompt_cfg.get("user", "")
        user_prompt = user_template.format(
            sections_block=sections_block,
            headlines_block=headlines_block,
            section_count=len(sections),
            headline_count=len(headlines),
        )

        try:
            data = await self._llm.complete_json(
                prompt=user_prompt,
                system=system,
                temperature=0.4,
                max_tokens=1024,
            )

            overview = data.get("overview", "").strip()
            key_developments = data.get("key_developments", [])
            tone = data.get("tone", "neutral").strip().lower()

            if not overview or len(overview) < 50:
                raise LLMResponseError(
                    "Executive summary overview too short or empty",
                    context={"overview_len": len(overview) if overview else 0},
                )

            # Validate key_developments is a list of strings
            if not isinstance(key_developments, list):
                key_developments = []
            key_developments = [
                str(d).strip() for d in key_developments if str(d).strip()
            ][:5]

            # Validate tone is a single word
            if not tone or len(tone.split()) > 3:
                tone = "neutral"

            result = ExecutiveSummary(
                overview=overview,
                key_developments=key_developments,
                tone=tone,
            )

            log.info(
                "executive_summary_written",
                overview_len=len(overview),
                development_count=len(key_developments),
                tone=tone,
            )
            return result

        except LLMResponseError as exc:
            log.warning("executive_llm_bad_response", error=str(exc))
        except LLMError as exc:
            log.error("executive_llm_error", error=str(exc))

        # Fallback
        fallback = self._build_fallback(sections, headlines)
        log.warning("executive_using_fallback")
        return fallback
