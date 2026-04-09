"""
BharatIntel — Pipeline Orchestrator

Wires together all agents into a single end-to-end pipeline:
  collect → curate → summarize → publish.

Responsibilities:
  - Initialize all four agents with shared config paths
  - Execute each stage sequentially, passing outputs downstream
  - Enforce minimum-article threshold before proceeding
  - Track wall-clock timing per stage
  - Capture and report per-stage errors without crashing the process
  - Return a PipelineResult with full diagnostics

This module has ZERO scheduling logic. It runs exactly once when called.
Scheduling is handled by pipeline/scheduler.py.

Usage:
    from pipeline.orchestrator import PipelineOrchestrator
    orch = PipelineOrchestrator()
    result = await orch.run()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from agents.collector.agent import CollectorAgent
from agents.collector.models import RawArticle
from agents.curator.agent import CuratorAgent
from agents.curator.models import RankedArticle
from agents.publisher.agent import PublisherAgent
from agents.publisher.models import PublishResult
from agents.summarizer.agent import SummarizerAgent
from agents.summarizer.models import Briefing
from core.exceptions import PipelineError
from core.logger import get_logger

log = get_logger("pipeline")


@dataclass
class StageResult:
    """Timing and status for a single pipeline stage."""

    name: str
    status: str = "not_started"   # not_started | success | failed | skipped
    duration_sec: float = 0.0
    item_count: int = 0
    error: str = ""


@dataclass
class PipelineResult:
    """
    Complete diagnostic output from a single pipeline run.

    Fields:
        success:        True if the full pipeline ran to completion
        stages:         Per-stage timing and status
        total_sec:      Total wall-clock time
        publish_result: Output metadata from the publisher (None if publish skipped)
        error:          Top-level error message if pipeline aborted
    """

    success: bool = False
    stages: list[StageResult] = field(default_factory=list)
    total_sec: float = 0.0
    publish_result: PublishResult | None = None
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "stages": [
                {
                    "name": s.name,
                    "status": s.status,
                    "duration_sec": round(s.duration_sec, 2),
                    "item_count": s.item_count,
                    "error": s.error,
                }
                for s in self.stages
            ],
            "total_sec": round(self.total_sec, 2),
            "publish_result": self.publish_result.to_dict() if self.publish_result else None,
            "error": self.error,
        }


class PipelineOrchestrator:
    """
    Runs the full collect → curate → summarize → publish pipeline.

    Args:
        settings_path: Path to settings.yaml
        sources_path:  Path to sources.yaml
        prompts_path:  Path to prompts.yaml
    """

    def __init__(
        self,
        settings_path: str = "config/settings.yaml",
        sources_path: str = "config/sources.yaml",
        prompts_path: str = "config/prompts.yaml",
    ):
        self._settings_path = settings_path
        self._sources_path = sources_path
        self._prompts_path = prompts_path

        settings = self._load_settings(settings_path)
        pipeline_cfg = settings.get("pipeline", {})
        self._min_articles = pipeline_cfg.get("min_articles_to_proceed", 5)

        log.info(
            "orchestrator_initialized",
            min_articles=self._min_articles,
        )

    @staticmethod
    def _load_settings(path: str) -> dict[str, Any]:
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Settings config not found: {path}")
        with open(filepath, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def _raw_to_ranked(articles: list[RawArticle]) -> list[RankedArticle]:
        """Convert raw articles to ranked articles for graceful curate fallback."""
        ranked = []
        cat_map = {
            "world": "geopolitics", "politics": "geopolitics", "conflict": "geopolitics",
            "tech": "technology", "ai": "technology", "cyber": "technology",
            "india": "indian_politics", "kerala": "indian_politics", "bengal": "indian_politics",
            "science": "science", "space": "science", "health": "science",
            "culture": "civilisation", "art": "civilisation", "music": "civilisation",
        }
        for i, a in enumerate(articles[:30]):
            # Assign category from source categories or title keywords
            category = "editors_picks"
            for cat in (a.categories or []):
                cat_lower = cat.lower()
                for keyword, mapped in cat_map.items():
                    if keyword in cat_lower:
                        category = mapped
                        break
                if category != "editors_picks":
                    break
            ranked.append(RankedArticle(
                title=a.title,
                url=a.url,
                source_name=a.source_name,
                relevance_score=max(1, 10 - i // 3),
                assigned_category=category,
                rank_reason="Fallback: curate stage unavailable",
                published_at=a.published_at,
                snippet=a.snippet,
                author=a.author,
                image_url=a.image_url,
                fetched_at=a.fetched_at,
                original_categories=a.categories,
            ))
        return ranked

    async def run(self) -> PipelineResult:
        """
        Execute the full pipeline end-to-end.

        Pipeline stages:
          1. Collect: Fetch articles from all enabled sources
          2. Curate:  Dedup + multi-dimensional ranking
          3. Summarize: LLM section writing + headlines + executive summary
          4. Publish:  HTML + PDF + JSON output

        If collection returns fewer articles than min_articles_to_proceed,
        the pipeline aborts with a clear error. Each subsequent stage
        catches its own exceptions and marks itself failed — but later
        stages still attempt to run on whatever partial data exists.

        Returns:
            PipelineResult with full diagnostics. Never raises.
        """
        pipeline_start = time.monotonic()
        stages: list[StageResult] = []

        log.info("pipeline_start")

        # ── Stage 1: Collect ─────────────────────────────────────────
        collect_stage = StageResult(name="collect")
        raw_articles = []
        try:
            t0 = time.monotonic()
            collector = CollectorAgent(
                sources_path=self._sources_path,
                settings_path=self._settings_path,
            )
            raw_articles = await collector.run()
            collect_stage.duration_sec = time.monotonic() - t0
            collect_stage.item_count = len(raw_articles)
            collect_stage.status = "success"
            log.info("stage_collect_done", articles=len(raw_articles))
        except Exception as exc:
            collect_stage.duration_sec = time.monotonic() - t0
            collect_stage.status = "failed"
            collect_stage.error = str(exc)
            log.error("stage_collect_failed", error=str(exc))

        stages.append(collect_stage)

        # ── Gate: minimum article threshold ──────────────────────────
        if len(raw_articles) < self._min_articles:
            msg = (
                f"Only {len(raw_articles)} articles collected "
                f"(minimum: {self._min_articles}). Pipeline aborted."
            )
            log.warning("pipeline_threshold_not_met", count=len(raw_articles))

            # Mark remaining stages as skipped
            for name in ("curate", "summarize", "publish"):
                stages.append(StageResult(name=name, status="skipped"))

            return PipelineResult(
                success=False,
                stages=stages,
                total_sec=time.monotonic() - pipeline_start,
                error=msg,
            )

        # ── Stage 2: Curate ──────────────────────────────────────────
        curate_stage = StageResult(name="curate")
        ranked_articles = []
        try:
            t0 = time.monotonic()
            curator = CuratorAgent(
                settings_path=self._settings_path,
                prompts_path=self._prompts_path,
            )
            ranked_articles = await curator.run(raw_articles)
            curate_stage.duration_sec = time.monotonic() - t0
            curate_stage.item_count = len(ranked_articles)
            curate_stage.status = "success"
            log.info("stage_curate_done", articles=len(ranked_articles))
        except Exception as exc:
            curate_stage.duration_sec = time.monotonic() - t0
            curate_stage.status = "failed"
            curate_stage.error = str(exc)
            log.error("stage_curate_failed", error=str(exc))

            # Graceful degradation: pass raw articles through as RankedArticles
            if not ranked_articles and raw_articles:
                log.warning("curate_fallback_passthrough", articles=len(raw_articles))
                ranked_articles = self._raw_to_ranked(raw_articles)
                curate_stage.item_count = len(ranked_articles)

        stages.append(curate_stage)

        # ── Stage 3: Summarize ───────────────────────────────────────
        summarize_stage = StageResult(name="summarize")
        briefing: Briefing | None = None
        try:
            t0 = time.monotonic()
            summarizer = SummarizerAgent(
                settings_path=self._settings_path,
                prompts_path=self._prompts_path,
            )
            briefing = await summarizer.run(ranked_articles)
            summarize_stage.duration_sec = time.monotonic() - t0
            summarize_stage.item_count = len(briefing.sections) if briefing else 0
            summarize_stage.status = "success"
            log.info(
                "stage_summarize_done",
                sections=summarize_stage.item_count,
            )
        except Exception as exc:
            summarize_stage.duration_sec = time.monotonic() - t0
            summarize_stage.status = "failed"
            summarize_stage.error = str(exc)
            log.error("stage_summarize_failed", error=str(exc))

        stages.append(summarize_stage)

        # ── Stage 4: Publish ─────────────────────────────────────────
        publish_stage = StageResult(name="publish")
        publish_result: PublishResult | None = None

        if briefing is None:
            publish_stage.status = "skipped"
            publish_stage.error = "No briefing to publish (summarize failed)"
            log.warning("stage_publish_skipped")
        else:
            try:
                t0 = time.monotonic()
                publisher = PublisherAgent(settings_path=self._settings_path)
                publish_result = publisher.publish(briefing)
                publish_stage.duration_sec = time.monotonic() - t0
                publish_stage.status = "success"
                log.info(
                    "stage_publish_done",
                    pdf=bool(publish_result.pdf_path),
                    html=publish_result.html_path,
                )
            except Exception as exc:
                publish_stage.duration_sec = time.monotonic() - t0
                publish_stage.status = "failed"
                publish_stage.error = str(exc)
                log.error("stage_publish_failed", error=str(exc))

        stages.append(publish_stage)

        # ── Final result ─────────────────────────────────────────────
        total_sec = time.monotonic() - pipeline_start
        all_success = all(s.status in ("success",) for s in stages)

        result = PipelineResult(
            success=all_success,
            stages=stages,
            total_sec=total_sec,
            publish_result=publish_result,
        )

        log.info(
            "pipeline_complete",
            success=all_success,
            total_sec=round(total_sec, 2),
            stage_summary={s.name: s.status for s in stages},
        )

        return result
