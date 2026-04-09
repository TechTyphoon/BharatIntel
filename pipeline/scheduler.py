"""
BharatIntel — Scheduler

Manages timed and on-demand execution of the pipeline.

Responsibilities:
  - Parse cron expression from settings for daily scheduling
  - Run the pipeline immediately on demand (run_once)
  - Run the pipeline on a recurring cron schedule (start)
  - Graceful shutdown on SIGINT / SIGTERM
  - Prevent overlapping runs (skip if previous run still active)
  - Log every scheduled trigger, completion, and skip

Dependencies: apscheduler

Usage:
    # One-shot:
    from pipeline.scheduler import Scheduler
    scheduler = Scheduler()
    await scheduler.run_once()

    # Cron loop:
    scheduler = Scheduler()
    scheduler.start()  # blocks until signal / stop()
"""

from __future__ import annotations

import asyncio
import signal
import sys
from pathlib import Path
from typing import Any

import yaml

from core.logger import get_logger, setup_logging
from pipeline.orchestrator import PipelineOrchestrator, PipelineResult

log = get_logger("scheduler")


class Scheduler:
    """
    Production scheduler: cron-based recurring pipeline execution.

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
        app_cfg = settings.get("app", {})
        pipeline_cfg = settings.get("pipeline", {})

        self._log_dir = app_cfg.get("log_dir", "logs")
        self._cron_expr = pipeline_cfg.get("schedule_cron", "0 6 * * *")

        # Guard against overlapping runs
        self._running = False
        self._shutdown_event: asyncio.Event | None = None

        log.info(
            "scheduler_initialized",
            cron=self._cron_expr,
        )

    @staticmethod
    def _load_settings(path: str) -> dict[str, Any]:
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Settings config not found: {path}")
        with open(filepath, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def _parse_cron(expr: str) -> dict[str, str]:
        """
        Parse a standard 5-field cron expression into APScheduler kwargs.

        Format: minute hour day_of_month month day_of_week
        Example: "0 6 * * *" → run at 06:00 every day

        Returns:
            Dict with keys: minute, hour, day, month, day_of_week
            Each value is a string compatible with APScheduler CronTrigger.

        Raises:
            ValueError: If expression doesn't have exactly 5 fields.
        """
        parts = expr.strip().split()
        if len(parts) != 5:
            raise ValueError(
                f"Invalid cron expression '{expr}': expected 5 fields, got {len(parts)}"
            )
        return {
            "minute": parts[0],
            "hour": parts[1],
            "day": parts[2],
            "month": parts[3],
            "day_of_week": parts[4],
        }

    async def _execute_pipeline(self) -> PipelineResult | None:
        """
        Run the pipeline once with overlap guard.

        Returns PipelineResult if executed, None if skipped (already running).
        """
        if self._running:
            log.warning("pipeline_skipped_overlap", reason="Previous run still active")
            return None

        self._running = True
        try:
            log.info("pipeline_triggered")
            orchestrator = PipelineOrchestrator(
                settings_path=self._settings_path,
                sources_path=self._sources_path,
                prompts_path=self._prompts_path,
            )
            result = await orchestrator.run()

            if result.success:
                log.info(
                    "pipeline_run_success",
                    total_sec=round(result.total_sec, 2),
                    stages={s.name: s.status for s in result.stages},
                )
            else:
                log.warning(
                    "pipeline_run_partial",
                    total_sec=round(result.total_sec, 2),
                    error=result.error,
                    stages={s.name: s.status for s in result.stages},
                )

            return result

        except Exception as exc:
            log.error("pipeline_run_crashed", error=str(exc))
            return None
        finally:
            self._running = False

    async def run_once(self) -> PipelineResult | None:
        """
        Execute the pipeline exactly once (on-demand).

        This is the primary entry point for CLI invocation
        and testing. No scheduling involved.

        Returns:
            PipelineResult with full diagnostics, or None on overlap.
        """
        log.info("run_once_start")
        result = await self._execute_pipeline()
        log.info("run_once_complete", success=result.success if result else False)
        return result

    def start(self) -> None:
        """
        Start the cron-based scheduler. Blocks until interrupted.

        Installs SIGINT/SIGTERM handlers for graceful shutdown.
        Uses APScheduler's AsyncIOScheduler for async pipeline execution.

        Raises:
            ImportError: If apscheduler is not installed.
        """
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
            from apscheduler.triggers.cron import CronTrigger
        except ImportError as exc:
            log.error(
                "apscheduler_not_installed",
                hint="pip install apscheduler>=3.10.0",
            )
            raise ImportError(
                "apscheduler is required for scheduled runs. "
                "Install with: pip install 'apscheduler>=3.10.0'"
            ) from exc

        cron_fields = self._parse_cron(self._cron_expr)

        trigger = CronTrigger(
            minute=cron_fields["minute"],
            hour=cron_fields["hour"],
            day=cron_fields["day"],
            month=cron_fields["month"],
            day_of_week=cron_fields["day_of_week"],
        )

        scheduler = AsyncIOScheduler()
        scheduler.add_job(
            self._execute_pipeline,
            trigger=trigger,
            id="bharatintel_daily",
            name="BharatIntel Daily Briefing",
            max_instances=1,           # APScheduler-level overlap guard
            misfire_grace_time=3600,   # Allow 1 hour late execution
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._shutdown_event = asyncio.Event()

        def _signal_handler(sig: int, _frame: Any) -> None:
            sig_name = signal.Signals(sig).name
            log.info("shutdown_signal_received", signal=sig_name)
            scheduler.shutdown(wait=False)
            if self._shutdown_event:
                self._shutdown_event.set()

        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

        scheduler.start()
        log.info(
            "scheduler_started",
            cron=self._cron_expr,
            next_run=str(scheduler.get_job("bharatintel_daily").next_run_time),
        )

        try:
            loop.run_until_complete(self._shutdown_event.wait())
        except KeyboardInterrupt:
            pass
        finally:
            scheduler.shutdown(wait=False)
            loop.close()
            log.info("scheduler_stopped")

    def stop(self) -> None:
        """Signal the scheduler to shut down gracefully."""
        if self._shutdown_event:
            self._shutdown_event.set()
