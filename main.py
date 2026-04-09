#!/usr/bin/env python3
"""
BharatIntel — Main Entry Point

CLI for running the daily news briefing pipeline.

Modes:
  python main.py              Run once immediately, then exit
  python main.py --schedule   Run on cron schedule (blocks until interrupted)
  python main.py --help       Show usage

Environment:
  Requires .env with API keys (OPENAI_API_KEY, GNEWS_API_KEY, etc.)
  All config lives in config/ (settings.yaml, sources.yaml, prompts.yaml)
  Output is written to output/ (HTML, PDF, JSON)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure project root is on sys.path for imports
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

from core.logger import get_logger, setup_logging


def _load_environment() -> None:
    """Load .env file and validate critical environment variables."""
    env_path = _PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    import os

    # Warn (don't crash) if keys are missing — some sources may not need them
    warnings: list[str] = []
    # Check for at least one LLM provider key
    has_llm_key = any(
        os.environ.get(k)
        for k in ("XAI_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")
    )
    if not has_llm_key:
        warnings.append("No LLM API key set (XAI_API_KEY, GEMINI_API_KEY, etc.) — LLM calls will fail")
    if not os.environ.get("GNEWS_API_KEY"):
        warnings.append("GNEWS_API_KEY not set — GNews API source will fail")

    log = get_logger("main")
    for w in warnings:
        log.warning("env_warning", message=w)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bharatintel",
        description="BharatIntel — AI-Powered Daily Intelligence Briefing",
    )
    parser.add_argument(
        "--schedule",
        action="store_true",
        default=False,
        help="Run on cron schedule (default: run once then exit)",
    )
    parser.add_argument(
        "--settings",
        type=str,
        default="config/settings.yaml",
        help="Path to settings.yaml",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default="config/sources.yaml",
        help="Path to sources.yaml",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="config/prompts.yaml",
        help="Path to prompts.yaml",
    )
    return parser


async def _run_once(
    settings: str,
    sources: str,
    prompts: str,
) -> int:
    """
    Execute the pipeline once and return exit code.

    Returns 0 on full success, 1 on partial/failure.
    """
    from pipeline.scheduler import Scheduler

    scheduler = Scheduler(
        settings_path=settings,
        sources_path=sources,
        prompts_path=prompts,
    )
    result = await scheduler.run_once()

    if result and result.success:
        pub = result.publish_result
        if pub:
            log = get_logger("main")
            log.info(
                "briefing_ready",
                pdf=pub.pdf_path or "(skipped)",
                html=pub.html_path,
                json=pub.json_path,
            )
        return 0

    return 1


def _run_scheduled(
    settings: str,
    sources: str,
    prompts: str,
) -> int:
    """Start the cron scheduler. Blocks until interrupted. Returns exit code."""
    from pipeline.scheduler import Scheduler

    scheduler = Scheduler(
        settings_path=settings,
        sources_path=sources,
        prompts_path=prompts,
    )

    log = get_logger("main")
    log.info("starting_scheduled_mode")

    try:
        scheduler.start()  # blocks
    except ImportError:
        log.error(
            "scheduler_dependency_missing",
            hint="pip install 'apscheduler>=3.10.0'",
        )
        return 1
    except KeyboardInterrupt:
        log.info("interrupted_by_user")

    return 0


def main() -> None:
    # Initialize logging before anything else
    setup_logging()
    log = get_logger("main")

    parser = _build_parser()
    args = parser.parse_args()

    # Load environment
    _load_environment()

    log.info(
        "bharatintel_start",
        mode="scheduled" if args.schedule else "once",
        settings=args.settings,
        sources=args.sources,
    )

    if args.schedule:
        exit_code = _run_scheduled(args.settings, args.sources, args.prompts)
    else:
        exit_code = asyncio.run(
            _run_once(args.settings, args.sources, args.prompts)
        )

    log.info("bharatintel_exit", code=exit_code)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
