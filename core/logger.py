"""
BharatIntel — Structured Logging

Uses structlog for JSON file output + human-readable console output.
Each agent binds its own context (agent name, source, etc.).

Usage:
    from core.logger import get_logger
    log = get_logger("collector")
    log.info("fetched articles", count=23, source="rss")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import structlog


_CONFIGURED = False


def _ensure_log_dir(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)


def setup_logging(
    log_dir: str = "logs",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> None:
    """
    Configure structlog + stdlib logging once per process.

    - Console: human-readable, colored
    - File: JSON lines, rotating daily by filename convention
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_path = Path(log_dir)
    _ensure_log_dir(log_path)

    from datetime import date

    today = date.today().isoformat()
    main_log = log_path / f"bharatintel_{today}.log"
    error_log = log_path / f"errors_{today}.log"

    # ── stdlib handlers ──────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)

    file_handler = logging.FileHandler(str(main_log), encoding="utf-8")
    file_handler.setLevel(file_level)

    error_handler = logging.FileHandler(str(error_log), encoding="utf-8")
    error_handler.setLevel(logging.ERROR)

    root = logging.getLogger()
    root.setLevel(file_level)
    root.handlers.clear()
    root.addHandler(console_handler)
    root.addHandler(file_handler)
    root.addHandler(error_handler)

    # ── structlog configuration ──────────────────────────────────────
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Attach structlog formatters to handlers
    console_formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer(colors=True),
        foreign_pre_chain=shared_processors,
    )
    json_formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
        foreign_pre_chain=shared_processors,
    )

    console_handler.setFormatter(console_formatter)
    file_handler.setFormatter(json_formatter)
    error_handler.setFormatter(json_formatter)

    _CONFIGURED = True


def get_logger(agent_name: str) -> structlog.stdlib.BoundLogger:
    """
    Return a logger bound to a specific agent context.

    Auto-initializes logging on first call.
    """
    setup_logging()
    return structlog.get_logger(agent_name).bind(agent=agent_name)
