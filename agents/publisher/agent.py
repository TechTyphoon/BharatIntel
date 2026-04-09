"""
BharatIntel — Publisher Agent

Orchestrates the full publishing pipeline:
  Briefing → HTML rendering → PDF generation → JSON archive.

This is the single entry point for the publisher module.

Responsibilities:
  - Load publisher settings (template path, PDF options, output dir)
  - Render Briefing to HTML via TemplateRenderer
  - Write HTML to disk (always — lightweight, fast)
  - Write PDF to disk via PDFWriter (with graceful skip on failure)
  - Write JSON archive to disk (machine-readable backup)
  - Return PublishResult with paths and metadata

Usage:
    from agents.publisher.agent import PublisherAgent
    agent = PublisherAgent()
    result = agent.publish(briefing)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from agents.publisher.models import PublishResult
from agents.publisher.pdf_writer import PDFWriter
from agents.publisher.renderer import TemplateRenderer
from agents.publisher.reportlab_writer import ReportLabWriter
from agents.summarizer.models import Briefing
from core.exceptions import RenderError
from core.logger import get_logger

log = get_logger("publisher")


def _parse_margin_cm(margin_str: str) -> float:
    """Extract numeric cm value from a CSS margin string like '1.5cm'."""
    cleaned = margin_str.strip().lower().replace("cm", "")
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return 1.5


class PublisherAgent:
    """
    Orchestrates HTML rendering → PDF generation → JSON archival.

    Args:
        settings_path: Path to settings.yaml
    """

    def __init__(self, settings_path: str = "config/settings.yaml"):
        settings = self._load_settings(settings_path)
        app_cfg = settings.get("app", {})
        pub_cfg = settings.get("publisher", {})

        self._app_name = app_cfg.get("name", "BharatIntel")
        self._app_version = app_cfg.get("version", "1.0.0")
        self._output_dir = app_cfg.get("output_dir", "output")

        template_path = pub_cfg.get("template", "agents/publisher/templates/briefing.html")
        pdf_options = pub_cfg.get("pdf_options", {})
        self._page_size = pdf_options.get("page_size", "A4")
        self._margin = pdf_options.get("margin", "1.5cm")
        self._pdf_backend = pub_cfg.get("pdf_backend", "reportlab")

        # Ensure output directory exists
        Path(self._output_dir).mkdir(parents=True, exist_ok=True)

        self._renderer = TemplateRenderer(
            template_path=template_path,
            app_name=self._app_name,
            app_version=self._app_version,
        )
        self._pdf_writer = PDFWriter(output_dir=self._output_dir)
        self._reportlab_writer = ReportLabWriter(
            output_dir=self._output_dir,
            app_name=self._app_name,
            app_version=self._app_version,
            page_size=self._page_size,
            margin_cm=_parse_margin_cm(self._margin),
        )

        log.info(
            "publisher_initialized",
            output_dir=self._output_dir,
            template=template_path,
            page_size=self._page_size,
            margin=self._margin,
            pdf_backend=self._pdf_backend,
        )

    @staticmethod
    def _load_settings(path: str) -> dict[str, Any]:
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Settings config not found: {path}")
        with open(filepath, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _write_html(self, html: str, date_str: str) -> Path:
        """Write rendered HTML to disk. Returns absolute path."""
        filename = f"briefing_{date_str}.html"
        out_path = Path(self._output_dir) / filename
        out_path.write_text(html, encoding="utf-8")
        log.info("html_saved", path=str(out_path.resolve()), size=len(html))
        return out_path.resolve()

    def _write_json(self, briefing: Briefing) -> Path:
        """Write Briefing as JSON archive. Returns absolute path."""
        filename = f"briefing_{briefing.date}.json"
        out_path = Path(self._output_dir) / filename

        data = briefing.to_dict()
        json_str = json.dumps(data, indent=2, ensure_ascii=False, default=str)
        out_path.write_text(json_str, encoding="utf-8")

        log.info("json_saved", path=str(out_path.resolve()), size=len(json_str))
        return out_path.resolve()

    def publish(self, briefing: Briefing) -> PublishResult:
        """
        Run the full publishing pipeline.

        Pipeline:
          1. Render Briefing → HTML via Jinja2 template
          2. Save HTML to disk (always succeeds if render did)
          3. Convert HTML → PDF via WeasyPrint (graceful skip on failure)
          4. Save JSON archive to disk (always succeeds)
          5. Return PublishResult with paths and metadata

        If PDF generation fails (e.g. weasyprint not installed),
        HTML + JSON are still produced. The caller gets a PublishResult
        with an empty pdf_path and a warning is logged.

        Args:
            briefing: Complete Briefing from the summarizer agent.

        Returns:
            PublishResult with file paths and metadata. Never None.

        Raises:
            RenderError: Only if HTML rendering itself fails (template broken).
        """
        date_str = briefing.date or "unknown"
        log.info("publish_start", date=date_str)

        # ── Step 1: Render HTML ──────────────────────────────────────
        html = self._renderer.render(
            briefing,
            page_size=self._page_size,
            margin=self._margin,
        )

        # ── Step 2: Save HTML ────────────────────────────────────────
        html_path = self._write_html(html, date_str)

        # ── Step 3: Generate PDF (graceful, with fallback) ────────────
        pdf_path_str = ""
        pdf_size = 0
        pdf_filename = f"briefing_{date_str}.pdf"

        if self._pdf_backend == "reportlab":
            # Primary: ReportLab (pure Python, no native deps)
            try:
                pdf_path = self._reportlab_writer.write(
                    briefing=briefing,
                    filename=pdf_filename,
                )
                pdf_path_str = str(pdf_path)
                pdf_size = pdf_path.stat().st_size
            except RenderError as exc:
                log.warning(
                    "reportlab_pdf_skipped",
                    date=date_str,
                    reason=str(exc),
                )
            except Exception as exc:
                log.error(
                    "reportlab_pdf_unexpected_error",
                    date=date_str,
                    error=str(exc),
                )
        else:
            # Fallback: WeasyPrint (HTML → PDF)
            try:
                pdf_path = self._pdf_writer.write(
                    html=html,
                    filename=pdf_filename,
                )
                pdf_path_str = str(pdf_path)
                pdf_size = pdf_path.stat().st_size
            except RenderError as exc:
                log.warning(
                    "weasyprint_pdf_skipped",
                    date=date_str,
                    reason=str(exc),
                )
            except Exception as exc:
                log.error(
                    "weasyprint_pdf_unexpected_error",
                    date=date_str,
                    error=str(exc),
                )

        # ── Step 4: Save JSON archive ────────────────────────────────
        json_path = self._write_json(briefing)

        # ── Step 5: Assemble result ──────────────────────────────────
        file_sizes: dict[str, int] = {
            "html": html_path.stat().st_size,
            "json": json_path.stat().st_size,
        }
        if pdf_size:
            file_sizes["pdf"] = pdf_size

        result = PublishResult(
            date=date_str,
            pdf_path=pdf_path_str,
            html_path=str(html_path),
            json_path=str(json_path),
            section_count=len(briefing.sections),
            headline_count=len(briefing.headlines),
            has_executive=briefing.executive_summary is not None,
            file_sizes=file_sizes,
        )

        log.info(
            "publish_complete",
            date=date_str,
            pdf=bool(pdf_path_str),
            html=str(html_path),
            json=str(json_path),
            file_sizes=file_sizes,
        )

        return result
