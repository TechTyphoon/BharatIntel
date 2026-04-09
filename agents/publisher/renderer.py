"""
BharatIntel — Template Renderer

Renders a Briefing object into an HTML string using Jinja2 templates.

Responsibilities:
  - Load + compile Jinja2 template from disk
  - Map Briefing dataclass into template context dict
  - Render HTML with auto-escaped user content (XSS prevention)
  - Provide fallback for missing template fields

Dependencies: jinja2
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, TemplateNotFound, select_autoescape

from agents.summarizer.models import Briefing
from core.exceptions import RenderError
from core.logger import get_logger

log = get_logger("publisher.renderer")


class TemplateRenderer:
    """
    Renders a Briefing into HTML via a Jinja2 template.

    Args:
        template_path: Path to the .html template file
                       (e.g. "agents/publisher/templates/briefing.html")
        app_name:      Application name for header/footer
        app_version:   Application version for footer
    """

    def __init__(
        self,
        template_path: str = "agents/publisher/templates/briefing.html",
        app_name: str = "BharatIntel",
        app_version: str = "1.0.0",
    ):
        self._template_path = Path(template_path)
        self._app_name = app_name
        self._app_version = app_version

        if not self._template_path.exists():
            raise FileNotFoundError(
                f"Template not found: {self._template_path.resolve()}"
            )

        template_dir = str(self._template_path.parent)
        template_name = self._template_path.name

        self._env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(["html"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        try:
            self._template = self._env.get_template(template_name)
        except TemplateNotFound as exc:
            raise FileNotFoundError(
                f"Template file not loadable: {template_name}"
            ) from exc

        log.info(
            "renderer_initialized",
            template=str(self._template_path),
        )

    def _build_context(self, briefing: Briefing) -> dict[str, Any]:
        """
        Map a Briefing dataclass into a flat template context dict.

        Converts dataclass instances to dicts for Jinja2 attribute access,
        and fills in display-friendly date + generation timestamp.
        """
        # Parse date for display
        try:
            parsed_date = datetime.fromisoformat(briefing.date)
            date_display = parsed_date.strftime("%A, %B %d, %Y")
        except (ValueError, TypeError):
            date_display = briefing.date or "Unknown Date"

        # Executive summary as dict (or None)
        exec_summary = None
        if briefing.executive_summary:
            exec_summary = briefing.executive_summary.to_dict()

        # Headlines as list of dicts
        headlines = [h.to_dict() for h in briefing.headlines]

        # Sections as list of dicts
        sections = [s.to_dict() for s in briefing.sections]

        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        return {
            "title": self._app_name,
            "version": self._app_version,
            "date": briefing.date,
            "date_display": date_display,
            "generated_at": generated_at,
            "executive_summary": exec_summary,
            "headlines": headlines,
            "sections": sections,
            "token_usage": briefing.token_usage or {},
            "page_size": "A4",
            "margin": "1.5cm",
        }

    def render(
        self,
        briefing: Briefing,
        page_size: str = "A4",
        margin: str = "1.5cm",
    ) -> str:
        """
        Render a Briefing into an HTML string.

        Args:
            briefing:   Complete Briefing dataclass
            page_size:  CSS page size for @page rule
            margin:     CSS margin for @page rule

        Returns:
            Fully rendered HTML string.

        Raises:
            RenderError: If template rendering fails.
        """
        context = self._build_context(briefing)
        context["page_size"] = page_size
        context["margin"] = margin

        try:
            html = self._template.render(**context)
        except Exception as exc:
            raise RenderError(
                f"Template rendering failed: {exc}",
                context={
                    "template": str(self._template_path),
                    "date": briefing.date,
                    "error": str(exc),
                },
            ) from exc

        log.info(
            "html_rendered",
            date=briefing.date,
            html_length=len(html),
            section_count=len(briefing.sections),
        )

        return html
