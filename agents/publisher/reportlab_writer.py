"""
BharatIntel — ReportLab PDF Generator

Generates a professional, structured PDF briefing directly from a Briefing
dataclass using ReportLab — no HTML intermediary needed.

Responsibilities:
  - Render executive summary, headlines, and category sections to PDF
  - Professional typography with consistent spacing and color palette
  - Category-wise sections with article titles, sources, and summaries
  - Key takeaways rendered as bullet lists per section
  - Header/footer on every page (title, date, page numbers)
  - Clean A4 layout with configurable margins
  - Sanitized filenames to prevent path traversal

Dependencies: reportlab

Usage:
    from agents.publisher.reportlab_writer import ReportLabWriter
    writer = ReportLabWriter(output_dir="output")
    path = writer.write(briefing, filename="briefing_2026-04-08.pdf")
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.exceptions import RenderError
from core.logger import get_logger

log = get_logger("publisher.reportlab")

# ── Color Palette (RGB tuples, 0-1 range) ────────────────────────────
_NAVY = (0.102, 0.322, 0.463)       # #1A5276 — headings, accents
_DARK = (0.1, 0.1, 0.1)             # #1A1A1A — body text
_GREY = (0.4, 0.4, 0.4)             # #666666 — meta text
_LIGHT_GREY = (0.6, 0.6, 0.6)       # #999999 — footnotes
_VERY_LIGHT = (0.94, 0.96, 0.97)    # #F0F4F8 — executive bg
_RULE = (0.85, 0.85, 0.85)          # #D9D9D9 — horizontal rules
_WHITE = (1, 1, 1)

# ── Spacing Constants (points) ───────────────────────────────────────
_SECTION_GAP = 18
_PARA_GAP = 6
_BULLET_INDENT = 18
_LINE_HEIGHT = 14


class ReportLabWriter:
    """
    Generates a structured PDF from a Briefing using ReportLab Platypus.

    Args:
        output_dir:  Directory where PDFs are written (created if missing)
        app_name:    Application name for header/footer
        app_version: Version string for footer
        page_size:   ReportLab page size string ("A4" or "letter")
        margin_cm:   Page margin in centimeters
    """

    def __init__(
        self,
        output_dir: str = "output",
        app_name: str = "BharatIntel",
        app_version: str = "1.0.0",
        page_size: str = "A4",
        margin_cm: float = 1.5,
    ):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._app_name = app_name
        self._app_version = app_version
        self._page_size_str = page_size
        self._margin_cm = margin_cm

        log.info(
            "reportlab_writer_initialized",
            output_dir=str(self._output_dir.resolve()),
            page_size=page_size,
            margin_cm=margin_cm,
        )

    def write(
        self,
        briefing: Any,
        filename: str = "briefing.pdf",
    ) -> Path:
        """
        Generate a PDF from a Briefing dataclass and save to disk.

        Args:
            briefing: Briefing dataclass from agents.summarizer.models
            filename: Output filename (sanitized internally)

        Returns:
            Absolute Path to the written PDF file.

        Raises:
            RenderError: If ReportLab is not installed or PDF generation fails.
        """
        try:
            from reportlab.lib import colors as rl_colors
            from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
            from reportlab.lib.pagesizes import A4, letter
            from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
            from reportlab.lib.units import cm
            from reportlab.platypus import (
                BaseDocTemplate,
                Frame,
                PageBreak,
                PageTemplate,
                Paragraph,
                Spacer,
                Table,
                TableStyle,
            )
        except ImportError as exc:
            raise RenderError(
                "reportlab is not installed. Install with: pip install reportlab",
                context={"error": str(exc)},
            ) from exc

        safe_name = self._sanitize_filename(filename)
        pdf_path = self._output_dir / safe_name

        # ── Page setup ───────────────────────────────────────────────
        page_size = A4 if self._page_size_str.upper() == "A4" else letter
        margin = self._margin_cm * cm

        # ── Styles ───────────────────────────────────────────────────
        styles = self._build_styles(TA_LEFT, TA_JUSTIFY, TA_CENTER)

        # ── Build story (flowable elements) ──────────────────────────
        story: list = []

        # Header
        story.extend(self._build_header(briefing, styles, Paragraph, Spacer))

        # Executive summary
        if briefing.executive_summary:
            story.extend(
                self._build_executive(
                    briefing.executive_summary, styles,
                    Paragraph, Spacer, Table, TableStyle, rl_colors,
                )
            )

        # Headlines
        if briefing.headlines:
            story.extend(
                self._build_headlines(briefing.headlines, styles, Paragraph, Spacer)
            )

        # Sections
        for section in briefing.sections:
            story.extend(
                self._build_section(section, styles, Paragraph, Spacer)
            )

        # Footer info
        story.extend(self._build_footer_content(briefing, styles, Paragraph, Spacer))

        # ── Build PDF ────────────────────────────────────────────────
        briefing_date = briefing.date or "Unknown"
        app_name = self._app_name

        def _header_footer(canvas: Any, doc: Any) -> None:
            """Draw header line and page footer on every page."""
            canvas.saveState()

            # Header rule
            canvas.setStrokeColorRGB(*_NAVY)
            canvas.setLineWidth(1.5)
            y_top = page_size[1] - margin + 10
            canvas.line(margin, y_top, page_size[0] - margin, y_top)

            # Footer
            canvas.setFont("Helvetica", 7)
            canvas.setFillColorRGB(*_LIGHT_GREY)
            footer_y = margin - 20
            canvas.drawString(
                margin, footer_y,
                f"{app_name} — {briefing_date}",
            )
            canvas.drawRightString(
                page_size[0] - margin, footer_y,
                f"Page {canvas.getPageNumber()}",
            )
            canvas.restoreState()

        try:
            frame = Frame(
                margin, margin,
                page_size[0] - 2 * margin,
                page_size[1] - 2 * margin,
                id="main",
            )
            template = PageTemplate(
                id="briefing",
                frames=[frame],
                onPage=_header_footer,
            )
            doc = BaseDocTemplate(
                str(pdf_path),
                pagesize=page_size,
                pageTemplates=[template],
                title=f"{self._app_name} — {briefing.date}",
                author=self._app_name,
            )
            doc.build(story)
        except Exception as exc:
            raise RenderError(
                f"ReportLab PDF generation failed: {exc}",
                context={"filename": safe_name, "error": str(exc)},
            ) from exc

        file_size = pdf_path.stat().st_size
        log.info(
            "reportlab_pdf_generated",
            path=str(pdf_path.resolve()),
            size_bytes=file_size,
            sections=len(briefing.sections),
            headlines=len(briefing.headlines),
        )

        return pdf_path.resolve()

    # ── Style Factory ────────────────────────────────────────────────

    @staticmethod
    def _build_styles(TA_LEFT: int, TA_JUSTIFY: int, TA_CENTER: int) -> dict[str, Any]:
        """Create all paragraph styles used in the PDF."""
        from reportlab.lib.styles import ParagraphStyle

        return {
            "title": ParagraphStyle(
                "BITitle",
                fontName="Helvetica-Bold",
                fontSize=22,
                leading=26,
                textColor=rl_colors_rgb(*_NAVY),
                spaceAfter=2,
                alignment=TA_LEFT,
            ),
            "subtitle": ParagraphStyle(
                "BISubtitle",
                fontName="Helvetica",
                fontSize=9,
                leading=12,
                textColor=rl_colors_rgb(*_GREY),
                spaceAfter=4,
            ),
            "date": ParagraphStyle(
                "BIDate",
                fontName="Helvetica-Bold",
                fontSize=11,
                leading=14,
                textColor=rl_colors_rgb(*_DARK),
                spaceAfter=_SECTION_GAP,
            ),
            "section_heading": ParagraphStyle(
                "BISectionHead",
                fontName="Helvetica-Bold",
                fontSize=13,
                leading=16,
                textColor=rl_colors_rgb(*_NAVY),
                spaceBefore=_SECTION_GAP,
                spaceAfter=6,
            ),
            "exec_heading": ParagraphStyle(
                "BIExecHead",
                fontName="Helvetica-Bold",
                fontSize=13,
                leading=16,
                textColor=rl_colors_rgb(*_NAVY),
                spaceAfter=6,
            ),
            "body": ParagraphStyle(
                "BIBody",
                fontName="Helvetica",
                fontSize=10,
                leading=14,
                textColor=rl_colors_rgb(*_DARK),
                alignment=TA_JUSTIFY,
                spaceAfter=_PARA_GAP,
            ),
            "headline_title": ParagraphStyle(
                "BIHeadlineTitle",
                fontName="Helvetica-Bold",
                fontSize=10.5,
                leading=13,
                textColor=rl_colors_rgb(*_DARK),
                spaceAfter=1,
            ),
            "headline_oneliner": ParagraphStyle(
                "BIHeadlineOneliner",
                fontName="Helvetica",
                fontSize=9,
                leading=12,
                textColor=rl_colors_rgb(*_GREY),
                spaceAfter=1,
            ),
            "meta": ParagraphStyle(
                "BIMeta",
                fontName="Helvetica",
                fontSize=8,
                leading=10,
                textColor=rl_colors_rgb(*_LIGHT_GREY),
                spaceAfter=2,
            ),
            "bullet": ParagraphStyle(
                "BIBullet",
                fontName="Helvetica",
                fontSize=9,
                leading=12,
                textColor=rl_colors_rgb(*_DARK),
                leftIndent=_BULLET_INDENT,
                bulletIndent=6,
                spaceAfter=2,
            ),
            "article_title": ParagraphStyle(
                "BIArticleTitle",
                fontName="Helvetica-Bold",
                fontSize=9,
                leading=11,
                textColor=rl_colors_rgb(*_DARK),
                leftIndent=10,
                spaceAfter=1,
            ),
            "article_meta": ParagraphStyle(
                "BIArticleMeta",
                fontName="Helvetica-Oblique",
                fontSize=7.5,
                leading=10,
                textColor=rl_colors_rgb(*_LIGHT_GREY),
                leftIndent=10,
                spaceAfter=1,
            ),
            "footer_text": ParagraphStyle(
                "BIFooter",
                fontName="Helvetica",
                fontSize=7.5,
                leading=10,
                textColor=rl_colors_rgb(*_LIGHT_GREY),
                alignment=TA_CENTER,
                spaceBefore=_SECTION_GAP,
            ),
            "tone_badge": ParagraphStyle(
                "BITone",
                fontName="Helvetica-Bold",
                fontSize=8,
                leading=10,
                textColor=rl_colors_rgb(*_NAVY),
                spaceAfter=4,
            ),
        }

    # ── Section Builders ─────────────────────────────────────────────

    def _build_header(
        self, briefing: Any, styles: dict, Paragraph: type, Spacer: type,
    ) -> list:
        """Build the title block at the top of the first page."""
        elements: list = []

        elements.append(Paragraph(self._app_name, styles["title"]))
        elements.append(
            Paragraph("AI-Powered Daily Intelligence Briefing", styles["subtitle"])
        )

        try:
            parsed = datetime.fromisoformat(briefing.date)
            display_date = parsed.strftime("%A, %B %d, %Y")
        except (ValueError, TypeError):
            display_date = briefing.date or "Unknown Date"

        elements.append(Paragraph(display_date, styles["date"]))
        return elements

    def _build_executive(
        self,
        exec_summary: Any,
        styles: dict,
        Paragraph: type,
        Spacer: type,
        Table: type,
        TableStyle: type,
        rl_colors: Any,
    ) -> list:
        """Build the executive summary block with background shading."""
        elements: list = []

        # Build inner content
        inner: list = []
        inner.append(Paragraph("Executive Summary", styles["exec_heading"]))

        # Overview paragraphs (split on double newlines)
        for para in exec_summary.overview.split("\n\n"):
            para = para.strip()
            if para:
                inner.append(Paragraph(_escape(para), styles["body"]))

        # Key developments as bullets
        if exec_summary.key_developments:
            inner.append(Spacer(1, 4))
            for dev in exec_summary.key_developments:
                inner.append(
                    Paragraph(
                        f"<bullet>&bull;</bullet> {_escape(dev)}",
                        styles["bullet"],
                    )
                )

        # Tone badge
        tone = getattr(exec_summary, "tone", "neutral")
        if tone and tone.lower() != "neutral":
            inner.append(Spacer(1, 4))
            inner.append(
                Paragraph(f"TONE: {tone.upper()}", styles["tone_badge"])
            )

        # Wrap in a table for background color
        content_table = Table(
            [[inner]],
            colWidths=["100%"],
        )
        content_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), rl_colors.Color(*_VERY_LIGHT)),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ("LINEBEFORESTARTOFFSETX", (0, 0), (0, -1), 0),
                ("LINEBEFORE", (0, 0), (0, -1), 3, rl_colors.Color(*_NAVY)),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ])
        )

        elements.append(content_table)
        elements.append(Spacer(1, _SECTION_GAP))
        return elements

    def _build_headlines(
        self,
        headlines: list,
        styles: dict,
        Paragraph: type,
        Spacer: type,
    ) -> list:
        """Build the top headlines block."""
        elements: list = []
        elements.append(Paragraph("Top Headlines", styles["section_heading"]))

        for idx, hl in enumerate(headlines, 1):
            title = _escape(hl.title)
            url = getattr(hl, "url", "")
            if url:
                title = f'<a href="{_escape(url)}" color="blue">{title}</a>'
            elements.append(
                Paragraph(f"{idx}. {title}", styles["headline_title"])
            )

            oneliner = getattr(hl, "oneliner", "")
            if oneliner:
                elements.append(
                    Paragraph(_escape(oneliner), styles["headline_oneliner"])
                )

            source = getattr(hl, "source", "")
            if source:
                elements.append(
                    Paragraph(f"Source: {_escape(source)}", styles["meta"])
                )

            elements.append(Spacer(1, 4))

        elements.append(Spacer(1, _SECTION_GAP - 4))
        return elements

    def _build_section(
        self,
        section: Any,
        styles: dict,
        Paragraph: type,
        Spacer: type,
    ) -> list:
        """Build a single category section with summary, takeaways, and sources."""
        elements: list = []

        # Section heading with article count
        count = getattr(section, "article_count", 0)
        count_str = f" ({count} source{'s' if count != 1 else ''})" if count else ""
        heading = f"{_escape(section.category)}"
        meta = f'<font size="8" color="#{_rgb_hex(_LIGHT_GREY)}">{count_str}</font>'
        elements.append(
            Paragraph(f"{heading}{meta}", styles["section_heading"])
        )

        # Summary narrative
        elements.append(
            Paragraph(_escape(section.summary), styles["body"])
        )

        # Key takeaways
        takeaways = getattr(section, "key_takeaways", [])
        if takeaways:
            for ta in takeaways:
                elements.append(
                    Paragraph(
                        f"<bullet>&bull;</bullet> {_escape(ta)}",
                        styles["bullet"],
                    )
                )
            elements.append(Spacer(1, 4))

        # Source article titles
        titles = getattr(section, "article_titles", [])
        urls = getattr(section, "article_urls", [])
        if titles:
            elements.append(
                Paragraph("Source Articles:", styles["meta"])
            )
            for i, title in enumerate(titles):
                url = urls[i] if i < len(urls) else ""
                display = _escape(title)
                if url:
                    display = f'<a href="{_escape(url)}" color="blue">{display}</a>'
                elements.append(
                    Paragraph(f"[{i + 1}] {display}", styles["article_title"])
                )

        elements.append(Spacer(1, _SECTION_GAP))
        return elements

    def _build_footer_content(
        self,
        briefing: Any,
        styles: dict,
        Paragraph: type,
        Spacer: type,
    ) -> list:
        """Build the closing footer block (generation info, token usage)."""
        elements: list = []
        elements.append(Spacer(1, _SECTION_GAP))

        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        elements.append(
            Paragraph(
                f"Generated by {self._app_name} v{self._app_version} — {generated_at}",
                styles["footer_text"],
            )
        )

        usage = getattr(briefing, "token_usage", {})
        if usage:
            calls = usage.get("total_calls", 0)
            prompt_t = usage.get("total_prompt_tokens", 0)
            comp_t = usage.get("total_completion_tokens", 0)
            elements.append(
                Paragraph(
                    f"LLM calls: {calls} | "
                    f"Prompt tokens: {prompt_t} | "
                    f"Completion tokens: {comp_t}",
                    styles["footer_text"],
                )
            )

        return elements

    # ── Utilities ────────────────────────────────────────────────────

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """
        Strip path separators and dangerous characters from the filename.

        Prevents path-traversal attacks.
        """
        name = Path(filename).name
        sanitized = "".join(
            c if (c.isalnum() or c in "-_.") else "_" for c in name
        )
        if not sanitized or sanitized.startswith("."):
            sanitized = "briefing.pdf"
        return sanitized


# ── Module-level helpers (no class state needed) ─────────────────────

def _escape(text: str) -> str:
    """Escape XML-special characters for ReportLab Paragraph markup."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _rgb_hex(rgb_tuple: tuple[float, float, float]) -> str:
    """Convert an (r, g, b) 0-1 tuple to a hex string like '999999'."""
    return "".join(f"{int(c * 255):02x}" for c in rgb_tuple)


def rl_colors_rgb(r: float, g: float, b: float) -> Any:
    """
    Create a ReportLab Color from 0-1 RGB floats.

    Uses lazy import to avoid top-level reportlab dependency.
    """
    from reportlab.lib.colors import Color
    return Color(r, g, b)
