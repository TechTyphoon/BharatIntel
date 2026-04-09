"""
BharatIntel — PDF Writer

Converts rendered HTML into a PDF file using WeasyPrint.

Responsibilities:
  - Accept an HTML string and write it to a PDF file
  - Apply page size / margin overrides at the WeasyPrint level
  - Handle missing fonts, broken CSS, and encoding issues gracefully
  - Log warnings emitted by WeasyPrint without crashing

Dependencies: weasyprint
"""

from __future__ import annotations

import logging
from pathlib import Path

from core.exceptions import RenderError
from core.logger import get_logger

log = get_logger("publisher.pdf")

# Suppress verbose WeasyPrint / fontconfig warnings to our own handler
logging.getLogger("weasyprint").setLevel(logging.ERROR)
logging.getLogger("fontTools").setLevel(logging.ERROR)


class PDFWriter:
    """
    Converts HTML content to a PDF file on disk.

    Args:
        output_dir: Directory where PDF files are written.
                    Created automatically if it doesn't exist.
    """

    def __init__(self, output_dir: str = "output"):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        log.info("pdf_writer_initialized", output_dir=str(self._output_dir.resolve()))

    def write(
        self,
        html: str,
        filename: str,
        base_url: str | None = None,
    ) -> Path:
        """
        Render HTML to a PDF and save it.

        Args:
            html:     Fully rendered HTML string (from TemplateRenderer)
            filename: Output filename (e.g. "briefing_2026-04-08.pdf")
            base_url: Base URL for resolving relative paths in HTML
                      (images, stylesheets). Defaults to the output directory.

        Returns:
            Absolute Path to the written PDF file.

        Raises:
            RenderError: If WeasyPrint fails to produce the PDF.
        """
        # Lazy import — weasyprint has heavy native dependencies; defer loading
        # until actually called so the rest of the system can run without it.
        try:
            from weasyprint import HTML as WeasyHTML
        except ImportError as exc:
            raise RenderError(
                "weasyprint is not installed. Install with: pip install weasyprint",
                context={"error": str(exc)},
            ) from exc

        if not html or not html.strip():
            raise RenderError(
                "Cannot generate PDF from empty HTML",
                context={"filename": filename},
            )

        # Sanitize filename
        safe_name = self._sanitize_filename(filename)
        pdf_path = self._output_dir / safe_name

        if base_url is None:
            base_url = str(self._output_dir.resolve())

        log.info(
            "pdf_generation_start",
            filename=safe_name,
            html_length=len(html),
        )

        try:
            doc = WeasyHTML(string=html, base_url=base_url)
            doc.write_pdf(str(pdf_path))
        except Exception as exc:
            raise RenderError(
                f"PDF generation failed: {exc}",
                context={"filename": safe_name, "error": str(exc)},
            ) from exc

        file_size = pdf_path.stat().st_size

        log.info(
            "pdf_generated",
            path=str(pdf_path.resolve()),
            size_bytes=file_size,
        )

        return pdf_path.resolve()

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """
        Strip path separators and dangerous characters from the filename.

        Ensures the file is always created inside output_dir (no path-traversal).
        """
        # Take only the basename in case someone passes a path
        name = Path(filename).name
        # Remove anything that isn't alphanumeric, dash, underscore, or dot
        sanitized = "".join(
            c if (c.isalnum() or c in "-_.") else "_" for c in name
        )
        if not sanitized or sanitized.startswith("."):
            sanitized = "briefing.pdf"
        return sanitized
