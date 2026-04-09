"""
BharatIntel — Publisher Data Models

Defines PublishResult — the output metadata of a publishing run.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PublishResult:
    """
    Metadata returned after a successful publish operation.

    Fields:
        date:           Briefing date (ISO format)
        pdf_path:       Absolute path to the generated PDF (empty if PDF skipped)
        html_path:      Absolute path to the generated HTML file
        json_path:      Absolute path to the generated JSON archive
        section_count:  Number of sections rendered
        headline_count: Number of headlines rendered
        has_executive:  Whether an executive summary was included
        file_sizes:     Mapping of output type → file size in bytes
    """

    date: str = ""
    pdf_path: str = ""
    html_path: str = ""
    json_path: str = ""
    section_count: int = 0
    headline_count: int = 0
    has_executive: bool = False
    file_sizes: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "pdf_path": self.pdf_path,
            "html_path": self.html_path,
            "json_path": self.json_path,
            "section_count": self.section_count,
            "headline_count": self.headline_count,
            "has_executive": self.has_executive,
            "file_sizes": self.file_sizes,
        }
