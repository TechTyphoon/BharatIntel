"""
BharatIntel — Summarizer Data Models

Defines Headline, BriefingSection, ExecutiveSummary, and Briefing.
These are consumed directly by the publisher to render the PDF.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Headline:
    """
    A single top headline for the briefing masthead.

    Fields:
        title:    Rewritten headline (concise, punchy)
        oneliner: One-sentence context for the headline
        url:      Link to original article
        source:   Origin source name
    """

    title: str
    oneliner: str
    url: str
    source: str = ""

    def __post_init__(self) -> None:
        if not self.title or not self.title.strip():
            raise ValueError("Headline.title must be non-empty")

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "oneliner": self.oneliner,
            "url": self.url,
            "source": self.source,
        }


@dataclass(frozen=True)
class BriefingSection:
    """
    A themed section of the daily briefing (e.g. "Technology", "World").

    Fields:
        category:       Section category name (title-cased for display)
        summary:        LLM-generated 3-5 sentence narrative summary
        key_takeaways:  2-4 actionable bullet points distilled from the summary
        article_count:  Number of source articles used to generate this section
        article_urls:   URLs of the source articles (for attribution)
        article_titles: Original titles (for reference / fallback rendering)
    """

    category: str
    summary: str
    key_takeaways: list[str] = field(default_factory=list)
    article_count: int = 0
    article_urls: list[str] = field(default_factory=list)
    article_titles: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.category or not self.category.strip():
            raise ValueError("BriefingSection.category must be non-empty")
        if not self.summary or not self.summary.strip():
            raise ValueError("BriefingSection.summary must be non-empty")

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "summary": self.summary,
            "key_takeaways": self.key_takeaways,
            "article_count": self.article_count,
            "article_urls": self.article_urls,
            "article_titles": self.article_titles,
        }


@dataclass(frozen=True)
class ExecutiveSummary:
    """
    Top-level overview of the entire day's briefing.

    Fields:
        overview:        2-3 paragraph narrative of the day's most important developments
        key_developments: 3-5 bullet points of the biggest stories across all categories
        tone:            Overall tone descriptor (e.g. "cautious", "optimistic", "volatile")
    """

    overview: str
    key_developments: list[str] = field(default_factory=list)
    tone: str = "neutral"

    def __post_init__(self) -> None:
        if not self.overview or not self.overview.strip():
            raise ValueError("ExecutiveSummary.overview must be non-empty")

    def to_dict(self) -> dict:
        return {
            "overview": self.overview,
            "key_developments": self.key_developments,
            "tone": self.tone,
        }


@dataclass
class Briefing:
    """
    Complete daily briefing — the final output of the summarizer agent.

    Fields:
        date:              Briefing date string (ISO format)
        executive_summary: Top-level overview of the day
        headlines:         Top N headlines for the masthead
        sections:          Themed summary sections
        token_usage:       LLM token consumption for this briefing
    """

    date: str
    executive_summary: ExecutiveSummary | None = None
    headlines: list[Headline] = field(default_factory=list)
    sections: list[BriefingSection] = field(default_factory=list)
    token_usage: dict[str, int] = field(default_factory=dict)
    generated_at: str = ""

    def to_dict(self) -> dict:
        from datetime import datetime, timezone

        return {
            "date": self.date,
            "generated_at": self.generated_at or datetime.now(timezone.utc).isoformat(),
            "executive_summary": self.executive_summary.to_dict() if self.executive_summary else None,
            "headlines": [h.to_dict() for h in self.headlines],
            "sections": [s.to_dict() for s in self.sections],
            "token_usage": self.token_usage,
        }
