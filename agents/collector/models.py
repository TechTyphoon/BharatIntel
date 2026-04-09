"""
BharatIntel — Collector Data Models

Defines the RawArticle dataclass — the canonical output of the collector agent.
Every source adapter must produce List[RawArticle].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class RawArticle:
    """
    A single news article as harvested from a source.

    Fields:
        title:        Article headline (required)
        url:          Canonical URL of the article (required)
        source_name:  Human-readable name of the source (e.g. "Reuters Top News")
        published_at: Publication datetime (UTC preferred); None if unavailable
        snippet:      First ~500 chars of article body or description
        categories:   Categories inherited from source config
        author:       Author name if available
        image_url:    Thumbnail / hero image URL if available
        fetched_at:   Timestamp when this article was collected
    """

    title: str
    url: str
    source_name: str
    published_at: Optional[datetime] = None
    snippet: str = ""
    categories: list[str] = field(default_factory=list)
    author: Optional[str] = None
    image_url: Optional[str] = None
    fetched_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        if not self.title or not self.title.strip():
            raise ValueError("RawArticle.title must be non-empty")
        if not self.url or not self.url.strip():
            raise ValueError("RawArticle.url must be non-empty")

    @property
    def has_content(self) -> bool:
        return bool(self.snippet and self.snippet.strip())

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "source_name": self.source_name,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "snippet": self.snippet,
            "categories": self.categories,
            "author": self.author,
            "image_url": self.image_url,
            "fetched_at": self.fetched_at.isoformat(),
        }
