"""
BharatIntel — Curator Data Models

Defines RankedArticle — the output of the curator agent.
Extends RawArticle data with LLM-assigned score, category, and reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class RankedArticle:
    """
    An article that has been scored and categorized by the curator.

    Fields:
        title:          Original headline
        url:            Canonical URL
        source_name:    Origin source name
        published_at:   Publication timestamp (UTC)
        snippet:        Article excerpt
        author:         Author if available
        image_url:      Image if available
        fetched_at:     Collection timestamp
        relevance_score: LLM-assigned score 1-10
        assigned_category: LLM-assigned category (normalized)
        rank_reason:    One-sentence justification from LLM
        original_categories: Categories from source config
    """

    title: str
    url: str
    source_name: str
    relevance_score: int
    assigned_category: str
    rank_reason: str = ""
    published_at: Optional[datetime] = None
    snippet: str = ""
    author: Optional[str] = None
    image_url: Optional[str] = None
    fetched_at: Optional[datetime] = None
    original_categories: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.title or not self.title.strip():
            raise ValueError("RankedArticle.title must be non-empty")
        if not self.url or not self.url.strip():
            raise ValueError("RankedArticle.url must be non-empty")
        if not (1 <= self.relevance_score <= 10):
            raise ValueError(f"relevance_score must be 1-10, got {self.relevance_score}")

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "source_name": self.source_name,
            "relevance_score": self.relevance_score,
            "assigned_category": self.assigned_category,
            "rank_reason": self.rank_reason,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "snippet": self.snippet,
            "author": self.author,
            "image_url": self.image_url,
            "fetched_at": self.fetched_at.isoformat() if self.fetched_at else None,
            "original_categories": self.original_categories,
        }
