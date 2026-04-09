"""
BharatIntel — Heuristic Signal Calculator

Computes non-LLM ranking signals from article metadata.

Responsibilities:
  - Recency score: exponential decay based on article age
  - Source authority score: configurable per-source weight
  - Content richness: reward articles with substantial snippets

These signals are combined with LLM dimensions in the composite ranker.
No API calls, no LLM cost — pure computation.

Dependencies: none (stdlib only)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from core.logger import get_logger

log = get_logger("ranker.signals")


# ── Source authority tiers ────────────────────────────────────────────
# Higher = more trusted. Range 1-10.
# Sources not listed default to _DEFAULT_AUTHORITY.

_SOURCE_AUTHORITY: dict[str, int] = {
    "reuters": 9,
    "bbc": 9,
    "al jazeera": 8,
    "the hindu": 8,
    "ndtv": 7,
    "ars technica": 7,
    "gnews": 5,
    "hacker news": 6,
}

_DEFAULT_AUTHORITY = 5


def _match_source_authority(source_name: str) -> int:
    """Look up source authority by substring match (case-insensitive)."""
    name_lower = source_name.lower()
    for key, score in _SOURCE_AUTHORITY.items():
        if key in name_lower:
            return score
    return _DEFAULT_AUTHORITY


def recency_score(
    published_at: Optional[datetime],
    now: Optional[datetime] = None,
    half_life_hours: float = 12.0,
) -> float:
    """
    Exponential decay score based on article age.

    Returns a float in [0.0, 1.0]:
      - 1.0 = just published
      - 0.5 = half_life_hours ago
      - ~0.0 = very old

    Articles with no publish date get a neutral 0.4.
    """
    if published_at is None:
        return 0.4

    if now is None:
        now = datetime.now(timezone.utc)

    # Ensure timezone-aware comparison
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)

    age_hours = max(0, (now - published_at).total_seconds() / 3600)

    # Exponential decay: score = 2^(-age / half_life)
    import math
    return math.pow(2, -(age_hours / half_life_hours))


def source_authority_score(source_name: str) -> float:
    """
    Normalized source authority score in [0.0, 1.0].

    Based on source name matching against known authority tiers.
    """
    raw = _match_source_authority(source_name)
    return raw / 10.0


def content_richness_score(snippet: str) -> float:
    """
    Score how substantial the article content is in [0.0, 1.0].

    Based on snippet length — longer snippets suggest more content.
    Caps at 400 chars.
    """
    if not snippet:
        return 0.1
    length = len(snippet.strip())
    return min(1.0, length / 400.0)


def compute_heuristic_signals(
    source_name: str,
    published_at: Optional[datetime],
    snippet: str,
    now: Optional[datetime] = None,
) -> dict[str, float]:
    """
    Compute all heuristic signals for a single article.

    Returns:
        Dict with keys: recency, source_authority, content_richness
        All values in [0.0, 1.0].
    """
    return {
        "recency": recency_score(published_at, now),
        "source_authority": source_authority_score(source_name),
        "content_richness": content_richness_score(snippet),
    }
