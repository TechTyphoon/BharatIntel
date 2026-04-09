"""
BharatIntel — Custom Exception Hierarchy

All exceptions inherit from BharatIntelError.
Each agent catches its own exceptions; no agent crashes the pipeline.
"""


class BharatIntelError(Exception):
    """Base exception for the entire system."""

    def __init__(self, message: str, context: dict | None = None):
        self.context = context or {}
        super().__init__(message)


# ── Collector Exceptions ─────────────────────────────────────────────

class SourceFetchError(BharatIntelError):
    """Raised when a news source fails to respond or returns bad status."""
    pass


class ParseError(BharatIntelError):
    """Raised when article content cannot be parsed from a source."""
    pass


# ── LLM Exceptions ──────────────────────────────────────────────────

class LLMError(BharatIntelError):
    """Base for all LLM-related failures."""
    pass


class LLMRateLimitError(LLMError):
    """LLM provider rate limit hit."""
    pass


class LLMResponseError(LLMError):
    """LLM returned malformed or unparseable response."""
    pass


class LLMTimeoutError(LLMError):
    """LLM call timed out."""
    pass


# ── Publisher Exceptions ─────────────────────────────────────────────

class RenderError(BharatIntelError):
    """PDF or template rendering failure."""
    pass


# ── Pipeline Exceptions ─────────────────────────────────────────────

class PipelineError(BharatIntelError):
    """Fatal pipeline-level failure (threshold not met, etc.)."""
    pass
