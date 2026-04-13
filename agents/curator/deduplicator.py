"""
BharatIntel — Semantic Deduplicator

Removes near-duplicate articles using sentence embedding cosine similarity.

Responsibilities:
  - Encode article titles + snippets into dense vectors
  - Compare all pairs using cosine similarity
  - Remove articles whose similarity exceeds threshold (keep the earlier one)
  - Works entirely locally — no API calls, no LLM cost

Dependencies: sentence-transformers, numpy

Design notes:
  - Uses a lightweight model (all-MiniLM-L6-v2, ~80MB) for speed
  - Model is loaded once and reused across calls
  - O(n²) pairwise comparison is fine for n < 500 articles/day
"""

from __future__ import annotations

from difflib import SequenceMatcher
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from core.logger import get_logger

if TYPE_CHECKING:
    from agents.collector.models import RawArticle

log = get_logger("curator.dedup")

_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_DEFAULT_MODE = "auto"
_LOW_MEMORY_THRESHOLD_MB = 650


def _read_memory_limit_bytes() -> int | None:
    """
    Best-effort detection of the container memory limit.

    Supports both cgroup v2 and v1. Returns None when unavailable.
    """
    candidates = (
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    )

    for raw_path in candidates:
        path = Path(raw_path)
        if not path.exists():
            continue

        try:
            raw = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue

        if not raw or raw == "max":
            continue

        try:
            limit = int(raw)
        except ValueError:
            continue

        # Ignore obviously bogus sentinel values from unrestricted hosts.
        if limit <= 0 or limit >= 1 << 60:
            continue

        return limit

    return None


class Deduplicator:
    """
    Removes near-duplicate articles by semantic similarity of title + snippet.

    Args:
        model_name:          Sentence-transformer model identifier
        similarity_threshold: Cosine similarity above which two articles are dupes (0.0–1.0)
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        similarity_threshold: float = 0.85,
        mode: str = _DEFAULT_MODE,
        low_memory_threshold_mb: int = _LOW_MEMORY_THRESHOLD_MB,
    ):
        self.similarity_threshold = similarity_threshold
        self._model_name = model_name
        self._model = None

        requested_mode = os.environ.get("DEDUP_MODE", mode).strip().lower()
        if requested_mode not in {"auto", "semantic", "lexical"}:
            requested_mode = _DEFAULT_MODE

        self._memory_limit_bytes = _read_memory_limit_bytes()
        self._low_memory_threshold_bytes = low_memory_threshold_mb * 1024 * 1024
        self._mode = self._resolve_mode(requested_mode)

        log.info(
            "dedup_mode_selected",
            requested_mode=requested_mode,
            active_mode=self._mode,
            memory_limit_mb=(
                round(self._memory_limit_bytes / (1024 * 1024), 1)
                if self._memory_limit_bytes is not None else None
            ),
            low_memory_threshold_mb=low_memory_threshold_mb,
        )

    def _resolve_mode(self, requested_mode: str) -> str:
        """Resolve the deduplication mode for the current environment."""
        if requested_mode in {"semantic", "lexical"}:
            return requested_mode

        if (
            self._memory_limit_bytes is not None and
            self._memory_limit_bytes <= self._low_memory_threshold_bytes
        ):
            return "lexical"

        return "semantic"

    def _load_model(self) -> None:
        """Lazy-load the sentence-transformer only when semantic mode is used."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            ) from exc

        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
        log.info("dedup_model_loading", model=self._model_name)
        self._model = SentenceTransformer(self._model_name)
        log.info("dedup_model_loaded", model=self._model_name)

    def _build_text(self, article: RawArticle) -> str:
        """Combine title and snippet into a single string for embedding."""
        parts = [article.title]
        if article.snippet:
            parts.append(article.snippet[:200])
        return " ".join(parts)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Create a compact token set for lightweight lexical similarity."""
        return {
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if len(token) > 2
        }

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for deterministic lexical comparisons."""
        return " ".join(re.findall(r"[a-z0-9]+", text.lower()))

    @staticmethod
    def _jaccard_similarity(left: set[str], right: set[str]) -> float:
        """Compute Jaccard similarity between two token sets."""
        if not left or not right:
            return 0.0
        union = left | right
        if not union:
            return 0.0
        return len(left & right) / len(union)

    def _cosine_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute pairwise cosine similarity for a matrix of row vectors.

        Args:
            embeddings: (n, dim) array of L2-normalized embeddings

        Returns:
            (n, n) similarity matrix
        """
        # sentence-transformers already returns normalized vectors by default
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Guard against zero-norm edge case
        norms = np.where(norms == 0, 1.0, norms)
        normalized = embeddings / norms
        return normalized @ normalized.T

    def _deduplicate_lexical(self, articles: list[RawArticle]) -> list[RawArticle]:
        """
        Lightweight near-duplicate removal using token overlap.

        This mode avoids loading PyTorch/sentence-transformers on low-memory
        deployments such as Render's 512Mi free tier.
        """
        title_tokens = [self._tokenize(article.title) for article in articles]
        body_tokens = [self._tokenize(self._build_text(article)) for article in articles]
        normalized_titles = [self._normalize_text(article.title) for article in articles]

        remove_indices: set[int] = set()
        n = len(articles)

        for i in range(n):
            if i in remove_indices:
                continue
            for j in range(i + 1, n):
                if j in remove_indices:
                    continue

                title_sim = self._jaccard_similarity(title_tokens[i], title_tokens[j])
                body_sim = self._jaccard_similarity(body_tokens[i], body_tokens[j])
                title_ratio = SequenceMatcher(
                    None,
                    normalized_titles[i],
                    normalized_titles[j],
                ).ratio()

                is_duplicate = (
                    title_ratio >= 0.92 or
                    title_sim >= 0.88 or
                    (title_ratio >= 0.78 and body_sim >= 0.80) or
                    (title_sim >= 0.72 and body_sim >= 0.82)
                )

                if is_duplicate:
                    remove_indices.add(j)
                    log.debug(
                        "dedup_pair_removed",
                        kept=articles[i].title[:80],
                        removed=articles[j].title[:80],
                        similarity=round(max(title_sim, body_sim, title_ratio), 3),
                        mode="lexical",
                    )

        result = [a for idx, a in enumerate(articles) if idx not in remove_indices]

        log.info(
            "dedup_complete",
            mode="lexical",
            input_count=len(articles),
            output_count=len(result),
            removed=len(remove_indices),
        )
        return result

    def deduplicate(self, articles: list[RawArticle]) -> list[RawArticle]:
        """
        Remove near-duplicate articles.

        Keeps the first occurrence (by input order) when duplicates are found.

        Args:
            articles: Input articles (order matters — earlier = kept)

        Returns:
            De-duplicated list of articles.
        """
        if len(articles) <= 1:
            return list(articles)

        if self._mode == "lexical":
            log.info("dedup_encoding", article_count=len(articles), mode="lexical")
            return self._deduplicate_lexical(articles)

        self._load_model()

        texts = [self._build_text(a) for a in articles]

        log.info("dedup_encoding", article_count=len(texts), mode="semantic")
        embeddings = self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        sim_matrix = self._cosine_similarity_matrix(embeddings)

        # Greedy dedup: iterate in order, mark later duplicates for removal
        remove_indices: set[int] = set()
        n = len(articles)

        for i in range(n):
            if i in remove_indices:
                continue
            for j in range(i + 1, n):
                if j in remove_indices:
                    continue
                if sim_matrix[i, j] >= self.similarity_threshold:
                    remove_indices.add(j)
                    log.debug(
                        "dedup_pair_removed",
                        kept=articles[i].title[:80],
                        removed=articles[j].title[:80],
                        similarity=round(float(sim_matrix[i, j]), 3),
                        mode="semantic",
                    )

        result = [a for idx, a in enumerate(articles) if idx not in remove_indices]

        log.info(
            "dedup_complete",
            mode="semantic",
            input_count=len(articles),
            output_count=len(result),
            removed=len(remove_indices),
        )
        return result
