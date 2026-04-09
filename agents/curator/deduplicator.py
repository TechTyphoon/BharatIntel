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

from typing import TYPE_CHECKING

import numpy as np

from core.logger import get_logger

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

if TYPE_CHECKING:
    from agents.collector.models import RawArticle

log = get_logger("curator.dedup")

_DEFAULT_MODEL = "all-MiniLM-L6-v2"


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
    ):
        self.similarity_threshold = similarity_threshold
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            )
        import os
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
        log.info("dedup_model_loading", model=model_name)
        self._model = SentenceTransformer(model_name)
        log.info("dedup_model_loaded", model=model_name)

    def _build_text(self, article: RawArticle) -> str:
        """Combine title and snippet into a single string for embedding."""
        parts = [article.title]
        if article.snippet:
            parts.append(article.snippet[:200])
        return " ".join(parts)

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

        texts = [self._build_text(a) for a in articles]

        log.info("dedup_encoding", article_count=len(texts))
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
                    )

        result = [a for idx, a in enumerate(articles) if idx not in remove_indices]

        log.info(
            "dedup_complete",
            input_count=len(articles),
            output_count=len(result),
            removed=len(remove_indices),
        )
        return result
