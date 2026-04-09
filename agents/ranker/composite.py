"""
BharatIntel — Composite Ranker with Diversity-Aware Reranking

Combines multi-dimensional LLM scores with heuristic signals into a single
composite score, then applies diversity-aware selection to ensure the final
list isn't dominated by one category.

Responsibilities:
  - Weighted composite score from LLM dimensions + heuristic signals
  - Diversity-aware greedy selection (Maximal Marginal Relevance variant)
  - Configurable weights for all signals
  - Maps scored articles to RankedArticle output

Dependencies: agents.ranker.llm_scorer, agents.ranker.signals
"""

from __future__ import annotations

from typing import Any

from agents.collector.models import RawArticle
from agents.curator.models import RankedArticle
from agents.ranker.llm_scorer import ArticleScores
from agents.ranker.signals import compute_heuristic_signals
from core.logger import get_logger

log = get_logger("ranker.composite")

# ── Default weights ──────────────────────────────────────────────────
# LLM dimensions (sum to ~0.7 of total signal)
# Heuristic signals (sum to ~0.3 of total signal)

DEFAULT_WEIGHTS: dict[str, float] = {
    # LLM dimensions (raw 1-10, normalized to 0-1 internally)
    "relevance": 0.25,
    "impact": 0.20,
    "novelty": 0.15,
    "timeliness": 0.10,
    # Heuristic signals (already 0-1)
    "recency": 0.15,
    "source_authority": 0.10,
    "content_richness": 0.05,
}


class CompositeRanker:
    """
    Computes weighted composite scores and applies diversity reranking.

    Args:
        weights:           Signal weight overrides (merged with defaults)
        diversity_penalty: How much to penalize same-category articles during
                           reranking. 0.0 = no diversity enforcement,
                           1.0 = heavily penalize repeats.
        min_per_category:  Minimum articles to include from any represented category
                           before diversity penalty kicks in.
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        diversity_penalty: float = 0.15,
        min_per_category: int = 1,
    ):
        self._weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self._weights.update(weights)
        self._diversity_penalty = diversity_penalty
        self._min_per_category = min_per_category

        # Normalize weights so they sum to 1.0
        total = sum(self._weights.values())
        if total > 0:
            self._weights = {k: v / total for k, v in self._weights.items()}

    def compute_composite_score(
        self,
        llm_scores: ArticleScores,
        heuristic_signals: dict[str, float],
    ) -> float:
        """
        Compute a single composite score in [0.0, 1.0].

        LLM dimensions (1-10) are normalized to [0, 1] before weighting.
        Heuristic signals are already [0, 1].
        """
        # Normalize LLM dimensions to 0-1
        llm_normalized = {k: v / 10.0 for k, v in llm_scores.dimensions.items()}

        # Combine all signals
        all_signals = {**llm_normalized, **heuristic_signals}

        composite = 0.0
        for signal_name, weight in self._weights.items():
            value = all_signals.get(signal_name, 0.0)
            composite += weight * value

        return round(composite, 4)

    def rank(
        self,
        articles: list[RawArticle],
        llm_scores: dict[int, ArticleScores],
        top_n: int = 25,
    ) -> list[RankedArticle]:
        """
        Compute composite scores and apply diversity-aware selection.

        Pipeline:
          1. Compute heuristic signals for each article
          2. Combine with LLM scores into composite
          3. Sort by composite descending
          4. Apply diversity-aware greedy reranking
          5. Truncate to top_n

        Args:
            articles:    Original articles (index-aligned with llm_scores keys)
            llm_scores:  Multi-dimensional scores from LLMScorer
            top_n:       Maximum articles to return

        Returns:
            List[RankedArticle] in final rank order, length <= top_n.
        """
        if not articles:
            return []

        # ── Step 1+2: Compute composite scores ──────────────────────
        scored: list[tuple[int, float, ArticleScores, dict[str, float]]] = []
        for i, article in enumerate(articles):
            ls = llm_scores.get(i)
            if ls is None:
                continue

            heuristic = compute_heuristic_signals(
                source_name=article.source_name,
                published_at=article.published_at,
                snippet=article.snippet,
            )

            composite = self.compute_composite_score(ls, heuristic)
            scored.append((i, composite, ls, heuristic))

        # ── Step 3: Sort by composite descending ────────────────────
        scored.sort(key=lambda x: x[1], reverse=True)

        log.info(
            "composite_scores_computed",
            total=len(scored),
            top_score=scored[0][1] if scored else 0,
            bottom_score=scored[-1][1] if scored else 0,
        )

        # ── Step 4: Diversity-aware greedy selection ─────────────────
        selected = self._diversity_rerank(scored, top_n)

        # ── Step 5: Convert to RankedArticle ─────────────────────────
        result: list[RankedArticle] = []
        for idx, composite, ls, heuristic in selected:
            article = articles[idx]

            # Map composite (0-1) back to 1-10 integer for RankedArticle
            final_score = max(1, min(10, round(composite * 10)))

            try:
                ranked = RankedArticle(
                    title=article.title,
                    url=article.url,
                    source_name=article.source_name,
                    relevance_score=final_score,
                    assigned_category=ls.category,
                    rank_reason=ls.reason,
                    published_at=article.published_at,
                    snippet=article.snippet,
                    author=article.author,
                    image_url=article.image_url,
                    fetched_at=article.fetched_at,
                    original_categories=list(article.categories),
                )
                result.append(ranked)
            except (ValueError, TypeError) as exc:
                log.warning("composite_article_error", index=idx, error=str(exc))
                continue

        log.info(
            "composite_ranking_done",
            output_count=len(result),
            categories={r.assigned_category for r in result},
        )

        return result

    def _diversity_rerank(
        self,
        scored: list[tuple[int, float, ArticleScores, dict[str, float]]],
        top_n: int,
    ) -> list[tuple[int, float, ArticleScores, dict[str, float]]]:
        """
        Greedy diversity-aware selection.

        For each candidate:
          - If its category already has >= min_per_category articles selected,
            apply a cumulative penalty to its effective score.
          - Select the candidate with the highest effective score.

        This ensures category diversity without rigidly enforcing quotas.
        Articles with very high scores still break through the penalty.
        """
        if self._diversity_penalty <= 0 or len(scored) <= top_n:
            return scored[:top_n]

        selected: list[tuple[int, float, ArticleScores, dict[str, float]]] = []
        remaining = list(scored)
        category_counts: dict[str, int] = {}

        while remaining and len(selected) < top_n:
            best_idx = -1
            best_effective = -1.0

            for ri, (idx, composite, ls, heuristic) in enumerate(remaining):
                cat_count = category_counts.get(ls.category, 0)

                # Apply cumulative penalty for over-represented categories
                if cat_count >= self._min_per_category:
                    excess = cat_count - self._min_per_category + 1
                    penalty = self._diversity_penalty * excess
                    effective = composite * (1.0 - min(penalty, 0.5))  # Cap penalty at 50%
                else:
                    effective = composite

                if effective > best_effective:
                    best_effective = effective
                    best_idx = ri

            if best_idx < 0:
                break

            pick = remaining.pop(best_idx)
            selected.append(pick)
            category_counts[pick[2].category] = category_counts.get(pick[2].category, 0) + 1

        if category_counts:
            log.debug("diversity_rerank_distribution", categories=dict(category_counts))

        return selected
