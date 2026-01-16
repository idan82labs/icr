"""
Non-generative aggregation operations for RLM.

Aggregates results from multiple retrieval iterations without
using generative models, preserving the non-generative property
of the retrieval pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog

if TYPE_CHECKING:
    from icd.config import Config
    from icd.retrieval.hybrid import Chunk
    from icd.rlm.planner import RetrievalPlan, SubQuery

logger = structlog.get_logger(__name__)


@dataclass
class AggregatedResult:
    """Result of aggregating multiple retrieval iterations."""

    chunks: list["Chunk"]
    scores: list[float]
    sources: dict[str, list[str]]  # chunk_id -> list of sub-query types
    coverage: dict[str, float]  # query_type -> coverage ratio
    entropy: float
    metadata: dict[str, Any] = field(default_factory=dict)


class Aggregator:
    """
    Non-generative aggregator for RLM results.

    Features:
    - Score fusion from multiple sources
    - Deduplication with score boosting
    - Source tracking
    - Coverage computation
    """

    def __init__(self, config: "Config" | None = None) -> None:
        """
        Initialize the aggregator.

        Args:
            config: ICD configuration (optional).
        """
        self.config = config

        # Aggregation parameters
        self.duplicate_boost = 0.2  # Boost for appearing in multiple results
        self.source_diversity_bonus = 0.1  # Bonus for diverse sources
        self.max_chunks = 50  # Maximum chunks in aggregated result

    def aggregate(
        self,
        plan: "RetrievalPlan",
        iteration_results: list[tuple["SubQuery", list["Chunk"], list[float]]],
    ) -> AggregatedResult:
        """
        Aggregate results from multiple iterations.

        Args:
            plan: The retrieval plan.
            iteration_results: List of (sub_query, chunks, scores) tuples.

        Returns:
            AggregatedResult with fused chunks and scores.
        """
        logger.debug(
            "Aggregating results",
            num_iterations=len(iteration_results),
        )

        # Collect all chunks with their scores and sources
        chunk_data: dict[str, dict[str, Any]] = {}

        for sub_query, chunks, scores in iteration_results:
            # Determine source type (initial results have None sub_query)
            source_type = sub_query.query_type.value if sub_query else "initial"

            for chunk, score in zip(chunks, scores):
                chunk_id = chunk.chunk_id

                if chunk_id not in chunk_data:
                    chunk_data[chunk_id] = {
                        "chunk": chunk,
                        "scores": [],
                        "sources": [],
                        "max_score": 0.0,
                    }

                chunk_data[chunk_id]["scores"].append(score)
                chunk_data[chunk_id]["sources"].append(source_type)
                chunk_data[chunk_id]["max_score"] = max(
                    chunk_data[chunk_id]["max_score"], score
                )

        # Compute fused scores
        fused_chunks: list[tuple["Chunk", float, list[str]]] = []

        for chunk_id, data in chunk_data.items():
            fused_score = self._fuse_scores(
                scores=data["scores"],
                sources=data["sources"],
                max_score=data["max_score"],
            )

            fused_chunks.append((data["chunk"], fused_score, data["sources"]))

        # Sort by fused score
        fused_chunks.sort(key=lambda x: x[1], reverse=True)

        # Limit to max chunks
        fused_chunks = fused_chunks[: self.max_chunks]

        # Build result
        chunks = [fc[0] for fc in fused_chunks]
        scores = [fc[1] for fc in fused_chunks]
        sources = {fc[0].chunk_id: fc[2] for fc in fused_chunks}

        # Compute coverage
        coverage = self._compute_coverage(plan, iteration_results)

        # Compute entropy
        from icd.retrieval.entropy import EntropyCalculator

        entropy_calc = EntropyCalculator()
        entropy = entropy_calc.compute_entropy(scores)

        logger.debug(
            "Aggregation complete",
            num_chunks=len(chunks),
            entropy=entropy,
            coverage=coverage,
        )

        return AggregatedResult(
            chunks=chunks,
            scores=scores,
            sources=sources,
            coverage=coverage,
            entropy=entropy,
            metadata={
                "total_iterations": len(iteration_results),
                "total_unique_chunks": len(chunk_data),
                "duplicate_ratio": 1.0 - len(chunk_data) / max(
                    1, sum(len(ir[1]) for ir in iteration_results)
                ),
            },
        )

    def _fuse_scores(
        self,
        scores: list[float],
        sources: list[str],
        max_score: float,
    ) -> float:
        """
        Fuse multiple scores for the same chunk.

        Uses a combination of:
        - Max score (primary signal)
        - Duplicate boost (appearing in multiple results)
        - Source diversity bonus
        """
        # Base: max score
        fused = max_score

        # Duplicate boost: log-scaled bonus for appearing multiple times
        if len(scores) > 1:
            duplicate_factor = np.log2(len(scores)) * self.duplicate_boost
            fused += duplicate_factor

        # Source diversity bonus
        unique_sources = len(set(sources))
        if unique_sources > 1:
            diversity_factor = (unique_sources - 1) * self.source_diversity_bonus
            fused += diversity_factor

        return float(fused)

    def _compute_coverage(
        self,
        plan: "RetrievalPlan",
        iteration_results: list[tuple["SubQuery", list["Chunk"], list[float]]],
    ) -> dict[str, float]:
        """Compute coverage ratio for each query type."""
        coverage: dict[str, float] = {}

        # Count sub-queries and results by type
        type_counts: dict[str, int] = {}
        type_results: dict[str, int] = {}

        for sq in plan.sub_queries:
            qt = sq.query_type.value
            type_counts[qt] = type_counts.get(qt, 0) + 1

        for sub_query, chunks, _ in iteration_results:
            # Skip initial results which have None sub_query
            if sub_query is None:
                continue
            qt = sub_query.query_type.value
            if chunks:
                type_results[qt] = type_results.get(qt, 0) + 1

        # Compute coverage ratios
        for qt, count in type_counts.items():
            results = type_results.get(qt, 0)
            coverage[qt] = results / count if count > 0 else 0.0

        return coverage

    def incremental_update(
        self,
        current_result: AggregatedResult,
        new_sub_query: "SubQuery",
        new_chunks: list["Chunk"],
        new_scores: list[float],
    ) -> AggregatedResult:
        """
        Incrementally update aggregated result with new iteration.

        Args:
            current_result: Current aggregated result.
            new_sub_query: The new sub-query that was executed.
            new_chunks: New chunks from the sub-query.
            new_scores: Scores for new chunks.

        Returns:
            Updated AggregatedResult.
        """
        # Create lookup for existing chunks
        existing_chunks = {c.chunk_id: i for i, c in enumerate(current_result.chunks)}

        updated_chunks = list(current_result.chunks)
        updated_scores = list(current_result.scores)
        updated_sources = dict(current_result.sources)

        for chunk, score in zip(new_chunks, new_scores):
            chunk_id = chunk.chunk_id
            source_type = new_sub_query.query_type.value

            if chunk_id in existing_chunks:
                # Update existing chunk
                idx = existing_chunks[chunk_id]

                # Update score (apply fusion)
                current_score = updated_scores[idx]
                current_sources = updated_sources[chunk_id]

                # Apply duplicate boost
                updated_scores[idx] = max(current_score, score) + self.duplicate_boost

                # Add source
                if source_type not in current_sources:
                    current_sources.append(source_type)
                    updated_scores[idx] += self.source_diversity_bonus

            else:
                # Add new chunk
                updated_chunks.append(chunk)
                updated_scores.append(score)
                updated_sources[chunk_id] = [source_type]
                existing_chunks[chunk_id] = len(updated_chunks) - 1

        # Sort by score and limit
        combined = list(zip(updated_chunks, updated_scores))
        combined.sort(key=lambda x: x[1], reverse=True)
        combined = combined[: self.max_chunks]

        final_chunks = [c[0] for c in combined]
        final_scores = [c[1] for c in combined]
        final_sources = {
            c.chunk_id: updated_sources.get(c.chunk_id, []) for c in final_chunks
        }

        # Recompute entropy
        from icd.retrieval.entropy import EntropyCalculator

        entropy_calc = EntropyCalculator()
        entropy = entropy_calc.compute_entropy(final_scores)

        return AggregatedResult(
            chunks=final_chunks,
            scores=final_scores,
            sources=final_sources,
            coverage=current_result.coverage,  # Would need plan to update
            entropy=entropy,
            metadata={
                **current_result.metadata,
                "last_update_source": source_type,
            },
        )


class ScoreFuser:
    """
    Various score fusion methods.

    Provides different fusion strategies that can be selected
    based on the use case.
    """

    @staticmethod
    def max_fusion(scores: list[float]) -> float:
        """Return maximum score."""
        return max(scores) if scores else 0.0

    @staticmethod
    def mean_fusion(scores: list[float]) -> float:
        """Return mean score."""
        return sum(scores) / len(scores) if scores else 0.0

    @staticmethod
    def rrf_fusion(
        scores: list[float],
        k: int = 60,
    ) -> float:
        """
        Reciprocal Rank Fusion.

        Args:
            scores: List of scores (higher = better).
            k: Smoothing constant.

        Returns:
            RRF score.
        """
        # Convert scores to ranks (1-indexed)
        # Note: This assumes scores are from different rankings
        # In practice, you'd have rank positions
        return sum(1.0 / (k + i) for i, _ in enumerate(scores, 1))

    @staticmethod
    def weighted_fusion(
        scores: list[float],
        weights: list[float],
    ) -> float:
        """
        Weighted score fusion.

        Args:
            scores: List of scores.
            weights: Corresponding weights.

        Returns:
            Weighted sum.
        """
        if not scores or len(scores) != len(weights):
            return 0.0

        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0

        return sum(s * w for s, w in zip(scores, weights)) / total_weight

    @staticmethod
    def geometric_fusion(scores: list[float]) -> float:
        """
        Geometric mean fusion.

        More conservative than arithmetic mean.
        """
        if not scores:
            return 0.0

        # Filter out zeros
        positive_scores = [s for s in scores if s > 0]
        if not positive_scores:
            return 0.0

        product = 1.0
        for s in positive_scores:
            product *= s

        return product ** (1.0 / len(positive_scores))


class ResultFilter:
    """
    Filter aggregated results based on various criteria.
    """

    @staticmethod
    def filter_by_score(
        result: AggregatedResult,
        min_score: float,
    ) -> AggregatedResult:
        """Filter chunks below minimum score."""
        filtered = [
            (c, s)
            for c, s in zip(result.chunks, result.scores)
            if s >= min_score
        ]

        return AggregatedResult(
            chunks=[f[0] for f in filtered],
            scores=[f[1] for f in filtered],
            sources={f[0].chunk_id: result.sources.get(f[0].chunk_id, []) for f in filtered},
            coverage=result.coverage,
            entropy=result.entropy,
            metadata=result.metadata,
        )

    @staticmethod
    def filter_by_source_count(
        result: AggregatedResult,
        min_sources: int,
    ) -> AggregatedResult:
        """Filter chunks that don't appear in enough sources."""
        filtered = [
            (c, s)
            for c, s in zip(result.chunks, result.scores)
            if len(result.sources.get(c.chunk_id, [])) >= min_sources
        ]

        return AggregatedResult(
            chunks=[f[0] for f in filtered],
            scores=[f[1] for f in filtered],
            sources={f[0].chunk_id: result.sources.get(f[0].chunk_id, []) for f in filtered},
            coverage=result.coverage,
            entropy=result.entropy,
            metadata=result.metadata,
        )

    @staticmethod
    def filter_contracts_only(
        result: AggregatedResult,
    ) -> AggregatedResult:
        """Filter to only contract chunks."""
        filtered = [
            (c, s)
            for c, s in zip(result.chunks, result.scores)
            if c.is_contract
        ]

        return AggregatedResult(
            chunks=[f[0] for f in filtered],
            scores=[f[1] for f in filtered],
            sources={f[0].chunk_id: result.sources.get(f[0].chunk_id, []) for f in filtered},
            coverage=result.coverage,
            entropy=result.entropy,
            metadata=result.metadata,
        )
