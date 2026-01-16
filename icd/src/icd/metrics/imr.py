"""
Impact Miss Rate (IMR) tracking.

IMR measures how often the retrieval system fails to include
chunks that were actually important/impactful for the task.

IMR = missed_impactful_chunks / total_impactful_chunks
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)


@dataclass
class IMRResult:
    """Result of IMR computation."""

    imr: float  # 0 = no misses, 1 = all misses
    total_impactful: int
    retrieved_impactful: int
    missed_impactful: int
    precision: float  # retrieved_impactful / total_retrieved
    recall: float  # 1 - IMR
    f1: float
    breakdown: dict[str, float] = field(default_factory=dict)


@dataclass
class FeedbackRecord:
    """Record of user feedback for IMR tracking."""

    query: str
    retrieved_ids: list[str]
    impactful_ids: list[str]  # IDs that were actually impactful
    missed_ids: list[str]  # IDs that should have been retrieved but weren't
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = "user"  # user, heuristic, etc.
    metadata: dict[str, Any] = field(default_factory=dict)


class IMRTracker:
    """
    Track Impact Miss Rate over time.

    Features:
    - User feedback integration
    - Heuristic impact detection
    - Per-query and aggregate IMR
    - Trend analysis
    """

    def __init__(self) -> None:
        """Initialize the IMR tracker."""
        self._records: list[FeedbackRecord] = []
        self._max_records = 1000

    def record_feedback(
        self,
        query: str,
        retrieved_ids: list[str],
        impactful_ids: list[str],
        missed_ids: list[str] | None = None,
        source: str = "user",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Record feedback about retrieval impact.

        Args:
            query: The query that was executed.
            retrieved_ids: IDs of all retrieved chunks.
            impactful_ids: IDs that were actually impactful.
            missed_ids: IDs that should have been retrieved.
            source: Feedback source (user, heuristic, etc.).
            metadata: Additional metadata.
        """
        record = FeedbackRecord(
            query=query,
            retrieved_ids=retrieved_ids,
            impactful_ids=impactful_ids,
            missed_ids=missed_ids or [],
            source=source,
            metadata=metadata or {},
        )

        self._records.append(record)

        # Trim old records
        if len(self._records) > self._max_records:
            self._records = self._records[-self._max_records :]

        logger.debug(
            "IMR feedback recorded",
            query=query[:50],
            impactful=len(impactful_ids),
            missed=len(missed_ids or []),
        )

    def compute_imr(
        self,
        retrieved_ids: list[str],
        impactful_ids: list[str],
        missed_ids: list[str] | None = None,
    ) -> IMRResult:
        """
        Compute IMR for a single retrieval.

        Args:
            retrieved_ids: IDs of all retrieved chunks.
            impactful_ids: IDs that were actually impactful.
            missed_ids: IDs that should have been retrieved.

        Returns:
            IMRResult with detailed metrics.
        """
        retrieved_set = set(retrieved_ids)
        impactful_set = set(impactful_ids)
        missed_set = set(missed_ids or [])

        # Compute metrics
        retrieved_impactful = len(impactful_set & retrieved_set)

        # Total impactful = union of explicitly impactful and missed (avoid double-counting)
        all_impactful = impactful_set | missed_set
        total_impactful = len(all_impactful)

        # Missed = impactful chunks that were not retrieved
        actually_missed = all_impactful - retrieved_set
        missed_impactful = len(actually_missed)

        # IMR
        imr = missed_impactful / total_impactful if total_impactful > 0 else 0.0

        # Precision: what fraction of retrieved was impactful
        precision = (
            retrieved_impactful / len(retrieved_set) if retrieved_set else 0.0
        )

        # Recall: 1 - IMR
        recall = 1.0 - imr

        # F1 score
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return IMRResult(
            imr=imr,
            total_impactful=total_impactful,
            retrieved_impactful=retrieved_impactful,
            missed_impactful=missed_impactful,
            precision=precision,
            recall=recall,
            f1=f1,
        )

    def compute_session_imr(self) -> IMRResult:
        """
        Compute aggregate IMR for the current session.

        Returns:
            Aggregate IMRResult.
        """
        if not self._records:
            return IMRResult(
                imr=0.0,
                total_impactful=0,
                retrieved_impactful=0,
                missed_impactful=0,
                precision=1.0,
                recall=1.0,
                f1=1.0,
            )

        total_impactful = 0
        total_retrieved_impactful = 0
        total_retrieved = 0

        for record in self._records:
            retrieved_set = set(record.retrieved_ids)
            impactful_set = set(record.impactful_ids)
            missed_set = set(record.missed_ids)

            # Use union to avoid double-counting
            all_impactful = impactful_set | missed_set
            total_impactful += len(all_impactful)
            total_retrieved_impactful += len(all_impactful & retrieved_set)
            total_retrieved += len(retrieved_set)

        missed_impactful = total_impactful - total_retrieved_impactful
        imr = missed_impactful / total_impactful if total_impactful > 0 else 0.0

        precision = (
            total_retrieved_impactful / total_retrieved if total_retrieved > 0 else 0.0
        )
        recall = 1.0 - imr
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return IMRResult(
            imr=imr,
            total_impactful=total_impactful,
            retrieved_impactful=total_retrieved_impactful,
            missed_impactful=missed_impactful,
            precision=precision,
            recall=recall,
            f1=f1,
        )

    def compute_trend(self, window: int = 10) -> list[float]:
        """
        Compute IMR trend over recent queries.

        Args:
            window: Number of recent queries to consider.

        Returns:
            List of IMR values (most recent first).
        """
        trend = []

        for record in self._records[-window:][::-1]:
            result = self.compute_imr(
                record.retrieved_ids,
                record.impactful_ids,
                record.missed_ids,
            )
            trend.append(result.imr)

        return trend

    def get_missed_patterns(self) -> dict[str, int]:
        """
        Analyze patterns in missed chunks.

        Returns:
            Dictionary mapping patterns to frequency.
        """
        # This would analyze metadata to find common patterns in missed chunks
        # For now, return basic statistics
        patterns: dict[str, int] = {}

        for record in self._records:
            for missed_id in record.missed_ids:
                # Extract pattern from metadata if available
                pattern = record.metadata.get(f"pattern_{missed_id}", "unknown")
                patterns[pattern] = patterns.get(pattern, 0) + 1

        return patterns

    def get_statistics(self) -> dict[str, Any]:
        """Get detailed IMR statistics."""
        if not self._records:
            return {"queries": 0}

        session_imr = self.compute_session_imr()
        trend = self.compute_trend()

        # Compute per-query metrics
        query_imrs = []
        query_precisions = []
        query_recalls = []

        for record in self._records:
            result = self.compute_imr(
                record.retrieved_ids,
                record.impactful_ids,
                record.missed_ids,
            )
            query_imrs.append(result.imr)
            query_precisions.append(result.precision)
            query_recalls.append(result.recall)

        import numpy as np

        return {
            "queries": len(self._records),
            "session_imr": session_imr.imr,
            "session_precision": session_imr.precision,
            "session_recall": session_imr.recall,
            "session_f1": session_imr.f1,
            "mean_imr": float(np.mean(query_imrs)),
            "std_imr": float(np.std(query_imrs)),
            "mean_precision": float(np.mean(query_precisions)),
            "mean_recall": float(np.mean(query_recalls)),
            "trend": trend,
            "feedback_sources": self._count_sources(),
        }

    def _count_sources(self) -> dict[str, int]:
        """Count feedback by source."""
        sources: dict[str, int] = {}
        for record in self._records:
            sources[record.source] = sources.get(record.source, 0) + 1
        return sources

    def reset(self) -> None:
        """Reset all records."""
        self._records.clear()


class HeuristicImpactDetector:
    """
    Detect impactful chunks using heuristics.

    Used when explicit user feedback is not available.
    """

    def __init__(self) -> None:
        """Initialize the detector."""
        # Heuristic weights
        self.contract_weight = 0.3
        self.pinned_weight = 0.4
        self.score_threshold = 0.7

    def detect_impactful(
        self,
        chunks: list[Any],  # list[Chunk]
        scores: list[float],
        threshold: float | None = None,
    ) -> list[str]:
        """
        Detect impactful chunks using heuristics.

        Args:
            chunks: Retrieved chunks.
            scores: Corresponding scores.
            threshold: Score threshold for impact.

        Returns:
            List of impactful chunk IDs.
        """
        threshold = threshold or self.score_threshold
        max_score = max(scores) if scores else 1.0
        if max_score == 0:
            max_score = 1.0

        impactful_ids = []

        for chunk, score in zip(chunks, scores):
            normalized_score = score / max_score

            # High score
            if normalized_score >= threshold:
                impactful_ids.append(chunk.chunk_id)
                continue

            # Contract bonus
            if hasattr(chunk, "is_contract") and chunk.is_contract:
                if normalized_score + self.contract_weight >= threshold:
                    impactful_ids.append(chunk.chunk_id)
                    continue

            # Pinned bonus
            if hasattr(chunk, "is_pinned") and chunk.is_pinned:
                if normalized_score + self.pinned_weight >= threshold:
                    impactful_ids.append(chunk.chunk_id)
                    continue

        return impactful_ids

    def estimate_missed(
        self,
        query: str,
        retrieved_ids: list[str],
        all_indexed_count: int,
    ) -> int:
        """
        Estimate number of missed impactful chunks.

        This is a rough heuristic based on query complexity
        and retrieval count.

        Args:
            query: The query.
            retrieved_ids: Retrieved chunk IDs.
            all_indexed_count: Total indexed chunks.

        Returns:
            Estimated missed count.
        """
        # Query complexity heuristic
        words = query.split()
        complexity_factor = min(1.0, len(words) / 10)

        # Coverage heuristic
        coverage = len(retrieved_ids) / max(1, all_indexed_count)
        coverage_factor = 1.0 - min(1.0, coverage * 10)

        # Estimate: higher complexity and lower coverage = more misses
        estimated_total = len(retrieved_ids) * (1 + complexity_factor)
        estimated_missed = int(estimated_total * coverage_factor * 0.1)

        return max(0, estimated_missed)
