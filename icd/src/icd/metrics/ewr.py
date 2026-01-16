"""
Exploration Waste Ratio (EWR) computation.

EWR measures how much retrieval effort was wasted on irrelevant
results. Lower EWR indicates more efficient retrieval.

EWR = (tokens_explored - tokens_used) / tokens_explored
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from icd.retrieval.hybrid import Chunk

logger = structlog.get_logger(__name__)


@dataclass
class EWRResult:
    """Result of EWR computation."""

    ewr: float  # 0 = no waste, 1 = all waste
    tokens_explored: int
    tokens_used: int
    tokens_wasted: int
    chunks_explored: int
    chunks_used: int
    efficiency: float  # 1 - EWR
    breakdown: dict[str, float] = field(default_factory=dict)


@dataclass
class ExplorationRecord:
    """Record of exploration for EWR tracking."""

    query: str
    chunks_retrieved: list["Chunk"]
    chunks_selected: list[str]  # IDs of chunks that were used
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


class EWRCalculator:
    """
    Calculate Exploration Waste Ratio.

    Features:
    - Session-based tracking
    - Per-query EWR
    - Aggregate EWR over time
    - Breakdown by source type
    """

    def __init__(self) -> None:
        """Initialize the EWR calculator."""
        self._records: list[ExplorationRecord] = []
        self._max_records = 1000

    def record_exploration(
        self,
        query: str,
        chunks_retrieved: list["Chunk"],
        chunks_selected: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Record an exploration for EWR tracking.

        Args:
            query: The query that was executed.
            chunks_retrieved: All chunks that were retrieved.
            chunks_selected: IDs of chunks that were actually used.
            metadata: Additional metadata.
        """
        record = ExplorationRecord(
            query=query,
            chunks_retrieved=chunks_retrieved,
            chunks_selected=chunks_selected,
            metadata=metadata or {},
        )

        self._records.append(record)

        # Trim old records
        if len(self._records) > self._max_records:
            self._records = self._records[-self._max_records :]

    def compute_ewr(
        self,
        chunks_retrieved: list["Chunk"],
        chunks_selected: list[str],
    ) -> EWRResult:
        """
        Compute EWR for a single retrieval.

        Args:
            chunks_retrieved: All chunks that were retrieved.
            chunks_selected: IDs of chunks that were actually used.

        Returns:
            EWRResult with detailed metrics.
        """
        selected_ids = set(chunks_selected)

        # Compute token counts
        tokens_explored = sum(c.token_count for c in chunks_retrieved)
        tokens_used = sum(
            c.token_count for c in chunks_retrieved if c.chunk_id in selected_ids
        )
        tokens_wasted = tokens_explored - tokens_used

        # Compute chunk counts
        chunks_explored = len(chunks_retrieved)
        chunks_used = len(selected_ids)

        # Compute EWR
        ewr = tokens_wasted / tokens_explored if tokens_explored > 0 else 0.0
        efficiency = 1.0 - ewr

        # Compute breakdown by source type (if available)
        breakdown: dict[str, float] = {}

        # Group by symbol type
        type_tokens: dict[str, dict[str, int]] = {}
        for chunk in chunks_retrieved:
            symbol_type = chunk.symbol_type or "unknown"
            if symbol_type not in type_tokens:
                type_tokens[symbol_type] = {"explored": 0, "used": 0}
            type_tokens[symbol_type]["explored"] += chunk.token_count
            if chunk.chunk_id in selected_ids:
                type_tokens[symbol_type]["used"] += chunk.token_count

        for symbol_type, tokens in type_tokens.items():
            if tokens["explored"] > 0:
                type_ewr = (tokens["explored"] - tokens["used"]) / tokens["explored"]
                breakdown[f"ewr_{symbol_type}"] = type_ewr

        return EWRResult(
            ewr=ewr,
            tokens_explored=tokens_explored,
            tokens_used=tokens_used,
            tokens_wasted=tokens_wasted,
            chunks_explored=chunks_explored,
            chunks_used=chunks_used,
            efficiency=efficiency,
            breakdown=breakdown,
        )

    def compute_session_ewr(self) -> EWRResult:
        """
        Compute aggregate EWR for the current session.

        Returns:
            Aggregate EWRResult.
        """
        if not self._records:
            return EWRResult(
                ewr=0.0,
                tokens_explored=0,
                tokens_used=0,
                tokens_wasted=0,
                chunks_explored=0,
                chunks_used=0,
                efficiency=1.0,
            )

        total_explored = 0
        total_used = 0
        total_chunks_explored = 0
        total_chunks_used = 0

        for record in self._records:
            selected_ids = set(record.chunks_selected)

            total_explored += sum(c.token_count for c in record.chunks_retrieved)
            total_used += sum(
                c.token_count
                for c in record.chunks_retrieved
                if c.chunk_id in selected_ids
            )
            total_chunks_explored += len(record.chunks_retrieved)
            total_chunks_used += len(selected_ids)

        total_wasted = total_explored - total_used
        ewr = total_wasted / total_explored if total_explored > 0 else 0.0

        return EWRResult(
            ewr=ewr,
            tokens_explored=total_explored,
            tokens_used=total_used,
            tokens_wasted=total_wasted,
            chunks_explored=total_chunks_explored,
            chunks_used=total_chunks_used,
            efficiency=1.0 - ewr,
        )

    def compute_trend(self, window: int = 10) -> list[float]:
        """
        Compute EWR trend over recent queries.

        Args:
            window: Number of recent queries to consider.

        Returns:
            List of EWR values (most recent first).
        """
        trend = []

        for record in self._records[-window:][::-1]:
            selected_ids = set(record.chunks_selected)
            tokens_explored = sum(c.token_count for c in record.chunks_retrieved)
            tokens_used = sum(
                c.token_count
                for c in record.chunks_retrieved
                if c.chunk_id in selected_ids
            )

            ewr = (tokens_explored - tokens_used) / tokens_explored if tokens_explored > 0 else 0.0
            trend.append(ewr)

        return trend

    def get_statistics(self) -> dict[str, Any]:
        """Get detailed EWR statistics."""
        if not self._records:
            return {"queries": 0}

        session_ewr = self.compute_session_ewr()
        trend = self.compute_trend()

        # Compute per-query EWR values
        query_ewrs = []
        for record in self._records:
            result = self.compute_ewr(
                record.chunks_retrieved, record.chunks_selected
            )
            query_ewrs.append(result.ewr)

        import numpy as np

        return {
            "queries": len(self._records),
            "session_ewr": session_ewr.ewr,
            "session_efficiency": session_ewr.efficiency,
            "total_tokens_explored": session_ewr.tokens_explored,
            "total_tokens_used": session_ewr.tokens_used,
            "total_tokens_wasted": session_ewr.tokens_wasted,
            "mean_ewr": float(np.mean(query_ewrs)),
            "std_ewr": float(np.std(query_ewrs)),
            "min_ewr": float(np.min(query_ewrs)),
            "max_ewr": float(np.max(query_ewrs)),
            "trend": trend,
        }

    def reset(self) -> None:
        """Reset all records."""
        self._records.clear()


class EWROptimizer:
    """
    Suggest optimizations to reduce EWR.

    Analyzes exploration patterns and suggests improvements.
    """

    def __init__(self, calculator: EWRCalculator) -> None:
        """
        Initialize the optimizer.

        Args:
            calculator: EWR calculator with historical data.
        """
        self.calculator = calculator

    def suggest_optimizations(self) -> list[str]:
        """
        Suggest optimizations based on EWR analysis.

        Returns:
            List of optimization suggestions.
        """
        suggestions = []

        stats = self.calculator.get_statistics()
        if stats.get("queries", 0) < 5:
            return ["Need more data for optimization suggestions."]

        session_ewr = stats.get("session_ewr", 0)

        # High overall EWR
        if session_ewr > 0.7:
            suggestions.append(
                "High EWR detected. Consider reducing initial retrieval count."
            )

        # Check for specific patterns
        if stats.get("max_ewr", 0) > 0.9:
            suggestions.append(
                "Some queries have very high waste. Review query formulation."
            )

        # Check trend
        trend = stats.get("trend", [])
        if len(trend) >= 3:
            recent_avg = sum(trend[:3]) / 3
            older_avg = sum(trend[3:]) / max(1, len(trend) - 3) if len(trend) > 3 else recent_avg

            if recent_avg > older_avg * 1.2:
                suggestions.append(
                    "EWR is increasing. Query patterns may need adjustment."
                )
            elif recent_avg < older_avg * 0.8:
                suggestions.append(
                    "EWR is decreasing. Current approach is improving."
                )

        # Chunk type analysis
        for record in self.calculator._records[-10:]:
            result = self.calculator.compute_ewr(
                record.chunks_retrieved, record.chunks_selected
            )
            for key, value in result.breakdown.items():
                if value > 0.9:
                    symbol_type = key.replace("ewr_", "")
                    suggestions.append(
                        f"Consider filtering '{symbol_type}' chunks - high waste."
                    )

        return suggestions if suggestions else ["No specific optimizations suggested."]
