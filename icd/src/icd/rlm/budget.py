"""
Budget tracking and stop conditions for RLM.

Manages token budgets, iteration limits, and determines
when to stop iterative retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from icd.config import Config
    from icd.rlm.aggregator import AggregatedResult
    from icd.rlm.planner import RetrievalPlan

logger = structlog.get_logger(__name__)


class StopReason(str, Enum):
    """Reasons for stopping RLM iteration."""

    MAX_ITERATIONS = "max_iterations"
    TOKEN_BUDGET = "token_budget"
    LOW_ENTROPY = "low_entropy"
    NO_NEW_RESULTS = "no_new_results"
    TIMEOUT = "timeout"
    USER_INTERRUPT = "user_interrupt"
    PLAN_COMPLETE = "plan_complete"
    QUALITY_THRESHOLD = "quality_threshold"


@dataclass
class BudgetState:
    """Current budget state."""

    tokens_used: int
    tokens_remaining: int
    iterations_used: int
    iterations_remaining: int
    time_elapsed: timedelta
    time_remaining: timedelta | None
    should_stop: bool
    stop_reason: StopReason | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IterationStats:
    """Statistics for a single iteration."""

    iteration_number: int
    tokens_used: int
    chunks_retrieved: int
    new_chunks: int
    entropy: float
    duration: timedelta
    sub_query_type: str | None = None


class BudgetTracker:
    """
    Track and manage RLM budgets.

    Features:
    - Token budget tracking
    - Iteration limit enforcement
    - Time limit enforcement
    - Quality-based stopping
    - Detailed statistics
    """

    def __init__(self, config: "Config") -> None:
        """
        Initialize the budget tracker.

        Args:
            config: ICD configuration.
        """
        self.config = config

        # Limits from config
        self.max_iterations = config.rlm.max_iterations
        self.budget_per_iteration = config.rlm.budget_per_iteration
        self.total_token_budget = self.max_iterations * self.budget_per_iteration

        # Quality thresholds
        self.entropy_threshold = config.rlm.entropy_threshold
        self.min_new_chunks_threshold = 2

        # Time limit (optional)
        self.time_limit: timedelta | None = None

        # State
        self._iterations: list[IterationStats] = []
        self._tokens_used = 0
        self._start_time: datetime | None = None
        self._stopped = False
        self._stop_reason: StopReason | None = None
        self._previous_chunk_ids: set[str] = set()

    def start(self) -> None:
        """Start budget tracking."""
        self._start_time = datetime.utcnow()
        self._iterations.clear()
        self._tokens_used = 0
        self._stopped = False
        self._stop_reason = None
        self._previous_chunk_ids.clear()

        logger.debug(
            "Budget tracking started",
            max_iterations=self.max_iterations,
            total_budget=self.total_token_budget,
        )

    def record_iteration(
        self,
        chunks_retrieved: int,
        chunk_ids: list[str],
        tokens_used: int,
        entropy: float,
        sub_query_type: str | None = None,
    ) -> IterationStats:
        """
        Record an iteration's statistics.

        Args:
            chunks_retrieved: Total chunks retrieved.
            chunk_ids: IDs of retrieved chunks.
            tokens_used: Tokens used in this iteration.
            entropy: Retrieval entropy.
            sub_query_type: Type of sub-query executed.

        Returns:
            IterationStats for this iteration.
        """
        # Compute new chunks
        current_ids = set(chunk_ids)
        new_chunks = len(current_ids - self._previous_chunk_ids)
        self._previous_chunk_ids.update(current_ids)

        # Compute duration
        duration = (
            datetime.utcnow() - self._start_time
            if self._start_time
            else timedelta(0)
        )

        stats = IterationStats(
            iteration_number=len(self._iterations) + 1,
            tokens_used=tokens_used,
            chunks_retrieved=chunks_retrieved,
            new_chunks=new_chunks,
            entropy=entropy,
            duration=duration,
            sub_query_type=sub_query_type,
        )

        self._iterations.append(stats)
        self._tokens_used += tokens_used

        logger.debug(
            "Iteration recorded",
            iteration=stats.iteration_number,
            tokens_used=tokens_used,
            new_chunks=new_chunks,
            entropy=entropy,
        )

        return stats

    def check_budget(self) -> BudgetState:
        """
        Check current budget state and determine if should stop.

        Returns:
            BudgetState with current state and stop decision.
        """
        elapsed = (
            datetime.utcnow() - self._start_time
            if self._start_time
            else timedelta(0)
        )

        # Check various stopping conditions
        should_stop = False
        stop_reason: StopReason | None = None

        # Max iterations
        if len(self._iterations) >= self.max_iterations:
            should_stop = True
            stop_reason = StopReason.MAX_ITERATIONS

        # Token budget
        elif self._tokens_used >= self.total_token_budget:
            should_stop = True
            stop_reason = StopReason.TOKEN_BUDGET

        # Time limit
        elif self.time_limit and elapsed >= self.time_limit:
            should_stop = True
            stop_reason = StopReason.TIMEOUT

        # Quality-based: low entropy (high confidence)
        elif self._iterations and self._iterations[-1].entropy < 0.3:
            should_stop = True
            stop_reason = StopReason.LOW_ENTROPY

        # Quality-based: no new results
        elif (
            len(self._iterations) >= 2
            and self._iterations[-1].new_chunks < self.min_new_chunks_threshold
            and self._iterations[-2].new_chunks < self.min_new_chunks_threshold
        ):
            should_stop = True
            stop_reason = StopReason.NO_NEW_RESULTS

        if should_stop:
            self._stopped = True
            self._stop_reason = stop_reason

        # Compute remaining
        time_remaining = None
        if self.time_limit:
            time_remaining = max(timedelta(0), self.time_limit - elapsed)

        return BudgetState(
            tokens_used=self._tokens_used,
            tokens_remaining=max(0, self.total_token_budget - self._tokens_used),
            iterations_used=len(self._iterations),
            iterations_remaining=max(0, self.max_iterations - len(self._iterations)),
            time_elapsed=elapsed,
            time_remaining=time_remaining,
            should_stop=should_stop,
            stop_reason=stop_reason,
            metadata={
                "last_entropy": self._iterations[-1].entropy if self._iterations else None,
                "last_new_chunks": self._iterations[-1].new_chunks if self._iterations else None,
            },
        )

    def should_continue(
        self,
        plan: "RetrievalPlan",
        aggregated_result: "AggregatedResult | None" = None,
    ) -> bool:
        """
        Determine if RLM should continue.

        Args:
            plan: Current retrieval plan.
            aggregated_result: Current aggregated result (optional).

        Returns:
            True if should continue iteration.
        """
        if self._stopped:
            return False

        state = self.check_budget()
        if state.should_stop:
            return False

        # Check if plan is complete
        if plan.completed:
            self._stopped = True
            self._stop_reason = StopReason.PLAN_COMPLETE
            return False

        # Check aggregated result quality
        if aggregated_result:
            if aggregated_result.entropy < 0.3:
                self._stopped = True
                self._stop_reason = StopReason.QUALITY_THRESHOLD
                return False

        return True

    def force_stop(self, reason: StopReason = StopReason.USER_INTERRUPT) -> None:
        """Force stop the RLM process."""
        self._stopped = True
        self._stop_reason = reason
        logger.info("Budget tracking force stopped", reason=reason.value)

    def get_statistics(self) -> dict[str, Any]:
        """Get detailed statistics."""
        if not self._iterations:
            return {
                "iterations": 0,
                "stopped": self._stopped,
                "stop_reason": self._stop_reason.value if self._stop_reason else None,
            }

        total_chunks = sum(it.chunks_retrieved for it in self._iterations)
        total_new_chunks = sum(it.new_chunks for it in self._iterations)
        avg_entropy = sum(it.entropy for it in self._iterations) / len(self._iterations)

        return {
            "iterations": len(self._iterations),
            "tokens_used": self._tokens_used,
            "tokens_budget": self.total_token_budget,
            "budget_utilization": self._tokens_used / self.total_token_budget,
            "total_chunks_retrieved": total_chunks,
            "total_new_chunks": total_new_chunks,
            "deduplication_ratio": 1.0 - total_new_chunks / max(1, total_chunks),
            "average_entropy": avg_entropy,
            "final_entropy": self._iterations[-1].entropy,
            "stopped": self._stopped,
            "stop_reason": self._stop_reason.value if self._stop_reason else None,
            "iteration_details": [
                {
                    "iteration": it.iteration_number,
                    "tokens": it.tokens_used,
                    "chunks": it.chunks_retrieved,
                    "new_chunks": it.new_chunks,
                    "entropy": it.entropy,
                    "sub_query_type": it.sub_query_type,
                }
                for it in self._iterations
            ],
        }

    def get_remaining_budget(self) -> int:
        """Get remaining token budget."""
        return max(0, self.total_token_budget - self._tokens_used)

    def get_iteration_budget(self) -> int:
        """Get budget for the next iteration."""
        remaining = self.get_remaining_budget()
        return min(remaining, self.budget_per_iteration)


class AdaptiveBudget:
    """
    Adaptive budget allocation based on retrieval progress.

    Allocates more budget to promising iterations and less
    to iterations with diminishing returns.
    """

    def __init__(
        self,
        total_budget: int,
        max_iterations: int,
        min_iteration_budget: int = 500,
    ) -> None:
        """
        Initialize adaptive budget.

        Args:
            total_budget: Total token budget.
            max_iterations: Maximum iterations.
            min_iteration_budget: Minimum budget per iteration.
        """
        self.total_budget = total_budget
        self.max_iterations = max_iterations
        self.min_iteration_budget = min_iteration_budget

        self._used = 0
        self._iteration = 0
        self._history: list[dict[str, Any]] = []

    def allocate_next(
        self,
        last_new_chunks: int = 0,
        last_entropy: float = 1.0,
    ) -> int:
        """
        Allocate budget for the next iteration.

        Args:
            last_new_chunks: New chunks from last iteration.
            last_entropy: Entropy from last iteration.

        Returns:
            Token budget for next iteration.
        """
        remaining = self.total_budget - self._used
        remaining_iterations = self.max_iterations - self._iteration

        if remaining_iterations <= 0 or remaining <= 0:
            return 0

        # Base allocation: equal distribution
        base_allocation = remaining // remaining_iterations

        # Adjust based on progress
        if self._iteration > 0:
            # Reduce if low new chunks
            if last_new_chunks < 3:
                base_allocation = int(base_allocation * 0.7)

            # Reduce if low entropy (already have good results)
            if last_entropy < 0.4:
                base_allocation = int(base_allocation * 0.8)

            # Increase if high entropy (need more exploration)
            elif last_entropy > 0.7:
                base_allocation = int(base_allocation * 1.2)

        # Ensure minimum
        allocation = max(self.min_iteration_budget, base_allocation)
        allocation = min(allocation, remaining)

        self._history.append(
            {
                "iteration": self._iteration,
                "allocated": allocation,
                "last_new_chunks": last_new_chunks,
                "last_entropy": last_entropy,
            }
        )

        return allocation

    def record_usage(self, tokens_used: int) -> None:
        """Record actual token usage."""
        self._used += tokens_used
        self._iteration += 1

    def get_efficiency(self) -> float:
        """Get budget efficiency (actual use vs allocation)."""
        if not self._history:
            return 1.0

        total_allocated = sum(h["allocated"] for h in self._history)
        return self._used / total_allocated if total_allocated > 0 else 1.0
