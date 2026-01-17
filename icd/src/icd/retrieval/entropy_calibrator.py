"""
Adaptive Entropy Calibration (AEC).

A novel approach to calibrating entropy thresholds per-project.
Standard entropy thresholds are magic numbers that don't adapt to:
- Different codebase characteristics
- Different embedding models
- Different query distributions

This module provides automatic calibration using synthetic queries
generated from indexed symbols.

Algorithm:
1. Generate synthetic queries from indexed symbols (known answers)
2. Run retrieval and measure entropy for each
3. Label queries as "easy" (answer in top-k) or "hard" (answer not in top-k)
4. Set threshold at the decision boundary (e.g., p75 of easy queries)

Reference: Novel contribution - no existing code retrieval system
automatically calibrates entropy thresholds.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog

if TYPE_CHECKING:
    from icd.config import Config
    from icd.retrieval.hybrid import HybridRetriever
    from icd.storage.sqlite_store import SQLiteStore

logger = structlog.get_logger(__name__)


@dataclass
class CalibrationResult:
    """Result of entropy calibration."""

    recommended_threshold: float
    easy_query_entropy_mean: float
    easy_query_entropy_std: float
    hard_query_entropy_mean: float
    hard_query_entropy_std: float
    separation_score: float  # How well easy/hard are separated
    num_samples: int
    confidence: float  # Confidence in the calibration
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SyntheticQuery:
    """A synthetic query with known answer."""

    query: str
    expected_chunk_id: str
    expected_symbol: str
    query_type: str  # "definition", "symbol", "function"


class SyntheticQueryGenerator:
    """
    Generates synthetic queries from indexed symbols.

    Query templates:
    - "What is {ClassName}?" → expects class definition
    - "{function_name} implementation" → expects function
    - "How does {MethodName} work?" → expects method
    """

    QUERY_TEMPLATES = {
        "definition": [
            "What is {symbol}?",
            "{symbol} class",
            "definition of {symbol}",
            "{symbol} interface",
        ],
        "implementation": [
            "How does {symbol} work?",
            "{symbol} implementation",
            "{symbol} function",
            "{symbol} method",
        ],
        "usage": [
            "Where is {symbol} used?",
            "{symbol} usage",
            "calls to {symbol}",
        ],
    }

    def __init__(self, sqlite_store: "SQLiteStore") -> None:
        self.sqlite_store = sqlite_store

    async def generate_queries(
        self,
        num_queries: int = 50,
        seed: int | None = None,
    ) -> list[SyntheticQuery]:
        """
        Generate synthetic queries from indexed symbols.

        Samples symbols from the index and creates queries with known answers.
        """
        if seed is not None:
            random.seed(seed)

        queries = []

        # Get all chunks with symbols
        chunk_ids = await self.sqlite_store.get_all_chunk_ids()

        # Sample chunks with symbol names
        symbol_chunks = []
        for chunk_id in chunk_ids:
            chunk = await self.sqlite_store.get_chunk(chunk_id)
            if chunk and chunk.symbol_name:
                symbol_chunks.append((chunk_id, chunk))

        if not symbol_chunks:
            logger.warning("No symbol chunks found for calibration")
            return []

        # Sample chunks
        sample_size = min(num_queries * 2, len(symbol_chunks))
        sampled = random.sample(symbol_chunks, sample_size)

        for chunk_id, chunk in sampled:
            if len(queries) >= num_queries:
                break

            symbol = chunk.symbol_name
            symbol_type = chunk.symbol_type or "unknown"

            # Choose query type based on symbol type
            if "class" in symbol_type.lower() or "interface" in symbol_type.lower():
                query_type = "definition"
            elif "function" in symbol_type.lower() or "method" in symbol_type.lower():
                query_type = "implementation"
            else:
                query_type = random.choice(["definition", "implementation"])

            # Generate query from template
            templates = self.QUERY_TEMPLATES[query_type]
            template = random.choice(templates)
            query_text = template.format(symbol=symbol)

            queries.append(SyntheticQuery(
                query=query_text,
                expected_chunk_id=chunk_id,
                expected_symbol=symbol,
                query_type=query_type,
            ))

        return queries


class EntropyCalibrator:
    """
    Calibrates entropy thresholds for a specific project.

    Uses synthetic queries with known answers to find optimal
    threshold for triggering RLM refinement.
    """

    def __init__(
        self,
        config: "Config",
        sqlite_store: "SQLiteStore",
        retriever: "HybridRetriever",
    ) -> None:
        self.config = config
        self.sqlite_store = sqlite_store
        self.retriever = retriever
        self.query_generator = SyntheticQueryGenerator(sqlite_store)

        # Calibration parameters
        self.num_calibration_queries = 50
        self.top_k_threshold = 5  # "Easy" if answer in top-k
        self.percentile_threshold = 75  # Use p75 of easy queries

    async def calibrate(
        self,
        seed: int | None = None,
    ) -> CalibrationResult:
        """
        Run calibration and compute recommended threshold.

        Returns:
            CalibrationResult with recommended threshold and statistics.
        """
        logger.info("Starting entropy calibration")

        # Step 1: Generate synthetic queries
        queries = await self.query_generator.generate_queries(
            num_queries=self.num_calibration_queries,
            seed=seed,
        )

        if len(queries) < 10:
            logger.warning(
                "Insufficient queries for calibration",
                num_queries=len(queries)
            )
            return self._default_result()

        # Step 2: Run retrieval and classify
        easy_entropies = []
        hard_entropies = []

        for query in queries:
            result = await self.retriever.retrieve(
                query=query.query,
                limit=self.top_k_threshold * 2,
            )

            # Check if expected chunk is in top-k
            top_k_ids = [c.chunk_id for c in result.chunks[:self.top_k_threshold]]
            is_easy = query.expected_chunk_id in top_k_ids

            if is_easy:
                easy_entropies.append(result.entropy)
            else:
                hard_entropies.append(result.entropy)

        # Step 3: Compute statistics
        if not easy_entropies:
            logger.warning("No easy queries found during calibration")
            return self._default_result()

        easy_mean = np.mean(easy_entropies)
        easy_std = np.std(easy_entropies)
        hard_mean = np.mean(hard_entropies) if hard_entropies else easy_mean + 1.0
        hard_std = np.std(hard_entropies) if hard_entropies else easy_std

        # Step 4: Compute recommended threshold
        # Use percentile of easy queries as threshold
        recommended_threshold = np.percentile(easy_entropies, self.percentile_threshold)

        # Compute separation score (how well easy/hard are separated)
        if hard_entropies:
            # Cohen's d effect size
            pooled_std = np.sqrt((easy_std**2 + hard_std**2) / 2)
            separation_score = (hard_mean - easy_mean) / max(pooled_std, 0.001)
        else:
            separation_score = 0.0

        # Compute confidence based on sample size and separation
        confidence = min(0.95, (len(queries) / 100) * (1 + separation_score / 2))

        logger.info(
            "Entropy calibration complete",
            recommended_threshold=recommended_threshold,
            easy_mean=easy_mean,
            hard_mean=hard_mean,
            separation=separation_score,
            confidence=confidence,
        )

        return CalibrationResult(
            recommended_threshold=recommended_threshold,
            easy_query_entropy_mean=easy_mean,
            easy_query_entropy_std=easy_std,
            hard_query_entropy_mean=hard_mean,
            hard_query_entropy_std=hard_std,
            separation_score=separation_score,
            num_samples=len(queries),
            confidence=confidence,
            metadata={
                "num_easy": len(easy_entropies),
                "num_hard": len(hard_entropies),
                "percentile_used": self.percentile_threshold,
                "top_k_threshold": self.top_k_threshold,
            },
        )

    def _default_result(self) -> CalibrationResult:
        """Return default calibration when calibration fails."""
        return CalibrationResult(
            recommended_threshold=self.config.rlm.entropy_threshold,
            easy_query_entropy_mean=0.0,
            easy_query_entropy_std=0.0,
            hard_query_entropy_mean=0.0,
            hard_query_entropy_std=0.0,
            separation_score=0.0,
            num_samples=0,
            confidence=0.0,
            metadata={"reason": "calibration_failed"},
        )

    async def should_trigger_rlm(
        self,
        entropy: float,
        calibration: CalibrationResult | None = None,
    ) -> bool:
        """
        Determine if RLM should be triggered based on calibrated threshold.

        Args:
            entropy: Current retrieval entropy.
            calibration: Optional pre-computed calibration result.

        Returns:
            True if RLM should be triggered.
        """
        if calibration is None or calibration.confidence < 0.5:
            # Fall back to config threshold
            return entropy >= self.config.rlm.entropy_threshold

        return entropy >= calibration.recommended_threshold


class AdaptiveEntropyGate:
    """
    Adaptive gate for RLM triggering using calibrated thresholds.

    Caches calibration results and provides fast threshold checks.
    """

    def __init__(
        self,
        config: "Config",
        sqlite_store: "SQLiteStore",
        retriever: "HybridRetriever",
    ) -> None:
        self.config = config
        self.calibrator = EntropyCalibrator(config, sqlite_store, retriever)
        self._calibration: CalibrationResult | None = None
        self._calibration_valid = False

    async def ensure_calibrated(self, force: bool = False) -> CalibrationResult:
        """Ensure calibration is done, running if necessary."""
        if not self._calibration_valid or force:
            self._calibration = await self.calibrator.calibrate()
            self._calibration_valid = self._calibration.confidence > 0.3

        return self._calibration

    def should_trigger_rlm(self, entropy: float) -> bool:
        """Fast check if RLM should be triggered."""
        if not self._calibration_valid or self._calibration is None:
            # Fall back to config
            return entropy >= self.config.rlm.entropy_threshold

        return entropy >= self._calibration.recommended_threshold

    def get_threshold(self) -> float:
        """Get current threshold."""
        if self._calibration_valid and self._calibration is not None:
            return self._calibration.recommended_threshold
        return self.config.rlm.entropy_threshold

    def invalidate(self) -> None:
        """Invalidate calibration (call after reindexing)."""
        self._calibration_valid = False


async def calibrate_entropy_threshold(
    config: "Config",
    sqlite_store: "SQLiteStore",
    retriever: "HybridRetriever",
) -> CalibrationResult:
    """
    Convenience function to run entropy calibration.

    Args:
        config: ICD configuration.
        sqlite_store: SQLite store with indexed chunks.
        retriever: Hybrid retriever for running queries.

    Returns:
        CalibrationResult with recommended threshold.
    """
    calibrator = EntropyCalibrator(config, sqlite_store, retriever)
    return await calibrator.calibrate()
