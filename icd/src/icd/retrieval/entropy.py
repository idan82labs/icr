"""
Retrieval entropy computation.

Computes the entropy of retrieval score distributions to measure
query ambiguity and result confidence.

High entropy indicates uncertain/ambiguous queries that may benefit
from RLM (retrieval-augmented language model) fallback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import structlog

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)


@dataclass
class EntropyResult:
    """Result of entropy computation."""

    entropy: float
    normalized_entropy: float
    max_entropy: float
    confidence: float
    score_distribution: list[float]


class EntropyCalculator:
    """
    Compute retrieval entropy from score distributions.

    Implements:
    p_i = exp(s_i/τ) / Σ exp(s_j/τ)
    H = -Σ p_i log(p_i)

    Features:
    - Temperature-scaled softmax
    - Normalized entropy for comparison
    - Confidence scoring
    """

    def __init__(self, temperature: float = 1.0) -> None:
        """
        Initialize the entropy calculator.

        Args:
            temperature: Softmax temperature (τ).
                        Lower values sharpen the distribution.
                        Higher values flatten it.
        """
        self.temperature = temperature

    def compute_entropy(
        self,
        scores: list[float],
        normalize: bool = True,
    ) -> float:
        """
        Compute entropy of a score distribution.

        Args:
            scores: List of retrieval scores.
            normalize: Whether to normalize by max entropy.

        Returns:
            Entropy value (or normalized entropy if normalize=True).
        """
        if not scores:
            return 0.0

        result = self.compute_full(scores)

        if normalize:
            return result.normalized_entropy

        return result.entropy

    def compute_full(self, scores: list[float]) -> EntropyResult:
        """
        Compute full entropy analysis.

        Args:
            scores: List of retrieval scores.

        Returns:
            EntropyResult with detailed metrics.
        """
        if not scores:
            return EntropyResult(
                entropy=0.0,
                normalized_entropy=0.0,
                max_entropy=0.0,
                confidence=1.0,
                score_distribution=[],
            )

        n = len(scores)

        # Convert to numpy for computation
        scores_array = np.array(scores, dtype=np.float64)

        # Apply temperature-scaled softmax
        # p_i = exp(s_i/τ) / Σ exp(s_j/τ)
        scaled = scores_array / self.temperature

        # Numerical stability: subtract max
        scaled = scaled - np.max(scaled)
        exp_scores = np.exp(scaled)
        probabilities = exp_scores / np.sum(exp_scores)

        # Compute entropy: H = -Σ p_i log(p_i)
        # Avoid log(0) by masking zero probabilities
        mask = probabilities > 1e-10
        entropy = -np.sum(probabilities[mask] * np.log(probabilities[mask]))

        # Maximum entropy (uniform distribution)
        max_entropy = np.log(n) if n > 1 else 0.0

        # Normalized entropy (0 = certain, 1 = uniform)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        # Confidence: 1 - normalized_entropy
        confidence = 1.0 - normalized_entropy

        return EntropyResult(
            entropy=float(entropy),
            normalized_entropy=float(normalized_entropy),
            max_entropy=float(max_entropy),
            confidence=float(confidence),
            score_distribution=probabilities.tolist(),
        )

    def should_trigger_rlm(
        self,
        scores: list[float],
        threshold: float = 0.7,
    ) -> bool:
        """
        Determine if RLM should be triggered based on entropy.

        High entropy suggests the query is ambiguous and may benefit
        from iterative retrieval-augmented generation.

        Args:
            scores: List of retrieval scores.
            threshold: Normalized entropy threshold for triggering RLM.

        Returns:
            True if RLM should be triggered.
        """
        if len(scores) < 3:
            return False

        result = self.compute_full(scores)
        return result.normalized_entropy > threshold

    def compute_score_gap(self, scores: list[float]) -> float:
        """
        Compute the gap between top score and others.

        Larger gaps indicate more confident retrievals.

        Args:
            scores: List of retrieval scores.

        Returns:
            Score gap metric.
        """
        if len(scores) < 2:
            return 1.0

        sorted_scores = sorted(scores, reverse=True)
        top_score = sorted_scores[0]
        second_score = sorted_scores[1]

        if top_score == 0:
            return 0.0

        return (top_score - second_score) / top_score

    def compute_concentration(
        self,
        scores: list[float],
        top_k: int = 3,
    ) -> float:
        """
        Compute what fraction of probability mass is in top-k results.

        Higher concentration indicates more confident retrieval.

        Args:
            scores: List of retrieval scores.
            top_k: Number of top results to consider.

        Returns:
            Concentration ratio (0-1).
        """
        result = self.compute_full(scores)

        if not result.score_distribution:
            return 0.0

        # Sort probabilities descending
        sorted_probs = sorted(result.score_distribution, reverse=True)

        # Sum top-k probabilities
        top_k_sum = sum(sorted_probs[: min(top_k, len(sorted_probs))])

        return float(top_k_sum)


class QueryDifficultyEstimator:
    """
    Estimate query difficulty from retrieval results.

    Uses multiple signals to determine if a query is likely
    difficult or ambiguous.
    """

    def __init__(
        self,
        entropy_weight: float = 0.4,
        gap_weight: float = 0.3,
        concentration_weight: float = 0.3,
    ) -> None:
        """
        Initialize the difficulty estimator.

        Args:
            entropy_weight: Weight for entropy signal.
            gap_weight: Weight for score gap signal.
            concentration_weight: Weight for concentration signal.
        """
        self.entropy_weight = entropy_weight
        self.gap_weight = gap_weight
        self.concentration_weight = concentration_weight
        self.entropy_calc = EntropyCalculator()

    def estimate_difficulty(self, scores: list[float]) -> float:
        """
        Estimate query difficulty on a 0-1 scale.

        0 = easy (clear top results)
        1 = difficult (ambiguous, many similar scores)

        Args:
            scores: List of retrieval scores.

        Returns:
            Difficulty estimate.
        """
        if not scores:
            return 0.5

        # Compute signals
        entropy_result = self.entropy_calc.compute_full(scores)
        normalized_entropy = entropy_result.normalized_entropy

        score_gap = self.entropy_calc.compute_score_gap(scores)
        gap_difficulty = 1.0 - score_gap

        concentration = self.entropy_calc.compute_concentration(scores)
        concentration_difficulty = 1.0 - concentration

        # Weighted combination
        difficulty = (
            self.entropy_weight * normalized_entropy
            + self.gap_weight * gap_difficulty
            + self.concentration_weight * concentration_difficulty
        )

        return float(difficulty)

    def get_difficulty_analysis(
        self,
        scores: list[float],
    ) -> dict:
        """
        Get detailed difficulty analysis.

        Args:
            scores: List of retrieval scores.

        Returns:
            Dictionary with difficulty metrics.
        """
        entropy_result = self.entropy_calc.compute_full(scores)

        return {
            "difficulty": self.estimate_difficulty(scores),
            "entropy": entropy_result.entropy,
            "normalized_entropy": entropy_result.normalized_entropy,
            "confidence": entropy_result.confidence,
            "score_gap": self.entropy_calc.compute_score_gap(scores),
            "top3_concentration": self.entropy_calc.compute_concentration(
                scores, top_k=3
            ),
            "top5_concentration": self.entropy_calc.compute_concentration(
                scores, top_k=5
            ),
            "num_results": len(scores),
        }


def compute_cross_entropy(
    scores_a: list[float],
    scores_b: list[float],
    temperature: float = 1.0,
) -> float:
    """
    Compute cross-entropy between two score distributions.

    Useful for comparing retrieval results from different queries
    or different retrieval methods.

    Args:
        scores_a: First score distribution.
        scores_b: Second score distribution (target).
        temperature: Softmax temperature.

    Returns:
        Cross-entropy value.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("Score lists must have same length")

    if not scores_a:
        return 0.0

    # Convert to probabilities
    def to_probs(scores: list[float]) -> np.ndarray:
        scaled = np.array(scores) / temperature
        scaled = scaled - np.max(scaled)
        exp_scores = np.exp(scaled)
        return exp_scores / np.sum(exp_scores)

    probs_a = to_probs(scores_a)
    probs_b = to_probs(scores_b)

    # Cross entropy: -Σ p_b log(p_a)
    mask = probs_a > 1e-10
    cross_entropy = -np.sum(probs_b[mask] * np.log(probs_a[mask]))

    return float(cross_entropy)


def compute_kl_divergence(
    scores_p: list[float],
    scores_q: list[float],
    temperature: float = 1.0,
) -> float:
    """
    Compute KL divergence between two score distributions.

    KL(P || Q) measures how much information is lost when Q is used
    to approximate P.

    Args:
        scores_p: Reference distribution scores.
        scores_q: Approximation distribution scores.
        temperature: Softmax temperature.

    Returns:
        KL divergence value.
    """
    if len(scores_p) != len(scores_q):
        raise ValueError("Score lists must have same length")

    if not scores_p:
        return 0.0

    # Convert to probabilities
    def to_probs(scores: list[float]) -> np.ndarray:
        scaled = np.array(scores) / temperature
        scaled = scaled - np.max(scaled)
        exp_scores = np.exp(scaled)
        return exp_scores / np.sum(exp_scores)

    probs_p = to_probs(scores_p)
    probs_q = to_probs(scores_q)

    # KL divergence: Σ p log(p/q)
    # Only where both are non-zero
    mask = (probs_p > 1e-10) & (probs_q > 1e-10)
    kl_div = np.sum(probs_p[mask] * np.log(probs_p[mask] / probs_q[mask]))

    return float(kl_div)
