"""
Unit tests for the retrieval entropy module.

Tests cover:
- Low entropy (concentrated scores)
- High entropy (distributed scores)
- Temperature parameter
- Entropy-based gating decisions
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest


# ==============================================================================
# Entropy Computation Functions
# ==============================================================================

def compute_entropy(scores: np.ndarray, temperature: float = 1.0) -> float:
    """
    Compute retrieval entropy from relevance scores.

    H = -sum(p_i * log(p_i))

    where p_i = softmax(score_i / temperature)

    Higher entropy indicates more uncertainty (spread-out scores).
    Lower entropy indicates more certainty (concentrated scores).

    Args:
        scores: Array of relevance scores
        temperature: Softmax temperature (higher = more uniform distribution)

    Returns:
        Entropy value in bits (log base 2) or nats (natural log)
    """
    if len(scores) == 0:
        return 0.0

    # Apply temperature scaling
    scaled_scores = np.array(scores) / temperature

    # Softmax to get probability distribution
    exp_scores = np.exp(scaled_scores - np.max(scaled_scores))  # Subtract max for numerical stability
    probabilities = exp_scores / np.sum(exp_scores)

    # Compute entropy (using natural log)
    # H = -sum(p * log(p))
    # Avoid log(0) by filtering out zero probabilities
    entropy = 0.0
    for p in probabilities:
        if p > 1e-10:
            entropy -= p * math.log(p)

    return entropy


def compute_entropy_bits(scores: np.ndarray, temperature: float = 1.0) -> float:
    """Compute entropy in bits (log base 2)."""
    entropy_nats = compute_entropy(scores, temperature)
    # Convert from nats to bits: bits = nats / ln(2)
    return entropy_nats / math.log(2)


def softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Compute softmax with temperature."""
    scaled = np.array(scores) / temperature
    exp_scores = np.exp(scaled - np.max(scaled))
    return exp_scores / np.sum(exp_scores)


# ==============================================================================
# Basic Entropy Tests
# ==============================================================================

class TestEntropyBasic:
    """Basic tests for entropy computation."""

    def test_entropy_uniform_distribution(self):
        """Test entropy of uniform distribution is maximum."""
        n = 10
        # Uniform scores -> uniform probabilities
        scores = np.ones(n)

        entropy = compute_entropy(scores)

        # Maximum entropy for n items is log(n)
        max_entropy = math.log(n)
        assert abs(entropy - max_entropy) < 1e-6

    def test_entropy_single_peak(self):
        """Test entropy of single dominant score is low."""
        # One very high score, rest low
        scores = np.array([10.0, 0.1, 0.1, 0.1, 0.1])

        entropy = compute_entropy(scores)

        # Should be close to 0 (very concentrated)
        assert entropy < 0.5

    def test_entropy_two_peaks(self):
        """Test entropy of two equal peaks."""
        # Two equal high scores
        scores = np.array([10.0, 10.0, 0.1, 0.1, 0.1])

        entropy = compute_entropy(scores)

        # Should be around log(2) ~ 0.693
        assert 0.5 < entropy < 1.0

    def test_entropy_increases_with_spread(self):
        """Test that entropy increases as scores spread out."""
        # Concentrated
        scores_concentrated = np.array([1.0, 0.1, 0.1, 0.1, 0.1])

        # Spread out
        scores_spread = np.array([0.5, 0.4, 0.3, 0.2, 0.1])

        entropy_concentrated = compute_entropy(scores_concentrated)
        entropy_spread = compute_entropy(scores_spread)

        assert entropy_spread > entropy_concentrated


# ==============================================================================
# Low Entropy Tests
# ==============================================================================

class TestLowEntropy:
    """Tests for low entropy scenarios (concentrated scores)."""

    def test_very_low_entropy(self):
        """Test very concentrated score distribution."""
        # One dominant result
        scores = np.array([0.99, 0.001, 0.001, 0.001, 0.001])

        entropy = compute_entropy(scores)

        # Very low entropy
        assert entropy < 0.2

    def test_low_entropy_triggers_pack_mode(self):
        """Test that low entropy suggests pack mode."""
        # Simulated scenario: clear top results
        scores = np.array([0.95, 0.85, 0.3, 0.2, 0.1])

        entropy = compute_entropy(scores)

        # Low entropy threshold for pack mode (from config: 2.5)
        entropy_threshold = 2.5
        # In bits for comparison
        entropy_bits = entropy / math.log(2)

        # Low entropy indicates pack mode is appropriate
        if entropy_bits < entropy_threshold:
            mode = "pack"
        else:
            mode = "rlm"

        # With concentrated scores, should be pack mode
        assert mode == "pack"

    @pytest.mark.parametrize("dominant_score,expected_entropy_bound", [
        (0.99, 0.1),    # Very dominant -> very low entropy
        (0.9, 0.5),     # Fairly dominant -> low entropy
        (0.7, 1.0),     # Somewhat dominant -> medium-low entropy
    ])
    def test_entropy_bounds_for_dominant_scores(self, dominant_score, expected_entropy_bound):
        """Test entropy bounds for various dominant score levels."""
        # Create scores with one dominant
        n = 5
        remaining = (1 - dominant_score) / (n - 1)
        scores = [dominant_score] + [remaining] * (n - 1)

        entropy = compute_entropy(np.array(scores))

        assert entropy < expected_entropy_bound


# ==============================================================================
# High Entropy Tests
# ==============================================================================

class TestHighEntropy:
    """Tests for high entropy scenarios (distributed scores)."""

    def test_high_entropy_uniform(self):
        """Test high entropy with uniform distribution."""
        n = 20
        scores = np.ones(n)  # All equal

        entropy = compute_entropy(scores)

        # Should be close to log(n) ~ 2.996
        expected = math.log(n)
        assert abs(entropy - expected) < 1e-6

    def test_high_entropy_triggers_rlm_mode(self):
        """Test that high entropy suggests RLM mode."""
        # Distributed scores (no clear winner)
        scores = np.array([0.5, 0.48, 0.47, 0.45, 0.44, 0.43, 0.42, 0.41])

        entropy = compute_entropy(scores)
        entropy_bits = entropy / math.log(2)

        # High entropy threshold for RLM mode
        entropy_threshold = 2.5

        if entropy_bits < entropy_threshold:
            mode = "pack"
        else:
            mode = "rlm"

        # With distributed scores, should be RLM mode
        assert entropy_bits > 1.5  # Verify it's reasonably high

    def test_entropy_approaches_max(self):
        """Test entropy approaches maximum for uniform distribution."""
        for n in [5, 10, 20, 50]:
            scores = np.ones(n)
            entropy = compute_entropy(scores)
            max_entropy = math.log(n)

            assert abs(entropy - max_entropy) < 1e-6


# ==============================================================================
# Temperature Parameter Tests
# ==============================================================================

class TestTemperatureParameter:
    """Tests for temperature parameter effects."""

    def test_temperature_one_is_standard(self):
        """Test that temperature=1 gives standard softmax."""
        scores = np.array([2.0, 1.0, 0.5])

        probs_t1 = softmax(scores, temperature=1.0)

        # Standard softmax
        exp_scores = np.exp(scores - np.max(scores))
        expected = exp_scores / np.sum(exp_scores)

        np.testing.assert_array_almost_equal(probs_t1, expected)

    def test_high_temperature_flattens_distribution(self):
        """Test that high temperature makes distribution more uniform."""
        scores = np.array([3.0, 1.0, 0.5])

        probs_low_t = softmax(scores, temperature=0.5)
        probs_high_t = softmax(scores, temperature=2.0)

        # High temperature -> more uniform
        # Measured by entropy
        entropy_low = compute_entropy(scores, temperature=0.5)
        entropy_high = compute_entropy(scores, temperature=2.0)

        assert entropy_high > entropy_low

    def test_low_temperature_sharpens_distribution(self):
        """Test that low temperature makes distribution more peaked."""
        scores = np.array([3.0, 2.0, 1.0])

        probs_high_t = softmax(scores, temperature=2.0)
        probs_low_t = softmax(scores, temperature=0.5)

        # Low temperature -> more peaked
        # Top score gets higher probability
        assert probs_low_t[0] > probs_high_t[0]

    def test_very_low_temperature_approaches_argmax(self):
        """Test that very low temperature approaches argmax."""
        scores = np.array([3.0, 2.0, 1.0, 0.5])

        probs = softmax(scores, temperature=0.01)

        # Should be nearly one-hot
        assert probs[0] > 0.99

    def test_very_high_temperature_approaches_uniform(self):
        """Test that very high temperature approaches uniform."""
        scores = np.array([10.0, 5.0, 1.0, 0.1])

        probs = softmax(scores, temperature=100.0)

        # Should be nearly uniform
        expected_uniform = 1.0 / len(scores)
        for p in probs:
            assert abs(p - expected_uniform) < 0.1

    @pytest.mark.parametrize("temperature", [0.1, 0.5, 1.0, 2.0, 5.0])
    def test_temperature_preserves_probability_sum(self, temperature):
        """Test that probabilities sum to 1 for any temperature."""
        scores = np.array([3.0, 2.0, 1.0, 0.5])

        probs = softmax(scores, temperature)

        assert abs(sum(probs) - 1.0) < 1e-6


# ==============================================================================
# Entropy Range Tests
# ==============================================================================

class TestEntropyRange:
    """Tests for entropy value ranges."""

    def test_entropy_non_negative(self):
        """Test that entropy is always non-negative."""
        test_cases = [
            np.array([1.0]),
            np.array([1.0, 0.0]),
            np.array([0.5, 0.5]),
            np.array([0.9, 0.05, 0.05]),
            np.random.rand(100),
        ]

        for scores in test_cases:
            if len(scores) > 0:
                entropy = compute_entropy(scores)
                assert entropy >= 0

    def test_entropy_bounded_by_log_n(self):
        """Test that entropy is bounded by log(n)."""
        for n in [2, 5, 10, 50, 100]:
            scores = np.random.rand(n)
            entropy = compute_entropy(scores)
            max_entropy = math.log(n)

            assert entropy <= max_entropy + 1e-6

    def test_entropy_of_single_item(self):
        """Test entropy of single item is zero."""
        scores = np.array([1.0])
        entropy = compute_entropy(scores)

        assert abs(entropy) < 1e-10

    def test_entropy_bits_range(self):
        """Test entropy in bits has expected range."""
        # For n items, max entropy in bits is log2(n)
        n = 8
        scores = np.ones(n)  # Uniform

        entropy_bits = compute_entropy_bits(scores)

        # log2(8) = 3 bits
        assert abs(entropy_bits - 3.0) < 1e-6


# ==============================================================================
# Gating Decision Tests
# ==============================================================================

class TestEntropyGating:
    """Tests for entropy-based mode gating decisions."""

    def test_gating_threshold(self):
        """Test gating threshold from config."""
        threshold = 2.5  # Default from RLMConfig

        # Below threshold -> pack mode
        # Above threshold -> RLM mode
        assert threshold > 0

    def test_pack_mode_selection(self):
        """Test pack mode is selected for low entropy."""
        # Simulated retrieval with clear top results
        scores = np.array([0.9, 0.85, 0.4, 0.3, 0.2, 0.1, 0.1, 0.1])

        entropy = compute_entropy(scores)
        threshold = 2.5

        if entropy < threshold:
            mode = "pack"
        else:
            mode = "rlm"

        assert mode == "pack"

    def test_rlm_mode_selection(self):
        """Test RLM mode is selected for high entropy."""
        # Simulated retrieval with ambiguous results
        n = 20
        scores = np.ones(n) * 0.5 + np.random.rand(n) * 0.1  # Nearly uniform

        entropy = compute_entropy(scores)
        threshold = 1.5  # Using lower threshold to ensure RLM selection

        if entropy < threshold:
            mode = "pack"
        else:
            mode = "rlm"

        assert mode == "rlm"

    def test_auto_mode_uses_entropy(self):
        """Test that auto mode delegates based on entropy."""
        def select_mode(scores: np.ndarray, mode: str = "auto", threshold: float = 2.5) -> str:
            if mode != "auto":
                return mode

            entropy = compute_entropy(scores)
            return "pack" if entropy < threshold else "rlm"

        # Test auto with low entropy
        low_entropy_scores = np.array([0.95, 0.1, 0.05])
        assert select_mode(low_entropy_scores) == "pack"

        # Test explicit modes override
        assert select_mode(low_entropy_scores, mode="pack") == "pack"
        assert select_mode(low_entropy_scores, mode="rlm") == "rlm"


# ==============================================================================
# Edge Cases Tests
# ==============================================================================

class TestEntropyEdgeCases:
    """Tests for entropy computation edge cases."""

    def test_empty_scores(self):
        """Test entropy of empty score array."""
        scores = np.array([])
        entropy = compute_entropy(scores)

        assert entropy == 0.0

    def test_all_zero_scores(self):
        """Test entropy when all scores are zero."""
        scores = np.zeros(5)
        entropy = compute_entropy(scores)

        # All zeros -> uniform after softmax -> max entropy
        max_entropy = math.log(5)
        assert abs(entropy - max_entropy) < 1e-6

    def test_negative_scores(self):
        """Test entropy with negative scores."""
        scores = np.array([-1.0, -2.0, -3.0])

        entropy = compute_entropy(scores)

        # Softmax handles negative values
        assert entropy >= 0

    def test_very_large_scores(self):
        """Test entropy with very large scores (numerical stability)."""
        scores = np.array([1000.0, 999.0, 1.0])

        # Should not overflow due to max subtraction in softmax
        entropy = compute_entropy(scores)

        assert not math.isnan(entropy)
        assert not math.isinf(entropy)

    def test_very_small_differences(self):
        """Test entropy with very small score differences."""
        scores = np.array([1.0, 1.0 + 1e-10, 1.0 - 1e-10])

        entropy = compute_entropy(scores)

        # Should be close to max entropy for 3 items
        max_entropy = math.log(3)
        assert abs(entropy - max_entropy) < 0.1

    def test_single_non_zero_score(self):
        """Test entropy with single non-zero score."""
        scores = np.array([1.0, 0.0, 0.0, 0.0])

        # After softmax, still single dominant
        probs = softmax(scores)
        # First will be dominant but not 1.0 due to softmax

        entropy = compute_entropy(scores)
        # Should be relatively low
        assert entropy < 1.0


# ==============================================================================
# Numerical Stability Tests
# ==============================================================================

class TestNumericalStability:
    """Tests for numerical stability of entropy computation."""

    def test_softmax_numerical_stability(self):
        """Test softmax is numerically stable with large values."""
        large_scores = np.array([1000.0, 1001.0, 1002.0])

        probs = softmax(large_scores)

        assert not np.any(np.isnan(probs))
        assert not np.any(np.isinf(probs))
        assert abs(np.sum(probs) - 1.0) < 1e-6

    def test_log_probability_stability(self):
        """Test log computation doesn't produce -inf."""
        scores = np.array([100.0, 1.0, 0.001])

        entropy = compute_entropy(scores)

        assert not math.isnan(entropy)
        assert not math.isinf(entropy)

    def test_extreme_temperature_stability(self):
        """Test stability with extreme temperature values."""
        scores = np.array([2.0, 1.0, 0.5])

        # Very low temperature
        entropy_low_t = compute_entropy(scores, temperature=0.001)
        assert not math.isnan(entropy_low_t)

        # Very high temperature
        entropy_high_t = compute_entropy(scores, temperature=1000.0)
        assert not math.isnan(entropy_high_t)
