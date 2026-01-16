"""
Unit tests for the mode gating/selection module.

Tests cover:
- Pack mode selection
- RLM mode triggers
- Entropy threshold
- Contract touch detection
- Auto mode logic
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pytest


# ==============================================================================
# Gating Data Types
# ==============================================================================

@dataclass
class GatingContext:
    """Context for mode gating decisions."""

    query: str
    retrieval_scores: list[float]
    entropy: float
    contract_touched: bool
    focus_paths: list[str]
    explicit_mode: Literal["auto", "pack", "rlm"] | None = None


@dataclass
class GatingDecision:
    """Result of mode gating."""

    mode: Literal["pack", "rlm"]
    reason: str
    entropy: float
    confidence: float


# ==============================================================================
# Gating Functions
# ==============================================================================

def compute_entropy(scores: list[float], temperature: float = 1.0) -> float:
    """Compute entropy from retrieval scores."""
    if not scores:
        return 0.0

    scores_arr = np.array(scores)
    scaled = scores_arr / temperature
    exp_scores = np.exp(scaled - np.max(scaled))
    probs = exp_scores / np.sum(exp_scores)

    entropy = 0.0
    for p in probs:
        if p > 1e-10:
            entropy -= p * math.log(p)

    return entropy


def detect_contract_touch(query: str, contract_keywords: list[str] | None = None) -> bool:
    """Detect if query touches contracts/interfaces."""
    keywords = contract_keywords or [
        "interface",
        "contract",
        "schema",
        "api",
        "type",
        "endpoint",
        "protocol",
        "specification",
        "definition",
    ]

    query_lower = query.lower()
    return any(kw in query_lower for kw in keywords)


def detect_audit_query(query: str) -> bool:
    """Detect if query requires comprehensive audit (RLM mode)."""
    audit_patterns = [
        "all usages",
        "every instance",
        "audit",
        "find all",
        "list all",
        "comprehensive",
        "throughout",
        "across the repo",
        "everywhere",
    ]

    query_lower = query.lower()
    return any(pattern in query_lower for pattern in audit_patterns)


def gate_mode(context: GatingContext, threshold: float = 2.5) -> GatingDecision:
    """
    Determine execution mode based on context.

    Decision logic:
    1. Explicit mode overrides all
    2. Audit queries -> RLM mode
    3. High entropy -> RLM mode
    4. Low entropy -> Pack mode

    Args:
        context: Gating context with query and retrieval info
        threshold: Entropy threshold for mode selection

    Returns:
        GatingDecision with selected mode and reason
    """
    # Explicit mode takes priority
    if context.explicit_mode and context.explicit_mode != "auto":
        return GatingDecision(
            mode=context.explicit_mode,
            reason=f"Explicit mode: {context.explicit_mode}",
            entropy=context.entropy,
            confidence=1.0,
        )

    # Check for audit-style queries
    if detect_audit_query(context.query):
        return GatingDecision(
            mode="rlm",
            reason="Query requires comprehensive audit",
            entropy=context.entropy,
            confidence=0.9,
        )

    # Entropy-based gating
    if context.entropy >= threshold:
        return GatingDecision(
            mode="rlm",
            reason=f"High entropy ({context.entropy:.2f} >= {threshold})",
            entropy=context.entropy,
            confidence=min(1.0, (context.entropy - threshold) / threshold + 0.5),
        )
    else:
        return GatingDecision(
            mode="pack",
            reason=f"Low entropy ({context.entropy:.2f} < {threshold})",
            entropy=context.entropy,
            confidence=min(1.0, (threshold - context.entropy) / threshold + 0.5),
        )


# ==============================================================================
# Pack Mode Selection Tests
# ==============================================================================

class TestPackModeSelection:
    """Tests for pack mode selection."""

    def test_low_entropy_selects_pack(self):
        """Test that low entropy selects pack mode."""
        context = GatingContext(
            query="Where is auth token validated?",
            retrieval_scores=[0.95, 0.85, 0.3, 0.2, 0.1],
            entropy=1.0,
            contract_touched=False,
            focus_paths=[],
        )

        decision = gate_mode(context, threshold=2.5)

        assert decision.mode == "pack"
        assert "Low entropy" in decision.reason

    def test_explicit_pack_mode(self):
        """Test explicit pack mode selection."""
        context = GatingContext(
            query="Complex query",
            retrieval_scores=[0.5, 0.5, 0.5, 0.5],
            entropy=3.0,  # High entropy
            contract_touched=False,
            focus_paths=[],
            explicit_mode="pack",
        )

        decision = gate_mode(context)

        assert decision.mode == "pack"
        assert "Explicit mode" in decision.reason

    def test_concentrated_scores_pack_mode(self):
        """Test that concentrated scores lead to pack mode."""
        # Scores with clear top result
        scores = [0.95, 0.4, 0.2, 0.1, 0.05]
        entropy = compute_entropy(scores)

        context = GatingContext(
            query="Simple question",
            retrieval_scores=scores,
            entropy=entropy,
            contract_touched=False,
            focus_paths=[],
        )

        decision = gate_mode(context)

        assert decision.mode == "pack"


# ==============================================================================
# RLM Mode Trigger Tests
# ==============================================================================

class TestRLMModeTrigers:
    """Tests for RLM mode triggers."""

    def test_high_entropy_selects_rlm(self):
        """Test that high entropy selects RLM mode."""
        context = GatingContext(
            query="Ambiguous query with many possible answers",
            retrieval_scores=[0.5, 0.49, 0.48, 0.47, 0.46, 0.45],
            entropy=3.0,
            contract_touched=False,
            focus_paths=[],
        )

        decision = gate_mode(context, threshold=2.5)

        assert decision.mode == "rlm"
        assert "High entropy" in decision.reason

    def test_explicit_rlm_mode(self):
        """Test explicit RLM mode selection."""
        context = GatingContext(
            query="Simple query",
            retrieval_scores=[0.95, 0.2, 0.1],
            entropy=0.5,  # Low entropy
            contract_touched=False,
            focus_paths=[],
            explicit_mode="rlm",
        )

        decision = gate_mode(context)

        assert decision.mode == "rlm"
        assert "Explicit mode" in decision.reason

    def test_audit_query_triggers_rlm(self):
        """Test that audit queries trigger RLM mode."""
        audit_queries = [
            "Find all usages of endpoint X across the repo",
            "Audit every instance of this function",
            "List all places where this is called",
            "Comprehensive review of authentication",
        ]

        for query in audit_queries:
            context = GatingContext(
                query=query,
                retrieval_scores=[0.9, 0.1, 0.05],
                entropy=0.5,  # Low entropy
                contract_touched=False,
                focus_paths=[],
            )

            decision = gate_mode(context)
            assert decision.mode == "rlm", f"Query should trigger RLM: {query}"

    def test_distributed_scores_rlm_mode(self):
        """Test that distributed scores lead to RLM mode."""
        # Uniform-ish scores
        scores = [0.5, 0.48, 0.47, 0.46, 0.45, 0.44, 0.43, 0.42]
        entropy = compute_entropy(scores)

        context = GatingContext(
            query="Which files use this pattern?",
            retrieval_scores=scores,
            entropy=entropy,
            contract_touched=False,
            focus_paths=[],
        )

        decision = gate_mode(context)

        # With uniform scores, entropy should be high
        assert entropy > 1.5


# ==============================================================================
# Entropy Threshold Tests
# ==============================================================================

class TestEntropyThreshold:
    """Tests for entropy threshold behavior."""

    def test_default_threshold(self):
        """Test default entropy threshold is 2.5."""
        threshold = 2.5

        # Just below threshold -> pack
        context_low = GatingContext(
            query="Test",
            retrieval_scores=[],
            entropy=2.4,
            contract_touched=False,
            focus_paths=[],
        )

        # Just above threshold -> rlm
        context_high = GatingContext(
            query="Test",
            retrieval_scores=[],
            entropy=2.6,
            contract_touched=False,
            focus_paths=[],
        )

        assert gate_mode(context_low, threshold).mode == "pack"
        assert gate_mode(context_high, threshold).mode == "rlm"

    def test_exact_threshold(self):
        """Test behavior at exact threshold."""
        threshold = 2.5

        context = GatingContext(
            query="Test",
            retrieval_scores=[],
            entropy=2.5,
            contract_touched=False,
            focus_paths=[],
        )

        decision = gate_mode(context, threshold)

        # At threshold, should trigger RLM (>= threshold)
        assert decision.mode == "rlm"

    @pytest.mark.parametrize("threshold", [1.0, 1.5, 2.0, 2.5, 3.0])
    def test_custom_thresholds(self, threshold):
        """Test custom threshold values."""
        # Low entropy context
        low_context = GatingContext(
            query="Test",
            retrieval_scores=[],
            entropy=threshold - 0.5,
            contract_touched=False,
            focus_paths=[],
        )

        # High entropy context
        high_context = GatingContext(
            query="Test",
            retrieval_scores=[],
            entropy=threshold + 0.5,
            contract_touched=False,
            focus_paths=[],
        )

        assert gate_mode(low_context, threshold).mode == "pack"
        assert gate_mode(high_context, threshold).mode == "rlm"


# ==============================================================================
# Contract Touch Detection Tests
# ==============================================================================

class TestContractTouchDetection:
    """Tests for contract touch detection."""

    @pytest.mark.parametrize("query,expected", [
        ("What is the API interface?", True),
        ("Show me the contract definition", True),
        ("Where is the schema defined?", True),
        ("List all endpoints", True),
        ("What types are used?", True),
        ("How does the function work?", False),
        ("Debug this code", False),
        ("Fix the bug", False),
    ])
    def test_contract_keywords(self, query, expected):
        """Test contract keyword detection."""
        detected = detect_contract_touch(query)
        assert detected == expected

    def test_custom_contract_keywords(self):
        """Test custom contract keywords."""
        custom_keywords = ["model", "entity", "dto"]

        query = "Show me the user model"
        detected = detect_contract_touch(query, custom_keywords)

        assert detected is True

    def test_case_insensitive(self):
        """Test case insensitive detection."""
        queries = [
            "What is the API?",
            "what is the api?",
            "WHAT IS THE API?",
            "What Is The Api?",
        ]

        for query in queries:
            assert detect_contract_touch(query) is True


# ==============================================================================
# Auto Mode Tests
# ==============================================================================

class TestAutoMode:
    """Tests for auto mode logic."""

    def test_auto_mode_uses_entropy(self):
        """Test that auto mode delegates based on entropy."""
        # Low entropy -> pack
        low_context = GatingContext(
            query="Simple question",
            retrieval_scores=[0.9, 0.1, 0.05],
            entropy=0.5,
            contract_touched=False,
            focus_paths=[],
            explicit_mode="auto",
        )

        decision = gate_mode(low_context)
        assert decision.mode == "pack"

    def test_auto_with_audit_query(self):
        """Test auto mode with audit query."""
        context = GatingContext(
            query="Find all usages of this function",
            retrieval_scores=[0.9, 0.1],
            entropy=0.3,  # Low entropy
            contract_touched=False,
            focus_paths=[],
            explicit_mode="auto",
        )

        decision = gate_mode(context)

        # Audit queries should trigger RLM regardless of entropy
        assert decision.mode == "rlm"

    def test_auto_none_same_as_auto(self):
        """Test that explicit_mode=None behaves like auto."""
        context_none = GatingContext(
            query="Test",
            retrieval_scores=[],
            entropy=1.0,
            contract_touched=False,
            focus_paths=[],
            explicit_mode=None,
        )

        context_auto = GatingContext(
            query="Test",
            retrieval_scores=[],
            entropy=1.0,
            contract_touched=False,
            focus_paths=[],
            explicit_mode="auto",
        )

        decision_none = gate_mode(context_none)
        decision_auto = gate_mode(context_auto)

        assert decision_none.mode == decision_auto.mode


# ==============================================================================
# Audit Query Detection Tests
# ==============================================================================

class TestAuditQueryDetection:
    """Tests for audit query detection."""

    @pytest.mark.parametrize("query,expected", [
        ("Audit all usages of endpoint X across repo", True),
        ("Find all places where this is used", True),
        ("List all instances of this pattern", True),
        ("Where is this called?", False),
        ("How does this work?", False),
        ("Fix the authentication bug", False),
    ])
    def test_audit_patterns(self, query, expected):
        """Test audit pattern detection."""
        detected = detect_audit_query(query)
        assert detected == expected

    def test_comprehensive_query(self):
        """Test comprehensive query detection."""
        query = "I need a comprehensive review of all authentication code"
        assert detect_audit_query(query) is True

    def test_throughout_query(self):
        """Test throughout query detection."""
        query = "Check this pattern throughout the codebase"
        assert detect_audit_query(query) is True


# ==============================================================================
# Confidence Score Tests
# ==============================================================================

class TestConfidenceScores:
    """Tests for gating confidence scores."""

    def test_explicit_mode_confidence(self):
        """Test explicit mode has confidence 1.0."""
        context = GatingContext(
            query="Test",
            retrieval_scores=[],
            entropy=1.5,
            contract_touched=False,
            focus_paths=[],
            explicit_mode="pack",
        )

        decision = gate_mode(context)

        assert decision.confidence == 1.0

    def test_clear_pack_high_confidence(self):
        """Test clear pack decision has high confidence."""
        context = GatingContext(
            query="Test",
            retrieval_scores=[],
            entropy=0.5,  # Well below threshold
            contract_touched=False,
            focus_paths=[],
        )

        decision = gate_mode(context, threshold=2.5)

        assert decision.confidence > 0.7

    def test_boundary_lower_confidence(self):
        """Test boundary decisions have lower confidence."""
        context = GatingContext(
            query="Test",
            retrieval_scores=[],
            entropy=2.4,  # Just below threshold
            contract_touched=False,
            focus_paths=[],
        )

        decision = gate_mode(context, threshold=2.5)

        # Near threshold should have moderate confidence
        assert 0.3 < decision.confidence < 0.9


# ==============================================================================
# Edge Cases Tests
# ==============================================================================

class TestGatingEdgeCases:
    """Tests for gating edge cases."""

    def test_empty_scores(self):
        """Test gating with empty retrieval scores."""
        context = GatingContext(
            query="Test",
            retrieval_scores=[],
            entropy=0.0,
            contract_touched=False,
            focus_paths=[],
        )

        decision = gate_mode(context)

        # Empty scores -> low entropy -> pack mode
        assert decision.mode == "pack"

    def test_single_score(self):
        """Test gating with single retrieval score."""
        context = GatingContext(
            query="Test",
            retrieval_scores=[0.9],
            entropy=0.0,
            contract_touched=False,
            focus_paths=[],
        )

        decision = gate_mode(context)

        assert decision.mode == "pack"

    def test_empty_query(self):
        """Test gating with empty query."""
        context = GatingContext(
            query="",
            retrieval_scores=[0.5, 0.5],
            entropy=1.0,
            contract_touched=False,
            focus_paths=[],
        )

        decision = gate_mode(context)

        # Should still work based on entropy
        assert decision.mode in ["pack", "rlm"]

    def test_very_high_entropy(self):
        """Test gating with very high entropy."""
        context = GatingContext(
            query="Test",
            retrieval_scores=[],
            entropy=10.0,
            contract_touched=False,
            focus_paths=[],
        )

        decision = gate_mode(context)

        assert decision.mode == "rlm"

    def test_negative_entropy(self):
        """Test gating with negative entropy (invalid but handled)."""
        context = GatingContext(
            query="Test",
            retrieval_scores=[],
            entropy=-1.0,
            contract_touched=False,
            focus_paths=[],
        )

        decision = gate_mode(context)

        # Should handle gracefully
        assert decision.mode == "pack"


# ==============================================================================
# Integration Tests
# ==============================================================================

class TestGatingIntegration:
    """Integration tests for mode gating."""

    def test_full_gating_pipeline(self):
        """Test complete gating pipeline."""
        # Simulate retrieval
        scores = [0.92, 0.88, 0.45, 0.32, 0.21]
        entropy = compute_entropy(scores)

        context = GatingContext(
            query="Where is the auth token validated?",
            retrieval_scores=scores,
            entropy=entropy,
            contract_touched=detect_contract_touch("Where is the auth token validated?"),
            focus_paths=["/src/auth/"],
        )

        decision = gate_mode(context)

        # With concentrated scores, should be pack mode
        assert decision.mode == "pack"
        assert decision.entropy == entropy

    def test_rlm_pipeline(self):
        """Test RLM gating pipeline."""
        # Distributed scores
        scores = [0.5, 0.48, 0.47, 0.46, 0.45, 0.44, 0.43, 0.42, 0.41, 0.40]
        entropy = compute_entropy(scores)

        context = GatingContext(
            query="Find all usages of endpoint X across the repo",
            retrieval_scores=scores,
            entropy=entropy,
            contract_touched=detect_contract_touch("Find all usages of endpoint X across the repo"),
            focus_paths=[],
        )

        decision = gate_mode(context)

        # Audit query should trigger RLM
        assert decision.mode == "rlm"
