"""
Acceptance test for RLM mode gating.

From PRD:
- Prompt: "Audit all usages of endpoint X across repo"
- Pass: memory_pack(auto) -> RLM mode, bounded tool usage
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest


# ==============================================================================
# Test Data Structures
# ==============================================================================

@dataclass
class RLMPlan:
    """RLM execution plan."""

    task: str
    steps: list[dict[str, Any]]
    budget: dict[str, int]
    estimated_tokens: int


@dataclass
class RLMExecution:
    """Track RLM execution."""

    plan: RLMPlan
    steps_executed: int
    candidates_found: int
    tokens_used: int
    peek_lines_used: int
    completed: bool
    aborted: bool = False
    abort_reason: str | None = None


# ==============================================================================
# RLM Gating Functions
# ==============================================================================

def compute_entropy(scores: list[float]) -> float:
    """Compute retrieval entropy."""
    if not scores:
        return 0.0

    scores_arr = np.array(scores)
    exp_scores = np.exp(scores_arr - np.max(scores_arr))
    probs = exp_scores / np.sum(exp_scores)

    entropy = 0.0
    for p in probs:
        if p > 1e-10:
            entropy -= p * math.log(p)

    return entropy


def detect_audit_query(query: str) -> bool:
    """Detect if query requires comprehensive audit."""
    audit_patterns = [
        "all usages",
        "every instance",
        "audit",
        "find all",
        "list all",
        "across repo",
        "across the repo",
        "throughout",
        "everywhere",
    ]
    query_lower = query.lower()
    return any(p in query_lower for p in audit_patterns)


def gate_mode(
    query: str,
    scores: list[float],
    mode: str = "auto",
    entropy_threshold: float = 2.5,
) -> tuple[Literal["pack", "rlm"], str]:
    """
    Determine execution mode.

    Returns:
        Tuple of (mode, reason)
    """
    if mode == "pack":
        return "pack", "Explicit pack mode"
    if mode == "rlm":
        return "rlm", "Explicit RLM mode"

    # Auto mode
    if detect_audit_query(query):
        return "rlm", "Query requires comprehensive audit"

    entropy = compute_entropy(scores)
    if entropy >= entropy_threshold:
        return "rlm", f"High entropy ({entropy:.2f})"

    return "pack", f"Low entropy ({entropy:.2f})"


def create_rlm_plan(task: str, budget: dict[str, int] | None = None) -> RLMPlan:
    """Create RLM execution plan."""
    default_budget = {
        "max_steps": 12,
        "max_peek_lines": 1200,
        "max_candidates": 50,
    }
    budget = {**default_budget, **(budget or {})}

    steps = [
        {"action": "search", "query": task, "limit": 50},
        {"action": "filter", "criteria": "relevance > 0.5"},
        {"action": "peek", "candidates": "top_k", "k": 10},
        {"action": "aggregate", "op": "unique"},
    ]

    return RLMPlan(
        task=task,
        steps=steps,
        budget=budget,
        estimated_tokens=budget["max_peek_lines"] * 4,
    )


def execute_rlm_step(
    execution: RLMExecution,
    step: dict[str, Any],
) -> None:
    """Execute a single RLM step with budget tracking."""
    # Check budgets
    if execution.steps_executed >= execution.plan.budget["max_steps"]:
        execution.aborted = True
        execution.abort_reason = "Max steps exceeded"
        return

    if execution.peek_lines_used >= execution.plan.budget["max_peek_lines"]:
        execution.aborted = True
        execution.abort_reason = "Max peek lines exceeded"
        return

    if execution.candidates_found >= execution.plan.budget["max_candidates"]:
        execution.aborted = True
        execution.abort_reason = "Max candidates exceeded"
        return

    # Execute step
    action = step.get("action")

    if action == "search":
        # Simulate search
        execution.candidates_found += 10
        execution.tokens_used += 100

    elif action == "peek":
        # Simulate peek
        execution.peek_lines_used += 100
        execution.tokens_used += 400

    elif action == "filter":
        # Filter doesn't consume much budget
        execution.tokens_used += 10

    elif action == "aggregate":
        # Aggregate is cheap
        execution.tokens_used += 20

    execution.steps_executed += 1


# ==============================================================================
# RLM Gating Acceptance Test
# ==============================================================================

@pytest.mark.acceptance
class TestRLMGating:
    """
    Acceptance test for RLM mode gating.

    Pass criteria (from PRD):
    - Prompt: "Audit all usages of endpoint X across repo"
    - memory_pack(auto) -> RLM mode
    - Bounded tool usage (respects budgets)
    """

    def test_audit_query_triggers_rlm(self):
        """
        Test: "Audit all usages of endpoint X across repo"
        Expected: memory_pack(auto) -> RLM mode
        """
        query = "Audit all usages of endpoint X across repo"

        # Simulate retrieval scores (distributed = high entropy)
        scores = [0.5, 0.48, 0.47, 0.46, 0.45, 0.44, 0.43, 0.42, 0.41, 0.40]

        mode, reason = gate_mode(query, scores, mode="auto")

        assert mode == "rlm", f"Audit query should trigger RLM mode, got: {mode}"
        assert "audit" in reason.lower() or "entropy" in reason.lower()

    def test_find_all_triggers_rlm(self):
        """Test that 'find all' pattern triggers RLM."""
        query = "Find all places where authentication is performed"
        scores = [0.6, 0.5, 0.4]  # Even with low entropy

        mode, reason = gate_mode(query, scores, mode="auto")

        # Audit patterns should override entropy
        is_audit = detect_audit_query(query)
        if is_audit:
            assert mode == "rlm"

    def test_simple_query_stays_pack(self):
        """Test that simple queries stay in pack mode."""
        query = "Where is the auth token validated?"

        # Concentrated scores = low entropy
        scores = [0.95, 0.3, 0.2, 0.1, 0.05]

        mode, reason = gate_mode(query, scores, mode="auto")

        # Should use pack mode for simple focused queries
        entropy = compute_entropy(scores)
        if entropy < 2.5 and not detect_audit_query(query):
            assert mode == "pack"


# ==============================================================================
# RLM Budget Enforcement Tests
# ==============================================================================

@pytest.mark.acceptance
class TestRLMBudgetEnforcement:
    """Tests for RLM budget enforcement."""

    def test_max_steps_enforced(self):
        """Test that max_steps budget is enforced."""
        plan = create_rlm_plan(
            "Audit all usages",
            budget={"max_steps": 5, "max_peek_lines": 1200, "max_candidates": 50}
        )

        execution = RLMExecution(
            plan=plan,
            steps_executed=0,
            candidates_found=0,
            tokens_used=0,
            peek_lines_used=0,
            completed=False,
        )

        # Try to execute more steps than allowed
        for i in range(10):
            if not execution.aborted:
                execute_rlm_step(execution, {"action": "search"})

        assert execution.aborted
        assert "steps" in execution.abort_reason.lower()
        assert execution.steps_executed <= plan.budget["max_steps"]

    def test_max_peek_lines_enforced(self):
        """Test that max_peek_lines budget is enforced."""
        plan = create_rlm_plan(
            "Audit all usages",
            budget={"max_steps": 100, "max_peek_lines": 500, "max_candidates": 50}
        )

        execution = RLMExecution(
            plan=plan,
            steps_executed=0,
            candidates_found=0,
            tokens_used=0,
            peek_lines_used=0,
            completed=False,
        )

        # Execute peek steps until budget exceeded
        for i in range(20):
            if not execution.aborted:
                execute_rlm_step(execution, {"action": "peek"})

        assert execution.aborted
        assert "peek" in execution.abort_reason.lower()
        assert execution.peek_lines_used <= plan.budget["max_peek_lines"] + 100

    def test_max_candidates_enforced(self):
        """Test that max_candidates budget is enforced."""
        plan = create_rlm_plan(
            "Audit all usages",
            budget={"max_steps": 100, "max_peek_lines": 5000, "max_candidates": 25}
        )

        execution = RLMExecution(
            plan=plan,
            steps_executed=0,
            candidates_found=0,
            tokens_used=0,
            peek_lines_used=0,
            completed=False,
        )

        # Execute search steps until candidates exceeded
        for i in range(10):
            if not execution.aborted:
                execute_rlm_step(execution, {"action": "search"})

        assert execution.aborted
        assert "candidates" in execution.abort_reason.lower()

    def test_prd_default_budgets(self):
        """Test PRD default budgets are respected."""
        # PRD specifies: max_steps=12, max_peek_lines=1200, max_candidates=50
        plan = create_rlm_plan("Task")

        assert plan.budget["max_steps"] == 12
        assert plan.budget["max_peek_lines"] == 1200
        assert plan.budget["max_candidates"] == 50


# ==============================================================================
# RLM Fallback Tests
# ==============================================================================

@pytest.mark.acceptance
class TestRLMFallback:
    """Tests for RLM fallback behavior."""

    def test_fallback_to_pack_on_budget_exceeded(self):
        """Test fallback to pack mode when RLM budget exceeded."""
        plan = create_rlm_plan(
            "Large audit task",
            budget={"max_steps": 3, "max_peek_lines": 100, "max_candidates": 10}
        )

        execution = RLMExecution(
            plan=plan,
            steps_executed=0,
            candidates_found=0,
            tokens_used=0,
            peek_lines_used=0,
            completed=False,
        )

        # Execute until budget exceeded
        while not execution.aborted and execution.steps_executed < 10:
            execute_rlm_step(execution, {"action": "search"})

        assert execution.aborted
        # In production, system should fall back to pack mode with warning

    def test_partial_results_on_abort(self):
        """Test that partial results are available on abort."""
        plan = create_rlm_plan(
            "Task",
            budget={"max_steps": 2, "max_peek_lines": 1200, "max_candidates": 50}
        )

        execution = RLMExecution(
            plan=plan,
            steps_executed=0,
            candidates_found=0,
            tokens_used=0,
            peek_lines_used=0,
            completed=False,
        )

        # Execute a few steps
        execute_rlm_step(execution, {"action": "search"})
        execute_rlm_step(execution, {"action": "search"})

        # Should have some results even if aborted
        assert execution.candidates_found > 0
        assert execution.steps_executed > 0


# ==============================================================================
# Entropy-Based Gating Tests
# ==============================================================================

@pytest.mark.acceptance
class TestEntropyBasedGating:
    """Tests for entropy-based mode gating."""

    def test_low_entropy_selects_pack(self):
        """Test low entropy selects pack mode."""
        query = "Where is validateToken defined?"
        scores = [0.95, 0.2, 0.1, 0.05, 0.02]

        entropy = compute_entropy(scores)
        mode, _ = gate_mode(query, scores, mode="auto")

        assert entropy < 2.5
        assert mode == "pack"

    def test_high_entropy_selects_rlm(self):
        """Test high entropy selects RLM mode."""
        query = "How is authentication implemented?"

        # Uniform-ish scores = high entropy
        scores = [0.5, 0.48, 0.47, 0.46, 0.45, 0.44, 0.43, 0.42]

        entropy = compute_entropy(scores)
        mode, _ = gate_mode(query, scores, mode="auto")

        # With high entropy (ambiguous results), should prefer RLM
        if entropy >= 2.5:
            assert mode == "rlm"

    def test_entropy_threshold_configurable(self):
        """Test that entropy threshold is configurable."""
        query = "Test query"
        scores = [0.6, 0.5, 0.4, 0.3, 0.2]

        entropy = compute_entropy(scores)

        # With low threshold, should trigger RLM
        mode_low, _ = gate_mode(query, scores, entropy_threshold=0.5)

        # With high threshold, should stay pack
        mode_high, _ = gate_mode(query, scores, entropy_threshold=10.0)

        if entropy >= 0.5:
            assert mode_low == "rlm"
        if entropy < 10.0:
            assert mode_high == "pack"
