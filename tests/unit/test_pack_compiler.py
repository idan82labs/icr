"""
Unit tests for the pack compiler (knapsack packing) module.

Tests cover:
- Knapsack optimization
- Mandatory items
- Budget constraints
- Overflow handling
- Pack formatting
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest


# ==============================================================================
# Test Data Types
# ==============================================================================

@dataclass
class PackItem:
    """Item to be packed into context."""

    id: str
    content: str
    tokens: int
    score: float
    mandatory: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PackResult:
    """Result of pack compilation."""

    content: str
    total_tokens: int
    items: list[str]
    overflow_items: list[str]
    citations: dict[str, str]


# ==============================================================================
# Knapsack Algorithm Implementation
# ==============================================================================

def knapsack_pack(
    items: list[PackItem],
    budget: int,
    mandatory_ids: set[str] | None = None,
) -> tuple[list[PackItem], list[PackItem]]:
    """
    Pack items using knapsack optimization.

    Maximizes total score while staying within token budget.
    Mandatory items are always included first.

    Args:
        items: List of items to pack
        budget: Token budget
        mandatory_ids: IDs of items that must be included

    Returns:
        Tuple of (selected items, overflow items)
    """
    mandatory_ids = mandatory_ids or set()

    # Separate mandatory and optional items
    mandatory = [item for item in items if item.id in mandatory_ids or item.mandatory]
    optional = [item for item in items if item.id not in mandatory_ids and not item.mandatory]

    # Start with mandatory items
    selected = list(mandatory)
    remaining_budget = budget - sum(item.tokens for item in mandatory)

    if remaining_budget < 0:
        # Mandatory items exceed budget - include what we can
        selected = []
        remaining_budget = budget
        for item in mandatory:
            if item.tokens <= remaining_budget:
                selected.append(item)
                remaining_budget -= item.tokens
        overflow = [item for item in mandatory if item not in selected]
        return selected, overflow + optional

    # Sort optional items by score/token ratio (greedy approximation)
    optional_sorted = sorted(
        optional,
        key=lambda x: x.score / max(x.tokens, 1),
        reverse=True
    )

    overflow = []
    for item in optional_sorted:
        if item.tokens <= remaining_budget:
            selected.append(item)
            remaining_budget -= item.tokens
        else:
            overflow.append(item)

    return selected, overflow


def format_pack(items: list[PackItem], include_citations: bool = True) -> str:
    """Format selected items into markdown pack."""
    lines = []

    for i, item in enumerate(items, 1):
        if include_citations:
            lines.append(f"<!-- [{i}] {item.id} -->")
        lines.append(item.content)
        lines.append("")

    return "\n".join(lines)


# ==============================================================================
# Knapsack Optimization Tests
# ==============================================================================

class TestKnapsackOptimization:
    """Tests for knapsack optimization algorithm."""

    def test_basic_packing(self):
        """Test basic item packing within budget."""
        items = [
            PackItem(id="a", content="Content A", tokens=100, score=0.9),
            PackItem(id="b", content="Content B", tokens=200, score=0.8),
            PackItem(id="c", content="Content C", tokens=150, score=0.7),
        ]
        budget = 300

        selected, overflow = knapsack_pack(items, budget)

        # Should select items that fit
        total_tokens = sum(item.tokens for item in selected)
        assert total_tokens <= budget

    def test_maximize_score(self):
        """Test that packing maximizes total score."""
        items = [
            PackItem(id="high_score", content="High", tokens=100, score=0.95),
            PackItem(id="low_score", content="Low", tokens=100, score=0.5),
        ]
        budget = 100

        selected, overflow = knapsack_pack(items, budget)

        # Should select high score item
        assert len(selected) == 1
        assert selected[0].id == "high_score"

    def test_score_per_token_ratio(self):
        """Test that score/token ratio is considered."""
        items = [
            PackItem(id="efficient", content="E", tokens=50, score=0.8),   # 0.016 per token
            PackItem(id="bulky", content="B", tokens=200, score=0.85),     # 0.00425 per token
        ]
        budget = 100

        selected, overflow = knapsack_pack(items, budget)

        # Should prefer efficient item
        assert any(item.id == "efficient" for item in selected)

    def test_respects_budget(self):
        """Test that budget is never exceeded."""
        items = [
            PackItem(id=f"item_{i}", content=f"Content {i}", tokens=100, score=0.5 + i * 0.1)
            for i in range(10)
        ]
        budget = 350

        selected, overflow = knapsack_pack(items, budget)

        total_tokens = sum(item.tokens for item in selected)
        assert total_tokens <= budget

    def test_all_items_fit(self):
        """Test when all items fit within budget."""
        items = [
            PackItem(id="a", content="A", tokens=100, score=0.9),
            PackItem(id="b", content="B", tokens=100, score=0.8),
        ]
        budget = 500

        selected, overflow = knapsack_pack(items, budget)

        assert len(selected) == 2
        assert len(overflow) == 0

    def test_no_items_fit(self):
        """Test when no items fit within budget."""
        items = [
            PackItem(id="a", content="A", tokens=500, score=0.9),
            PackItem(id="b", content="B", tokens=600, score=0.8),
        ]
        budget = 100

        selected, overflow = knapsack_pack(items, budget)

        assert len(selected) == 0
        assert len(overflow) == 2


# ==============================================================================
# Mandatory Items Tests
# ==============================================================================

class TestMandatoryItems:
    """Tests for mandatory item handling."""

    def test_mandatory_always_included(self):
        """Test that mandatory items are always included."""
        items = [
            PackItem(id="mandatory", content="Must include", tokens=100, score=0.5, mandatory=True),
            PackItem(id="optional", content="Optional", tokens=100, score=0.9, mandatory=False),
        ]
        budget = 150

        selected, overflow = knapsack_pack(items, budget)

        # Mandatory must be included
        assert any(item.id == "mandatory" for item in selected)

    def test_mandatory_by_id(self):
        """Test mandatory items specified by ID."""
        items = [
            PackItem(id="item_a", content="A", tokens=100, score=0.5),
            PackItem(id="item_b", content="B", tokens=100, score=0.9),
        ]
        mandatory_ids = {"item_a"}
        budget = 150

        selected, overflow = knapsack_pack(items, budget, mandatory_ids)

        assert any(item.id == "item_a" for item in selected)

    def test_mandatory_exceeds_budget(self):
        """Test when mandatory items alone exceed budget."""
        items = [
            PackItem(id="m1", content="M1", tokens=200, score=0.5, mandatory=True),
            PackItem(id="m2", content="M2", tokens=200, score=0.5, mandatory=True),
            PackItem(id="opt", content="Opt", tokens=100, score=0.9),
        ]
        budget = 300

        selected, overflow = knapsack_pack(items, budget)

        # Should include what mandatory items fit
        total = sum(item.tokens for item in selected)
        assert total <= budget

    def test_mandatory_priority_over_score(self):
        """Test that mandatory takes priority over high scores."""
        items = [
            PackItem(id="mandatory_low", content="M", tokens=100, score=0.3, mandatory=True),
            PackItem(id="optional_high", content="O", tokens=100, score=0.99),
        ]
        budget = 100

        selected, overflow = knapsack_pack(items, budget)

        # Mandatory should be selected despite lower score
        assert selected[0].id == "mandatory_low"


# ==============================================================================
# Budget Constraint Tests
# ==============================================================================

class TestBudgetConstraints:
    """Tests for budget constraint handling."""

    def test_exact_budget_fit(self):
        """Test items that exactly fit budget."""
        items = [
            PackItem(id="a", content="A", tokens=250, score=0.9),
            PackItem(id="b", content="B", tokens=250, score=0.8),
        ]
        budget = 500

        selected, overflow = knapsack_pack(items, budget)

        total = sum(item.tokens for item in selected)
        assert total == budget

    def test_zero_budget(self):
        """Test with zero budget."""
        items = [
            PackItem(id="a", content="A", tokens=100, score=0.9),
        ]
        budget = 0

        selected, overflow = knapsack_pack(items, budget)

        assert len(selected) == 0

    def test_large_budget(self):
        """Test with budget larger than all items."""
        items = [
            PackItem(id=f"item_{i}", content=f"Content {i}", tokens=100, score=0.5)
            for i in range(10)
        ]
        budget = 10000

        selected, overflow = knapsack_pack(items, budget)

        assert len(selected) == 10
        assert len(overflow) == 0

    @pytest.mark.parametrize("budget", [1000, 4000, 8000, 16000, 32000])
    def test_various_budgets(self, budget):
        """Test packing with various budget sizes."""
        items = [
            PackItem(id=f"item_{i}", content=f"Content {i}", tokens=500, score=0.5 + i * 0.01)
            for i in range(20)
        ]

        selected, overflow = knapsack_pack(items, budget)

        total = sum(item.tokens for item in selected)
        assert total <= budget


# ==============================================================================
# Overflow Handling Tests
# ==============================================================================

class TestOverflowHandling:
    """Tests for overflow item handling."""

    def test_overflow_contains_rejected_items(self):
        """Test that overflow contains items that didn't fit."""
        items = [
            PackItem(id="fits", content="Fits", tokens=100, score=0.9),
            PackItem(id="overflow", content="Overflow", tokens=500, score=0.8),
        ]
        budget = 200

        selected, overflow = knapsack_pack(items, budget)

        assert any(item.id == "overflow" for item in overflow)

    def test_overflow_preserves_items(self):
        """Test that all items are in either selected or overflow."""
        items = [
            PackItem(id=f"item_{i}", content=f"Content {i}", tokens=100 + i * 50, score=0.5)
            for i in range(10)
        ]
        budget = 500

        selected, overflow = knapsack_pack(items, budget)

        all_ids = {item.id for item in selected} | {item.id for item in overflow}
        original_ids = {item.id for item in items}

        assert all_ids == original_ids

    def test_empty_overflow_when_all_fit(self):
        """Test overflow is empty when all items fit."""
        items = [
            PackItem(id="a", content="A", tokens=100, score=0.9),
            PackItem(id="b", content="B", tokens=100, score=0.8),
        ]
        budget = 500

        selected, overflow = knapsack_pack(items, budget)

        assert len(overflow) == 0


# ==============================================================================
# Pack Formatting Tests
# ==============================================================================

class TestPackFormatting:
    """Tests for pack output formatting."""

    def test_format_with_citations(self):
        """Test pack formatting with citations."""
        items = [
            PackItem(id="chunk_123", content="def hello():\n    pass", tokens=50, score=0.9),
        ]

        pack = format_pack(items, include_citations=True)

        assert "<!-- [1] chunk_123 -->" in pack
        assert "def hello():" in pack

    def test_format_without_citations(self):
        """Test pack formatting without citations."""
        items = [
            PackItem(id="chunk_123", content="def hello():\n    pass", tokens=50, score=0.9),
        ]

        pack = format_pack(items, include_citations=False)

        assert "<!--" not in pack
        assert "def hello():" in pack

    def test_format_multiple_items(self):
        """Test formatting multiple items."""
        items = [
            PackItem(id="a", content="Content A", tokens=50, score=0.9),
            PackItem(id="b", content="Content B", tokens=50, score=0.8),
            PackItem(id="c", content="Content C", tokens=50, score=0.7),
        ]

        pack = format_pack(items, include_citations=True)

        assert "<!-- [1] a -->" in pack
        assert "<!-- [2] b -->" in pack
        assert "<!-- [3] c -->" in pack

    def test_format_preserves_content(self):
        """Test that formatting preserves item content."""
        code = """def complex_function():
    x = 1
    y = 2
    return x + y"""

        items = [PackItem(id="test", content=code, tokens=50, score=0.9)]

        pack = format_pack(items)

        assert code in pack

    def test_empty_items(self):
        """Test formatting with no items."""
        pack = format_pack([])
        assert pack == ""


# ==============================================================================
# Edge Cases Tests
# ==============================================================================

class TestPackCompilerEdgeCases:
    """Tests for edge cases in pack compilation."""

    def test_empty_items_list(self):
        """Test packing with empty items list."""
        selected, overflow = knapsack_pack([], budget=1000)

        assert len(selected) == 0
        assert len(overflow) == 0

    def test_single_item_fits(self):
        """Test packing single item that fits."""
        items = [PackItem(id="only", content="Only item", tokens=100, score=0.9)]

        selected, overflow = knapsack_pack(items, budget=200)

        assert len(selected) == 1
        assert len(overflow) == 0

    def test_single_item_too_large(self):
        """Test packing single item too large for budget."""
        items = [PackItem(id="large", content="Large item", tokens=500, score=0.9)]

        selected, overflow = knapsack_pack(items, budget=100)

        assert len(selected) == 0
        assert len(overflow) == 1

    def test_zero_token_items(self):
        """Test handling of zero-token items."""
        items = [
            PackItem(id="zero", content="", tokens=0, score=0.9),
            PackItem(id="normal", content="Normal", tokens=100, score=0.8),
        ]

        selected, overflow = knapsack_pack(items, budget=50)

        # Zero-token items should always fit
        assert any(item.id == "zero" for item in selected)

    def test_equal_score_items(self):
        """Test packing items with equal scores."""
        items = [
            PackItem(id=f"item_{i}", content=f"Content {i}", tokens=100, score=0.5)
            for i in range(5)
        ]

        selected, overflow = knapsack_pack(items, budget=300)

        # Should select 3 items (300 tokens)
        assert len(selected) == 3

    def test_very_large_tokens(self):
        """Test handling of very large token counts."""
        items = [
            PackItem(id="huge", content="Huge", tokens=1_000_000, score=0.9),
            PackItem(id="tiny", content="Tiny", tokens=10, score=0.1),
        ]

        selected, overflow = knapsack_pack(items, budget=100)

        assert any(item.id == "tiny" for item in selected)
        assert any(item.id == "huge" for item in overflow)


# ==============================================================================
# Performance Tests
# ==============================================================================

class TestPackCompilerPerformance:
    """Tests for pack compiler performance."""

    def test_many_items(self):
        """Test packing with many items."""
        items = [
            PackItem(id=f"item_{i}", content=f"Content {i}", tokens=50 + i, score=0.5 + i * 0.001)
            for i in range(100)
        ]

        selected, overflow = knapsack_pack(items, budget=2000)

        total = sum(item.tokens for item in selected)
        assert total <= 2000

    def test_varied_sizes(self):
        """Test packing with varied item sizes."""
        import random
        random.seed(42)

        items = [
            PackItem(
                id=f"item_{i}",
                content=f"Content {i}",
                tokens=random.randint(50, 500),
                score=random.random()
            )
            for i in range(50)
        ]

        selected, overflow = knapsack_pack(items, budget=3000)

        # Should utilize budget efficiently
        total = sum(item.tokens for item in selected)
        assert total <= 3000
        # Should use at least 80% of budget (greedy should be efficient)
        assert total >= 2400 or len(overflow) == 0


# ==============================================================================
# Integration Tests
# ==============================================================================

class TestPackCompilerIntegration:
    """Integration tests for pack compilation."""

    def test_full_pack_pipeline(self):
        """Test complete pack compilation pipeline."""
        # Simulate retrieval results
        chunks = [
            PackItem(
                id="chunk_auth_handler",
                content="async function handleAuth(token) { ... }",
                tokens=150,
                score=0.95,
            ),
            PackItem(
                id="chunk_validator",
                content="function validateToken(token) { ... }",
                tokens=100,
                score=0.85,
            ),
            PackItem(
                id="chunk_types",
                content="interface AuthToken { ... }",
                tokens=80,
                score=0.75,
                metadata={"is_contract": True},
            ),
            PackItem(
                id="chunk_utils",
                content="export const helpers = { ... }",
                tokens=200,
                score=0.4,
            ),
        ]

        budget = 400

        # Pack
        selected, overflow = knapsack_pack(chunks, budget)

        # Format
        pack = format_pack(selected, include_citations=True)

        # Verify
        total_tokens = sum(item.tokens for item in selected)
        assert total_tokens <= budget
        assert len(pack) > 0

    def test_pack_with_pinned_items(self):
        """Test pack with pinned (mandatory) items."""
        items = [
            PackItem(id="pinned_1", content="Pinned content", tokens=200, score=0.3, mandatory=True),
            PackItem(id="high_score", content="High score", tokens=150, score=0.99),
            PackItem(id="low_score", content="Low score", tokens=100, score=0.2),
        ]

        selected, overflow = knapsack_pack(items, budget=300)

        # Pinned must be included
        selected_ids = {item.id for item in selected}
        assert "pinned_1" in selected_ids
