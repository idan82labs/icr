"""
Acceptance test for compaction survival.

From PRD:
- After simulated /compact
- Pass: pinned invariants and ledger decisions persist
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest


# ==============================================================================
# Test Data Structures
# ==============================================================================

@dataclass
class PinnedItem:
    """A pinned item that must survive compaction."""

    id: str
    content: str
    reason: str
    created_at: str


@dataclass
class LedgerEntry:
    """A ledger entry that must survive compaction."""

    id: str
    category: str  # decision, invariant, observation
    content: str
    created_at: str


@dataclass
class ConversationState:
    """State of conversation before/after compaction."""

    messages: list[dict[str, Any]]
    pinned_items: list[PinnedItem]
    ledger_entries: list[LedgerEntry]
    total_tokens: int


# ==============================================================================
# Compaction Simulation
# ==============================================================================

def simulate_compact(
    state: ConversationState,
    target_tokens: int = 10000,
    preserve_pinned: bool = True,
    preserve_ledger: bool = True,
) -> ConversationState:
    """
    Simulate conversation compaction.

    Compaction rules:
    1. Pinned items MUST be preserved
    2. Ledger entries MUST be preserved
    3. Recent messages are prioritized
    4. Summarize older messages
    """
    # Start with empty compacted state
    compacted_messages = []
    preserved_pinned = []
    preserved_ledger = []
    tokens_used = 0

    # Always preserve pinned items first
    if preserve_pinned:
        for item in state.pinned_items:
            preserved_pinned.append(item)
            tokens_used += len(item.content) // 4

    # Always preserve ledger entries
    if preserve_ledger:
        for entry in state.ledger_entries:
            preserved_ledger.append(entry)
            tokens_used += len(entry.content) // 4

    # Add messages from most recent, respecting budget
    remaining_budget = target_tokens - tokens_used

    for message in reversed(state.messages):
        message_tokens = len(message.get("content", "")) // 4
        if message_tokens <= remaining_budget:
            compacted_messages.insert(0, message)
            remaining_budget -= message_tokens
        else:
            # Summarize older messages
            summary = f"[Summarized: {len(state.messages) - len(compacted_messages)} earlier messages]"
            compacted_messages.insert(0, {"role": "system", "content": summary})
            break

    return ConversationState(
        messages=compacted_messages,
        pinned_items=preserved_pinned,
        ledger_entries=preserved_ledger,
        total_tokens=target_tokens - remaining_budget,
    )


def verify_survival(
    before: ConversationState,
    after: ConversationState,
) -> tuple[bool, list[str]]:
    """
    Verify that critical items survived compaction.

    Returns:
        Tuple of (all_survived, missing_items)
    """
    missing = []

    # Check pinned items
    after_pinned_ids = {p.id for p in after.pinned_items}
    for item in before.pinned_items:
        if item.id not in after_pinned_ids:
            missing.append(f"Pinned item: {item.id}")

    # Check ledger entries
    after_ledger_ids = {e.id for e in after.ledger_entries}
    for entry in before.ledger_entries:
        if entry.id not in after_ledger_ids:
            missing.append(f"Ledger entry: {entry.id}")

    return len(missing) == 0, missing


# ==============================================================================
# Compaction Survival Acceptance Test
# ==============================================================================

@pytest.mark.acceptance
class TestCompactionSurvival:
    """
    Acceptance test for compaction survival.

    Pass criteria (from PRD):
    - After /compact, pinned invariants persist
    - After /compact, ledger decisions persist
    """

    @pytest.fixture
    def conversation_with_pinned(self) -> ConversationState:
        """Create conversation state with pinned items."""
        return ConversationState(
            messages=[
                {"role": "user", "content": "How does auth work?"},
                {"role": "assistant", "content": "Auth uses JWT tokens..."},
                {"role": "user", "content": "What about refresh?"},
                {"role": "assistant", "content": "Refresh tokens are stored..."},
                {"role": "user", "content": "Another question"},
                {"role": "assistant", "content": "Another response"},
            ] * 10,  # Many messages to trigger compaction
            pinned_items=[
                PinnedItem(
                    id="pin_1",
                    content="Authentication invariant: All tokens must be validated",
                    reason="Critical security requirement",
                    created_at="2026-01-16T10:00:00Z",
                ),
                PinnedItem(
                    id="pin_2",
                    content="Rate limiting: Max 100 requests per minute",
                    reason="Performance requirement",
                    created_at="2026-01-16T10:05:00Z",
                ),
            ],
            ledger_entries=[
                LedgerEntry(
                    id="ledger_1",
                    category="decision",
                    content="Use JWT for authentication",
                    created_at="2026-01-16T10:00:00Z",
                ),
                LedgerEntry(
                    id="ledger_2",
                    category="invariant",
                    content="Tokens expire after 1 hour",
                    created_at="2026-01-16T10:01:00Z",
                ),
                LedgerEntry(
                    id="ledger_3",
                    category="decision",
                    content="Store refresh tokens in httpOnly cookies",
                    created_at="2026-01-16T10:02:00Z",
                ),
            ],
            total_tokens=50000,
        )

    def test_pinned_items_survive_compaction(self, conversation_with_pinned):
        """
        Test: Pinned items survive compaction

        Expected: All pinned items present after /compact
        """
        before = conversation_with_pinned

        # Compact to smaller size
        after = simulate_compact(before, target_tokens=5000)

        # Verify pinned items survived
        all_survived, missing = verify_survival(before, after)

        assert all_survived, f"Missing items after compaction: {missing}"
        assert len(after.pinned_items) == len(before.pinned_items)

    def test_ledger_entries_survive_compaction(self, conversation_with_pinned):
        """
        Test: Ledger entries survive compaction

        Expected: All ledger entries present after /compact
        """
        before = conversation_with_pinned

        after = simulate_compact(before, target_tokens=5000)

        # Verify ledger entries survived
        before_ledger_ids = {e.id for e in before.ledger_entries}
        after_ledger_ids = {e.id for e in after.ledger_entries}

        assert before_ledger_ids == after_ledger_ids, \
            "All ledger entries must survive compaction"

    def test_decisions_persist(self, conversation_with_pinned):
        """Test that decision entries persist."""
        before = conversation_with_pinned
        after = simulate_compact(before, target_tokens=5000)

        before_decisions = [e for e in before.ledger_entries if e.category == "decision"]
        after_decisions = [e for e in after.ledger_entries if e.category == "decision"]

        assert len(after_decisions) == len(before_decisions)

    def test_invariants_persist(self, conversation_with_pinned):
        """Test that invariant entries persist."""
        before = conversation_with_pinned
        after = simulate_compact(before, target_tokens=5000)

        before_invariants = [e for e in before.ledger_entries if e.category == "invariant"]
        after_invariants = [e for e in after.ledger_entries if e.category == "invariant"]

        assert len(after_invariants) == len(before_invariants)

    def test_aggressive_compaction_preserves_critical(self, conversation_with_pinned):
        """Test that even aggressive compaction preserves critical items."""
        before = conversation_with_pinned

        # Very aggressive compaction
        after = simulate_compact(before, target_tokens=1000)

        all_survived, missing = verify_survival(before, after)

        assert all_survived, \
            f"Critical items must survive even aggressive compaction: {missing}"


# ==============================================================================
# Content Preservation Tests
# ==============================================================================

@pytest.mark.acceptance
class TestContentPreservation:
    """Tests for content preservation during compaction."""

    def test_pinned_content_unchanged(self):
        """Test that pinned item content is unchanged."""
        original_content = "Critical invariant: Never store plaintext passwords"

        state = ConversationState(
            messages=[{"role": "user", "content": "test"}] * 100,
            pinned_items=[
                PinnedItem(
                    id="pin_1",
                    content=original_content,
                    reason="Security",
                    created_at="2026-01-16T10:00:00Z",
                )
            ],
            ledger_entries=[],
            total_tokens=10000,
        )

        after = simulate_compact(state, target_tokens=500)

        assert after.pinned_items[0].content == original_content

    def test_ledger_content_unchanged(self):
        """Test that ledger entry content is unchanged."""
        original_content = "Decision: Use bcrypt for password hashing"

        state = ConversationState(
            messages=[{"role": "user", "content": "test"}] * 100,
            pinned_items=[],
            ledger_entries=[
                LedgerEntry(
                    id="ledger_1",
                    category="decision",
                    content=original_content,
                    created_at="2026-01-16T10:00:00Z",
                )
            ],
            total_tokens=10000,
        )

        after = simulate_compact(state, target_tokens=500)

        assert after.ledger_entries[0].content == original_content


# ==============================================================================
# Compaction Budget Tests
# ==============================================================================

@pytest.mark.acceptance
class TestCompactionBudget:
    """Tests for compaction token budget handling."""

    def test_compaction_respects_target(self):
        """Test that compaction respects target token count."""
        state = ConversationState(
            messages=[{"role": "user", "content": "x" * 1000}] * 100,
            pinned_items=[],
            ledger_entries=[],
            total_tokens=100000,
        )

        target = 5000
        after = simulate_compact(state, target_tokens=target)

        assert after.total_tokens <= target

    def test_pinned_items_dont_exceed_budget_alone(self):
        """Test handling when pinned items alone approach budget."""
        # Large pinned content
        large_pinned = [
            PinnedItem(
                id=f"pin_{i}",
                content="x" * 1000,  # ~250 tokens each
                reason="test",
                created_at="2026-01-16T10:00:00Z",
            )
            for i in range(10)
        ]

        state = ConversationState(
            messages=[{"role": "user", "content": "test"}],
            pinned_items=large_pinned,
            ledger_entries=[],
            total_tokens=10000,
        )

        # Small target that pinned items nearly fill
        after = simulate_compact(state, target_tokens=3000)

        # All pinned items should still survive
        assert len(after.pinned_items) == len(large_pinned)


# ==============================================================================
# Edge Cases Tests
# ==============================================================================

@pytest.mark.acceptance
class TestCompactionEdgeCases:
    """Tests for compaction edge cases."""

    def test_empty_conversation(self):
        """Test compaction of empty conversation."""
        state = ConversationState(
            messages=[],
            pinned_items=[],
            ledger_entries=[],
            total_tokens=0,
        )

        after = simulate_compact(state, target_tokens=1000)

        assert len(after.messages) == 0
        assert len(after.pinned_items) == 0
        assert len(after.ledger_entries) == 0

    def test_only_pinned_items(self):
        """Test compaction with only pinned items."""
        state = ConversationState(
            messages=[],
            pinned_items=[
                PinnedItem(
                    id="pin_1",
                    content="Important",
                    reason="Critical",
                    created_at="2026-01-16T10:00:00Z",
                )
            ],
            ledger_entries=[],
            total_tokens=100,
        )

        after = simulate_compact(state, target_tokens=50)

        assert len(after.pinned_items) == 1

    def test_only_ledger_entries(self):
        """Test compaction with only ledger entries."""
        state = ConversationState(
            messages=[],
            pinned_items=[],
            ledger_entries=[
                LedgerEntry(
                    id="ledger_1",
                    category="decision",
                    content="Decision 1",
                    created_at="2026-01-16T10:00:00Z",
                )
            ],
            total_tokens=100,
        )

        after = simulate_compact(state, target_tokens=50)

        assert len(after.ledger_entries) == 1

    def test_multiple_compactions(self):
        """Test that multiple compactions preserve items."""
        state = ConversationState(
            messages=[{"role": "user", "content": "test"}] * 50,
            pinned_items=[
                PinnedItem(
                    id="pin_1",
                    content="Must survive",
                    reason="Critical",
                    created_at="2026-01-16T10:00:00Z",
                )
            ],
            ledger_entries=[
                LedgerEntry(
                    id="ledger_1",
                    category="invariant",
                    content="Must survive",
                    created_at="2026-01-16T10:00:00Z",
                )
            ],
            total_tokens=5000,
        )

        # Multiple compactions
        state = simulate_compact(state, target_tokens=3000)
        state = simulate_compact(state, target_tokens=2000)
        state = simulate_compact(state, target_tokens=1000)

        # Items should survive all compactions
        assert len(state.pinned_items) == 1
        assert len(state.ledger_entries) == 1
