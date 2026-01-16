"""
Unit tests for Impact Miss Rate (IMR) tracking.

Tests the REAL implementation from icd.metrics.imr.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add icd/src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "icd" / "src"))

from icd.metrics.imr import IMRTracker, IMRResult, HeuristicImpactDetector


# ==============================================================================
# IMR Computation Tests
# ==============================================================================

class TestIMRComputation:
    """Tests for IMR computation correctness."""

    @pytest.fixture
    def tracker(self) -> IMRTracker:
        """Create an IMR tracker."""
        return IMRTracker()

    def test_perfect_retrieval(self, tracker: IMRTracker):
        """Test IMR = 0 when all impactful chunks are retrieved."""
        retrieved_ids = ["a", "b", "c"]
        impactful_ids = ["a", "b", "c"]
        missed_ids = []

        result = tracker.compute_imr(retrieved_ids, impactful_ids, missed_ids)

        assert result.imr == 0.0
        assert result.recall == 1.0
        assert result.missed_impactful == 0

    def test_complete_miss(self, tracker: IMRTracker):
        """Test IMR = 1 when no impactful chunks are retrieved."""
        retrieved_ids = ["x", "y", "z"]
        impactful_ids = []
        missed_ids = ["a", "b", "c"]

        result = tracker.compute_imr(retrieved_ids, impactful_ids, missed_ids)

        assert result.imr == 1.0
        assert result.recall == 0.0
        assert result.missed_impactful == 3

    def test_partial_miss(self, tracker: IMRTracker):
        """Test IMR with partial misses."""
        retrieved_ids = ["a", "b", "x"]  # Retrieved a, b
        impactful_ids = ["a", "b"]  # a, b were impactful
        missed_ids = ["c", "d"]  # c, d should have been retrieved

        result = tracker.compute_imr(retrieved_ids, impactful_ids, missed_ids)

        # Total impactful = {a, b, c, d} = 4
        # Retrieved impactful = {a, b} = 2
        # Missed = 4 - 2 = 2
        # IMR = 2/4 = 0.5
        assert result.total_impactful == 4
        assert result.retrieved_impactful == 2
        assert result.missed_impactful == 2
        assert result.imr == 0.5

    def test_no_double_counting_with_overlap(self, tracker: IMRTracker):
        """Test that overlapping impactful and missed sets don't double-count."""
        retrieved_ids = ["a", "b"]
        # Overlap: "c" appears in both impactful (from feedback) and missed
        impactful_ids = ["a", "c"]
        missed_ids = ["c", "d"]

        result = tracker.compute_imr(retrieved_ids, impactful_ids, missed_ids)

        # All impactful = {a, c, d} = 3 (NOT 4 due to double-counting)
        # Retrieved impactful = {a} = 1
        # Actually missed = {c, d} = 2
        # IMR = 2/3
        assert result.total_impactful == 3
        assert result.retrieved_impactful == 1
        assert result.missed_impactful == 2
        assert abs(result.imr - 2/3) < 0.01

    def test_empty_retrieval(self, tracker: IMRTracker):
        """Test with empty retrieval."""
        result = tracker.compute_imr([], [], None)

        assert result.imr == 0.0
        assert result.total_impactful == 0

    def test_precision_calculation(self, tracker: IMRTracker):
        """Test precision is calculated correctly."""
        retrieved_ids = ["a", "b", "x", "y"]  # 4 retrieved
        impactful_ids = ["a", "b"]  # 2 impactful
        missed_ids = []

        result = tracker.compute_imr(retrieved_ids, impactful_ids, missed_ids)

        # Precision = 2/4 = 0.5
        assert result.precision == 0.5

    def test_f1_score(self, tracker: IMRTracker):
        """Test F1 score calculation."""
        retrieved_ids = ["a", "b", "c"]
        impactful_ids = ["a", "b"]
        missed_ids = ["d"]

        result = tracker.compute_imr(retrieved_ids, impactful_ids, missed_ids)

        # Precision = 2/3, Recall = 2/3
        # F1 = 2 * (2/3) * (2/3) / ((2/3) + (2/3)) = 2/3
        assert abs(result.f1 - 2/3) < 0.01


# ==============================================================================
# Session IMR Tests
# ==============================================================================

class TestSessionIMR:
    """Tests for session-level IMR aggregation."""

    @pytest.fixture
    def tracker(self) -> IMRTracker:
        return IMRTracker()

    def test_empty_session(self, tracker: IMRTracker):
        """Test session IMR with no records."""
        result = tracker.compute_session_imr()

        assert result.imr == 0.0
        assert result.precision == 1.0
        assert result.recall == 1.0

    def test_multiple_records(self, tracker: IMRTracker):
        """Test aggregation across multiple records."""
        # Record 1: Perfect
        tracker.record_feedback(
            query="query1",
            retrieved_ids=["a", "b"],
            impactful_ids=["a", "b"],
            missed_ids=[],
        )
        # Record 2: 50% miss
        tracker.record_feedback(
            query="query2",
            retrieved_ids=["c"],
            impactful_ids=["c"],
            missed_ids=["d"],
        )

        result = tracker.compute_session_imr()

        # Total impactful across records: {a, b} + {c, d} = 4
        # Total retrieved impactful: {a, b} + {c} = 3
        # IMR = 1/4 = 0.25
        assert result.total_impactful == 4
        assert result.retrieved_impactful == 3
        assert result.imr == 0.25

    def test_record_trimming(self, tracker: IMRTracker):
        """Test that old records are trimmed."""
        tracker._max_records = 5

        # Add more records than max
        for i in range(10):
            tracker.record_feedback(
                query=f"query{i}",
                retrieved_ids=[f"chunk_{i}"],
                impactful_ids=[f"chunk_{i}"],
            )

        assert len(tracker._records) == 5


# ==============================================================================
# Trend Analysis Tests
# ==============================================================================

class TestTrendAnalysis:
    """Tests for IMR trend analysis."""

    @pytest.fixture
    def tracker(self) -> IMRTracker:
        return IMRTracker()

    def test_trend_order(self, tracker: IMRTracker):
        """Test that trend returns most recent first."""
        tracker.record_feedback("q1", ["a"], ["a"], [])  # IMR = 0
        tracker.record_feedback("q2", ["b"], [], ["c"])  # IMR = 1
        tracker.record_feedback("q3", ["d"], ["d"], ["e"])  # IMR = 0.5

        trend = tracker.compute_trend(window=3)

        # Most recent first: 0.5, 1.0, 0.0
        assert len(trend) == 3
        assert trend[0] == 0.5
        assert trend[1] == 1.0
        assert trend[2] == 0.0

    def test_trend_window(self, tracker: IMRTracker):
        """Test trend respects window size."""
        for i in range(10):
            tracker.record_feedback(f"q{i}", [f"c{i}"], [f"c{i}"], [])

        trend = tracker.compute_trend(window=3)

        assert len(trend) == 3


# ==============================================================================
# Heuristic Impact Detector Tests
# ==============================================================================

class TestHeuristicImpactDetector:
    """Tests for heuristic impact detection."""

    @pytest.fixture
    def detector(self) -> HeuristicImpactDetector:
        return HeuristicImpactDetector()

    def test_high_score_detection(self, detector: HeuristicImpactDetector):
        """Test that high-scoring chunks are detected as impactful."""
        # Create mock chunks with chunk_id attribute
        class MockChunk:
            def __init__(self, chunk_id: str, is_contract: bool = False, is_pinned: bool = False):
                self.chunk_id = chunk_id
                self.is_contract = is_contract
                self.is_pinned = is_pinned

        chunks = [
            MockChunk("high"),
            MockChunk("low"),
        ]
        scores = [0.95, 0.3]

        impactful = detector.detect_impactful(chunks, scores, threshold=0.7)

        assert "high" in impactful
        assert "low" not in impactful

    def test_contract_bonus(self, detector: HeuristicImpactDetector):
        """Test that contracts get bonus towards impactful threshold."""
        class MockChunk:
            def __init__(self, chunk_id: str, is_contract: bool = False, is_pinned: bool = False):
                self.chunk_id = chunk_id
                self.is_contract = is_contract
                self.is_pinned = is_pinned

        chunks = [
            MockChunk("contract", is_contract=True),
            MockChunk("normal"),
        ]
        # Both below threshold, but contract should pass with bonus
        scores = [0.5, 0.5]

        impactful = detector.detect_impactful(chunks, scores, threshold=0.7)

        assert "contract" in impactful  # 0.5 + 0.3 bonus >= 0.7

    def test_pinned_bonus(self, detector: HeuristicImpactDetector):
        """Test that pinned items get bonus towards impactful threshold."""
        class MockChunk:
            def __init__(self, chunk_id: str, is_contract: bool = False, is_pinned: bool = False):
                self.chunk_id = chunk_id
                self.is_contract = is_contract
                self.is_pinned = is_pinned

        chunks = [
            MockChunk("pinned", is_pinned=True),
            MockChunk("normal"),
        ]
        scores = [0.4, 0.4]

        impactful = detector.detect_impactful(chunks, scores, threshold=0.7)

        assert "pinned" in impactful  # 0.4 + 0.4 bonus >= 0.7


# ==============================================================================
# Statistics Tests
# ==============================================================================

class TestStatistics:
    """Tests for IMR statistics reporting."""

    @pytest.fixture
    def tracker(self) -> IMRTracker:
        return IMRTracker()

    def test_get_statistics_empty(self, tracker: IMRTracker):
        """Test statistics with no records."""
        stats = tracker.get_statistics()

        assert stats["queries"] == 0

    def test_get_statistics_with_records(self, tracker: IMRTracker):
        """Test statistics with multiple records."""
        tracker.record_feedback("q1", ["a"], ["a"], [], source="user")
        tracker.record_feedback("q2", ["b"], ["b"], ["c"], source="heuristic")

        stats = tracker.get_statistics()

        assert stats["queries"] == 2
        assert "session_imr" in stats
        assert "mean_imr" in stats
        assert "feedback_sources" in stats
        assert stats["feedback_sources"]["user"] == 1
        assert stats["feedback_sources"]["heuristic"] == 1

    def test_reset(self, tracker: IMRTracker):
        """Test resetting tracker."""
        tracker.record_feedback("q1", ["a"], ["a"], [])
        tracker.reset()

        assert len(tracker._records) == 0
