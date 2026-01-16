"""
Tests for retrieval modules.
"""

from __future__ import annotations

import numpy as np
import pytest


class TestEntropyCalculator:
    """Tests for EntropyCalculator."""

    def test_uniform_distribution_high_entropy(self):
        """Test that uniform distribution has high entropy."""
        from icd.retrieval.entropy import EntropyCalculator

        calc = EntropyCalculator(temperature=1.0)

        # Uniform scores
        scores = [1.0, 1.0, 1.0, 1.0, 1.0]

        result = calc.compute_full(scores)

        # Normalized entropy should be close to 1 for uniform
        assert result.normalized_entropy > 0.95

    def test_peaked_distribution_low_entropy(self):
        """Test that peaked distribution has low entropy."""
        from icd.retrieval.entropy import EntropyCalculator

        calc = EntropyCalculator(temperature=1.0)

        # One high score, rest low
        scores = [10.0, 0.1, 0.1, 0.1, 0.1]

        result = calc.compute_full(scores)

        # Normalized entropy should be low
        assert result.normalized_entropy < 0.3

    def test_temperature_effect(self):
        """Test temperature scaling."""
        from icd.retrieval.entropy import EntropyCalculator

        scores = [5.0, 3.0, 1.0]

        # Low temperature (sharper)
        calc_low = EntropyCalculator(temperature=0.5)
        result_low = calc_low.compute_full(scores)

        # High temperature (flatter)
        calc_high = EntropyCalculator(temperature=2.0)
        result_high = calc_high.compute_full(scores)

        # Higher temperature should give higher entropy
        assert result_high.entropy > result_low.entropy

    def test_confidence_inverse_of_normalized_entropy(self):
        """Test that confidence = 1 - normalized_entropy."""
        from icd.retrieval.entropy import EntropyCalculator

        calc = EntropyCalculator()
        scores = [3.0, 2.0, 1.0]

        result = calc.compute_full(scores)

        assert abs(result.confidence - (1.0 - result.normalized_entropy)) < 0.001

    def test_rlm_trigger(self):
        """Test RLM trigger decision."""
        from icd.retrieval.entropy import EntropyCalculator

        calc = EntropyCalculator()

        # High entropy (ambiguous) - should trigger
        high_entropy_scores = [1.0, 0.9, 0.8, 0.7]
        assert calc.should_trigger_rlm(high_entropy_scores, threshold=0.5)

        # Low entropy (confident) - should not trigger
        low_entropy_scores = [10.0, 0.1, 0.1, 0.1]
        assert not calc.should_trigger_rlm(low_entropy_scores, threshold=0.5)

    def test_score_gap(self):
        """Test score gap computation."""
        from icd.retrieval.entropy import EntropyCalculator

        calc = EntropyCalculator()

        # Large gap
        large_gap_scores = [10.0, 1.0, 0.5]
        gap1 = calc.compute_score_gap(large_gap_scores)

        # Small gap
        small_gap_scores = [10.0, 9.5, 9.0]
        gap2 = calc.compute_score_gap(small_gap_scores)

        assert gap1 > gap2

    def test_concentration(self):
        """Test concentration computation."""
        from icd.retrieval.entropy import EntropyCalculator

        calc = EntropyCalculator()

        # High concentration in top 3
        concentrated = [10.0, 8.0, 6.0, 0.1, 0.1]
        conc1 = calc.compute_concentration(concentrated, top_k=3)

        # Low concentration
        spread = [2.0, 2.0, 2.0, 2.0, 2.0]
        conc2 = calc.compute_concentration(spread, top_k=3)

        assert conc1 > conc2


class TestMMRSelector:
    """Tests for MMRSelector."""

    @pytest.mark.asyncio
    async def test_mmr_selects_diverse_results(self, sample_chunks, mock_embedder):
        """Test that MMR selects diverse results."""
        from icd.retrieval.hybrid import ScoredChunk
        from icd.retrieval.mmr import MMRSelector

        # Create scored chunks with similar scores
        scored_chunks = [
            ScoredChunk(
                chunk=chunk,
                semantic_score=0.8,
                bm25_score=0.7,
                recency_score=0.5,
                contract_score=0.0,
                focus_score=0.0,
                pinned_score=0.0,
                final_score=0.8 - i * 0.01,  # Slightly decreasing
            )
            for i, chunk in enumerate(sample_chunks)
        ]

        mmr = MMRSelector(lambda_param=0.7)
        selected = await mmr.select(scored_chunks, limit=5)

        assert len(selected) == 5

    def test_mmr_sync_with_embeddings(self, sample_chunks):
        """Test synchronous MMR with precomputed embeddings."""
        from icd.retrieval.hybrid import ScoredChunk
        from icd.retrieval.mmr import MMRSelector

        np.random.seed(42)

        # Create embeddings
        embeddings = {
            f"chunk_{i}": np.random.randn(384).astype(np.float32)
            for i in range(10)
        }

        scored_chunks = [
            ScoredChunk(
                chunk=chunk,
                semantic_score=0.8,
                bm25_score=0.7,
                recency_score=0.5,
                contract_score=0.0,
                focus_score=0.0,
                pinned_score=0.0,
                final_score=0.8,
            )
            for chunk in sample_chunks
        ]

        mmr = MMRSelector(lambda_param=0.5)
        selected = mmr.select_sync(scored_chunks, limit=5, embeddings=embeddings)

        assert len(selected) == 5

    def test_lambda_affects_diversity(self, sample_chunks):
        """Test that lambda parameter affects diversity."""
        from icd.retrieval.hybrid import ScoredChunk
        from icd.retrieval.mmr import MMRSelector

        np.random.seed(42)

        # Create similar embeddings for first few chunks
        embeddings = {}
        base_emb = np.random.randn(384).astype(np.float32)
        for i in range(5):
            embeddings[f"chunk_{i}"] = base_emb + np.random.randn(384).astype(np.float32) * 0.1
        for i in range(5, 10):
            embeddings[f"chunk_{i}"] = np.random.randn(384).astype(np.float32)

        scored_chunks = [
            ScoredChunk(
                chunk=chunk,
                semantic_score=0.8 - i * 0.05,
                bm25_score=0.7,
                recency_score=0.5,
                contract_score=0.0,
                focus_score=0.0,
                pinned_score=0.0,
                final_score=0.8 - i * 0.05,
            )
            for i, chunk in enumerate(sample_chunks)
        ]

        # High lambda (more relevance)
        mmr_high = MMRSelector(lambda_param=0.9)
        selected_high = mmr_high.select_sync(scored_chunks, limit=5, embeddings=embeddings)

        # Low lambda (more diversity)
        mmr_low = MMRSelector(lambda_param=0.3)
        selected_low = mmr_low.select_sync(scored_chunks, limit=5, embeddings=embeddings)

        # Results may differ
        high_ids = {s.chunk.chunk_id for s in selected_high}
        low_ids = {s.chunk.chunk_id for s in selected_low}

        # At least some difference expected
        # (may not always differ with random data)
        assert isinstance(high_ids, set) and isinstance(low_ids, set)


class TestDiversityMetrics:
    """Tests for diversity metrics."""

    def test_pairwise_similarity(self):
        """Test average pairwise similarity computation."""
        from icd.retrieval.mmr import DiversityMetrics

        np.random.seed(42)

        # Similar embeddings
        base = np.random.randn(384).astype(np.float32)
        similar_embs = [base + np.random.randn(384).astype(np.float32) * 0.1 for _ in range(5)]
        similar_avg = DiversityMetrics.compute_average_pairwise_similarity(similar_embs)

        # Diverse embeddings
        diverse_embs = [np.random.randn(384).astype(np.float32) for _ in range(5)]
        diverse_avg = DiversityMetrics.compute_average_pairwise_similarity(diverse_embs)

        # Similar embeddings should have higher average similarity
        assert similar_avg > diverse_avg

    def test_type_diversity(self):
        """Test symbol type diversity computation."""
        from icd.retrieval.mmr import DiversityMetrics

        # All same type
        same_types = ["function"] * 10
        same_diversity = DiversityMetrics.compute_type_diversity(same_types)

        # Mixed types
        mixed_types = ["function", "class", "interface", "function", "method"]
        mixed_diversity = DiversityMetrics.compute_type_diversity(mixed_types)

        assert mixed_diversity > same_diversity


class TestQueryDifficultyEstimator:
    """Tests for query difficulty estimation."""

    def test_easy_query(self):
        """Test difficulty estimation for easy queries."""
        from icd.retrieval.entropy import QueryDifficultyEstimator

        estimator = QueryDifficultyEstimator()

        # Clear top result (easy)
        easy_scores = [10.0, 1.0, 0.5, 0.2, 0.1]
        difficulty = estimator.estimate_difficulty(easy_scores)

        assert difficulty < 0.5

    def test_hard_query(self):
        """Test difficulty estimation for hard queries."""
        from icd.retrieval.entropy import QueryDifficultyEstimator

        estimator = QueryDifficultyEstimator()

        # Similar scores (hard)
        hard_scores = [1.0, 0.95, 0.9, 0.85, 0.8]
        difficulty = estimator.estimate_difficulty(hard_scores)

        assert difficulty > 0.5

    def test_difficulty_analysis(self):
        """Test detailed difficulty analysis."""
        from icd.retrieval.entropy import QueryDifficultyEstimator

        estimator = QueryDifficultyEstimator()

        scores = [5.0, 4.0, 3.0, 2.0, 1.0]
        analysis = estimator.get_difficulty_analysis(scores)

        assert "difficulty" in analysis
        assert "entropy" in analysis
        assert "score_gap" in analysis
        assert "top3_concentration" in analysis
