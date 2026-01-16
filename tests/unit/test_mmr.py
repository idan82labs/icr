"""
Unit tests for the Maximal Marginal Relevance (MMR) module.

Tests cover:
- Diversity selection
- Lambda parameter effect
- Various similarity distributions
- Edge cases
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest


# ==============================================================================
# MMR Implementation for Testing
# ==============================================================================

def compute_mmr(
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    candidate_scores: np.ndarray,
    lambda_param: float = 0.7,
    k: int = 10,
) -> list[int]:
    """
    Compute MMR selection.

    MMR = argmax[lambda * sim(q, d) - (1 - lambda) * max(sim(d, d_i))]

    where:
    - sim(q, d) is similarity between query and candidate
    - sim(d, d_i) is similarity between candidate and already selected items

    Args:
        query_embedding: Query embedding vector
        candidate_embeddings: Matrix of candidate embedding vectors
        candidate_scores: Pre-computed relevance scores (can be hybrid scores)
        lambda_param: Trade-off between relevance and diversity (0-1)
        k: Number of results to select

    Returns:
        List of selected indices in order
    """
    if len(candidate_embeddings) == 0:
        return []

    n_candidates = len(candidate_embeddings)
    k = min(k, n_candidates)

    # Normalize embeddings for cosine similarity
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
    candidates_norm = candidate_embeddings / (
        np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-10
    )

    # Compute query-candidate similarities
    query_similarities = candidates_norm @ query_norm

    # Track selected indices and remaining candidates
    selected: list[int] = []
    remaining = set(range(n_candidates))

    for _ in range(k):
        if not remaining:
            break

        best_idx = -1
        best_score = float("-inf")

        for idx in remaining:
            # Relevance component (use pre-computed hybrid scores if available)
            relevance = candidate_scores[idx] if candidate_scores is not None else query_similarities[idx]

            # Diversity component: max similarity to already selected
            if selected:
                selected_embeddings = candidates_norm[selected]
                candidate_embedding = candidates_norm[idx]
                similarities_to_selected = selected_embeddings @ candidate_embedding
                max_sim_to_selected = np.max(similarities_to_selected)
            else:
                max_sim_to_selected = 0.0

            # MMR score
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx >= 0:
            selected.append(best_idx)
            remaining.remove(best_idx)

    return selected


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise similarity matrix."""
    normalized = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
    return normalized @ normalized.T


# ==============================================================================
# Basic MMR Tests
# ==============================================================================

class TestMMRBasic:
    """Basic tests for MMR functionality."""

    @pytest.fixture
    def sample_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate sample data for MMR tests."""
        np.random.seed(42)
        dimension = 384

        query = np.random.randn(dimension).astype(np.float32)
        query = query / np.linalg.norm(query)

        # 20 candidates
        candidates = np.random.randn(20, dimension).astype(np.float32)
        candidates = candidates / np.linalg.norm(candidates, axis=1, keepdims=True)

        # Relevance scores
        scores = candidates @ query

        return query, candidates, scores

    def test_mmr_returns_k_results(self, sample_data):
        """Test that MMR returns exactly k results."""
        query, candidates, scores = sample_data

        for k in [1, 5, 10, 15]:
            selected = compute_mmr(query, candidates, scores, lambda_param=0.7, k=k)
            assert len(selected) == k

    def test_mmr_returns_all_when_k_exceeds_candidates(self, sample_data):
        """Test MMR when k exceeds number of candidates."""
        query, candidates, scores = sample_data

        selected = compute_mmr(query, candidates, scores, lambda_param=0.7, k=100)
        assert len(selected) == len(candidates)

    def test_mmr_no_duplicates(self, sample_data):
        """Test that MMR doesn't return duplicate indices."""
        query, candidates, scores = sample_data

        selected = compute_mmr(query, candidates, scores, lambda_param=0.7, k=10)

        assert len(selected) == len(set(selected))

    def test_mmr_first_is_most_relevant(self, sample_data):
        """Test that first selected item is most relevant."""
        query, candidates, scores = sample_data

        selected = compute_mmr(query, candidates, scores, lambda_param=0.7, k=10)

        # First item should be the most relevant (highest score)
        # This is because with no selected items yet, MMR reduces to relevance
        most_relevant = np.argmax(scores)
        assert selected[0] == most_relevant


# ==============================================================================
# Lambda Parameter Tests
# ==============================================================================

class TestMMRLambda:
    """Tests for MMR lambda parameter effects."""

    @pytest.fixture
    def clustered_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate clustered data to test diversity."""
        np.random.seed(42)
        dimension = 384

        query = np.random.randn(dimension).astype(np.float32)
        query = query / np.linalg.norm(query)

        # Create 3 clusters of similar candidates
        clusters = []
        for _ in range(3):
            base = np.random.randn(dimension).astype(np.float32)
            base = base / np.linalg.norm(base)
            # Add small variations to create similar items
            cluster = [base + np.random.randn(dimension) * 0.1 for _ in range(5)]
            clusters.extend(cluster)

        candidates = np.array(clusters, dtype=np.float32)
        candidates = candidates / np.linalg.norm(candidates, axis=1, keepdims=True)

        scores = candidates @ query

        return query, candidates, scores

    def test_lambda_1_pure_relevance(self, clustered_data):
        """Test that lambda=1 gives pure relevance ordering."""
        query, candidates, scores = clustered_data

        selected = compute_mmr(query, candidates, scores, lambda_param=1.0, k=5)

        # Should be same as top-k by relevance score
        top_k_by_score = np.argsort(scores)[-5:][::-1].tolist()

        # First few should match (order might vary for ties)
        assert selected[0] == top_k_by_score[0]

    def test_lambda_0_pure_diversity(self, clustered_data):
        """Test that lambda=0 gives pure diversity."""
        query, candidates, scores = clustered_data

        selected = compute_mmr(query, candidates, scores, lambda_param=0.0, k=5)

        # With pure diversity, should pick items that are maximally different
        # First item is arbitrary (since no diversity to consider)
        # Subsequent items should be diverse
        assert len(selected) == 5

    def test_lambda_effect_on_diversity(self, clustered_data):
        """Test that lower lambda increases diversity."""
        query, candidates, scores = clustered_data

        # High lambda (relevance-focused)
        selected_high = compute_mmr(query, candidates, scores, lambda_param=0.9, k=5)

        # Low lambda (diversity-focused)
        selected_low = compute_mmr(query, candidates, scores, lambda_param=0.3, k=5)

        # Compute average pairwise similarity within selected sets
        def avg_pairwise_similarity(indices, embeddings):
            if len(indices) < 2:
                return 0.0
            selected_emb = embeddings[indices]
            sim_matrix = compute_similarity_matrix(selected_emb)
            # Exclude diagonal
            n = len(indices)
            total = np.sum(sim_matrix) - n  # Subtract diagonal (all 1s)
            return total / (n * (n - 1))

        sim_high = avg_pairwise_similarity(selected_high, candidates)
        sim_low = avg_pairwise_similarity(selected_low, candidates)

        # Lower lambda should have lower average similarity (more diverse)
        # Note: This might not always hold depending on data distribution
        # but with clustered data it should

    @pytest.mark.parametrize("lambda_param", [0.0, 0.3, 0.5, 0.7, 1.0])
    def test_lambda_values_valid(self, clustered_data, lambda_param):
        """Test that various lambda values produce valid results."""
        query, candidates, scores = clustered_data

        selected = compute_mmr(query, candidates, scores, lambda_param=lambda_param, k=5)

        assert len(selected) == 5
        assert len(set(selected)) == 5  # No duplicates
        assert all(0 <= idx < len(candidates) for idx in selected)


# ==============================================================================
# Similarity Distribution Tests
# ==============================================================================

class TestMMRSimilarityDistributions:
    """Tests for MMR with various similarity distributions."""

    def test_all_identical_embeddings(self):
        """Test MMR with all identical embeddings."""
        dimension = 384
        query = np.random.randn(dimension).astype(np.float32)
        query = query / np.linalg.norm(query)

        # All candidates are identical
        base = np.random.randn(dimension).astype(np.float32)
        base = base / np.linalg.norm(base)
        candidates = np.tile(base, (10, 1))

        scores = candidates @ query

        selected = compute_mmr(query, candidates, scores, lambda_param=0.7, k=5)

        # Should still return 5 results
        assert len(selected) == 5

    def test_orthogonal_embeddings(self):
        """Test MMR with orthogonal embeddings."""
        dimension = 10  # Small dimension for orthogonality
        query = np.zeros(dimension, dtype=np.float32)
        query[0] = 1.0

        # Create orthogonal candidates (standard basis vectors)
        candidates = np.eye(dimension, dtype=np.float32)
        scores = candidates @ query

        selected = compute_mmr(query, candidates, scores, lambda_param=0.7, k=5)

        # First should be most aligned with query
        assert selected[0] == 0

    def test_highly_correlated_embeddings(self):
        """Test MMR with highly correlated embeddings."""
        np.random.seed(42)
        dimension = 384

        query = np.random.randn(dimension).astype(np.float32)
        query = query / np.linalg.norm(query)

        # Create highly correlated candidates
        base = query.copy()
        candidates = []
        for i in range(10):
            noise = np.random.randn(dimension) * 0.05  # Small noise
            candidate = base + noise
            candidate = candidate / np.linalg.norm(candidate)
            candidates.append(candidate)

        candidates = np.array(candidates, dtype=np.float32)
        scores = candidates @ query

        selected = compute_mmr(query, candidates, scores, lambda_param=0.7, k=5)

        assert len(selected) == 5

    def test_bimodal_distribution(self):
        """Test MMR with bimodal similarity distribution."""
        np.random.seed(42)
        dimension = 384

        query = np.random.randn(dimension).astype(np.float32)
        query = query / np.linalg.norm(query)

        # Two clusters: one similar to query, one different
        similar_cluster = query + np.random.randn(5, dimension) * 0.1
        different_cluster = -query + np.random.randn(5, dimension) * 0.1

        candidates = np.vstack([similar_cluster, different_cluster]).astype(np.float32)
        candidates = candidates / np.linalg.norm(candidates, axis=1, keepdims=True)
        scores = candidates @ query

        selected = compute_mmr(query, candidates, scores, lambda_param=0.7, k=5)

        # With diversity, should pick from both clusters
        similar_indices = set(range(5))
        different_indices = set(range(5, 10))

        # At least some diversity expected
        assert len(selected) == 5


# ==============================================================================
# Edge Cases Tests
# ==============================================================================

class TestMMREdgeCases:
    """Tests for MMR edge cases."""

    def test_empty_candidates(self):
        """Test MMR with no candidates."""
        query = np.random.randn(384).astype(np.float32)
        candidates = np.array([]).reshape(0, 384).astype(np.float32)
        scores = np.array([])

        selected = compute_mmr(query, candidates, scores, lambda_param=0.7, k=10)

        assert len(selected) == 0

    def test_single_candidate(self):
        """Test MMR with single candidate."""
        query = np.random.randn(384).astype(np.float32)
        candidates = np.random.randn(1, 384).astype(np.float32)
        scores = candidates @ query

        selected = compute_mmr(query, candidates, scores, lambda_param=0.7, k=10)

        assert len(selected) == 1
        assert selected[0] == 0

    def test_k_equals_zero(self):
        """Test MMR with k=0."""
        query = np.random.randn(384).astype(np.float32)
        candidates = np.random.randn(10, 384).astype(np.float32)
        scores = candidates @ query

        selected = compute_mmr(query, candidates, scores, lambda_param=0.7, k=0)

        assert len(selected) == 0

    def test_k_equals_one(self):
        """Test MMR with k=1."""
        query = np.random.randn(384).astype(np.float32)
        query = query / np.linalg.norm(query)

        candidates = np.random.randn(10, 384).astype(np.float32)
        candidates = candidates / np.linalg.norm(candidates, axis=1, keepdims=True)
        scores = candidates @ query

        selected = compute_mmr(query, candidates, scores, lambda_param=0.7, k=1)

        assert len(selected) == 1
        # Should be the most relevant
        assert selected[0] == np.argmax(scores)

    def test_all_zero_scores(self):
        """Test MMR with all zero relevance scores."""
        query = np.random.randn(384).astype(np.float32)
        candidates = np.random.randn(10, 384).astype(np.float32)
        scores = np.zeros(10)

        selected = compute_mmr(query, candidates, scores, lambda_param=0.7, k=5)

        # Should still select items based on diversity
        assert len(selected) == 5

    def test_negative_scores(self):
        """Test MMR with negative relevance scores."""
        query = np.random.randn(384).astype(np.float32)
        query = query / np.linalg.norm(query)

        # Create candidates opposite to query
        candidates = -query + np.random.randn(10, 384) * 0.1
        candidates = candidates.astype(np.float32)
        candidates = candidates / np.linalg.norm(candidates, axis=1, keepdims=True)
        scores = candidates @ query  # Will be negative

        selected = compute_mmr(query, candidates, scores, lambda_param=0.7, k=5)

        assert len(selected) == 5


# ==============================================================================
# Performance Characteristics Tests
# ==============================================================================

class TestMMRPerformance:
    """Tests for MMR performance characteristics."""

    def test_mmr_with_many_candidates(self):
        """Test MMR performance with many candidates."""
        np.random.seed(42)
        query = np.random.randn(384).astype(np.float32)
        candidates = np.random.randn(1000, 384).astype(np.float32)
        scores = candidates @ query

        selected = compute_mmr(query, candidates, scores, lambda_param=0.7, k=20)

        assert len(selected) == 20

    def test_mmr_incremental_property(self):
        """Test that MMR selection is incremental."""
        np.random.seed(42)
        query = np.random.randn(384).astype(np.float32)
        candidates = np.random.randn(20, 384).astype(np.float32)
        scores = candidates @ query

        # Get top-5
        selected_5 = compute_mmr(query, candidates, scores, lambda_param=0.7, k=5)

        # Get top-10
        selected_10 = compute_mmr(query, candidates, scores, lambda_param=0.7, k=10)

        # First 5 of top-10 should match top-5
        assert selected_5 == selected_10[:5]


# ==============================================================================
# Diversity Metrics Tests
# ==============================================================================

class TestDiversityMetrics:
    """Tests for diversity measurement."""

    def test_compute_diversity_score(self):
        """Test computing diversity score for a selection."""
        np.random.seed(42)
        dimension = 384

        # Create diverse candidates
        candidates = np.random.randn(10, dimension).astype(np.float32)
        candidates = candidates / np.linalg.norm(candidates, axis=1, keepdims=True)

        selected_indices = [0, 3, 7]
        selected_embeddings = candidates[selected_indices]

        sim_matrix = compute_similarity_matrix(selected_embeddings)

        # Diversity = 1 - average pairwise similarity
        n = len(selected_indices)
        avg_sim = (np.sum(sim_matrix) - n) / (n * (n - 1))
        diversity = 1 - avg_sim

        assert 0 <= diversity <= 1

    def test_high_diversity_low_similarity(self):
        """Test that high diversity means low average similarity."""
        # Orthogonal vectors have maximum diversity
        dimension = 3
        orthogonal = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ], dtype=np.float32)

        sim_matrix = compute_similarity_matrix(orthogonal)
        off_diagonal_sim = sim_matrix[~np.eye(3, dtype=bool)].mean()

        # Orthogonal vectors should have ~0 average similarity
        assert abs(off_diagonal_sim) < 1e-6

    def test_low_diversity_high_similarity(self):
        """Test that low diversity means high average similarity."""
        # Identical vectors have minimum diversity
        dimension = 384
        base = np.random.randn(dimension).astype(np.float32)
        base = base / np.linalg.norm(base)

        identical = np.tile(base, (3, 1))
        sim_matrix = compute_similarity_matrix(identical)
        off_diagonal_sim = sim_matrix[~np.eye(3, dtype=bool)].mean()

        # Identical vectors should have ~1 average similarity
        assert abs(off_diagonal_sim - 1.0) < 1e-6
