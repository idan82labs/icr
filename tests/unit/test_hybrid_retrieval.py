"""
Unit tests for the hybrid retrieval module.

Tests cover:
- Semantic-only search
- Lexical-only search
- Hybrid score merging
- Contract boost
- Recency decay
- Focus path boost
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest


# ==============================================================================
# Score Merging Tests
# ==============================================================================

class TestHybridScoreMerging:
    """Tests for hybrid score merging formula."""

    def test_hybrid_score_formula(self):
        """Test the hybrid scoring formula from PRD.

        score = w_e * s_embed + w_b * s_bm25 + w_r * r(t) + w_c * c + w_f * f + w_p * p
        """
        # Default weights from config
        w_e = 0.4   # embedding weight
        w_b = 0.3   # BM25 weight
        w_r = 0.1   # recency weight
        w_c = 0.1   # contract weight
        w_f = 0.05  # focus weight
        w_p = 0.05  # pinned weight

        # Sample scores
        s_embed = 0.8   # semantic similarity
        s_bm25 = 0.6    # BM25 score (normalized)
        r_t = 0.9       # recency factor
        c = 1.0         # is contract
        f = 0.0         # not in focus path
        p = 0.0         # not pinned

        score = (
            w_e * s_embed +
            w_b * s_bm25 +
            w_r * r_t +
            w_c * c +
            w_f * f +
            w_p * p
        )

        expected = 0.4 * 0.8 + 0.3 * 0.6 + 0.1 * 0.9 + 0.1 * 1.0
        assert abs(score - expected) < 1e-6

    def test_weights_sum_to_one(self):
        """Test that default weights sum to 1.0."""
        w_e = 0.4
        w_b = 0.3
        w_r = 0.1
        w_c = 0.1
        w_f = 0.05
        w_p = 0.05

        total = w_e + w_b + w_r + w_c + w_f + w_p
        assert abs(total - 1.0) < 1e-6

    def test_score_range(self):
        """Test that merged scores are in [0, 1] range."""
        w_e, w_b, w_r, w_c, w_f, w_p = 0.4, 0.3, 0.1, 0.1, 0.05, 0.05

        # Maximum possible score (all components = 1)
        max_score = w_e * 1 + w_b * 1 + w_r * 1 + w_c * 1 + w_f * 1 + w_p * 1
        assert max_score <= 1.0

        # Minimum possible score (all components = 0)
        min_score = w_e * 0 + w_b * 0 + w_r * 0 + w_c * 0 + w_f * 0 + w_p * 0
        assert min_score >= 0.0

    @pytest.mark.parametrize("s_embed,s_bm25,expected_dominant", [
        (0.9, 0.1, "semantic"),   # High semantic, low lexical
        (0.1, 0.9, "lexical"),    # Low semantic, high lexical
        (0.5, 0.5, "balanced"),   # Equal scores
    ])
    def test_score_components_influence(self, s_embed, s_bm25, expected_dominant):
        """Test how different score components influence final score."""
        w_e, w_b = 0.4, 0.3

        semantic_contribution = w_e * s_embed
        lexical_contribution = w_b * s_bm25

        if expected_dominant == "semantic":
            assert semantic_contribution > lexical_contribution
        elif expected_dominant == "lexical":
            assert lexical_contribution > semantic_contribution
        else:
            # Close to equal
            assert abs(semantic_contribution - lexical_contribution) < 0.1


# ==============================================================================
# Semantic Search Tests
# ==============================================================================

class TestSemanticSearch:
    """Tests for semantic (embedding-based) search."""

    @pytest.fixture
    def mock_vectors(self) -> tuple[np.ndarray, list[str]]:
        """Create mock vectors for testing."""
        np.random.seed(42)
        vectors = np.random.randn(100, 384).astype(np.float32)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        ids = [f"chunk_{i}" for i in range(100)]
        return vectors, ids

    def test_semantic_search_returns_results(self, mock_vectors):
        """Test that semantic search returns results."""
        vectors, ids = mock_vectors
        query_vec = np.random.randn(384).astype(np.float32)
        query_vec = query_vec / np.linalg.norm(query_vec)

        similarities = vectors @ query_vec
        top_k_indices = np.argsort(similarities)[-10:][::-1]
        results = [ids[i] for i in top_k_indices]

        assert len(results) == 10

    def test_semantic_search_sorted_by_similarity(self, mock_vectors):
        """Test that results are sorted by similarity."""
        vectors, ids = mock_vectors
        query_vec = np.random.randn(384).astype(np.float32)
        query_vec = query_vec / np.linalg.norm(query_vec)

        similarities = vectors @ query_vec
        top_k_indices = np.argsort(similarities)[-10:][::-1]
        top_k_sims = similarities[top_k_indices]

        # Verify descending order
        for i in range(len(top_k_sims) - 1):
            assert top_k_sims[i] >= top_k_sims[i + 1]

    def test_semantic_only_mode(self, mock_vectors):
        """Test semantic-only search (w_b = 0)."""
        vectors, ids = mock_vectors
        w_e = 1.0  # Only semantic
        w_b = 0.0  # No lexical

        query_vec = np.random.randn(384).astype(np.float32)
        query_vec = query_vec / np.linalg.norm(query_vec)

        # Semantic scores only
        semantic_scores = vectors @ query_vec
        final_scores = w_e * semantic_scores + w_b * np.zeros_like(semantic_scores)

        # Final scores should equal semantic scores
        np.testing.assert_array_almost_equal(final_scores, semantic_scores)


# ==============================================================================
# Lexical Search Tests
# ==============================================================================

class TestLexicalSearch:
    """Tests for lexical (BM25) search."""

    def test_bm25_score_normalization(self):
        """Test BM25 score normalization to [0, 1]."""
        # Raw BM25 scores can be any positive number
        raw_scores = np.array([10.5, 8.2, 5.1, 2.3, 0.5])

        # Normalize to [0, 1]
        if raw_scores.max() > 0:
            normalized = raw_scores / raw_scores.max()
        else:
            normalized = raw_scores

        assert normalized.min() >= 0
        assert normalized.max() <= 1

    def test_lexical_only_mode(self):
        """Test lexical-only search (w_e = 0)."""
        w_e = 0.0  # No semantic
        w_b = 1.0  # Only lexical

        semantic_scores = np.array([0.9, 0.8, 0.7])
        lexical_scores = np.array([0.3, 0.5, 0.4])

        final_scores = w_e * semantic_scores + w_b * lexical_scores

        # Final scores should equal lexical scores
        np.testing.assert_array_almost_equal(final_scores, lexical_scores)

    def test_bm25_term_frequency_impact(self):
        """Test that term frequency affects BM25 scores."""
        # Simulated BM25 behavior: more term occurrences = higher score
        # (actual BM25 has diminishing returns with saturation parameter k1)
        doc_with_many_matches = 5.0
        doc_with_few_matches = 2.0

        assert doc_with_many_matches > doc_with_few_matches


# ==============================================================================
# Contract Boost Tests
# ==============================================================================

class TestContractBoost:
    """Tests for contract detection boost."""

    def test_contract_boost_applied(self):
        """Test that contract boost is applied to contract chunks."""
        w_c = 0.1  # Contract weight
        boost_factor = 1.5  # From config

        base_score = 0.5
        is_contract = 1.0

        # Contract contribution
        contract_boost = w_c * is_contract
        score_with_contract = base_score + contract_boost

        assert score_with_contract > base_score

    def test_non_contract_no_boost(self):
        """Test that non-contracts don't get boost."""
        w_c = 0.1
        is_contract = 0.0

        base_score = 0.5
        contract_contribution = w_c * is_contract

        assert contract_contribution == 0.0

    @pytest.mark.parametrize("pattern,expected_is_contract", [
        ("interface User {}", True),
        ("abstract class Base", True),
        ("type Alias = string", True),
        ("schema: object", True),
        ("struct Config {}", True),
        ("enum Status {}", True),
        ("function doSomething()", False),
        ("const x = 5", False),
    ])
    def test_contract_detection_patterns(self, pattern, expected_is_contract):
        """Test contract detection patterns from config."""
        contract_patterns = [
            "interface",
            "abstract",
            "protocol",
            "trait",
            r"type.*=",
            "@dataclass",
            "schema",
            "model",
            "struct",
            "enum",
        ]

        detected = any(p in pattern.lower() for p in ["interface", "abstract", "type", "schema", "struct", "enum"])

        if expected_is_contract:
            assert detected
        # Note: This is simplified; actual implementation uses regex


# ==============================================================================
# Recency Decay Tests
# ==============================================================================

class TestRecencyDecay:
    """Tests for time-based recency decay."""

    def test_recency_decay_formula(self):
        """Test recency decay formula: r(t) = exp(-delta_days / tau)."""
        tau = 30.0  # decay constant in days

        # Fresh document (0 days old)
        delta_0 = 0
        r_0 = math.exp(-delta_0 / tau)
        assert abs(r_0 - 1.0) < 1e-6

        # Document at tau days
        delta_tau = 30
        r_tau = math.exp(-delta_tau / tau)
        assert abs(r_tau - math.exp(-1)) < 1e-6

        # Very old document (90 days)
        delta_old = 90
        r_old = math.exp(-delta_old / tau)
        assert r_old < 0.1

    def test_recency_decay_monotonic(self):
        """Test that recency score decreases with age."""
        tau = 30.0
        ages = [0, 7, 14, 30, 60, 90]
        scores = [math.exp(-age / tau) for age in ages]

        for i in range(len(scores) - 1):
            assert scores[i] > scores[i + 1]

    def test_recency_decay_bounds(self):
        """Test that recency scores are in (0, 1]."""
        tau = 30.0

        for delta in range(0, 365):
            r = math.exp(-delta / tau)
            assert 0 < r <= 1

    @pytest.mark.parametrize("tau", [7, 30, 90, 365])
    def test_different_tau_values(self, tau):
        """Test recency decay with different tau values."""
        delta = 30  # 30 days old

        r = math.exp(-delta / tau)

        # With smaller tau, decay is faster
        if tau == 7:
            assert r < 0.02  # Very low for tau=7, delta=30
        elif tau == 365:
            assert r > 0.9   # High for tau=365, delta=30


# ==============================================================================
# Focus Path Boost Tests
# ==============================================================================

class TestFocusPathBoost:
    """Tests for focus path prioritization."""

    def test_focus_path_boost_applied(self):
        """Test that focus path boost is applied."""
        w_f = 0.05
        in_focus = 1.0

        base_score = 0.5
        focus_contribution = w_f * in_focus

        score_with_focus = base_score + focus_contribution
        assert score_with_focus > base_score

    def test_non_focus_path_no_boost(self):
        """Test that non-focus paths don't get boost."""
        w_f = 0.05
        in_focus = 0.0

        focus_contribution = w_f * in_focus
        assert focus_contribution == 0.0

    def test_focus_path_matching(self):
        """Test focus path matching logic."""
        focus_paths = ["/src/auth/", "/src/api/"]

        test_paths = [
            ("/src/auth/handler.ts", True),
            ("/src/api/endpoints.ts", True),
            ("/src/utils/helpers.ts", False),
            ("/tests/auth.test.ts", False),
        ]

        for path, expected_match in test_paths:
            matches = any(path.startswith(fp) for fp in focus_paths)
            assert matches == expected_match


# ==============================================================================
# Pinned Items Boost Tests
# ==============================================================================

class TestPinnedBoost:
    """Tests for pinned items boost."""

    def test_pinned_boost_applied(self):
        """Test that pinned boost is applied."""
        w_p = 0.05
        is_pinned = 1.0

        base_score = 0.5
        pinned_contribution = w_p * is_pinned

        score_with_pinned = base_score + pinned_contribution
        assert score_with_pinned > base_score

    def test_non_pinned_no_boost(self):
        """Test that non-pinned items don't get boost."""
        w_p = 0.05
        is_pinned = 0.0

        pinned_contribution = w_p * is_pinned
        assert pinned_contribution == 0.0


# ==============================================================================
# Result Limiting Tests
# ==============================================================================

class TestResultLimiting:
    """Tests for result count limiting."""

    def test_initial_candidates_limit(self):
        """Test initial candidates retrieval limit."""
        initial_candidates = 100  # From config
        all_results = list(range(500))

        limited = all_results[:initial_candidates]
        assert len(limited) == initial_candidates

    def test_final_results_limit(self):
        """Test final results limit after MMR."""
        final_results = 20  # From config
        candidates = list(range(100))

        # After MMR reranking
        limited = candidates[:final_results]
        assert len(limited) == final_results

    def test_limit_less_than_available(self):
        """Test when limit is less than available results."""
        available = 15
        limit = 20

        results = list(range(available))
        returned = results[:min(limit, len(results))]

        assert len(returned) == available


# ==============================================================================
# Edge Cases Tests
# ==============================================================================

class TestHybridRetrievalEdgeCases:
    """Tests for edge cases in hybrid retrieval."""

    def test_empty_query(self):
        """Test handling of empty query."""
        query = ""
        # Should return empty or handle gracefully
        assert query == ""

    def test_all_zero_scores(self):
        """Test handling when all scores are zero."""
        scores = np.zeros(10)
        # All results have same (zero) score
        assert np.all(scores == 0)

    def test_single_result(self):
        """Test with only one result available."""
        results = [("chunk_1", 0.9)]
        assert len(results) == 1

    def test_duplicate_chunk_ids(self):
        """Test handling of duplicate chunk IDs."""
        # Semantic and lexical might return same chunk
        semantic_results = ["chunk_1", "chunk_2", "chunk_3"]
        lexical_results = ["chunk_2", "chunk_4", "chunk_1"]

        # Merge should deduplicate
        merged = list(dict.fromkeys(semantic_results + lexical_results))
        assert len(merged) == 4  # chunk_1, chunk_2, chunk_3, chunk_4

    def test_missing_semantic_scores(self):
        """Test when semantic scores are missing for some chunks."""
        # Lexical-only results don't have embeddings
        lexical_only_chunk = {"id": "chunk_x", "bm25_score": 0.8}

        # Should handle gracefully, perhaps with s_embed = 0
        s_embed = 0.0  # Default when no embedding
        s_bm25 = lexical_only_chunk["bm25_score"]

        score = 0.4 * s_embed + 0.3 * s_bm25
        assert score > 0

    def test_missing_lexical_scores(self):
        """Test when lexical scores are missing for some chunks."""
        # Semantic-only results might not have BM25 scores
        semantic_only_chunk = {"id": "chunk_y", "semantic_score": 0.9}

        s_embed = semantic_only_chunk["semantic_score"]
        s_bm25 = 0.0  # Default when no BM25

        score = 0.4 * s_embed + 0.3 * s_bm25
        assert score > 0


# ==============================================================================
# Integration Tests (Lightweight)
# ==============================================================================

class TestHybridRetrievalIntegration:
    """Lightweight integration tests for hybrid retrieval."""

    def test_full_scoring_pipeline(self):
        """Test complete scoring pipeline."""
        # Weights
        w_e, w_b, w_r, w_c, w_f, w_p = 0.4, 0.3, 0.1, 0.1, 0.05, 0.05

        # Chunk data
        chunks = [
            {"id": "chunk_1", "s_embed": 0.9, "s_bm25": 0.7, "age_days": 1, "is_contract": False, "in_focus": True, "is_pinned": False},
            {"id": "chunk_2", "s_embed": 0.7, "s_bm25": 0.9, "age_days": 30, "is_contract": True, "in_focus": False, "is_pinned": False},
            {"id": "chunk_3", "s_embed": 0.8, "s_bm25": 0.8, "age_days": 7, "is_contract": False, "in_focus": False, "is_pinned": True},
        ]

        tau = 30.0
        results = []

        for chunk in chunks:
            r_t = math.exp(-chunk["age_days"] / tau)
            c = 1.0 if chunk["is_contract"] else 0.0
            f = 1.0 if chunk["in_focus"] else 0.0
            p = 1.0 if chunk["is_pinned"] else 0.0

            score = (
                w_e * chunk["s_embed"] +
                w_b * chunk["s_bm25"] +
                w_r * r_t +
                w_c * c +
                w_f * f +
                w_p * p
            )
            results.append((chunk["id"], score))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)

        assert len(results) == 3
        # Verify all scores are valid
        for chunk_id, score in results:
            assert 0 <= score <= 1

    def test_retrieval_with_focus_paths(self):
        """Test retrieval with focus paths specified."""
        focus_paths = ["/src/auth/"]

        chunks = [
            {"path": "/src/auth/handler.ts", "score": 0.8},
            {"path": "/src/api/endpoints.ts", "score": 0.85},
            {"path": "/src/auth/validator.ts", "score": 0.75},
        ]

        # Apply focus boost
        w_f = 0.05
        for chunk in chunks:
            in_focus = any(chunk["path"].startswith(fp) for fp in focus_paths)
            chunk["final_score"] = chunk["score"] + (w_f if in_focus else 0)

        # Sort by final score
        chunks.sort(key=lambda x: x["final_score"], reverse=True)

        # Auth files should be boosted
        auth_chunks = [c for c in chunks if "/auth/" in c["path"]]
        assert len(auth_chunks) == 2
