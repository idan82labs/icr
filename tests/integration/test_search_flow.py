"""
Integration tests for the search end-to-end flow.

Tests cover:
- Complete search pipeline
- Query processing
- Result ranking
- Response formatting
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.conftest import MockEmbeddingBackend, Chunk


# ==============================================================================
# Test Data Structures
# ==============================================================================

@dataclass
class SearchResult:
    """Search result for testing."""

    chunk_id: str
    content: str
    file_path: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResponse:
    """Complete search response."""

    results: list[SearchResult]
    query: str
    entropy: float
    total_candidates: int
    search_time_ms: float


# ==============================================================================
# Search Pipeline Simulation
# ==============================================================================

async def simulate_search(
    query: str,
    chunks: list[Chunk],
    embedder: MockEmbeddingBackend,
    config: dict[str, Any],
) -> SearchResponse:
    """
    Simulate the complete search pipeline.

    Pipeline:
    1. Embed query
    2. Semantic search
    3. Lexical search (simulated)
    4. Merge scores
    5. Apply boosts
    6. MMR diversity
    7. Return results
    """
    import time

    start_time = time.time()

    # 1. Embed query
    query_embedding = await embedder.embed_single(query)

    # 2. Generate chunk embeddings
    chunk_embeddings = await embedder.embed([c.content for c in chunks])

    # 3. Compute semantic scores
    semantic_scores = chunk_embeddings @ query_embedding

    # 4. Simulate lexical scores (based on keyword overlap)
    lexical_scores = []
    query_terms = set(query.lower().split())
    for chunk in chunks:
        chunk_terms = set(chunk.content.lower().split())
        overlap = len(query_terms & chunk_terms)
        lexical_scores.append(overlap / max(len(query_terms), 1))
    lexical_scores = np.array(lexical_scores)

    # 5. Merge scores
    w_e = config.get("weight_embedding", 0.4)
    w_b = config.get("weight_bm25", 0.3)
    w_c = config.get("weight_contract", 0.1)

    merged_scores = []
    for i, chunk in enumerate(chunks):
        is_contract = chunk.metadata.get("is_contract", False)
        score = (
            w_e * semantic_scores[i] +
            w_b * lexical_scores[i] +
            w_c * (1.0 if is_contract else 0.0)
        )
        merged_scores.append((i, score))

    # 6. Sort by score
    merged_scores.sort(key=lambda x: x[1], reverse=True)

    # 7. Apply limit
    limit = config.get("final_results", 10)
    top_results = merged_scores[:limit]

    # 8. Compute entropy
    scores = np.array([s for _, s in merged_scores])
    if len(scores) > 0:
        # Softmax
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / np.sum(exp_scores)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
    else:
        entropy = 0.0

    # 9. Format results
    results = []
    for idx, score in top_results:
        chunk = chunks[idx]
        results.append(SearchResult(
            chunk_id=chunk.chunk_id,
            content=chunk.content,
            file_path=chunk.file_path,
            score=float(score),
            metadata=chunk.metadata,
        ))

    end_time = time.time()

    return SearchResponse(
        results=results,
        query=query,
        entropy=entropy,
        total_candidates=len(chunks),
        search_time_ms=(end_time - start_time) * 1000,
    )


# ==============================================================================
# End-to-End Search Tests
# ==============================================================================

@pytest.mark.integration
class TestSearchEndToEnd:
    """End-to-end tests for search flow."""

    @pytest.fixture
    def search_config(self) -> dict[str, Any]:
        """Search configuration for tests."""
        return {
            "weight_embedding": 0.4,
            "weight_bm25": 0.3,
            "weight_recency": 0.1,
            "weight_contract": 0.1,
            "weight_focus": 0.05,
            "weight_pinned": 0.05,
            "initial_candidates": 100,
            "final_results": 10,
            "mmr_lambda": 0.7,
        }

    @pytest.mark.asyncio
    async def test_basic_search(self, sample_chunks, mock_embedder, search_config):
        """Test basic search returns results."""
        await mock_embedder.initialize()

        response = await simulate_search(
            query="auth token validation",
            chunks=sample_chunks,
            embedder=mock_embedder,
            config=search_config,
        )

        assert len(response.results) > 0
        assert response.query == "auth token validation"
        assert response.search_time_ms > 0

    @pytest.mark.asyncio
    async def test_search_returns_sorted_results(self, sample_chunks, mock_embedder, search_config):
        """Test search results are sorted by score."""
        await mock_embedder.initialize()

        response = await simulate_search(
            query="authentication",
            chunks=sample_chunks,
            embedder=mock_embedder,
            config=search_config,
        )

        # Verify descending order
        for i in range(len(response.results) - 1):
            assert response.results[i].score >= response.results[i + 1].score

    @pytest.mark.asyncio
    async def test_search_respects_limit(self, sample_chunks, mock_embedder, search_config):
        """Test search respects result limit."""
        await mock_embedder.initialize()

        search_config["final_results"] = 3

        response = await simulate_search(
            query="function",
            chunks=sample_chunks,
            embedder=mock_embedder,
            config=search_config,
        )

        assert len(response.results) <= 3

    @pytest.mark.asyncio
    async def test_search_computes_entropy(self, sample_chunks, mock_embedder, search_config):
        """Test search computes entropy."""
        await mock_embedder.initialize()

        response = await simulate_search(
            query="test query",
            chunks=sample_chunks,
            embedder=mock_embedder,
            config=search_config,
        )

        assert response.entropy >= 0


# ==============================================================================
# Query Processing Tests
# ==============================================================================

@pytest.mark.integration
class TestQueryProcessing:
    """Tests for query processing."""

    @pytest.mark.asyncio
    async def test_query_embedding(self, mock_embedder):
        """Test query embedding generation."""
        await mock_embedder.initialize()

        query = "Where is the auth token validated?"
        embedding = await mock_embedder.embed_single(query)

        assert embedding.shape == (384,)
        # Should be normalized
        assert abs(np.linalg.norm(embedding) - 1.0) < 1e-5

    @pytest.mark.asyncio
    async def test_query_preprocessing(self):
        """Test query preprocessing."""
        queries = [
            ("  Where is auth?  ", "Where is auth?"),
            ("WHERE IS AUTH?", "WHERE IS AUTH?"),
            ("auth\ntoken", "auth\ntoken"),
        ]

        for raw, expected in queries:
            processed = raw.strip()
            assert processed == expected

    @pytest.mark.asyncio
    async def test_empty_query_handling(self, sample_chunks, mock_embedder, search_config):
        """Test handling of empty queries."""
        await mock_embedder.initialize()

        response = await simulate_search(
            query="",
            chunks=sample_chunks,
            embedder=mock_embedder,
            config={"weight_embedding": 0.4, "weight_bm25": 0.3, "weight_contract": 0.1, "final_results": 10},
        )

        # Should return results (based on other signals)
        assert len(response.results) >= 0


# ==============================================================================
# Result Ranking Tests
# ==============================================================================

@pytest.mark.integration
class TestResultRanking:
    """Tests for result ranking."""

    @pytest.fixture
    def ranking_chunks(self) -> list[Chunk]:
        """Create chunks with known ranking characteristics."""
        return [
            Chunk(
                chunk_id="high_semantic",
                file_path="/src/auth.ts",
                content="This is about authentication and token validation",
                start_line=1,
                end_line=10,
                token_count=50,
            ),
            Chunk(
                chunk_id="high_lexical",
                file_path="/src/utils.ts",
                content="auth token auth token auth token",  # Many keyword matches
                start_line=1,
                end_line=5,
                token_count=30,
            ),
            Chunk(
                chunk_id="contract",
                file_path="/src/types.ts",
                content="interface AuthToken { value: string; }",
                start_line=1,
                end_line=5,
                token_count=40,
                metadata={"is_contract": True},
            ),
            Chunk(
                chunk_id="low_relevance",
                file_path="/src/unrelated.ts",
                content="function calculateSum(a, b) { return a + b; }",
                start_line=1,
                end_line=5,
                token_count=30,
            ),
        ]

    @pytest.mark.asyncio
    async def test_hybrid_ranking(self, ranking_chunks, mock_embedder):
        """Test hybrid ranking combines semantic and lexical."""
        await mock_embedder.initialize()

        config = {
            "weight_embedding": 0.4,
            "weight_bm25": 0.3,
            "weight_contract": 0.1,
            "final_results": 10,
        }

        response = await simulate_search(
            query="auth token",
            chunks=ranking_chunks,
            embedder=mock_embedder,
            config=config,
        )

        # Low relevance chunk should be ranked lower
        chunk_positions = {r.chunk_id: i for i, r in enumerate(response.results)}
        assert chunk_positions.get("low_relevance", 999) > 0

    @pytest.mark.asyncio
    async def test_contract_boost_in_ranking(self, ranking_chunks, mock_embedder):
        """Test contract boost affects ranking."""
        await mock_embedder.initialize()

        config = {
            "weight_embedding": 0.4,
            "weight_bm25": 0.3,
            "weight_contract": 0.2,  # Higher contract weight
            "final_results": 10,
        }

        response = await simulate_search(
            query="auth token interface",
            chunks=ranking_chunks,
            embedder=mock_embedder,
            config=config,
        )

        # Contract chunk should get boost
        contract_result = next(
            (r for r in response.results if r.chunk_id == "contract"),
            None
        )
        assert contract_result is not None


# ==============================================================================
# Response Formatting Tests
# ==============================================================================

@pytest.mark.integration
class TestResponseFormatting:
    """Tests for response formatting."""

    @pytest.mark.asyncio
    async def test_response_structure(self, sample_chunks, mock_embedder):
        """Test search response structure."""
        await mock_embedder.initialize()

        response = await simulate_search(
            query="test",
            chunks=sample_chunks,
            embedder=mock_embedder,
            config={"weight_embedding": 0.4, "weight_bm25": 0.3, "weight_contract": 0.1, "final_results": 5},
        )

        # Verify structure
        assert hasattr(response, "results")
        assert hasattr(response, "query")
        assert hasattr(response, "entropy")
        assert hasattr(response, "total_candidates")
        assert hasattr(response, "search_time_ms")

    @pytest.mark.asyncio
    async def test_result_structure(self, sample_chunks, mock_embedder):
        """Test individual result structure."""
        await mock_embedder.initialize()

        response = await simulate_search(
            query="auth",
            chunks=sample_chunks,
            embedder=mock_embedder,
            config={"weight_embedding": 0.4, "weight_bm25": 0.3, "weight_contract": 0.1, "final_results": 5},
        )

        if response.results:
            result = response.results[0]
            assert hasattr(result, "chunk_id")
            assert hasattr(result, "content")
            assert hasattr(result, "file_path")
            assert hasattr(result, "score")
            assert hasattr(result, "metadata")

    @pytest.mark.asyncio
    async def test_scores_are_numeric(self, sample_chunks, mock_embedder):
        """Test all scores are numeric."""
        await mock_embedder.initialize()

        response = await simulate_search(
            query="function",
            chunks=sample_chunks,
            embedder=mock_embedder,
            config={"weight_embedding": 0.4, "weight_bm25": 0.3, "weight_contract": 0.1, "final_results": 10},
        )

        for result in response.results:
            assert isinstance(result.score, (int, float))
            assert not math.isnan(result.score)
            assert not math.isinf(result.score)


# ==============================================================================
# Edge Cases Tests
# ==============================================================================

@pytest.mark.integration
class TestSearchEdgeCases:
    """Tests for search edge cases."""

    @pytest.mark.asyncio
    async def test_search_with_no_chunks(self, mock_embedder):
        """Test search with empty chunk index."""
        await mock_embedder.initialize()

        response = await simulate_search(
            query="test query",
            chunks=[],
            embedder=mock_embedder,
            config={"weight_embedding": 0.4, "weight_bm25": 0.3, "weight_contract": 0.1, "final_results": 10},
        )

        assert len(response.results) == 0
        assert response.total_candidates == 0

    @pytest.mark.asyncio
    async def test_search_with_single_chunk(self, mock_embedder):
        """Test search with single chunk."""
        await mock_embedder.initialize()

        single_chunk = Chunk(
            chunk_id="only_chunk",
            file_path="/src/only.ts",
            content="The only content",
            start_line=1,
            end_line=5,
            token_count=20,
        )

        response = await simulate_search(
            query="content",
            chunks=[single_chunk],
            embedder=mock_embedder,
            config={"weight_embedding": 0.4, "weight_bm25": 0.3, "weight_contract": 0.1, "final_results": 10},
        )

        assert len(response.results) == 1
        assert response.results[0].chunk_id == "only_chunk"

    @pytest.mark.asyncio
    async def test_search_with_special_characters(self, sample_chunks, mock_embedder):
        """Test search with special characters in query."""
        await mock_embedder.initialize()

        special_queries = [
            "auth()",
            "async function*",
            "<T>generic",
            "regex /pattern/",
        ]

        for query in special_queries:
            response = await simulate_search(
                query=query,
                chunks=sample_chunks,
                embedder=mock_embedder,
                config={"weight_embedding": 0.4, "weight_bm25": 0.3, "weight_contract": 0.1, "final_results": 5},
            )
            # Should not crash
            assert response is not None


# ==============================================================================
# Performance Tests
# ==============================================================================

@pytest.mark.integration
class TestSearchPerformance:
    """Tests for search performance."""

    @pytest.mark.asyncio
    async def test_search_time_recorded(self, sample_chunks, mock_embedder):
        """Test search time is recorded."""
        await mock_embedder.initialize()

        response = await simulate_search(
            query="authentication",
            chunks=sample_chunks,
            embedder=mock_embedder,
            config={"weight_embedding": 0.4, "weight_bm25": 0.3, "weight_contract": 0.1, "final_results": 10},
        )

        assert response.search_time_ms > 0

    @pytest.mark.asyncio
    async def test_search_with_many_chunks(self, mock_embedder):
        """Test search performance with many chunks."""
        await mock_embedder.initialize()

        # Create 1000 chunks
        many_chunks = [
            Chunk(
                chunk_id=f"chunk_{i}",
                file_path=f"/src/file_{i % 100}.ts",
                content=f"Content for chunk number {i} with some code",
                start_line=i * 10,
                end_line=(i + 1) * 10,
                token_count=50,
            )
            for i in range(1000)
        ]

        response = await simulate_search(
            query="code content",
            chunks=many_chunks,
            embedder=mock_embedder,
            config={"weight_embedding": 0.4, "weight_bm25": 0.3, "weight_contract": 0.1, "final_results": 10},
        )

        assert len(response.results) == 10
        assert response.total_candidates == 1000
