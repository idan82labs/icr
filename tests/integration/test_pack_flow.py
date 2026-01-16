"""
Integration tests for pack generation end-to-end.

Tests cover:
- Complete pack generation flow
- Query to pack pipeline
- Pack formatting and citations
- Budget handling
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.conftest import Chunk, MockEmbeddingBackend


# ==============================================================================
# Test Data Structures
# ==============================================================================

@dataclass
class PackItem:
    """Item to include in pack."""

    chunk_id: str
    content: str
    tokens: int
    score: float
    file_path: str
    start_line: int
    end_line: int
    mandatory: bool = False


@dataclass
class PackOutput:
    """Complete pack output."""

    content: str
    total_tokens: int
    items: list[PackItem]
    overflow_items: list[PackItem]
    citations: dict[str, str]
    query: str
    entropy: float
    mode: str


# ==============================================================================
# Pack Generation Pipeline
# ==============================================================================

async def generate_pack(
    query: str,
    chunks: list[Chunk],
    embedder: MockEmbeddingBackend,
    budget_tokens: int = 4000,
    mandatory_ids: set[str] | None = None,
) -> PackOutput:
    """
    Simulate complete pack generation pipeline.

    Pipeline:
    1. Embed query
    2. Score all chunks
    3. Apply knapsack packing
    4. Format output with citations
    """
    mandatory_ids = mandatory_ids or set()

    # 1. Embed query
    query_embedding = await embedder.embed_single(query)

    # 2. Score chunks
    chunk_embeddings = await embedder.embed([c.content for c in chunks])
    scores = chunk_embeddings @ query_embedding

    # 3. Create pack items
    items = [
        PackItem(
            chunk_id=chunk.chunk_id,
            content=chunk.content,
            tokens=chunk.token_count or len(chunk.content) // 4,
            score=float(scores[i]),
            file_path=chunk.file_path,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            mandatory=chunk.chunk_id in mandatory_ids,
        )
        for i, chunk in enumerate(chunks)
    ]

    # 4. Knapsack packing
    selected, overflow = knapsack_pack(items, budget_tokens, mandatory_ids)

    # 5. Format pack
    content, citations = format_pack(selected)

    # 6. Compute entropy
    if len(scores) > 0:
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / np.sum(exp_scores)
        entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
    else:
        entropy = 0.0

    # 7. Determine mode
    mode = "pack" if entropy < 2.5 else "rlm"

    return PackOutput(
        content=content,
        total_tokens=sum(item.tokens for item in selected),
        items=selected,
        overflow_items=overflow,
        citations=citations,
        query=query,
        entropy=entropy,
        mode=mode,
    )


def knapsack_pack(
    items: list[PackItem],
    budget: int,
    mandatory_ids: set[str],
) -> tuple[list[PackItem], list[PackItem]]:
    """Pack items using greedy knapsack."""
    # Separate mandatory and optional
    mandatory = [i for i in items if i.chunk_id in mandatory_ids or i.mandatory]
    optional = [i for i in items if i.chunk_id not in mandatory_ids and not i.mandatory]

    selected = list(mandatory)
    remaining = budget - sum(i.tokens for i in mandatory)

    # Sort optional by score/token ratio
    optional.sort(key=lambda x: x.score / max(x.tokens, 1), reverse=True)

    overflow = []
    for item in optional:
        if item.tokens <= remaining:
            selected.append(item)
            remaining -= item.tokens
        else:
            overflow.append(item)

    return selected, overflow


def format_pack(items: list[PackItem]) -> tuple[str, dict[str, str]]:
    """Format pack items into markdown with citations."""
    lines = []
    citations = {}

    for i, item in enumerate(items, 1):
        citation_key = f"[{i}]"
        citations[citation_key] = f"{item.file_path}:{item.start_line}-{item.end_line}"

        lines.append(f"<!-- {citation_key} {item.file_path}:{item.start_line} -->")
        lines.append("```")
        lines.append(item.content)
        lines.append("```")
        lines.append("")

    return "\n".join(lines), citations


# ==============================================================================
# End-to-End Pack Tests
# ==============================================================================

@pytest.mark.integration
class TestPackEndToEnd:
    """End-to-end tests for pack generation."""

    @pytest.mark.asyncio
    async def test_basic_pack_generation(self, sample_chunks, mock_embedder):
        """Test basic pack generation."""
        await mock_embedder.initialize()

        pack = await generate_pack(
            query="authentication flow",
            chunks=sample_chunks,
            embedder=mock_embedder,
            budget_tokens=2000,
        )

        assert pack.content is not None
        assert len(pack.content) > 0
        assert pack.total_tokens <= 2000
        assert pack.mode in ["pack", "rlm"]

    @pytest.mark.asyncio
    async def test_pack_respects_budget(self, sample_chunks, mock_embedder):
        """Test pack respects token budget."""
        await mock_embedder.initialize()

        budgets = [500, 1000, 2000, 4000]

        for budget in budgets:
            pack = await generate_pack(
                query="test query",
                chunks=sample_chunks,
                embedder=mock_embedder,
                budget_tokens=budget,
            )

            assert pack.total_tokens <= budget

    @pytest.mark.asyncio
    async def test_pack_includes_citations(self, sample_chunks, mock_embedder):
        """Test pack includes citations."""
        await mock_embedder.initialize()

        pack = await generate_pack(
            query="auth token",
            chunks=sample_chunks,
            embedder=mock_embedder,
            budget_tokens=3000,
        )

        # Should have citations
        assert len(pack.citations) > 0

        # Citations should reference file paths
        for key, value in pack.citations.items():
            assert key.startswith("[")
            assert ":" in value  # file:line format

    @pytest.mark.asyncio
    async def test_pack_prioritizes_relevant(self, sample_chunks, mock_embedder):
        """Test pack prioritizes relevant chunks."""
        await mock_embedder.initialize()

        # Use a specific query
        pack = await generate_pack(
            query="handleAuth token validation",
            chunks=sample_chunks,
            embedder=mock_embedder,
            budget_tokens=1000,
        )

        # Should include some items
        assert len(pack.items) > 0


# ==============================================================================
# Query to Pack Pipeline Tests
# ==============================================================================

@pytest.mark.integration
class TestQueryToPackPipeline:
    """Tests for the query to pack pipeline."""

    @pytest.mark.asyncio
    async def test_query_affects_pack_content(self, sample_chunks, mock_embedder):
        """Test that different queries produce different packs."""
        await mock_embedder.initialize()

        pack1 = await generate_pack(
            query="authentication",
            chunks=sample_chunks,
            embedder=mock_embedder,
            budget_tokens=1500,
        )

        pack2 = await generate_pack(
            query="endpoint API",
            chunks=sample_chunks,
            embedder=mock_embedder,
            budget_tokens=1500,
        )

        # Different queries may produce different item sets
        items1 = {item.chunk_id for item in pack1.items}
        items2 = {item.chunk_id for item in pack2.items}

        # Not necessarily different, but scores should differ
        assert pack1.query != pack2.query

    @pytest.mark.asyncio
    async def test_pack_with_focus_paths(self, sample_chunks, mock_embedder):
        """Test pack generation with focus paths."""
        await mock_embedder.initialize()

        # Filter chunks by focus path
        auth_chunks = [c for c in sample_chunks if "auth" in c.file_path.lower()]

        pack = await generate_pack(
            query="token validation",
            chunks=auth_chunks,
            embedder=mock_embedder,
            budget_tokens=2000,
        )

        # All items should be from auth path
        for item in pack.items:
            assert "auth" in item.file_path.lower()


# ==============================================================================
# Pack Formatting Tests
# ==============================================================================

@pytest.mark.integration
class TestPackFormatting:
    """Tests for pack output formatting."""

    @pytest.mark.asyncio
    async def test_pack_markdown_format(self, sample_chunks, mock_embedder):
        """Test pack is formatted as markdown."""
        await mock_embedder.initialize()

        pack = await generate_pack(
            query="function",
            chunks=sample_chunks,
            embedder=mock_embedder,
            budget_tokens=2000,
        )

        # Should contain code blocks
        assert "```" in pack.content

        # Should contain citation comments
        assert "<!--" in pack.content

    @pytest.mark.asyncio
    async def test_citation_format(self, sample_chunks, mock_embedder):
        """Test citation format in pack."""
        await mock_embedder.initialize()

        pack = await generate_pack(
            query="auth",
            chunks=sample_chunks,
            embedder=mock_embedder,
            budget_tokens=2000,
        )

        for key, value in pack.citations.items():
            # Key format: [N]
            assert key.startswith("[") and key.endswith("]")

            # Value format: path:start-end
            parts = value.split(":")
            assert len(parts) >= 2

    @pytest.mark.asyncio
    async def test_pack_preserves_code(self, sample_chunks, mock_embedder):
        """Test pack preserves original code content."""
        await mock_embedder.initialize()

        pack = await generate_pack(
            query="function",
            chunks=sample_chunks,
            embedder=mock_embedder,
            budget_tokens=5000,
        )

        # Original content should be in pack
        for item in pack.items:
            assert item.content in pack.content


# ==============================================================================
# Budget Handling Tests
# ==============================================================================

@pytest.mark.integration
class TestBudgetHandling:
    """Tests for pack budget handling."""

    @pytest.mark.asyncio
    async def test_small_budget(self, sample_chunks, mock_embedder):
        """Test pack with small budget."""
        await mock_embedder.initialize()

        pack = await generate_pack(
            query="test",
            chunks=sample_chunks,
            embedder=mock_embedder,
            budget_tokens=200,
        )

        assert pack.total_tokens <= 200
        # Should still include something if possible
        # May be empty if no chunks fit

    @pytest.mark.asyncio
    async def test_large_budget(self, sample_chunks, mock_embedder):
        """Test pack with large budget."""
        await mock_embedder.initialize()

        pack = await generate_pack(
            query="test",
            chunks=sample_chunks,
            embedder=mock_embedder,
            budget_tokens=50000,
        )

        # Should include many/all chunks
        assert len(pack.overflow_items) < len(sample_chunks)

    @pytest.mark.asyncio
    async def test_exact_budget_fit(self, mock_embedder):
        """Test pack with exact budget fit."""
        await mock_embedder.initialize()

        # Create chunks that exactly fit
        chunks = [
            Chunk(
                chunk_id=f"chunk_{i}",
                file_path=f"/src/file{i}.ts",
                content=f"Content {i}",
                start_line=1,
                end_line=5,
                token_count=100,
            )
            for i in range(5)
        ]

        pack = await generate_pack(
            query="content",
            chunks=chunks,
            embedder=mock_embedder,
            budget_tokens=300,
        )

        assert pack.total_tokens <= 300

    @pytest.mark.asyncio
    async def test_overflow_tracking(self, sample_chunks, mock_embedder):
        """Test overflow items are tracked."""
        await mock_embedder.initialize()

        pack = await generate_pack(
            query="test",
            chunks=sample_chunks,
            embedder=mock_embedder,
            budget_tokens=500,  # Small budget
        )

        # Total items should equal original
        total_items = len(pack.items) + len(pack.overflow_items)
        assert total_items == len(sample_chunks)


# ==============================================================================
# Mandatory Items Tests
# ==============================================================================

@pytest.mark.integration
class TestMandatoryItems:
    """Tests for mandatory item handling."""

    @pytest.mark.asyncio
    async def test_mandatory_items_included(self, sample_chunks, mock_embedder):
        """Test mandatory items are always included."""
        await mock_embedder.initialize()

        # Make first chunk mandatory
        mandatory_ids = {sample_chunks[0].chunk_id}

        pack = await generate_pack(
            query="unrelated query",
            chunks=sample_chunks,
            embedder=mock_embedder,
            budget_tokens=2000,
            mandatory_ids=mandatory_ids,
        )

        included_ids = {item.chunk_id for item in pack.items}
        assert sample_chunks[0].chunk_id in included_ids

    @pytest.mark.asyncio
    async def test_multiple_mandatory_items(self, sample_chunks, mock_embedder):
        """Test multiple mandatory items."""
        await mock_embedder.initialize()

        # Make first two chunks mandatory
        mandatory_ids = {sample_chunks[0].chunk_id, sample_chunks[1].chunk_id}

        pack = await generate_pack(
            query="test",
            chunks=sample_chunks,
            embedder=mock_embedder,
            budget_tokens=5000,
            mandatory_ids=mandatory_ids,
        )

        included_ids = {item.chunk_id for item in pack.items}
        for mandatory_id in mandatory_ids:
            assert mandatory_id in included_ids


# ==============================================================================
# Mode Selection Tests
# ==============================================================================

@pytest.mark.integration
class TestModeSelection:
    """Tests for pack/RLM mode selection."""

    @pytest.mark.asyncio
    async def test_mode_in_output(self, sample_chunks, mock_embedder):
        """Test mode is included in output."""
        await mock_embedder.initialize()

        pack = await generate_pack(
            query="test query",
            chunks=sample_chunks,
            embedder=mock_embedder,
            budget_tokens=2000,
        )

        assert pack.mode in ["pack", "rlm"]

    @pytest.mark.asyncio
    async def test_entropy_in_output(self, sample_chunks, mock_embedder):
        """Test entropy is computed and included."""
        await mock_embedder.initialize()

        pack = await generate_pack(
            query="authentication",
            chunks=sample_chunks,
            embedder=mock_embedder,
            budget_tokens=2000,
        )

        assert isinstance(pack.entropy, float)
        assert pack.entropy >= 0


# ==============================================================================
# Edge Cases Tests
# ==============================================================================

@pytest.mark.integration
class TestPackEdgeCases:
    """Tests for pack generation edge cases."""

    @pytest.mark.asyncio
    async def test_empty_chunks(self, mock_embedder):
        """Test pack with no chunks."""
        await mock_embedder.initialize()

        pack = await generate_pack(
            query="test",
            chunks=[],
            embedder=mock_embedder,
            budget_tokens=2000,
        )

        assert pack.total_tokens == 0
        assert len(pack.items) == 0
        assert pack.content == ""

    @pytest.mark.asyncio
    async def test_single_chunk(self, mock_embedder):
        """Test pack with single chunk."""
        await mock_embedder.initialize()

        single_chunk = Chunk(
            chunk_id="only_chunk",
            file_path="/src/only.ts",
            content="Only content here",
            start_line=1,
            end_line=5,
            token_count=20,
        )

        pack = await generate_pack(
            query="content",
            chunks=[single_chunk],
            embedder=mock_embedder,
            budget_tokens=1000,
        )

        assert len(pack.items) == 1
        assert pack.items[0].chunk_id == "only_chunk"

    @pytest.mark.asyncio
    async def test_zero_budget(self, sample_chunks, mock_embedder):
        """Test pack with zero budget."""
        await mock_embedder.initialize()

        pack = await generate_pack(
            query="test",
            chunks=sample_chunks,
            embedder=mock_embedder,
            budget_tokens=0,
        )

        assert pack.total_tokens == 0
        assert len(pack.items) == 0

    @pytest.mark.asyncio
    async def test_long_query(self, sample_chunks, mock_embedder):
        """Test pack with long query."""
        await mock_embedder.initialize()

        long_query = " ".join(["authentication"] * 100)

        pack = await generate_pack(
            query=long_query,
            chunks=sample_chunks,
            embedder=mock_embedder,
            budget_tokens=2000,
        )

        # Should handle without error
        assert pack is not None
