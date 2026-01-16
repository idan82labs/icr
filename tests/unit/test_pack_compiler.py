"""
Unit tests for the pack compiler (knapsack packing) module.

Tests the REAL implementation from icd.pack.compiler.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field

import pytest
import numpy as np

# Add icd/src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "icd" / "src"))

from icd.config import Config
from icd.pack.compiler import PackCompiler, PackItem, PackResult, IncrementalPackCompiler
from icd.retrieval.hybrid import Chunk


# ==============================================================================
# Test Fixtures
# ==============================================================================

@pytest.fixture
def test_config(tmp_path: Path) -> Config:
    """Create a test configuration."""
    return Config(
        project_root=tmp_path,
        data_dir=tmp_path / ".icd",
    )


@pytest.fixture
def pack_compiler(test_config: Config) -> PackCompiler:
    """Create a pack compiler with test config."""
    return PackCompiler(test_config)


def make_chunk(
    chunk_id: str,
    content: str,
    token_count: int,
    file_path: str = "test.py",
    start_line: int = 1,
    end_line: int = 10,
    symbol_name: str | None = None,
    symbol_type: str | None = None,
    is_contract: bool = False,
    is_pinned: bool = False,
) -> Chunk:
    """Helper to create test chunks."""
    return Chunk(
        chunk_id=chunk_id,
        file_path=file_path,
        content=content,
        start_line=start_line,
        end_line=end_line,
        symbol_name=symbol_name,
        symbol_type=symbol_type,
        language="python",
        token_count=token_count,
        is_contract=is_contract,
        is_pinned=is_pinned,
    )


# ==============================================================================
# Knapsack Optimization Tests
# ==============================================================================

class TestKnapsackOptimization:
    """Tests for knapsack optimization algorithm."""

    @pytest.mark.asyncio
    async def test_basic_packing(self, pack_compiler: PackCompiler):
        """Test basic item packing within budget."""
        chunks = [
            make_chunk("a", "Content A", 100),
            make_chunk("b", "Content B", 200),
            make_chunk("c", "Content C", 150),
        ]
        scores = [0.9, 0.8, 0.7]
        budget = 300

        result = await pack_compiler.compile(chunks, scores, budget_tokens=budget)

        # Should select items that fit
        assert result.token_count <= budget

    @pytest.mark.asyncio
    async def test_maximize_utility(self, pack_compiler: PackCompiler):
        """Test that packing maximizes utility."""
        chunks = [
            make_chunk("high", "High score", 100),
            make_chunk("low", "Low score", 100),
        ]
        scores = [0.95, 0.1]
        budget = 100

        result = await pack_compiler.compile(chunks, scores, budget_tokens=budget)

        # Should select high score item
        assert len(result.chunk_ids) == 1
        assert result.chunk_ids[0] == "high"

    @pytest.mark.asyncio
    async def test_respects_budget(self, pack_compiler: PackCompiler):
        """Test that budget is never exceeded."""
        chunks = [
            make_chunk(f"item_{i}", f"Content {i}", 100)
            for i in range(10)
        ]
        scores = [0.5 + i * 0.05 for i in range(10)]
        budget = 350

        result = await pack_compiler.compile(chunks, scores, budget_tokens=budget)

        assert result.token_count <= budget

    @pytest.mark.asyncio
    async def test_all_items_fit(self, pack_compiler: PackCompiler):
        """Test when all items fit within budget."""
        chunks = [
            make_chunk("a", "A", 100),
            make_chunk("b", "B", 100),
        ]
        scores = [0.9, 0.8]
        budget = 500

        result = await pack_compiler.compile(chunks, scores, budget_tokens=budget)

        assert len(result.chunk_ids) == 2

    @pytest.mark.asyncio
    async def test_no_items_fit(self, pack_compiler: PackCompiler):
        """Test when no items fit within budget."""
        chunks = [
            make_chunk("a", "A", 500),
            make_chunk("b", "B", 600),
        ]
        scores = [0.9, 0.8]
        budget = 100

        result = await pack_compiler.compile(chunks, scores, budget_tokens=budget)

        assert len(result.chunk_ids) == 0
        assert result.token_count == 0

    @pytest.mark.asyncio
    async def test_empty_chunks(self, pack_compiler: PackCompiler):
        """Test with empty chunks list."""
        result = await pack_compiler.compile([], [], budget_tokens=1000)

        assert len(result.chunk_ids) == 0
        assert result.content == ""


# ==============================================================================
# Contract and Pinned Item Tests
# ==============================================================================

class TestContractAndPinnedItems:
    """Tests for contract and pinned item handling."""

    @pytest.mark.asyncio
    async def test_contract_bonus(self, pack_compiler: PackCompiler):
        """Test that contracts get utility bonus."""
        chunks = [
            make_chunk("contract", "Interface definition", 100, is_contract=True),
            make_chunk("normal", "Normal code", 100, is_contract=False),
        ]
        # Same base score, but contract should have higher utility
        scores = [0.5, 0.5]
        budget = 100

        result = await pack_compiler.compile(chunks, scores, budget_tokens=budget)

        # Contract should be selected due to bonus
        assert "contract" in result.chunk_ids

    @pytest.mark.asyncio
    async def test_pinned_bonus(self, pack_compiler: PackCompiler):
        """Test that pinned items get utility bonus."""
        chunks = [
            make_chunk("pinned", "Pinned content", 100, is_pinned=True),
            make_chunk("normal", "Normal code", 100, is_pinned=False),
        ]
        scores = [0.5, 0.5]
        budget = 100

        result = await pack_compiler.compile(chunks, scores, budget_tokens=budget)

        # Pinned should be selected due to bonus
        assert "pinned" in result.chunk_ids

    @pytest.mark.asyncio
    async def test_contract_and_pinned_combined(self, pack_compiler: PackCompiler):
        """Test combined contract and pinned bonus."""
        chunks = [
            make_chunk("both", "Interface", 100, is_contract=True, is_pinned=True),
            make_chunk("normal", "Normal", 100),
        ]
        scores = [0.3, 0.6]  # Lower base score but bonuses should help
        budget = 100

        result = await pack_compiler.compile(chunks, scores, budget_tokens=budget)

        # Combined bonuses should overcome lower score
        assert "both" in result.chunk_ids


# ==============================================================================
# Budget Constraint Tests
# ==============================================================================

class TestBudgetConstraints:
    """Tests for budget constraint handling."""

    @pytest.mark.asyncio
    async def test_exact_budget_fit(self, pack_compiler: PackCompiler):
        """Test items that exactly fit budget."""
        chunks = [
            make_chunk("a", "A", 250),
            make_chunk("b", "B", 250),
        ]
        scores = [0.9, 0.8]
        budget = 500

        result = await pack_compiler.compile(chunks, scores, budget_tokens=budget)

        assert result.token_count <= budget
        assert len(result.chunk_ids) == 2

    @pytest.mark.asyncio
    async def test_very_small_budget(self, pack_compiler: PackCompiler):
        """Test with very small budget that can't fit any items."""
        chunks = [make_chunk("a", "A", 100)]
        scores = [0.9]
        budget = 10  # Too small to fit the 100-token chunk

        result = await pack_compiler.compile(chunks, scores, budget_tokens=budget)

        assert len(result.chunk_ids) == 0

    @pytest.mark.asyncio
    async def test_max_budget_cap(self, pack_compiler: PackCompiler):
        """Test that max budget is enforced."""
        chunks = [make_chunk(f"item_{i}", f"Content {i}", 5000) for i in range(10)]
        scores = [0.9] * 10
        # Request more than max_budget_tokens
        budget = 1000000

        result = await pack_compiler.compile(chunks, scores, budget_tokens=budget)

        # Should be capped by max_budget_tokens
        assert result.token_count <= pack_compiler.max_budget


# ==============================================================================
# Pack Result Tests
# ==============================================================================

class TestPackResult:
    """Tests for pack result generation."""

    @pytest.mark.asyncio
    async def test_result_contains_chunks(self, pack_compiler: PackCompiler):
        """Test that result contains selected chunk IDs."""
        chunks = [
            make_chunk("chunk_1", "def foo():\n    pass", 50),
            make_chunk("chunk_2", "def bar():\n    pass", 50),
        ]
        scores = [0.9, 0.8]
        budget = 200

        result = await pack_compiler.compile(chunks, scores, budget_tokens=budget)

        assert "chunk_1" in result.chunk_ids
        assert "chunk_2" in result.chunk_ids

    @pytest.mark.asyncio
    async def test_result_has_citations(self, pack_compiler: PackCompiler):
        """Test that result includes citations."""
        chunks = [make_chunk("test_chunk", "def test():\n    pass", 50, symbol_name="test")]
        scores = [0.9]
        budget = 200

        result = await pack_compiler.compile(chunks, scores, budget_tokens=budget)

        assert len(result.citations) > 0
        assert "[1]" in result.citations

    @pytest.mark.asyncio
    async def test_result_has_metadata(self, pack_compiler: PackCompiler):
        """Test that result includes metadata."""
        chunks = [
            make_chunk("c1", "Content", 50, is_contract=True),
            make_chunk("c2", "Content", 50, is_pinned=True),
        ]
        scores = [0.9, 0.8]
        budget = 200

        result = await pack_compiler.compile(chunks, scores, budget_tokens=budget)

        assert "num_chunks" in result.metadata
        assert "num_contracts" in result.metadata
        assert "num_pinned" in result.metadata

    @pytest.mark.asyncio
    async def test_content_formatting(self, pack_compiler: PackCompiler):
        """Test that content is properly formatted."""
        chunks = [
            make_chunk("chunk_1", "def hello():\n    print('world')", 50, symbol_name="hello"),
        ]
        scores = [0.9]
        budget = 200

        result = await pack_compiler.compile(chunks, scores, budget_tokens=budget)

        # Content should contain the chunk
        assert "hello" in result.content or "print" in result.content


# ==============================================================================
# Incremental Compiler Tests
# ==============================================================================

class TestIncrementalCompiler:
    """Tests for incremental pack compilation."""

    def test_add_chunk_basic(self, test_config: Config):
        """Test basic chunk addition."""
        compiler = IncrementalPackCompiler(test_config)
        chunk = make_chunk("test", "Content", 100)

        added = compiler.add_chunk(chunk, score=0.9)

        assert added is True

    def test_add_chunk_exceeds_budget(self, test_config: Config):
        """Test adding chunk that exceeds budget."""
        compiler = IncrementalPackCompiler(test_config)
        compiler.budget = 100

        # Fill up budget
        chunk1 = make_chunk("a", "A", 80)
        compiler.add_chunk(chunk1, score=0.5)

        # Try to add another that doesn't fit
        chunk2 = make_chunk("b", "B", 80)
        added = compiler.add_chunk(chunk2, score=0.4)

        # Should not be added (lower utility, doesn't fit)
        assert added is False

    def test_replace_lower_utility(self, test_config: Config):
        """Test replacing lower utility item."""
        compiler = IncrementalPackCompiler(test_config)
        compiler.budget = 100

        # Add low utility item
        chunk1 = make_chunk("low", "Low", 50)
        compiler.add_chunk(chunk1, score=0.3)

        # Add high utility item that replaces it
        chunk2 = make_chunk("high", "High", 50)
        added = compiler.add_chunk(chunk2, score=0.9)

        assert added is True

    def test_get_pack(self, test_config: Config):
        """Test getting compiled pack."""
        compiler = IncrementalPackCompiler(test_config)
        chunk = make_chunk("test", "Content", 100)
        compiler.add_chunk(chunk, score=0.9)

        result = compiler.get_pack(query="test query")

        assert len(result.chunk_ids) == 1
        assert result.token_count == 100

    def test_reset(self, test_config: Config):
        """Test resetting compiler state."""
        compiler = IncrementalPackCompiler(test_config)
        chunk = make_chunk("test", "Content", 100)
        compiler.add_chunk(chunk, score=0.9)

        compiler.reset()

        result = compiler.get_pack()
        assert len(result.chunk_ids) == 0


# ==============================================================================
# Edge Cases
# ==============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_zero_token_chunk(self, pack_compiler: PackCompiler):
        """Test handling of zero-token chunks."""
        chunks = [
            make_chunk("zero", "", 0),
            make_chunk("normal", "Content", 100),
        ]
        scores = [0.9, 0.8]
        budget = 50

        result = await pack_compiler.compile(chunks, scores, budget_tokens=budget)

        # Zero-token chunk should fit
        assert "zero" in result.chunk_ids

    @pytest.mark.asyncio
    async def test_single_chunk(self, pack_compiler: PackCompiler):
        """Test with single chunk."""
        chunks = [make_chunk("only", "Content", 100)]
        scores = [0.9]
        budget = 200

        result = await pack_compiler.compile(chunks, scores, budget_tokens=budget)

        assert len(result.chunk_ids) == 1

    @pytest.mark.asyncio
    async def test_equal_scores(self, pack_compiler: PackCompiler):
        """Test with equal scores."""
        chunks = [
            make_chunk(f"item_{i}", f"Content {i}", 100)
            for i in range(5)
        ]
        scores = [0.5] * 5
        budget = 300

        result = await pack_compiler.compile(chunks, scores, budget_tokens=budget)

        # Should select 3 items (300 tokens / 100 per item = 3)
        assert len(result.chunk_ids) == 3


# ==============================================================================
# Performance Tests
# ==============================================================================

class TestPerformance:
    """Tests for pack compiler performance."""

    @pytest.mark.asyncio
    async def test_many_items(self, pack_compiler: PackCompiler):
        """Test packing with many items."""
        chunks = [
            make_chunk(f"item_{i}", f"Content {i}", 50 + i)
            for i in range(100)
        ]
        scores = [0.5 + i * 0.005 for i in range(100)]
        budget = 2000

        result = await pack_compiler.compile(chunks, scores, budget_tokens=budget)

        assert result.token_count <= budget
        assert len(result.chunk_ids) > 0

    @pytest.mark.asyncio
    async def test_varied_sizes(self, pack_compiler: PackCompiler):
        """Test packing with varied item sizes."""
        import random
        random.seed(42)

        chunks = [
            make_chunk(f"item_{i}", f"Content {i}", random.randint(50, 500))
            for i in range(50)
        ]
        scores = [random.random() for _ in range(50)]
        budget = 3000

        result = await pack_compiler.compile(chunks, scores, budget_tokens=budget)

        assert result.token_count <= budget


# ==============================================================================
# Integration Tests
# ==============================================================================

class TestIntegration:
    """Integration tests for pack compilation."""

    @pytest.mark.asyncio
    async def test_full_pipeline(self, pack_compiler: PackCompiler):
        """Test complete pack compilation pipeline."""
        chunks = [
            make_chunk(
                "chunk_auth",
                "async function handleAuth(token) { ... }",
                150,
                file_path="src/auth/handler.ts",
                symbol_name="handleAuth",
                symbol_type="function",
            ),
            make_chunk(
                "chunk_validator",
                "function validateToken(token) { ... }",
                100,
                file_path="src/auth/validator.ts",
                symbol_name="validateToken",
                symbol_type="function",
            ),
            make_chunk(
                "chunk_types",
                "interface AuthToken { ... }",
                80,
                file_path="src/types/shared.ts",
                symbol_name="AuthToken",
                symbol_type="interface",
                is_contract=True,
            ),
        ]
        scores = [0.95, 0.85, 0.75]
        budget = 400

        result = await pack_compiler.compile(
            chunks, scores, budget_tokens=budget, query="How does auth work?"
        )

        # Verify result structure
        assert result.token_count <= budget
        assert len(result.chunk_ids) >= 1
        assert result.content  # Non-empty content
        assert result.metadata.get("query") == "How does auth work?"
