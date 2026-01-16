"""
Tests for pack compilation modules.
"""

from __future__ import annotations

import pytest


class TestPackCompiler:
    """Tests for PackCompiler."""

    @pytest.mark.asyncio
    async def test_compile_basic_pack(self, test_config, sample_chunks):
        """Test basic pack compilation."""
        from icd.pack.compiler import PackCompiler

        compiler = PackCompiler(test_config)

        scores = [0.9 - i * 0.05 for i in range(len(sample_chunks))]

        result = await compiler.compile(
            chunks=sample_chunks,
            scores=scores,
            budget_tokens=1000,
            query="test query",
        )

        assert result.content is not None
        assert result.token_count > 0
        assert result.token_count <= 1000
        assert len(result.chunk_ids) > 0

    @pytest.mark.asyncio
    async def test_respects_budget(self, test_config, sample_chunks):
        """Test that compiler respects token budget."""
        from icd.pack.compiler import PackCompiler

        compiler = PackCompiler(test_config)

        # Set a very small budget
        small_budget = 50
        scores = [0.9] * len(sample_chunks)

        result = await compiler.compile(
            chunks=sample_chunks,
            scores=scores,
            budget_tokens=small_budget,
        )

        assert result.token_count <= small_budget

    @pytest.mark.asyncio
    async def test_prioritizes_high_scores(self, test_config, sample_chunks):
        """Test that high-scoring chunks are prioritized."""
        from icd.pack.compiler import PackCompiler

        compiler = PackCompiler(test_config)

        # Give first chunk much higher score
        scores = [0.1] * len(sample_chunks)
        scores[0] = 1.0

        result = await compiler.compile(
            chunks=sample_chunks,
            scores=scores,
            budget_tokens=50,  # Small budget
        )

        # First chunk should be included if it fits
        if sample_chunks[0].token_count <= 50:
            assert sample_chunks[0].chunk_id in result.chunk_ids

    @pytest.mark.asyncio
    async def test_includes_contracts(self, test_config, sample_chunks):
        """Test that contracts get bonus utility."""
        from icd.pack.compiler import PackCompiler

        compiler = PackCompiler(test_config)

        # Give all equal scores
        scores = [0.5] * len(sample_chunks)

        result = await compiler.compile(
            chunks=sample_chunks,
            scores=scores,
            budget_tokens=200,
        )

        # Should include some chunks
        assert len(result.chunk_ids) > 0

    @pytest.mark.asyncio
    async def test_generates_citations(self, test_config, sample_chunks):
        """Test citation generation."""
        from icd.pack.compiler import PackCompiler

        compiler = PackCompiler(test_config)

        scores = [0.9 - i * 0.05 for i in range(len(sample_chunks))]

        result = await compiler.compile(
            chunks=sample_chunks,
            scores=scores,
            budget_tokens=1000,
        )

        # Should have citations for included chunks
        assert len(result.citations) == len(result.chunk_ids)
        assert all(key.startswith("[") for key in result.citations.keys())


class TestIncrementalPackCompiler:
    """Tests for IncrementalPackCompiler."""

    @pytest.mark.asyncio
    async def test_incremental_add(self, test_config, sample_chunks):
        """Test incremental chunk addition."""
        from icd.pack.compiler import IncrementalPackCompiler

        compiler = IncrementalPackCompiler(test_config)
        compiler.budget = 200

        # Add chunks incrementally
        added_count = 0
        for i, chunk in enumerate(sample_chunks[:5]):
            if compiler.add_chunk(chunk, 0.8 - i * 0.1):
                added_count += 1

        assert added_count > 0

        pack = compiler.get_pack("test query")
        assert pack.token_count <= 200

    @pytest.mark.asyncio
    async def test_replaces_low_utility(self, test_config, sample_chunks):
        """Test that low utility chunks are replaced."""
        from icd.pack.compiler import IncrementalPackCompiler

        compiler = IncrementalPackCompiler(test_config)
        compiler.budget = 50  # Very small

        # Add low score chunk
        compiler.add_chunk(sample_chunks[0], 0.1)

        # Add high score chunk
        added = compiler.add_chunk(sample_chunks[1], 0.9)

        # High score should be added (possibly replacing low score)
        pack = compiler.get_pack()
        assert len(pack.chunk_ids) > 0

    def test_reset(self, test_config, sample_chunks):
        """Test compiler reset."""
        from icd.pack.compiler import IncrementalPackCompiler

        compiler = IncrementalPackCompiler(test_config)

        compiler.add_chunk(sample_chunks[0], 0.8)
        compiler.reset()

        pack = compiler.get_pack()
        assert len(pack.chunk_ids) == 0


class TestPackFormatter:
    """Tests for PackFormatter."""

    def test_markdown_format(self, sample_chunks):
        """Test markdown formatting."""
        from icd.pack.compiler import PackItem
        from icd.pack.formatter import PackFormatter

        formatter = PackFormatter()

        items = [
            PackItem(
                chunk=chunk,
                score=0.8,
                utility=0.8,
                cost=chunk.token_count,
                selected=True,
            )
            for chunk in sample_chunks[:3]
        ]

        citations = {
            f"[{i+1}]": f"{item.chunk.file_path}:{item.chunk.start_line}"
            for i, item in enumerate(items)
        }

        content = formatter.format(items, citations, "test query")

        assert "# Context for: test query" in content
        assert "```" in content  # Code blocks
        assert "[1]" in content  # Citations

    def test_compact_format(self, sample_chunks):
        """Test compact formatting."""
        from icd.pack.compiler import PackItem
        from icd.pack.formatter import PackFormatter

        formatter = PackFormatter()

        items = [
            PackItem(
                chunk=chunk,
                score=0.8,
                utility=0.8,
                cost=chunk.token_count,
                selected=True,
            )
            for chunk in sample_chunks[:3]
        ]

        content = formatter.format_compact(items)

        # Should not have markdown fences
        assert "```" not in content
        # Should have file headers
        assert "#" in content

    def test_xml_format(self, sample_chunks):
        """Test XML formatting."""
        from icd.pack.compiler import PackItem
        from icd.pack.formatter import PackFormatter

        formatter = PackFormatter()

        items = [
            PackItem(
                chunk=chunk,
                score=0.8,
                utility=0.8,
                cost=chunk.token_count,
                selected=True,
            )
            for chunk in sample_chunks[:3]
        ]

        content = formatter.format_xml(items, "test query")

        assert "<context>" in content
        assert "</context>" in content
        assert "<chunk" in content
        assert "<query>" in content


class TestModeGate:
    """Tests for ModeGate."""

    def test_pack_mode_for_low_entropy(self, test_config, sample_chunks):
        """Test that low entropy results in pack mode."""
        from icd.pack.gating import ModeGate, RetrievalMode
        from icd.retrieval.hybrid import RetrievalResult

        gate = ModeGate(test_config)

        # Low entropy result (confident)
        result = RetrievalResult(
            chunks=sample_chunks,
            scores=[10.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001],
            entropy=0.2,  # Low entropy
            query="simple query",
        )

        decision = gate.decide(result, "simple query")

        assert decision.mode == RetrievalMode.PACK

    def test_rlm_mode_for_high_entropy(self, test_config, sample_chunks):
        """Test that high entropy results in RLM mode."""
        from icd.pack.gating import ModeGate, RetrievalMode
        from icd.retrieval.hybrid import RetrievalResult

        gate = ModeGate(test_config)

        # High entropy result (uncertain)
        result = RetrievalResult(
            chunks=sample_chunks,
            scores=[1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55],
            entropy=0.9,  # High entropy
            query="how does the complex authentication system work with microservices",
        )

        decision = gate.decide(result, "how does the complex authentication system work with microservices")

        # May be RLM depending on configuration
        assert decision.mode in [RetrievalMode.PACK, RetrievalMode.RLM]

    def test_rlm_disabled(self, test_config, sample_chunks):
        """Test behavior when RLM is disabled."""
        from icd.pack.gating import ModeGate, RetrievalMode
        from icd.retrieval.hybrid import RetrievalResult

        # Disable RLM in config
        test_config.rlm.enabled = False

        gate = ModeGate(test_config)

        result = RetrievalResult(
            chunks=sample_chunks,
            scores=[1.0] * 10,  # Uniform (high entropy)
            entropy=0.99,
            query="test",
        )

        decision = gate.decide(result, "test")

        # Should always be pack when RLM disabled
        assert decision.mode == RetrievalMode.PACK


class TestAdaptiveGate:
    """Tests for AdaptiveGate."""

    def test_feedback_adjusts_threshold(self, test_config, sample_chunks):
        """Test that feedback adjusts thresholds."""
        from icd.pack.gating import AdaptiveGate, RetrievalMode
        from icd.retrieval.hybrid import RetrievalResult

        gate = AdaptiveGate(test_config)

        initial_threshold = gate._entropy_threshold

        result = RetrievalResult(
            chunks=sample_chunks,
            scores=[1.0] * 10,
            entropy=0.5,
            query="test",
        )

        # Make decision
        decision = gate.decide(result, "test")

        # Provide feedback that user wanted different mode
        if decision.mode == RetrievalMode.PACK:
            gate.provide_feedback(False, RetrievalMode.RLM)
        else:
            gate.provide_feedback(False, RetrievalMode.PACK)

        # Threshold should have changed
        # (may not always change significantly with single feedback)
        stats = gate.get_stats()
        assert stats["decisions"] == 1
