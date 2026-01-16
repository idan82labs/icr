"""
Tests for indexing modules.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from icd.config import Config


class TestChunker:
    """Tests for Chunker."""

    def test_chunk_python_file(self, chunker, sample_code_files):
        """Test chunking a Python file."""
        python_path = sample_code_files["python"]
        content = python_path.read_text()

        chunks = chunker.chunk_file(python_path, content)

        assert len(chunks) > 0

        # Check chunk properties
        for chunk in chunks:
            assert chunk.file_path == str(python_path)
            assert chunk.language == "python"
            assert chunk.token_count > 0
            assert chunk.start_line >= 1
            assert chunk.end_line >= chunk.start_line

    def test_chunk_typescript_file(self, chunker, sample_code_files):
        """Test chunking a TypeScript file."""
        ts_path = sample_code_files["typescript"]
        content = ts_path.read_text()

        chunks = chunker.chunk_file(ts_path, content)

        assert len(chunks) > 0

        # Should detect interfaces and classes
        symbol_types = {c.symbol_type for c in chunks if c.symbol_type}
        assert len(symbol_types) > 0

    def test_chunk_go_file(self, chunker, sample_code_files):
        """Test chunking a Go file."""
        go_path = sample_code_files["go"]
        content = go_path.read_text()

        chunks = chunker.chunk_file(go_path, content)

        assert len(chunks) > 0

        # Should detect interfaces and structs
        symbol_names = {c.symbol_name for c in chunks if c.symbol_name}
        assert len(symbol_names) > 0

    def test_chunk_id_stability(self, chunker, sample_code_files):
        """Test that chunk IDs are stable for same content."""
        python_path = sample_code_files["python"]
        content = python_path.read_text()

        chunks1 = chunker.chunk_file(python_path, content)
        chunks2 = chunker.chunk_file(python_path, content)

        # Same content should produce same chunk IDs
        ids1 = {c.chunk_id for c in chunks1}
        ids2 = {c.chunk_id for c in chunks2}

        assert ids1 == ids2

    def test_token_estimation(self, chunker):
        """Test token counting."""
        short_text = "def hello(): pass"
        long_text = "def " * 100

        short_tokens = chunker.estimate_tokens(short_text)
        long_tokens = chunker.estimate_tokens(long_text)

        assert short_tokens < long_tokens
        assert short_tokens > 0

    def test_respects_max_tokens(self, chunker, temp_dir):
        """Test that chunks respect max token limit."""
        # Create a large file
        large_content = "\n".join([
            f"def function_{i}():" + "\n    " + "x = 1\n" * 50
            for i in range(20)
        ])
        large_path = temp_dir / "large.py"
        large_path.write_text(large_content)

        chunks = chunker.chunk_file(large_path, large_content)

        # All chunks should be within limits
        for chunk in chunks:
            assert chunk.token_count <= chunker.max_tokens


class TestContractDetector:
    """Tests for ContractDetector."""

    def test_detect_python_protocol(self, test_config, sample_code_files, chunker):
        """Test detecting Python protocols."""
        from icd.indexing.contract_detector import ContractDetector

        detector = ContractDetector(test_config)

        python_path = sample_code_files["python"]
        content = python_path.read_text()
        chunks = chunker.chunk_file(python_path, content)

        # Find protocol chunk
        protocol_chunks = [c for c in chunks if "Protocol" in c.content]
        assert len(protocol_chunks) > 0

        for chunk in protocol_chunks:
            matches = detector.detect_contracts(chunk)
            assert len(matches) > 0
            assert any(m.contract_type == "protocol" for m in matches)

    def test_detect_typescript_interface(self, test_config, sample_code_files, chunker):
        """Test detecting TypeScript interfaces."""
        from icd.indexing.contract_detector import ContractDetector

        detector = ContractDetector(test_config)

        ts_path = sample_code_files["typescript"]
        content = ts_path.read_text()
        chunks = chunker.chunk_file(ts_path, content)

        # Find interface chunk
        interface_chunks = [c for c in chunks if "interface" in c.content.lower()]
        assert len(interface_chunks) > 0

        for chunk in interface_chunks:
            matches = detector.detect_contracts(chunk)
            assert len(matches) > 0

    def test_detect_go_interface(self, test_config, sample_code_files, chunker):
        """Test detecting Go interfaces."""
        from icd.indexing.contract_detector import ContractDetector

        detector = ContractDetector(test_config)

        go_path = sample_code_files["go"]
        content = go_path.read_text()
        chunks = chunker.chunk_file(go_path, content)

        # Find interface chunk
        interface_chunks = [
            c for c in chunks
            if "interface" in c.content.lower() and "type" in c.content.lower()
        ]
        assert len(interface_chunks) > 0

        for chunk in interface_chunks:
            matches = detector.detect_contracts(chunk)
            assert len(matches) > 0

    def test_extract_signature(self, test_config, sample_code_files, chunker):
        """Test signature extraction."""
        from icd.indexing.contract_detector import ContractDetector

        detector = ContractDetector(test_config)

        python_path = sample_code_files["python"]
        content = python_path.read_text()
        chunks = chunker.chunk_file(python_path, content)

        for chunk in chunks:
            if chunk.symbol_name:
                signature = detector.extract_signature(chunk)
                if signature:
                    assert len(signature) > 0
                    assert signature.strip()

    def test_extract_dependencies(self, test_config, sample_code_files, chunker):
        """Test dependency extraction."""
        from icd.indexing.contract_detector import ContractDetector

        detector = ContractDetector(test_config)

        python_path = sample_code_files["python"]
        content = python_path.read_text()
        chunks = chunker.chunk_file(python_path, content)

        # Find class with inheritance
        class_chunks = [c for c in chunks if "class" in c.content and "(" in c.content]

        for chunk in class_chunks:
            deps = detector.extract_dependencies(chunk)
            # May or may not have dependencies
            assert isinstance(deps, list)


class TestEmbedder:
    """Tests for embedding backends."""

    @pytest.mark.asyncio
    async def test_mock_embedder(self, mock_embedder):
        """Test mock embedder for testing."""
        await mock_embedder.initialize()

        embedding = await mock_embedder.embed("test text")
        assert embedding.shape == (384,)

        embeddings = await mock_embedder.embed_batch(["text 1", "text 2"])
        assert len(embeddings) == 2
        assert all(e.shape == (384,) for e in embeddings)

    @pytest.mark.asyncio
    async def test_deterministic_embeddings(self, mock_embedder):
        """Test that same text produces same embedding."""
        await mock_embedder.initialize()

        emb1 = await mock_embedder.embed("hello world")
        emb2 = await mock_embedder.embed("hello world")

        import numpy as np
        np.testing.assert_array_equal(emb1, emb2)

    @pytest.mark.asyncio
    async def test_different_texts_different_embeddings(self, mock_embedder):
        """Test that different texts produce different embeddings."""
        await mock_embedder.initialize()

        emb1 = await mock_embedder.embed("hello")
        emb2 = await mock_embedder.embed("world")

        import numpy as np
        assert not np.array_equal(emb1, emb2)


class TestCachedEmbeddingBackend:
    """Tests for cached embedding backend."""

    @pytest.mark.asyncio
    async def test_caching(self, mock_embedder):
        """Test that caching works."""
        from icd.indexing.embedder import CachedEmbeddingBackend

        cached = CachedEmbeddingBackend(mock_embedder, cache_size=100)
        await cached.initialize()

        # First call
        emb1 = await cached.embed("test text")

        # Second call (should be cached)
        emb2 = await cached.embed("test text")

        import numpy as np
        np.testing.assert_array_equal(emb1, emb2)

        stats = cached.get_cache_stats()
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 1

    @pytest.mark.asyncio
    async def test_batch_caching(self, mock_embedder):
        """Test batch caching."""
        from icd.indexing.embedder import CachedEmbeddingBackend

        cached = CachedEmbeddingBackend(mock_embedder, cache_size=100)
        await cached.initialize()

        # First batch
        texts = ["text 1", "text 2", "text 3"]
        embs1 = await cached.embed_batch(texts)

        # Second batch with overlap
        texts2 = ["text 2", "text 4"]
        embs2 = await cached.embed_batch(texts2)

        stats = cached.get_cache_stats()
        assert stats["cache_hits"] >= 1  # "text 2" should be cached
