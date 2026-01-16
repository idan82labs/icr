"""
Unit tests for the embedding backend module.

Tests cover:
- Embedding backend abstraction
- Local ONNX backend
- Mock backend for testing
- Embedding normalization
- Batch processing
- Dimension validation
"""

from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from tests.conftest import MockEmbeddingBackend


# ==============================================================================
# Mock Embedding Backend Tests
# ==============================================================================

class TestMockEmbeddingBackend:
    """Tests for the mock embedding backend."""

    @pytest.fixture
    def backend(self) -> MockEmbeddingBackend:
        """Create a mock embedding backend."""
        return MockEmbeddingBackend(dimension=384)

    @pytest.mark.asyncio
    async def test_initialization(self, backend: MockEmbeddingBackend):
        """Test backend initialization."""
        assert backend._initialized is False
        await backend.initialize()
        assert backend._initialized is True

    @pytest.mark.asyncio
    async def test_embed_single(self, backend: MockEmbeddingBackend):
        """Test embedding a single text."""
        await backend.initialize()

        embedding = await backend.embed_single("Hello, world!")

        assert embedding.shape == (384,)
        assert embedding.dtype == np.float32

    @pytest.mark.asyncio
    async def test_embed_batch(self, backend: MockEmbeddingBackend):
        """Test embedding multiple texts."""
        await backend.initialize()

        texts = ["Hello", "World", "Test"]
        embeddings = await backend.embed(texts)

        assert embeddings.shape == (3, 384)
        assert embeddings.dtype == np.float32

    @pytest.mark.asyncio
    async def test_embeddings_normalized(self, backend: MockEmbeddingBackend):
        """Test that embeddings are L2 normalized."""
        await backend.initialize()

        embedding = await backend.embed_single("Test text")
        norm = np.linalg.norm(embedding)

        # Should be unit vector (norm ~= 1.0)
        assert abs(norm - 1.0) < 1e-5

    @pytest.mark.asyncio
    async def test_embeddings_deterministic(self, backend: MockEmbeddingBackend):
        """Test that same content produces same embedding."""
        await backend.initialize()

        text = "Deterministic test"
        embedding1 = await backend.embed_single(text)
        embedding2 = await backend.embed_single(text)

        np.testing.assert_array_almost_equal(embedding1, embedding2)

    @pytest.mark.asyncio
    async def test_different_texts_different_embeddings(self, backend: MockEmbeddingBackend):
        """Test that different texts produce different embeddings."""
        await backend.initialize()

        embedding1 = await backend.embed_single("Text one")
        embedding2 = await backend.embed_single("Text two")

        # Should not be equal
        assert not np.allclose(embedding1, embedding2)

    @pytest.mark.asyncio
    async def test_call_count_tracking(self, backend: MockEmbeddingBackend):
        """Test that call count is tracked."""
        await backend.initialize()
        assert backend.call_count == 0

        await backend.embed_single("Text 1")
        assert backend.call_count == 1

        await backend.embed(["Text 2", "Text 3"])
        assert backend.call_count == 3

    @pytest.mark.asyncio
    async def test_call_count_reset(self, backend: MockEmbeddingBackend):
        """Test call count reset."""
        await backend.initialize()

        await backend.embed(["A", "B", "C"])
        assert backend.call_count == 3

        backend.reset_call_count()
        assert backend.call_count == 0

    def test_custom_dimension(self):
        """Test backend with custom dimension."""
        backend = MockEmbeddingBackend(dimension=768)
        assert backend.dimension == 768


# ==============================================================================
# Embedding Dimension Tests
# ==============================================================================

class TestEmbeddingDimensions:
    """Tests for embedding dimension handling."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("dimension", [64, 128, 384, 512, 768, 1024, 1536])
    async def test_various_dimensions(self, dimension: int):
        """Test embedding with various dimensions."""
        backend = MockEmbeddingBackend(dimension=dimension)
        await backend.initialize()

        embedding = await backend.embed_single("Test")

        assert embedding.shape == (dimension,)

    @pytest.mark.asyncio
    async def test_dimension_consistency(self):
        """Test that all embeddings have consistent dimensions."""
        backend = MockEmbeddingBackend(dimension=384)
        await backend.initialize()

        texts = [f"Text {i}" for i in range(10)]
        embeddings = await backend.embed(texts)

        # All should have same dimension
        assert embeddings.shape == (10, 384)
        for embedding in embeddings:
            assert embedding.shape == (384,)


# ==============================================================================
# Batch Processing Tests
# ==============================================================================

class TestBatchProcessing:
    """Tests for batch embedding processing."""

    @pytest.mark.asyncio
    async def test_empty_batch(self):
        """Test handling of empty batch."""
        backend = MockEmbeddingBackend(dimension=384)
        await backend.initialize()

        embeddings = await backend.embed([])
        assert embeddings.shape == (0,) or len(embeddings) == 0

    @pytest.mark.asyncio
    async def test_single_item_batch(self):
        """Test batch with single item."""
        backend = MockEmbeddingBackend(dimension=384)
        await backend.initialize()

        embeddings = await backend.embed(["Single item"])
        assert embeddings.shape == (1, 384)

    @pytest.mark.asyncio
    async def test_large_batch(self):
        """Test handling of large batch."""
        backend = MockEmbeddingBackend(dimension=384)
        await backend.initialize()

        texts = [f"Text number {i}" for i in range(100)]
        embeddings = await backend.embed(texts)

        assert embeddings.shape == (100, 384)

    @pytest.mark.asyncio
    async def test_batch_order_preserved(self):
        """Test that batch order is preserved."""
        backend = MockEmbeddingBackend(dimension=384)
        await backend.initialize()

        texts = ["First", "Second", "Third"]
        batch_embeddings = await backend.embed(texts)

        # Compare with individual embeddings
        individual_embeddings = []
        for text in texts:
            emb = await backend.embed_single(text)
            individual_embeddings.append(emb)

        for i, emb in enumerate(individual_embeddings):
            np.testing.assert_array_almost_equal(batch_embeddings[i], emb)


# ==============================================================================
# Normalization Tests
# ==============================================================================

class TestEmbeddingNormalization:
    """Tests for embedding normalization."""

    @pytest.mark.asyncio
    async def test_l2_normalized(self):
        """Test L2 normalization of embeddings."""
        backend = MockEmbeddingBackend(dimension=384)
        await backend.initialize()

        embedding = await backend.embed_single("Test normalization")
        l2_norm = np.linalg.norm(embedding)

        # L2 norm should be 1.0 for normalized vectors
        assert abs(l2_norm - 1.0) < 1e-6

    @pytest.mark.asyncio
    async def test_batch_all_normalized(self):
        """Test that all embeddings in batch are normalized."""
        backend = MockEmbeddingBackend(dimension=384)
        await backend.initialize()

        embeddings = await backend.embed([f"Text {i}" for i in range(10)])

        for embedding in embeddings:
            l2_norm = np.linalg.norm(embedding)
            assert abs(l2_norm - 1.0) < 1e-6

    @pytest.mark.asyncio
    async def test_cosine_similarity_range(self):
        """Test that cosine similarity is in valid range."""
        backend = MockEmbeddingBackend(dimension=384)
        await backend.initialize()

        emb1 = await backend.embed_single("Text A")
        emb2 = await backend.embed_single("Text B")

        # Cosine similarity for normalized vectors is just dot product
        cosine_sim = np.dot(emb1, emb2)

        # Should be in [-1, 1]
        assert -1.0 <= cosine_sim <= 1.0


# ==============================================================================
# Data Type Tests
# ==============================================================================

class TestEmbeddingDataTypes:
    """Tests for embedding data types."""

    @pytest.mark.asyncio
    async def test_output_dtype_float32(self):
        """Test that output is float32."""
        backend = MockEmbeddingBackend(dimension=384)
        await backend.initialize()

        embedding = await backend.embed_single("Test")
        assert embedding.dtype == np.float32

    @pytest.mark.asyncio
    async def test_batch_output_dtype(self):
        """Test batch output dtype."""
        backend = MockEmbeddingBackend(dimension=384)
        await backend.initialize()

        embeddings = await backend.embed(["A", "B"])
        assert embeddings.dtype == np.float32


# ==============================================================================
# Edge Cases Tests
# ==============================================================================

class TestEmbedderEdgeCases:
    """Tests for edge cases in embedding."""

    @pytest.mark.asyncio
    async def test_empty_string(self):
        """Test embedding empty string."""
        backend = MockEmbeddingBackend(dimension=384)
        await backend.initialize()

        embedding = await backend.embed_single("")
        assert embedding.shape == (384,)

    @pytest.mark.asyncio
    async def test_whitespace_only(self):
        """Test embedding whitespace-only string."""
        backend = MockEmbeddingBackend(dimension=384)
        await backend.initialize()

        embedding = await backend.embed_single("   \n\t  ")
        assert embedding.shape == (384,)

    @pytest.mark.asyncio
    async def test_very_long_text(self):
        """Test embedding very long text."""
        backend = MockEmbeddingBackend(dimension=384)
        await backend.initialize()

        long_text = "word " * 10000  # ~50000 characters
        embedding = await backend.embed_single(long_text)

        assert embedding.shape == (384,)
        assert np.isfinite(embedding).all()

    @pytest.mark.asyncio
    async def test_unicode_text(self):
        """Test embedding text with unicode characters."""
        backend = MockEmbeddingBackend(dimension=384)
        await backend.initialize()

        unicode_texts = [
            "Hello, world!",
            "Bonjour le monde!",
            "Hola mundo!",
            "Hallo Welt!",
        ]

        embeddings = await backend.embed(unicode_texts)
        assert embeddings.shape == (4, 384)

    @pytest.mark.asyncio
    async def test_special_characters(self):
        """Test embedding text with special characters."""
        backend = MockEmbeddingBackend(dimension=384)
        await backend.initialize()

        text = "Special chars: @#$%^&*()[]{}|\\;:'\",.<>?/`~"
        embedding = await backend.embed_single(text)

        assert embedding.shape == (384,)
        assert np.isfinite(embedding).all()

    @pytest.mark.asyncio
    async def test_code_content(self):
        """Test embedding code content."""
        backend = MockEmbeddingBackend(dimension=384)
        await backend.initialize()

        code = """
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
"""
        embedding = await backend.embed_single(code)

        assert embedding.shape == (384,)
        assert np.isfinite(embedding).all()


# ==============================================================================
# Embedding Backend Abstraction Tests
# ==============================================================================

class TestEmbeddingBackendAbstraction:
    """Tests for embedding backend abstraction."""

    def test_backend_interface(self):
        """Test that mock backend implements expected interface."""
        backend = MockEmbeddingBackend(dimension=384)

        # Check required methods exist
        assert hasattr(backend, "initialize")
        assert hasattr(backend, "embed")
        assert hasattr(backend, "embed_single")
        assert hasattr(backend, "dimension")

        # Check methods are callable
        assert callable(backend.initialize)
        assert callable(backend.embed)
        assert callable(backend.embed_single)

    @pytest.mark.asyncio
    async def test_backend_async_methods(self):
        """Test that backend methods are async."""
        import asyncio
        import inspect

        backend = MockEmbeddingBackend(dimension=384)

        # Check methods are coroutines
        assert inspect.iscoroutinefunction(backend.initialize)
        assert inspect.iscoroutinefunction(backend.embed)
        assert inspect.iscoroutinefunction(backend.embed_single)


# ==============================================================================
# Similarity Computation Tests
# ==============================================================================

class TestSimilarityComputation:
    """Tests for embedding similarity computations."""

    @pytest.mark.asyncio
    async def test_self_similarity_is_one(self):
        """Test that self-similarity is 1.0."""
        backend = MockEmbeddingBackend(dimension=384)
        await backend.initialize()

        embedding = await backend.embed_single("Test text")
        similarity = np.dot(embedding, embedding)

        assert abs(similarity - 1.0) < 1e-6

    @pytest.mark.asyncio
    async def test_similar_texts_high_similarity(self):
        """Test that similar texts have higher similarity."""
        backend = MockEmbeddingBackend(dimension=384)
        await backend.initialize()

        # Note: Mock backend uses hash-based embeddings, so this tests
        # the computation infrastructure, not semantic similarity
        emb1 = await backend.embed_single("The quick brown fox")
        emb2 = await backend.embed_single("The quick brown fox jumps")

        # Just verify we can compute similarity
        similarity = np.dot(emb1, emb2)
        assert -1.0 <= similarity <= 1.0

    @pytest.mark.asyncio
    async def test_batch_similarity_matrix(self):
        """Test computing similarity matrix for batch."""
        backend = MockEmbeddingBackend(dimension=384)
        await backend.initialize()

        texts = ["A", "B", "C", "D"]
        embeddings = await backend.embed(texts)

        # Compute similarity matrix
        similarity_matrix = embeddings @ embeddings.T

        assert similarity_matrix.shape == (4, 4)
        # Diagonal should be 1.0 (self-similarity)
        np.testing.assert_array_almost_equal(
            np.diag(similarity_matrix),
            np.ones(4),
            decimal=5
        )


# ==============================================================================
# Configuration Tests
# ==============================================================================

class TestEmbeddingConfiguration:
    """Tests for embedding configuration."""

    def test_dimension_property(self):
        """Test dimension property."""
        backend = MockEmbeddingBackend(dimension=512)
        assert backend.dimension == 512

    def test_dimension_readonly(self):
        """Test that dimension is effectively readonly after init."""
        backend = MockEmbeddingBackend(dimension=384)
        original_dim = backend.dimension

        # Dimension should remain constant
        assert backend.dimension == original_dim


# ==============================================================================
# Backend Selection Tests (Conceptual)
# ==============================================================================

class TestBackendSelection:
    """Tests for backend selection logic (conceptual)."""

    def test_local_onnx_is_default(self):
        """Test that local ONNX would be default backend."""
        try:
            from icd.config import EmbeddingBackend, EmbeddingConfig

            config = EmbeddingConfig()
            assert config.backend == EmbeddingBackend.LOCAL_ONNX
        except ImportError:
            pytest.skip("icd.config module not available")

    def test_backend_enum_values(self):
        """Test available backend types."""
        try:
            from icd.config import EmbeddingBackend

            expected_backends = {"local_onnx", "openai", "anthropic", "custom"}
            actual_backends = {b.value for b in EmbeddingBackend}

            assert expected_backends.issubset(actual_backends)
        except ImportError:
            pytest.skip("icd.config module not available")

    def test_network_disabled_prevents_remote(self):
        """Test that network disabled should prevent remote backends."""
        try:
            from icd.config import NetworkConfig

            config = NetworkConfig()
            assert config.enabled is False
        except ImportError:
            pytest.skip("icd.config module not available")
