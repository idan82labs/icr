"""
Embedding backend abstraction with local ONNX implementation.

Provides:
- Abstract base class for embedding backends
- Local ONNX backend (default, no network required)
- Lazy model loading
- Batch processing support
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.request import urlretrieve

import numpy as np
import structlog

if TYPE_CHECKING:
    from icd.config import Config

logger = structlog.get_logger(__name__)


# Model configurations
MODEL_CONFIGS = {
    "all-MiniLM-L6-v2": {
        "dimension": 384,
        "max_tokens": 256,
        "onnx_url": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx",
        "tokenizer_url": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json",
    },
    "all-mpnet-base-v2": {
        "dimension": 768,
        "max_tokens": 384,
        "onnx_url": "https://huggingface.co/sentence-transformers/all-mpnet-base-v2/resolve/main/onnx/model.onnx",
        "tokenizer_url": "https://huggingface.co/sentence-transformers/all-mpnet-base-v2/resolve/main/tokenizer.json",
    },
}


class EmbeddingBackend(ABC):
    """Abstract base class for embedding backends."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the backend (load models, etc.)."""
        pass

    @abstractmethod
    async def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text.

        Returns:
            Embedding vector as numpy array.
        """
        pass

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts.

        Returns:
            List of embedding vectors.
        """
        pass

    async def close(self) -> None:
        """Cleanup resources."""
        pass


class LocalONNXBackend(EmbeddingBackend):
    """
    Local ONNX embedding backend.

    Features:
    - No network required after initial model download
    - Lazy model loading
    - Efficient batch processing
    - Mean pooling with attention mask
    """

    def __init__(
        self,
        config: "Config",
        model_name: str | None = None,
        model_path: Path | None = None,
    ) -> None:
        """
        Initialize the ONNX backend.

        Args:
            config: ICD configuration.
            model_name: Model name (from MODEL_CONFIGS).
            model_path: Path to custom ONNX model.
        """
        self.config = config
        self.model_name = model_name or config.embedding.model_name
        self.model_path = model_path or config.embedding.model_path
        self._dimension = config.embedding.dimension
        self.batch_size = config.embedding.batch_size
        self.max_tokens = config.embedding.max_tokens
        self.normalize = config.embedding.normalize

        self._session: Any = None
        self._tokenizer: Any = None
        self._initialized = False
        self._lock = asyncio.Lock()

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension

    async def initialize(self) -> None:
        """Initialize the ONNX model and tokenizer."""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            logger.info(
                "Initializing ONNX embedding backend",
                model=self.model_name,
            )

            # Ensure model files exist
            model_dir = self._get_model_dir()
            model_file = model_dir / "model.onnx"
            tokenizer_file = model_dir / "tokenizer.json"

            if not model_file.exists() or not tokenizer_file.exists():
                await self._download_model(model_dir)

            # Load ONNX model
            try:
                import onnxruntime as ort

                # Configure ONNX runtime
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                )
                sess_options.intra_op_num_threads = min(4, os.cpu_count() or 1)

                # Try GPU first, fall back to CPU
                providers = ["CPUExecutionProvider"]
                try:
                    if "CUDAExecutionProvider" in ort.get_available_providers():
                        providers.insert(0, "CUDAExecutionProvider")
                except Exception:
                    pass

                self._session = ort.InferenceSession(
                    str(model_file),
                    sess_options,
                    providers=providers,
                )

                logger.info(
                    "ONNX model loaded",
                    providers=self._session.get_providers(),
                )

            except ImportError:
                raise RuntimeError(
                    "onnxruntime not installed. Run: pip install onnxruntime"
                )

            # Load tokenizer
            try:
                from tokenizers import Tokenizer

                self._tokenizer = Tokenizer.from_file(str(tokenizer_file))
                self._tokenizer.enable_truncation(max_length=self.max_tokens)
                self._tokenizer.enable_padding(
                    length=self.max_tokens,
                    pad_id=0,
                    pad_token="[PAD]",
                )

            except ImportError:
                raise RuntimeError(
                    "tokenizers not installed. Run: pip install tokenizers"
                )

            self._initialized = True
            logger.info("ONNX embedding backend initialized")

    def _get_model_dir(self) -> Path:
        """Get the model directory path."""
        if self.model_path and self.model_path.is_dir():
            return self.model_path

        # Use default location in data directory
        model_dir = self.config.absolute_data_dir / "models" / self.model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    async def _download_model(self, model_dir: Path) -> None:
        """Download model files if not present."""
        if self.model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {self.model_name}")

        config = MODEL_CONFIGS[self.model_name]
        model_file = model_dir / "model.onnx"
        tokenizer_file = model_dir / "tokenizer.json"

        # Check if network is enabled
        if not self.config.network.enabled:
            raise RuntimeError(
                f"Model files not found and network disabled. "
                f"Please download model files to {model_dir} or enable network access."
            )

        logger.info("Downloading model files", model=self.model_name)

        # Download model
        if not model_file.exists():
            logger.info("Downloading ONNX model...")
            await asyncio.get_event_loop().run_in_executor(
                None,
                urlretrieve,
                config["onnx_url"],
                str(model_file),
            )

        # Download tokenizer
        if not tokenizer_file.exists():
            logger.info("Downloading tokenizer...")
            await asyncio.get_event_loop().run_in_executor(
                None,
                urlretrieve,
                config["tokenizer_url"],
                str(tokenizer_file),
            )

        logger.info("Model files downloaded")

    async def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        if not self._initialized:
            await self.initialize()

        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for multiple texts."""
        if not self._initialized:
            await self.initialize()

        if not texts:
            return []

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            embeddings = await self._embed_batch_internal(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    async def _embed_batch_internal(self, texts: list[str]) -> list[np.ndarray]:
        """Internal batch embedding implementation."""
        # Run in executor to avoid blocking
        return await asyncio.get_event_loop().run_in_executor(
            None,
            self._embed_sync,
            texts,
        )

    def _embed_sync(self, texts: list[str]) -> list[np.ndarray]:
        """Synchronous embedding implementation."""
        # Tokenize
        encodings = self._tokenizer.encode_batch(texts)

        # Prepare inputs
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array(
            [e.attention_mask for e in encodings], dtype=np.int64
        )
        token_type_ids = np.zeros_like(input_ids)

        # Run inference
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

        # Get model outputs
        outputs = self._session.run(None, inputs)

        # Get last hidden state (first output)
        last_hidden_state = outputs[0]

        # Mean pooling with attention mask
        embeddings = self._mean_pooling(last_hidden_state, attention_mask)

        # Normalize if configured
        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embeddings = embeddings / norms

        return [embeddings[i] for i in range(len(texts))]

    def _mean_pooling(
        self,
        last_hidden_state: np.ndarray,
        attention_mask: np.ndarray,
    ) -> np.ndarray:
        """Apply mean pooling with attention mask."""
        # Expand attention mask
        mask_expanded = np.expand_dims(attention_mask, -1)
        mask_expanded = np.broadcast_to(mask_expanded, last_hidden_state.shape)

        # Sum embeddings
        sum_embeddings = np.sum(last_hidden_state * mask_expanded, axis=1)

        # Sum mask
        sum_mask = np.clip(np.sum(mask_expanded, axis=1), a_min=1e-9, a_max=None)

        # Mean
        return sum_embeddings / sum_mask

    async def close(self) -> None:
        """Cleanup ONNX session."""
        self._session = None
        self._tokenizer = None
        self._initialized = False


class RemoteEmbeddingBackend(EmbeddingBackend):
    """
    Remote embedding backend for API-based embeddings.

    Supports OpenAI and Anthropic embedding APIs.
    """

    def __init__(
        self,
        config: "Config",
        provider: str = "openai",
        model: str | None = None,
    ) -> None:
        """
        Initialize the remote backend.

        Args:
            config: ICD configuration.
            provider: API provider (openai, anthropic).
            model: Model name for the provider.
        """
        self.config = config
        self.provider = provider
        self.model = model or self._default_model()
        self._dimension = config.embedding.dimension
        self.batch_size = config.embedding.batch_size

        self._client: Any = None
        self._initialized = False

    def _default_model(self) -> str:
        """Get default model for provider."""
        if self.provider == "openai":
            return "text-embedding-3-small"
        elif self.provider == "anthropic":
            return "claude-3-haiku-20240307"
        return "unknown"

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension

    async def initialize(self) -> None:
        """Initialize the API client."""
        if self._initialized:
            return

        if not self.config.network.enabled:
            raise RuntimeError("Network access disabled for remote embedding backend")

        if self.provider == "openai":
            try:
                from openai import AsyncOpenAI

                self._client = AsyncOpenAI(
                    api_key=self.config.network.api_key,
                    base_url=self.config.network.api_base_url,
                    timeout=self.config.network.timeout_seconds,
                )
            except ImportError:
                raise RuntimeError("openai package not installed")

        elif self.provider == "anthropic":
            # Anthropic doesn't have a native embedding API yet
            # This is a placeholder for future support
            raise NotImplementedError("Anthropic embeddings not yet supported")

        self._initialized = True
        logger.info(
            "Remote embedding backend initialized",
            provider=self.provider,
            model=self.model,
        )

    async def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for multiple texts."""
        if not self._initialized:
            await self.initialize()

        if not texts:
            return []

        if self.provider == "openai":
            return await self._embed_openai(texts)

        raise NotImplementedError(f"Provider {self.provider} not implemented")

    async def _embed_openai(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings using OpenAI API."""
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            response = await self._client.embeddings.create(
                input=batch,
                model=self.model,
            )

            for item in response.data:
                embedding = np.array(item.embedding, dtype=np.float32)
                all_embeddings.append(embedding)

        return all_embeddings

    async def close(self) -> None:
        """Cleanup client."""
        self._client = None
        self._initialized = False


class CachedEmbeddingBackend(EmbeddingBackend):
    """
    Wrapper that adds caching to any embedding backend.

    Uses content-based hashing for cache keys.
    """

    def __init__(
        self,
        backend: EmbeddingBackend,
        cache_size: int = 10000,
    ) -> None:
        """
        Initialize the cached backend.

        Args:
            backend: Underlying embedding backend.
            cache_size: Maximum cache entries.
        """
        self._backend = backend
        self._cache: dict[str, np.ndarray] = {}
        self._cache_size = cache_size
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._backend.dimension

    async def initialize(self) -> None:
        """Initialize the underlying backend."""
        await self._backend.initialize()

    def _cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    async def embed(self, text: str) -> np.ndarray:
        """Generate embedding with caching."""
        key = self._cache_key(text)

        if key in self._cache:
            self._cache_hits += 1
            return self._cache[key].copy()

        self._cache_misses += 1
        embedding = await self._backend.embed(text)

        # Add to cache
        if len(self._cache) >= self._cache_size:
            # Simple eviction: remove first item
            first_key = next(iter(self._cache))
            del self._cache[first_key]

        self._cache[key] = embedding.copy()
        return embedding

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings with caching."""
        results: list[np.ndarray | None] = [None] * len(texts)
        uncached_texts: list[tuple[int, str]] = []

        # Check cache
        for i, text in enumerate(texts):
            key = self._cache_key(text)
            if key in self._cache:
                self._cache_hits += 1
                results[i] = self._cache[key].copy()
            else:
                self._cache_misses += 1
                uncached_texts.append((i, text))

        # Embed uncached texts
        if uncached_texts:
            indices, texts_to_embed = zip(*uncached_texts)
            embeddings = await self._backend.embed_batch(list(texts_to_embed))

            for idx, embedding in zip(indices, embeddings):
                results[idx] = embedding
                key = self._cache_key(texts[idx])

                if len(self._cache) >= self._cache_size:
                    first_key = next(iter(self._cache))
                    del self._cache[first_key]

                self._cache[key] = embedding.copy()

        return [r for r in results if r is not None]

    async def close(self) -> None:
        """Cleanup."""
        await self._backend.close()
        self._cache.clear()

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0

        return {
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
        }


def create_embedder(config: "Config") -> EmbeddingBackend:
    """
    Create an embedding backend based on configuration.

    Args:
        config: ICD configuration.

    Returns:
        Configured EmbeddingBackend instance.
    """
    from icd.config import EmbeddingBackend as BackendType

    backend: EmbeddingBackend

    if config.embedding.backend == BackendType.LOCAL_ONNX:
        backend = LocalONNXBackend(config)
    elif config.embedding.backend == BackendType.OPENAI:
        backend = RemoteEmbeddingBackend(config, provider="openai")
    elif config.embedding.backend == BackendType.ANTHROPIC:
        backend = RemoteEmbeddingBackend(config, provider="anthropic")
    else:
        # Default to local ONNX
        backend = LocalONNXBackend(config)

    # Wrap with caching
    return CachedEmbeddingBackend(backend)
