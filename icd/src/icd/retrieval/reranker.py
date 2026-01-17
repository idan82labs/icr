"""
Cross-encoder reranking for improved retrieval precision.

Reranks initial retrieval results using a cross-encoder model that
jointly encodes query and document for more accurate relevance scoring.

Research shows +5-10% precision improvement over bi-encoder only retrieval.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog

if TYPE_CHECKING:
    from icd.config import Config
    from icd.retrieval.hybrid import ScoredChunk

logger = structlog.get_logger(__name__)


# Supported cross-encoder models
RERANKER_MODELS = {
    # Fast, good quality (recommended default)
    "cross-encoder/ms-marco-MiniLM-L-6-v2": {
        "max_length": 512,
        "score_range": (0, 1),  # Already normalized
    },
    # Higher quality, slower
    "cross-encoder/ms-marco-MiniLM-L-12-v2": {
        "max_length": 512,
        "score_range": (0, 1),
    },
    # Code-specific (experimental)
    "BAAI/bge-reranker-base": {
        "max_length": 512,
        "score_range": (-10, 10),  # Needs sigmoid normalization
    },
    "BAAI/bge-reranker-large": {
        "max_length": 512,
        "score_range": (-10, 10),
    },
}


class CrossEncoderReranker:
    """
    Cross-encoder reranker for retrieval results.

    Unlike bi-encoders (which encode query and document separately),
    cross-encoders jointly encode the pair for more accurate relevance.

    Trade-off: O(n) inference calls vs O(1) for bi-encoder.
    Typically used on top-k results from bi-encoder retrieval.
    """

    def __init__(
        self,
        config: "Config",
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        """
        Initialize the cross-encoder reranker.

        Args:
            config: ICD configuration.
            model_name: Cross-encoder model name from HuggingFace.
        """
        self.config = config
        self.model_name = model_name
        self._model_config = RERANKER_MODELS.get(model_name, {})
        self._max_length = self._model_config.get("max_length", 512)
        self._score_range = self._model_config.get("score_range", (0, 1))

        self._model: Any = None
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the cross-encoder model."""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            logger.info(
                "Initializing cross-encoder reranker",
                model=self.model_name,
            )

            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(
                    self.model_name,
                    max_length=self._max_length,
                )

                logger.info(
                    "Cross-encoder loaded",
                    model=self.model_name,
                    max_length=self._max_length,
                )

            except ImportError:
                raise RuntimeError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )

            self._initialized = True

    async def rerank(
        self,
        query: str,
        chunks: list["ScoredChunk"],
        top_k: int | None = None,
    ) -> list["ScoredChunk"]:
        """
        Rerank chunks using cross-encoder.

        Args:
            query: The search query.
            chunks: List of scored chunks from initial retrieval.
            top_k: Number of results to return (None = all).

        Returns:
            Reranked list of ScoredChunks with updated final_score.
        """
        if not self._initialized:
            await self.initialize()

        if not chunks:
            return []

        # Prepare pairs for cross-encoder
        pairs = [(query, chunk.chunk.content) for chunk in chunks]

        # Run cross-encoder scoring
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None,
            self._score_pairs,
            pairs,
        )

        # Normalize scores to [0, 1]
        normalized_scores = self._normalize_scores(scores)

        # Update chunks with cross-encoder scores
        reranked = []
        for chunk, ce_score in zip(chunks, normalized_scores):
            # Blend cross-encoder score with original score
            # CE score is more accurate, so weight it heavily
            blended_score = 0.7 * ce_score + 0.3 * chunk.final_score

            # Create updated chunk with new score
            from dataclasses import replace
            updated = replace(
                chunk,
                final_score=blended_score,
            )

            # Store cross-encoder score in metadata for analysis
            if hasattr(updated.chunk, 'metadata'):
                updated.chunk.metadata['cross_encoder_score'] = float(ce_score)

            reranked.append(updated)

        # Sort by new score
        reranked.sort(key=lambda x: x.final_score, reverse=True)

        # Return top_k if specified
        if top_k is not None:
            reranked = reranked[:top_k]

        logger.debug(
            "Reranking complete",
            input_count=len(chunks),
            output_count=len(reranked),
        )

        return reranked

    def _score_pairs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Score query-document pairs with cross-encoder."""
        scores = self._model.predict(
            pairs,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.array(scores)

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range."""
        min_score, max_score = self._score_range

        if min_score == 0 and max_score == 1:
            # Already normalized
            return np.clip(scores, 0, 1)

        # Apply sigmoid for unbounded scores (like BGE reranker)
        normalized = 1 / (1 + np.exp(-scores))
        return normalized

    async def close(self) -> None:
        """Cleanup resources."""
        self._model = None
        self._initialized = False


def create_reranker(config: "Config") -> CrossEncoderReranker | None:
    """
    Create a reranker if enabled in config.

    Args:
        config: ICD configuration.

    Returns:
        CrossEncoderReranker instance or None if disabled.
    """
    # Check if reranking is enabled
    if not getattr(config.retrieval, 'reranker_enabled', False):
        return None

    model_name = getattr(
        config.retrieval,
        'reranker_model',
        'cross-encoder/ms-marco-MiniLM-L-6-v2'
    )

    return CrossEncoderReranker(config, model_name=model_name)
