"""
Maximal Marginal Relevance (MMR) for diversity selection.

Implements:
d* = argmax[λ·score(d,q) - (1-λ)·max_s∈S cos(E_d, E_s)]

This balances relevance with diversity to avoid redundant results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import structlog

if TYPE_CHECKING:
    from icd.retrieval.hybrid import ScoredChunk
    from icd.storage.vector_store import VectorStore

logger = structlog.get_logger(__name__)


class MMRSelector:
    """
    Maximal Marginal Relevance selector for diverse result sets.

    Features:
    - Balances relevance and diversity
    - Configurable lambda parameter
    - Efficient incremental selection
    """

    def __init__(
        self,
        lambda_param: float = 0.7,
        vector_store: "VectorStore | None" = None,
    ) -> None:
        """
        Initialize the MMR selector.

        Args:
            lambda_param: Balance between relevance (1.0) and diversity (0.0).
            vector_store: Vector store for similarity computation.
        """
        self.lambda_param = lambda_param
        self.vector_store = vector_store

    async def select(
        self,
        candidates: list["ScoredChunk"],
        limit: int,
    ) -> list["ScoredChunk"]:
        """
        Select diverse results using MMR.

        Args:
            candidates: Scored candidate chunks.
            limit: Maximum results to select.

        Returns:
            Selected chunks in MMR order.
        """
        if not candidates:
            return []

        if len(candidates) <= limit:
            return candidates

        # Normalize scores to [0, 1]
        max_score = max(c.final_score for c in candidates)
        min_score = min(c.final_score for c in candidates)
        score_range = max_score - min_score if max_score > min_score else 1.0

        normalized_scores = {
            c.chunk.chunk_id: (c.final_score - min_score) / score_range
            for c in candidates
        }

        # Build candidate lookup
        candidate_map = {c.chunk.chunk_id: c for c in candidates}

        # Selected chunks
        selected: list["ScoredChunk"] = []
        selected_ids: set[str] = set()
        remaining_ids = set(candidate_map.keys())

        # Precompute similarities if vector store available
        similarity_cache: dict[tuple[str, str], float] = {}

        while len(selected) < limit and remaining_ids:
            best_mmr_score = float("-inf")
            best_id: str | None = None

            for cand_id in remaining_ids:
                # Relevance score (normalized)
                relevance = normalized_scores[cand_id]

                # Diversity penalty: max similarity to already selected
                max_similarity = 0.0

                if selected and self.vector_store:
                    for sel_chunk in selected:
                        sel_id = sel_chunk.chunk.chunk_id
                        cache_key = (cand_id, sel_id)

                        if cache_key not in similarity_cache:
                            sim = await self._compute_similarity(cand_id, sel_id)
                            similarity_cache[cache_key] = sim
                            similarity_cache[(sel_id, cand_id)] = sim

                        max_similarity = max(
                            max_similarity,
                            similarity_cache[cache_key],
                        )

                # MMR score: λ·relevance - (1-λ)·max_similarity
                mmr_score = (
                    self.lambda_param * relevance
                    - (1 - self.lambda_param) * max_similarity
                )

                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_id = cand_id

            if best_id is None:
                break

            # Select this candidate
            selected.append(candidate_map[best_id])
            selected_ids.add(best_id)
            remaining_ids.remove(best_id)

        logger.debug(
            "MMR selection complete",
            candidates=len(candidates),
            selected=len(selected),
            lambda_param=self.lambda_param,
        )

        return selected

    async def _compute_similarity(
        self,
        chunk_id_a: str,
        chunk_id_b: str,
    ) -> float:
        """Compute cosine similarity between two chunks."""
        if not self.vector_store:
            return 0.0

        sim = await self.vector_store.compute_similarity(chunk_id_a, chunk_id_b)
        return sim if sim is not None else 0.0

    def select_sync(
        self,
        candidates: list["ScoredChunk"],
        limit: int,
        embeddings: dict[str, np.ndarray] | None = None,
    ) -> list["ScoredChunk"]:
        """
        Synchronous MMR selection (for when embeddings are precomputed).

        Args:
            candidates: Scored candidate chunks.
            limit: Maximum results to select.
            embeddings: Precomputed embeddings by chunk ID.

        Returns:
            Selected chunks in MMR order.
        """
        if not candidates:
            return []

        if len(candidates) <= limit:
            return candidates

        embeddings = embeddings or {}

        # Normalize scores
        max_score = max(c.final_score for c in candidates)
        min_score = min(c.final_score for c in candidates)
        score_range = max_score - min_score if max_score > min_score else 1.0

        normalized_scores = {
            c.chunk.chunk_id: (c.final_score - min_score) / score_range
            for c in candidates
        }

        candidate_map = {c.chunk.chunk_id: c for c in candidates}
        selected: list["ScoredChunk"] = []
        selected_ids: set[str] = set()
        remaining_ids = set(candidate_map.keys())

        while len(selected) < limit and remaining_ids:
            best_mmr_score = float("-inf")
            best_id: str | None = None

            for cand_id in remaining_ids:
                relevance = normalized_scores[cand_id]
                max_similarity = 0.0

                if selected and cand_id in embeddings:
                    cand_emb = embeddings[cand_id]

                    for sel_chunk in selected:
                        sel_id = sel_chunk.chunk.chunk_id
                        if sel_id in embeddings:
                            sel_emb = embeddings[sel_id]
                            sim = self._cosine_similarity(cand_emb, sel_emb)
                            max_similarity = max(max_similarity, sim)

                mmr_score = (
                    self.lambda_param * relevance
                    - (1 - self.lambda_param) * max_similarity
                )

                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_id = cand_id

            if best_id is None:
                break

            selected.append(candidate_map[best_id])
            selected_ids.add(best_id)
            remaining_ids.remove(best_id)

        return selected

    def _cosine_similarity(
        self,
        vec_a: np.ndarray,
        vec_b: np.ndarray,
    ) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))


class DiversityMetrics:
    """Utilities for measuring result diversity."""

    @staticmethod
    def compute_average_pairwise_similarity(
        embeddings: list[np.ndarray],
    ) -> float:
        """
        Compute average pairwise cosine similarity.

        Lower values indicate more diverse results.

        Args:
            embeddings: List of embedding vectors.

        Returns:
            Average pairwise similarity.
        """
        if len(embeddings) < 2:
            return 0.0

        total_similarity = 0.0
        count = 0

        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                vec_a = embeddings[i]
                vec_b = embeddings[j]

                dot_product = np.dot(vec_a, vec_b)
                norm_a = np.linalg.norm(vec_a)
                norm_b = np.linalg.norm(vec_b)

                if norm_a > 0 and norm_b > 0:
                    similarity = dot_product / (norm_a * norm_b)
                    total_similarity += similarity
                    count += 1

        return total_similarity / count if count > 0 else 0.0

    @staticmethod
    def compute_coverage_ratio(
        result_paths: list[str],
        all_paths: list[str],
    ) -> float:
        """
        Compute the ratio of unique directories covered.

        Higher values indicate better coverage.

        Args:
            result_paths: File paths in results.
            all_paths: All indexed file paths.

        Returns:
            Coverage ratio.
        """
        if not all_paths:
            return 0.0

        def get_directories(paths: list[str]) -> set[str]:
            dirs = set()
            for p in paths:
                parts = p.split("/")[:-1]  # Remove filename
                for i in range(len(parts)):
                    dirs.add("/".join(parts[: i + 1]))
            return dirs

        result_dirs = get_directories(result_paths)
        all_dirs = get_directories(all_paths)

        if not all_dirs:
            return 0.0

        return len(result_dirs) / len(all_dirs)

    @staticmethod
    def compute_type_diversity(
        symbol_types: list[str | None],
    ) -> float:
        """
        Compute diversity of symbol types in results.

        Args:
            symbol_types: List of symbol types.

        Returns:
            Normalized entropy of symbol type distribution.
        """
        if not symbol_types:
            return 0.0

        # Count types
        type_counts: dict[str, int] = {}
        for t in symbol_types:
            key = t or "unknown"
            type_counts[key] = type_counts.get(key, 0) + 1

        # Compute entropy
        total = len(symbol_types)
        entropy = 0.0

        for count in type_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)

        # Normalize by max possible entropy
        max_entropy = np.log2(len(type_counts)) if len(type_counts) > 1 else 1.0

        return entropy / max_entropy if max_entropy > 0 else 0.0
