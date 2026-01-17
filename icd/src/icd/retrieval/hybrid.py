"""
Hybrid retrieval combining semantic and lexical search.

Implements the hybrid scoring formula:
score(d,q) = w_e·cos(E_d, E_q) + w_b·BM25(d,q) + w_r·exp(-dt/τ) + w_c·𝟙_contract + w_f·𝟙_focus + w_p·𝟙_pinned
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog

if TYPE_CHECKING:
    from icd.config import Config
    from icd.indexing.embedder import EmbeddingBackend
    from icd.storage.contract_store import ContractStore
    from icd.storage.memory_store import MemoryStore
    from icd.storage.sqlite_store import SQLiteStore
    from icd.storage.vector_store import VectorStore

logger = structlog.get_logger(__name__)


@dataclass
class Chunk:
    """Chunk data for retrieval results."""

    chunk_id: str
    file_path: str
    content: str
    start_line: int
    end_line: int
    symbol_name: str | None
    symbol_type: str | None
    language: str
    token_count: int
    is_contract: bool = False
    is_pinned: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredChunk:
    """A chunk with its computed score components."""

    chunk: Chunk
    semantic_score: float
    bm25_score: float
    recency_score: float
    contract_score: float
    focus_score: float
    pinned_score: float
    final_score: float


@dataclass
class RetrievalResult:
    """Result from hybrid retrieval."""

    chunks: list[Chunk]
    scores: list[float]
    entropy: float
    query: str
    metadata: dict[str, Any] = field(default_factory=dict)


class HybridRetriever:
    """
    Hybrid retrieval system combining semantic and lexical search.

    Features:
    - Vector similarity search (semantic)
    - BM25 lexical search
    - Recency boosting
    - Contract boosting
    - Focus path boosting
    - Pinned chunk boosting
    - MMR diversity selection
    """

    def __init__(
        self,
        config: "Config",
        sqlite_store: "SQLiteStore",
        vector_store: "VectorStore",
        embedder: "EmbeddingBackend",
        contract_store: "ContractStore",
        memory_store: "MemoryStore",
    ) -> None:
        """
        Initialize the hybrid retriever.

        Args:
            config: ICD configuration.
            sqlite_store: SQLite store for metadata and BM25.
            vector_store: Vector store for semantic search.
            embedder: Embedding backend.
            contract_store: Contract store.
            memory_store: Memory store for pinned chunks.
        """
        self.config = config
        self.sqlite_store = sqlite_store
        self.vector_store = vector_store
        self.embedder = embedder
        self.contract_store = contract_store
        self.memory_store = memory_store

        # Scoring weights from config
        self.w_e = config.retrieval.weight_embedding
        self.w_b = config.retrieval.weight_bm25
        self.w_r = config.retrieval.weight_recency
        self.w_c = config.retrieval.weight_contract
        self.w_f = config.retrieval.weight_focus
        self.w_p = config.retrieval.weight_pinned

        # Recency decay
        self.tau_days = config.retrieval.recency_tau_days

        # Retrieval parameters
        self.initial_candidates = config.retrieval.initial_candidates
        self.final_results = config.retrieval.final_results
        self.mmr_lambda = config.retrieval.mmr_lambda

        # Entropy settings
        self.entropy_temperature = config.retrieval.entropy_temperature

        # Reranker (optional, improves precision by +5-10%)
        self._reranker = None
        if config.retrieval.reranker_enabled:
            from icd.retrieval.reranker import CrossEncoderReranker
            self._reranker = CrossEncoderReranker(
                config,
                model_name=config.retrieval.reranker_model,
            )

    async def retrieve(
        self,
        query: str,
        limit: int | None = None,
        focus_paths: list[Path] | None = None,
        include_contracts: bool = True,
        include_pinned: bool = True,
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: Natural language query.
            limit: Maximum results to return.
            focus_paths: Paths to prioritize.
            include_contracts: Include contract chunks.
            include_pinned: Always include pinned chunks.

        Returns:
            RetrievalResult with ranked chunks.
        """
        limit = limit or self.final_results
        focus_path_strs = [str(p) for p in (focus_paths or [])]

        logger.debug(
            "Starting hybrid retrieval",
            query=query[:100],
            limit=limit,
            focus_paths=focus_path_strs[:5],
        )

        # Step 1: Get query embedding
        query_embedding = await self.embedder.embed(query)

        # Step 2: Get pinned chunks (always included)
        pinned_ids: set[str] = set()
        if include_pinned:
            pinned_ids = set(await self.memory_store.get_pinned_chunks())

        # Step 3: Get contract chunk IDs
        contract_ids: set[str] = set()
        if include_contracts:
            contract_ids = set(await self.contract_store.get_all_contract_chunk_ids())

        # Step 4: Semantic search
        semantic_results = await self.vector_store.search(
            query_embedding,
            k=self.initial_candidates,
        )

        # Step 5: BM25 search
        bm25_results = await self.sqlite_store.search_bm25(
            query,
            limit=self.initial_candidates,
        )

        # Step 6: Combine candidates
        candidate_ids = set()
        for r in semantic_results:
            candidate_ids.add(r.chunk_id)
        for r in bm25_results:
            candidate_ids.add(r.chunk_id)

        # Always include pinned chunks
        candidate_ids.update(pinned_ids)

        # Step 7: Score all candidates
        scored_chunks = await self._score_candidates(
            candidate_ids=candidate_ids,
            query_embedding=query_embedding,
            semantic_results={r.chunk_id: r.score for r in semantic_results},
            bm25_results={r.chunk_id: r.score for r in bm25_results},
            contract_ids=contract_ids,
            pinned_ids=pinned_ids,
            focus_paths=focus_path_strs,
        )

        # Step 7.5: Optional cross-encoder reranking (improves precision +5-10%)
        if self._reranker is not None:
            logger.debug("Applying cross-encoder reranking", count=len(scored_chunks))
            scored_chunks = await self._reranker.rerank(
                query=query,
                chunks=scored_chunks,
            )

        # Step 8: Apply MMR for diversity
        from icd.retrieval.mmr import MMRSelector

        mmr = MMRSelector(
            lambda_param=self.mmr_lambda,
            vector_store=self.vector_store,
        )

        selected = await mmr.select(
            candidates=scored_chunks,
            limit=limit,
        )

        # Step 9: Compute retrieval entropy
        from icd.retrieval.entropy import EntropyCalculator

        entropy_calc = EntropyCalculator(temperature=self.entropy_temperature)
        scores = [s.final_score for s in selected]
        entropy = entropy_calc.compute_entropy(scores)

        # Step 10: Build result
        chunks = [s.chunk for s in selected]
        final_scores = [s.final_score for s in selected]

        logger.debug(
            "Hybrid retrieval complete",
            candidates=len(candidate_ids),
            selected=len(chunks),
            entropy=entropy,
        )

        return RetrievalResult(
            chunks=chunks,
            scores=final_scores,
            entropy=entropy,
            query=query,
            metadata={
                "candidates": len(candidate_ids),
                "semantic_hits": len(semantic_results),
                "bm25_hits": len(bm25_results),
                "pinned_included": len(pinned_ids & set(c.chunk_id for c in chunks)),
            },
        )

    async def _score_candidates(
        self,
        candidate_ids: set[str],
        query_embedding: np.ndarray,
        semantic_results: dict[str, float],
        bm25_results: dict[str, float],
        contract_ids: set[str],
        pinned_ids: set[str],
        focus_paths: list[str],
    ) -> list[ScoredChunk]:
        """
        Score all candidate chunks.

        Implements:
        score(d,q) = w_e·cos(E_d, E_q) + w_b·BM25(d,q) + w_r·exp(-dt/τ) + w_c·𝟙_contract + w_f·𝟙_focus + w_p·𝟙_pinned
        """
        scored_chunks = []
        now = datetime.utcnow()

        # Normalize BM25 scores
        max_bm25 = max(bm25_results.values()) if bm25_results else 1.0
        if max_bm25 == 0:
            max_bm25 = 1.0

        for chunk_id in candidate_ids:
            # Get chunk metadata
            metadata = await self.sqlite_store.get_chunk(chunk_id)
            if not metadata:
                continue

            content = await self.sqlite_store.get_chunk_content(chunk_id)
            if not content:
                continue

            chunk = Chunk(
                chunk_id=chunk_id,
                file_path=metadata.file_path,
                content=content,
                start_line=metadata.start_line,
                end_line=metadata.end_line,
                symbol_name=metadata.symbol_name,
                symbol_type=metadata.symbol_type,
                language=metadata.language,
                token_count=metadata.token_count,
                is_contract=chunk_id in contract_ids,
                is_pinned=chunk_id in pinned_ids,
            )

            # Semantic score (already normalized 0-1)
            semantic_score = semantic_results.get(chunk_id, 0.0)

            # BM25 score (normalize)
            bm25_score = bm25_results.get(chunk_id, 0.0) / max_bm25

            # Recency score: exp(-dt/τ)
            days_old = (now - metadata.updated_at).total_seconds() / 86400
            recency_score = math.exp(-days_old / self.tau_days)

            # Contract indicator
            contract_score = 1.0 if chunk.is_contract else 0.0

            # Focus indicator
            focus_score = 0.0
            for focus_path in focus_paths:
                if chunk.file_path.startswith(focus_path) or focus_path in chunk.file_path:
                    focus_score = 1.0
                    break

            # Pinned indicator
            pinned_score = 1.0 if chunk.is_pinned else 0.0

            # Compute final score
            final_score = (
                self.w_e * semantic_score
                + self.w_b * bm25_score
                + self.w_r * recency_score
                + self.w_c * contract_score
                + self.w_f * focus_score
                + self.w_p * pinned_score
            )

            scored_chunks.append(
                ScoredChunk(
                    chunk=chunk,
                    semantic_score=semantic_score,
                    bm25_score=bm25_score,
                    recency_score=recency_score,
                    contract_score=contract_score,
                    focus_score=focus_score,
                    pinned_score=pinned_score,
                    final_score=final_score,
                )
            )

        # Sort by final score
        scored_chunks.sort(key=lambda x: x.final_score, reverse=True)

        return scored_chunks

    async def retrieve_by_ids(
        self,
        chunk_ids: list[str],
    ) -> list[Chunk]:
        """
        Retrieve chunks by their IDs.

        Args:
            chunk_ids: List of chunk IDs.

        Returns:
            List of chunks.
        """
        chunks = []
        for chunk_id in chunk_ids:
            metadata = await self.sqlite_store.get_chunk(chunk_id)
            if not metadata:
                continue

            content = await self.sqlite_store.get_chunk_content(chunk_id)
            if not content:
                continue

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    file_path=metadata.file_path,
                    content=content,
                    start_line=metadata.start_line,
                    end_line=metadata.end_line,
                    symbol_name=metadata.symbol_name,
                    symbol_type=metadata.symbol_type,
                    language=metadata.language,
                    token_count=metadata.token_count,
                    is_contract=metadata.is_contract,
                    is_pinned=metadata.is_pinned,
                )
            )

        return chunks

    async def expand_context(
        self,
        chunk_ids: list[str],
        max_additional: int = 5,
    ) -> list[Chunk]:
        """
        Expand context by finding related chunks.

        Args:
            chunk_ids: Starting chunk IDs.
            max_additional: Maximum additional chunks to add.

        Returns:
            List of additional context chunks.
        """
        additional_chunks = []
        seen_ids = set(chunk_ids)

        for chunk_id in chunk_ids:
            # Get chunk metadata
            metadata = await self.sqlite_store.get_chunk(chunk_id)
            if not metadata:
                continue

            # Find other chunks in the same file
            file_chunks = await self.sqlite_store.get_chunks_by_file(
                metadata.file_path
            )

            for fc in file_chunks:
                if fc.chunk_id not in seen_ids:
                    # Check if adjacent
                    if (
                        abs(fc.start_line - metadata.end_line) <= 5
                        or abs(metadata.start_line - fc.end_line) <= 5
                    ):
                        content = await self.sqlite_store.get_chunk_content(
                            fc.chunk_id
                        )
                        if content:
                            additional_chunks.append(
                                Chunk(
                                    chunk_id=fc.chunk_id,
                                    file_path=fc.file_path,
                                    content=content,
                                    start_line=fc.start_line,
                                    end_line=fc.end_line,
                                    symbol_name=fc.symbol_name,
                                    symbol_type=fc.symbol_type,
                                    language=fc.language,
                                    token_count=fc.token_count,
                                    is_contract=fc.is_contract,
                                    is_pinned=fc.is_pinned,
                                )
                            )
                            seen_ids.add(fc.chunk_id)

                            if len(additional_chunks) >= max_additional:
                                return additional_chunks

            # Find related contracts
            if metadata.is_contract:
                related = await self.contract_store.find_related_contracts(
                    chunk_id, max_depth=1
                )
                for contract in related:
                    if contract.chunk_id not in seen_ids:
                        content = await self.sqlite_store.get_chunk_content(
                            contract.chunk_id
                        )
                        if content:
                            fc = await self.sqlite_store.get_chunk(contract.chunk_id)
                            if fc:
                                additional_chunks.append(
                                    Chunk(
                                        chunk_id=contract.chunk_id,
                                        file_path=fc.file_path,
                                        content=content,
                                        start_line=fc.start_line,
                                        end_line=fc.end_line,
                                        symbol_name=fc.symbol_name,
                                        symbol_type=fc.symbol_type,
                                        language=fc.language,
                                        token_count=fc.token_count,
                                        is_contract=True,
                                        is_pinned=fc.is_pinned,
                                    )
                                )
                                seen_ids.add(contract.chunk_id)

                                if len(additional_chunks) >= max_additional:
                                    return additional_chunks

        return additional_chunks
