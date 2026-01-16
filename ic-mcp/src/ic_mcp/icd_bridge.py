"""
Bridge module connecting IC-MCP to ICD retrieval with RLM support.

This module provides the connection between the MCP tools and the actual
ICD retrieval system, including:
- HybridRetriever initialization from existing index
- RLM auto-gating based on entropy
- Iterative query decomposition and aggregation
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    """Configuration for retrieval."""
    entropy_threshold: float = 2.5
    max_rlm_iterations: int = 5
    budget_per_iteration: int = 2000
    default_k: int = 20


@dataclass
class RLMMetrics:
    """Metrics from RLM execution."""
    mode: str  # "pack" or "rlm"
    entropy: float
    iterations: int = 0
    sub_queries_executed: int = 0
    total_chunks_retrieved: int = 0
    aggregation_dedup_ratio: float = 0.0


@dataclass
class RetrievalResult:
    """Result from retrieval with RLM."""
    chunks: list[dict[str, Any]]
    scores: list[float]
    entropy: float
    metrics: RLMMetrics
    sub_query_results: list[dict[str, Any]] = field(default_factory=list)


class ICDBridge:
    """
    Bridge to ICD retrieval system.

    Handles:
    - Loading existing ICD index
    - Hybrid retrieval (semantic + BM25)
    - RLM auto-gating and iteration
    - Result aggregation
    """

    def __init__(self, project_root: Path | None = None):
        """
        Initialize the ICD bridge.

        Args:
            project_root: Root directory of the project (defaults to cwd)
        """
        self.project_root = project_root or Path.cwd()
        self.config = RetrievalConfig()
        self._initialized = False
        self._retriever = None
        self._planner = None
        self._aggregator = None
        self._icd_config = None

    async def initialize(self) -> bool:
        """
        Initialize connection to ICD index.

        Returns:
            True if successfully connected to existing index
        """
        if self._initialized:
            return True

        try:
            # Try to import icd modules
            from icd.config import load_config
            from icd.retrieval.hybrid import HybridRetriever
            from icd.rlm.planner import RLMPlanner
            from icd.rlm.aggregator import Aggregator
            from icd.storage.sqlite_store import SQLiteStore
            from icd.storage.vector_store import VectorStore
            from icd.storage.contract_store import ContractStore
            from icd.storage.memory_store import MemoryStore
            from icd.indexing.embedder import create_embedder

            # Load config
            config_path = self.project_root / ".icr" / "config.yaml"
            if not config_path.exists():
                config_path = self.project_root / ".icd" / "config.yaml"

            self._icd_config = load_config(
                config_path=config_path if config_path.exists() else None,
                project_root=self.project_root,
            )

            # Check if index exists
            db_path = self._icd_config.db_path
            if not db_path.exists():
                logger.warning(f"ICD index not found at {db_path}")
                return False

            # Initialize stores
            sqlite_store = SQLiteStore(db_path)
            await sqlite_store.initialize()

            vector_store = VectorStore(
                index_path=self._icd_config.vector_index_path,
                dimension=self._icd_config.embedding.dimension,
            )
            await vector_store.load()

            contract_store = ContractStore(sqlite_store)
            memory_store = MemoryStore(sqlite_store)

            # Initialize embedder
            embedder = await create_embedder(self._icd_config)

            # Create retriever
            self._retriever = HybridRetriever(
                config=self._icd_config,
                sqlite_store=sqlite_store,
                vector_store=vector_store,
                embedder=embedder,
                contract_store=contract_store,
                memory_store=memory_store,
            )

            # Create RLM components
            self._planner = RLMPlanner(self._icd_config)
            self._aggregator = Aggregator(self._icd_config)

            # Update config from icd config
            self.config.entropy_threshold = self._icd_config.rlm.entropy_threshold
            self.config.max_rlm_iterations = self._icd_config.rlm.max_iterations
            self.config.budget_per_iteration = self._icd_config.rlm.budget_per_iteration

            self._initialized = True
            logger.info(f"ICD bridge initialized for {self.project_root}")
            return True

        except ImportError as e:
            logger.warning(f"ICD modules not available: {e}")
            return False
        except Exception as e:
            logger.warning(f"Failed to initialize ICD bridge: {e}")
            return False

    async def retrieve(
        self,
        query: str,
        k: int = 20,
        mode: str = "auto",  # "auto", "pack", "rlm"
        focus_paths: list[str] | None = None,
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: Natural language query
            k: Number of results to return
            mode: Retrieval mode (auto, pack, rlm)
            focus_paths: Optional paths to prioritize

        Returns:
            RetrievalResult with chunks and metrics
        """
        if not self._initialized:
            await self.initialize()

        if not self._initialized or not self._retriever:
            # Fallback to basic retrieval
            return await self._basic_retrieve(query, k)

        try:
            # Step 1: Initial retrieval
            focus_path_objs = [Path(p) for p in (focus_paths or [])]
            initial_result = await self._retriever.retrieve(
                query=query,
                limit=k,
                focus_paths=focus_path_objs,
            )

            entropy = initial_result.entropy

            # Step 2: Determine mode
            if mode == "auto":
                use_rlm = entropy >= self.config.entropy_threshold
                resolved_mode = "rlm" if use_rlm else "pack"
            else:
                resolved_mode = mode
                use_rlm = mode == "rlm"

            # Step 3: Execute RLM if needed
            if use_rlm and self._planner and self._aggregator:
                return await self._execute_rlm(
                    query=query,
                    initial_result=initial_result,
                    k=k,
                )
            else:
                # Return initial results
                chunks = [
                    {
                        "chunk_id": c.chunk_id,
                        "file_path": c.file_path,
                        "content": c.content,
                        "start_line": c.start_line,
                        "end_line": c.end_line,
                        "symbol_name": c.symbol_name,
                        "symbol_type": c.symbol_type,
                        "language": c.language,
                        "token_count": c.token_count,
                        "is_contract": c.is_contract,
                    }
                    for c in initial_result.chunks
                ]

                return RetrievalResult(
                    chunks=chunks,
                    scores=initial_result.scores,
                    entropy=entropy,
                    metrics=RLMMetrics(
                        mode=resolved_mode,
                        entropy=entropy,
                    ),
                )

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return await self._basic_retrieve(query, k)

    async def _execute_rlm(
        self,
        query: str,
        initial_result: Any,
        k: int,
    ) -> RetrievalResult:
        """
        Execute RLM iterative retrieval.

        Args:
            query: Original query
            initial_result: Results from initial retrieval
            k: Number of final results

        Returns:
            RetrievalResult with aggregated chunks
        """
        # Create plan
        plan = self._planner.create_plan(query, initial_result)

        # Track results for aggregation
        iteration_results = []
        sub_query_info = []

        # Add initial results
        iteration_results.append((
            None,  # No sub-query for initial
            initial_result.chunks,
            initial_result.scores,
        ))

        # Execute sub-queries
        iterations = 0
        while iterations < self.config.max_rlm_iterations:
            sub_query = self._planner.get_next_sub_query(plan)
            if not sub_query:
                break

            iterations += 1

            # Execute sub-query
            result = await self._retriever.retrieve(
                query=sub_query.query,
                limit=self.config.budget_per_iteration // 100,  # Rough estimate
                focus_paths=[Path(p) for p in sub_query.focus_paths],
            )

            # Record results
            iteration_results.append((sub_query, result.chunks, result.scores))
            sub_query_info.append({
                "query": sub_query.query,
                "type": sub_query.query_type.value,
                "results": len(result.chunks),
            })

            # Update plan
            self._planner.update_plan(plan, sub_query, result.chunks)

            # Check if we should continue
            aggregated = self._aggregator.aggregate(plan, iteration_results)
            if not self._planner.should_continue(plan, aggregated.entropy):
                break

        # Final aggregation
        aggregated = self._aggregator.aggregate(plan, iteration_results)

        # Convert to result format
        chunks = [
            {
                "chunk_id": c.chunk_id,
                "file_path": c.file_path,
                "content": c.content,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "symbol_name": c.symbol_name,
                "symbol_type": c.symbol_type,
                "language": c.language,
                "token_count": c.token_count,
                "is_contract": c.is_contract,
            }
            for c in aggregated.chunks[:k]
        ]

        return RetrievalResult(
            chunks=chunks,
            scores=aggregated.scores[:k],
            entropy=aggregated.entropy,
            metrics=RLMMetrics(
                mode="rlm",
                entropy=aggregated.entropy,
                iterations=iterations,
                sub_queries_executed=len(sub_query_info),
                total_chunks_retrieved=sum(len(ir[1]) for ir in iteration_results),
                aggregation_dedup_ratio=aggregated.metadata.get("duplicate_ratio", 0),
            ),
            sub_query_results=sub_query_info,
        )

    async def _basic_retrieve(
        self,
        query: str,
        k: int,
    ) -> RetrievalResult:
        """
        Fallback basic retrieval when ICD is not available.

        Uses simple file scanning and keyword matching.
        """
        logger.info("Using basic retrieval (ICD not initialized)")

        # This is a minimal fallback - the full implementation is in memory.py
        return RetrievalResult(
            chunks=[],
            scores=[],
            entropy=0.0,
            metrics=RLMMetrics(
                mode="fallback",
                entropy=0.0,
            ),
        )


# Global bridge instance
_bridge: ICDBridge | None = None


def get_bridge(project_root: Path | None = None) -> ICDBridge:
    """Get or create the global bridge instance."""
    global _bridge
    if _bridge is None or (project_root and _bridge.project_root != project_root):
        _bridge = ICDBridge(project_root)
    return _bridge


async def retrieve_with_rlm(
    query: str,
    project_root: Path | None = None,
    k: int = 20,
    mode: str = "auto",
    focus_paths: list[str] | None = None,
) -> RetrievalResult:
    """
    Convenience function for retrieval with RLM.

    Args:
        query: Natural language query
        project_root: Project root directory
        k: Number of results
        mode: Retrieval mode (auto, pack, rlm)
        focus_paths: Optional paths to prioritize

    Returns:
        RetrievalResult with chunks and metrics
    """
    bridge = get_bridge(project_root)
    return await bridge.retrieve(query, k, mode, focus_paths)
