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
    used_llm_decomposition: bool = False
    llm_reasoning: str = ""


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
    - Auto-indexing when index doesn't exist
    - Hybrid retrieval (semantic + BM25)
    - RLM auto-gating and iteration
    - Result aggregation
    - File watching for incremental updates
    """

    def __init__(
        self,
        project_root: Path | None = None,
        auto_index: bool = True,
        watch_files: bool = False,
    ):
        """
        Initialize the ICD bridge.

        Args:
            project_root: Root directory of the project (defaults to cwd)
            auto_index: If True, automatically index if no index exists
            watch_files: If True, start file watching after indexing
        """
        self.project_root = project_root or Path.cwd()
        self.config = RetrievalConfig()
        self.auto_index = auto_index
        self.watch_files = watch_files
        self._initialized = False
        self._retriever = None
        self._planner = None
        self._aggregator = None
        self._icd_config = None
        self._icd_service = None
        self._indexing_in_progress = False

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
                if self.auto_index:
                    logger.info(f"ICD index not found at {db_path}, auto-indexing...")
                    success = await self._auto_index()
                    if not success:
                        logger.warning("Auto-indexing failed")
                        return False
                else:
                    logger.warning(f"ICD index not found at {db_path}")
                    return False

            # Initialize stores
            sqlite_store = SQLiteStore(self._icd_config)
            await sqlite_store.initialize()

            vector_store = VectorStore(self._icd_config)
            await vector_store.initialize()

            contract_store = ContractStore(self._icd_config, sqlite_store)
            memory_store = MemoryStore(self._icd_config, sqlite_store)

            # Initialize embedder
            embedder = create_embedder(self._icd_config)

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

    async def _auto_index(self) -> bool:
        """
        Automatically index the project.

        Returns:
            True if indexing succeeded
        """
        if self._indexing_in_progress:
            logger.warning("Indexing already in progress")
            return False

        self._indexing_in_progress = True
        try:
            from icd.config import load_config
            from icd.main import ICDService

            # Load config for this project
            config_path = self.project_root / ".icr" / "config.yaml"
            if not config_path.exists():
                config_path = self.project_root / ".icd" / "config.yaml"

            config = load_config(
                config_path=config_path if config_path.exists() else None,
                project_root=self.project_root,
            )

            # Create service instance with config
            self._icd_service = ICDService(config=config)
            await self._icd_service.initialize()

            # Index the directory
            logger.info(f"Auto-indexing project: {self.project_root}")
            stats = await self._icd_service.index_directory()
            logger.info(
                f"Auto-indexing complete",
                files=stats.get("files", 0),
                chunks=stats.get("chunks", 0),
                errors=stats.get("errors", 0),
            )

            # Start file watching if requested
            if self.watch_files:
                await self._icd_service.start_watching()
                logger.info("File watching started")

            return stats.get("errors", 0) == 0 or stats.get("chunks", 0) > 0

        except Exception as e:
            logger.error(f"Auto-indexing failed: {e}")
            return False
        finally:
            self._indexing_in_progress = False

    async def reindex_file(self, file_path: str | Path) -> dict[str, int]:
        """
        Re-index a specific file.

        Args:
            file_path: Path to the file to re-index

        Returns:
            Statistics about the re-indexed file
        """
        if not self._initialized:
            await self.initialize()

        try:
            from icd.config import load_config
            from icd.main import ICDService

            if not self._icd_service:
                # Load config for this project
                config_path = self.project_root / ".icr" / "config.yaml"
                if not config_path.exists():
                    config_path = self.project_root / ".icd" / "config.yaml"

                config = load_config(
                    config_path=config_path if config_path.exists() else None,
                    project_root=self.project_root,
                )
                self._icd_service = ICDService(config=config)
                await self._icd_service.initialize()

            return await self._icd_service.reindex_file(Path(file_path))

        except Exception as e:
            logger.error(f"Failed to reindex file {file_path}: {e}")
            return {"errors": 1}

    def is_indexing(self) -> bool:
        """Check if indexing is currently in progress."""
        return self._indexing_in_progress

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

        Uses LLM-based query decomposition if ANTHROPIC_API_KEY is set,
        otherwise falls back to heuristic decomposition.

        Args:
            query: Original query
            initial_result: Results from initial retrieval
            k: Number of final results

        Returns:
            RetrievalResult with aggregated chunks
        """
        # Create plan (with LLM if available)
        plan = await self._planner.create_plan_with_llm(query, initial_result)

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

        # Extract LLM metadata from plan
        used_llm = plan.metadata.get("used_llm", False)
        llm_reasoning = plan.metadata.get("llm_reasoning", "")

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
                used_llm_decomposition=used_llm,
                llm_reasoning=llm_reasoning,
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
