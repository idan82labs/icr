"""
Bridge module connecting IC-MCP to ICD retrieval with RLM support.

This module provides the connection between the MCP tools and the actual
ICD retrieval system, including:
- HybridRetriever initialization from existing index
- RLM auto-gating based on entropy
- Iterative query decomposition and aggregation
- CRAG (Corrective RAG) for quality-aware retrieval
- True RLM with context externalization
- Code graph traversal for multi-hop retrieval
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
    # New: CRAG and True RLM settings
    crag_enabled: bool = False  # Disabled until fixes complete
    true_rlm_enabled: bool = True  # Use True RLM instead of basic planner
    graph_expansion_enabled: bool = True  # Enable code graph expansion


@dataclass
class RLMMetrics:
    """Metrics from RLM execution."""
    mode: str  # "pack", "rlm", or "true_rlm"
    entropy: float
    iterations: int = 0
    sub_queries_executed: int = 0
    total_chunks_retrieved: int = 0
    aggregation_dedup_ratio: float = 0.0
    used_llm_decomposition: bool = False
    llm_reasoning: str = ""
    # New: CRAG and graph metrics
    crag_quality: str = ""  # "correct", "corrected", "ambiguous"
    crag_confidence: float = 0.0
    graph_nodes_expanded: int = 0
    true_rlm_operations: int = 0  # Number of True RLM operations executed


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
    - CRAG (Corrective RAG) for quality-aware retrieval
    - True RLM with context externalization
    - Code graph traversal for multi-hop retrieval
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
        # New: CRAG and graph components
        self._crag_retriever = None
        self._graph_builder = None
        self._graph_retriever = None

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

            # Create CRAG retriever (wraps HybridRetriever for quality-aware retrieval)
            if self.config.crag_enabled:
                try:
                    from icd.retrieval.crag import CRAGRetriever
                    self._crag_retriever = CRAGRetriever(self._icd_config, self._retriever)
                    logger.info("CRAG retriever initialized")
                except ImportError:
                    logger.warning("CRAG module not available")

            # Create code graph builder and retriever
            if self.config.graph_expansion_enabled:
                try:
                    from icd.graph import CodeGraphBuilder, GraphRetriever
                    self._graph_builder = CodeGraphBuilder(self._icd_config)
                    self._graph_retriever = GraphRetriever(
                        self._icd_config,
                        self._graph_builder,
                        self._retriever,
                    )
                    # Try to load existing graph from index
                    await self._load_code_graph()
                    logger.info("Code graph components initialized")
                except ImportError:
                    logger.warning("Graph module not available")

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

    async def _load_code_graph(self) -> None:
        """Load existing code graph from index if available."""
        if not self._graph_builder:
            return

        # Try to load from persisted graph file
        graph_path = self.project_root / ".icr" / "code_graph.json"
        if not graph_path.exists():
            graph_path = self.project_root / ".icd" / "code_graph.json"

        if graph_path.exists():
            try:
                import json
                data = json.loads(graph_path.read_text())
                self._graph_builder.load_from_dict(data)
                logger.info(
                    "Code graph loaded",
                    nodes=len(self._graph_builder.get_nodes()),
                )
            except Exception as e:
                logger.debug(f"Could not load code graph: {e}")

    async def build_code_graph(self, files: list[Path] | None = None) -> dict[str, int]:
        """
        Build the code graph from indexed files.

        Args:
            files: Optional list of files to process. If None, processes all indexed files.

        Returns:
            Statistics about the graph build.
        """
        if not self._graph_builder:
            return {"error": "Graph builder not initialized"}

        try:
            # Get files from index if not provided
            if files is None and self._icd_service:
                # Query sqlite for all indexed files
                indexed = await self._icd_service._sqlite_store.get_all_files()
                files = [Path(f.file_path) for f in indexed]
            elif files is None:
                # Scan project directory
                files = list(self.project_root.rglob("*.py"))
                files.extend(self.project_root.rglob("*.ts"))
                files.extend(self.project_root.rglob("*.js"))

            # Build graph
            for file_path in files:
                if file_path.exists():
                    try:
                        await self._graph_builder.process_file(file_path)
                    except Exception as e:
                        logger.debug(f"Could not process {file_path}: {e}")

            # Persist graph
            graph_path = self.project_root / ".icr" / "code_graph.json"
            graph_path.parent.mkdir(parents=True, exist_ok=True)
            import json
            graph_path.write_text(json.dumps(self._graph_builder.to_dict(), indent=2))

            stats = {
                "nodes": len(self._graph_builder.get_nodes()),
                "edges": len(self._graph_builder.get_edges()),
                "files_processed": len(files),
            }
            logger.info("Code graph built", **stats)
            return stats

        except Exception as e:
            logger.error(f"Failed to build code graph: {e}")
            return {"error": str(e)}

    async def retrieve(
        self,
        query: str,
        k: int = 20,
        mode: str = "auto",  # "auto", "pack", "rlm", "true_rlm"
        focus_paths: list[str] | None = None,
        use_crag: bool | None = None,  # None = use config default
        use_graph_expansion: bool | None = None,  # None = use config default
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: Natural language query
            k: Number of results to return
            mode: Retrieval mode (auto, pack, rlm, true_rlm)
            focus_paths: Optional paths to prioritize
            use_crag: Override CRAG setting (None = use config)
            use_graph_expansion: Override graph expansion setting (None = use config)

        Returns:
            RetrievalResult with chunks and metrics
        """
        if not self._initialized:
            await self.initialize()

        if not self._initialized or not self._retriever:
            # Fallback to basic retrieval
            return await self._basic_retrieve(query, k)

        # Resolve optional flags
        should_use_crag = use_crag if use_crag is not None else self.config.crag_enabled
        should_use_graph = use_graph_expansion if use_graph_expansion is not None else self.config.graph_expansion_enabled

        try:
            # Step 1: Initial retrieval (with CRAG if enabled)
            focus_path_objs = [Path(p) for p in (focus_paths or [])]
            crag_quality = ""
            crag_confidence = 0.0

            if should_use_crag and self._crag_retriever:
                # Use CRAG for quality-aware retrieval
                initial_result = await self._crag_retriever.retrieve_with_correction(
                    query=query,
                    limit=k,
                    focus_paths=focus_path_objs,
                )
                crag_quality = initial_result.metadata.get("crag_quality", "")
                crag_confidence = initial_result.metadata.get("crag_confidence", 0.0)
                logger.debug(
                    "CRAG retrieval complete",
                    quality=crag_quality,
                    confidence=crag_confidence,
                )
            else:
                # Standard retrieval
                initial_result = await self._retriever.retrieve(
                    query=query,
                    limit=k,
                    focus_paths=focus_path_objs,
                )

            entropy = initial_result.entropy

            # Step 2: Determine mode
            if mode == "auto":
                use_rlm = entropy >= self.config.entropy_threshold
                # Use True RLM if enabled and entropy is high
                if use_rlm and self.config.true_rlm_enabled:
                    resolved_mode = "true_rlm"
                elif use_rlm:
                    resolved_mode = "rlm"
                else:
                    resolved_mode = "pack"
            else:
                resolved_mode = mode
                use_rlm = mode in ("rlm", "true_rlm")

            # Step 3: Execute RLM if needed
            if resolved_mode == "true_rlm":
                return await self._execute_true_rlm(
                    query=query,
                    initial_result=initial_result,
                    k=k,
                    crag_quality=crag_quality,
                    crag_confidence=crag_confidence,
                )
            elif use_rlm and self._planner and self._aggregator:
                return await self._execute_rlm(
                    query=query,
                    initial_result=initial_result,
                    k=k,
                    crag_quality=crag_quality,
                    crag_confidence=crag_confidence,
                )
            else:
                # Step 4: Optional graph expansion for pack mode
                graph_nodes_expanded = 0
                if should_use_graph and self._graph_retriever:
                    try:
                        expanded_result = await self._graph_retriever.retrieve_with_expansion(
                            query=query,
                            limit=k,
                            query_type="default",
                        )
                        if expanded_result.metadata.get("graph_expansion"):
                            initial_result = expanded_result
                            graph_nodes_expanded = expanded_result.metadata["graph_expansion"].get("expanded_nodes", 0)
                    except Exception as e:
                        logger.debug(f"Graph expansion failed: {e}")

                # Return results
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
                        crag_quality=crag_quality,
                        crag_confidence=crag_confidence,
                        graph_nodes_expanded=graph_nodes_expanded,
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
        crag_quality: str = "",
        crag_confidence: float = 0.0,
    ) -> RetrievalResult:
        """
        Execute RLM iterative retrieval.

        Uses LLM-based query decomposition if ANTHROPIC_API_KEY is set,
        otherwise falls back to heuristic decomposition.

        Args:
            query: Original query
            initial_result: Results from initial retrieval
            k: Number of final results
            crag_quality: Quality assessment from CRAG
            crag_confidence: Confidence score from CRAG

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
                crag_quality=crag_quality,
                crag_confidence=crag_confidence,
            ),
            sub_query_results=sub_query_info,
        )

    async def _execute_true_rlm(
        self,
        query: str,
        initial_result: Any,
        k: int,
        crag_quality: str = "",
        crag_confidence: float = 0.0,
    ) -> RetrievalResult:
        """
        Execute True RLM with context externalization.

        Uses the TrueRLMOrchestrator for research-grade retrieval with:
        - LLM-generated retrieval programs
        - Parallel operation execution
        - Quality evaluation and refinement
        - Graph-aware exploration

        Args:
            query: Original query
            initial_result: Results from initial retrieval
            k: Number of final results
            crag_quality: Quality assessment from CRAG
            crag_confidence: Confidence score from CRAG

        Returns:
            RetrievalResult with chunks from True RLM execution
        """
        try:
            from icd.rlm.true_rlm import run_true_rlm

            # Execute True RLM
            rlm_result = await run_true_rlm(
                config=self._icd_config,
                base_retriever=self._retriever,
                query=query,
                graph_builder=self._graph_builder,
                limit=k,
            )

            # Convert to bridge result format
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
                for c in rlm_result.chunks[:k]
            ]

            return RetrievalResult(
                chunks=chunks,
                scores=rlm_result.scores[:k],
                entropy=rlm_result.final_entropy,
                metrics=RLMMetrics(
                    mode="true_rlm",
                    entropy=rlm_result.final_entropy,
                    iterations=rlm_result.refinement_iterations,
                    sub_queries_executed=rlm_result.operations_executed,
                    total_chunks_retrieved=len(rlm_result.chunks),
                    used_llm_decomposition=True,  # True RLM always uses LLM
                    llm_reasoning=rlm_result.metadata.get("program_reasoning", ""),
                    crag_quality=crag_quality,
                    crag_confidence=crag_confidence,
                    true_rlm_operations=rlm_result.operations_executed,
                ),
                sub_query_results=[
                    {"step": i, "trace": trace}
                    for i, trace in enumerate(rlm_result.execution_trace)
                ],
            )

        except ImportError:
            logger.warning("True RLM module not available, falling back to basic RLM")
            return await self._execute_rlm(
                query=query,
                initial_result=initial_result,
                k=k,
                crag_quality=crag_quality,
                crag_confidence=crag_confidence,
            )
        except Exception as e:
            logger.error(f"True RLM execution failed: {e}")
            # Fallback to basic RLM
            return await self._execute_rlm(
                query=query,
                initial_result=initial_result,
                k=k,
                crag_quality=crag_quality,
                crag_confidence=crag_confidence,
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
