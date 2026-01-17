"""
Enhanced Retriever with Novel Components.

Integrates all novel ICR contributions:
1. Query Intent Router (QIR) - Intent-aware weight adjustment
2. Multi-Hop Graph Retrieval (MHGR) - Query-guided graph traversal
3. Adaptive Entropy Calibration (AEC) - Per-project threshold tuning
4. Dependency-Aware Packing (DAC-Pack) - Coherent context compilation

This is the main entry point for research-grade retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from icd.config import Config
    from icd.graph.builder import CodeGraphBuilder
    from icd.retrieval.hybrid import HybridRetriever, Chunk, RetrievalResult

logger = structlog.get_logger(__name__)


@dataclass
class EnhancedRetrievalResult:
    """Result from enhanced retrieval with novel components."""

    chunks: list["Chunk"]
    scores: list[float]
    entropy: float
    query: str

    # Novel component outputs
    intent: str  # Classified query intent
    intent_confidence: float
    graph_paths_found: int  # Multi-hop paths discovered
    calibrated_threshold: float | None  # Entropy threshold used

    metadata: dict[str, Any] = field(default_factory=dict)


class EnhancedRetriever:
    """
    Enhanced retrieval system integrating all novel ICR components.

    This is the recommended entry point for production use, providing:
    - Automatic query intent classification and weight adjustment
    - Multi-hop graph traversal for related code discovery
    - Calibrated entropy thresholds for RLM triggering
    - Dependency-aware context packing
    """

    def __init__(
        self,
        config: "Config",
        base_retriever: "HybridRetriever",
        graph_builder: "CodeGraphBuilder | None" = None,
        sqlite_store: Any = None,
    ) -> None:
        self.config = config
        self.base_retriever = base_retriever
        self.graph_builder = graph_builder
        self.sqlite_store = sqlite_store

        # Initialize novel components
        self._init_query_router()
        self._init_multihop_retriever()
        self._init_entropy_calibrator()

        # Cached calibration
        self._calibration_result = None

    def _init_query_router(self) -> None:
        """Initialize query intent router."""
        try:
            from icd.retrieval.query_router import QueryRouter
            self.query_router = QueryRouter(self.config)
            logger.info("Query intent router initialized")
        except Exception as e:
            logger.warning(f"Query router initialization failed: {e}")
            self.query_router = None

    def _init_multihop_retriever(self) -> None:
        """Initialize multi-hop graph retriever."""
        if self.graph_builder is None:
            self.multihop_retriever = None
            return

        try:
            from icd.retrieval.multihop import MultiHopRetriever
            self.multihop_retriever = MultiHopRetriever(
                self.config,
                self.graph_builder,
                self.base_retriever,
            )
            logger.info("Multi-hop retriever initialized")
        except Exception as e:
            logger.warning(f"Multi-hop retriever initialization failed: {e}")
            self.multihop_retriever = None

    def _init_entropy_calibrator(self) -> None:
        """Initialize adaptive entropy calibrator."""
        if self.sqlite_store is None:
            self.entropy_gate = None
            return

        try:
            from icd.retrieval.entropy_calibrator import AdaptiveEntropyGate
            self.entropy_gate = AdaptiveEntropyGate(
                self.config,
                self.sqlite_store,
                self.base_retriever,
            )
            logger.info("Adaptive entropy gate initialized")
        except Exception as e:
            logger.warning(f"Entropy calibrator initialization failed: {e}")
            self.entropy_gate = None

    async def calibrate(self, force: bool = False) -> dict[str, Any]:
        """
        Run entropy calibration for this project.

        Should be called after indexing or periodically.
        """
        if self.entropy_gate is None:
            return {"status": "unavailable"}

        result = await self.entropy_gate.ensure_calibrated(force=force)
        self._calibration_result = result

        return {
            "status": "calibrated",
            "threshold": result.recommended_threshold,
            "confidence": result.confidence,
            "samples": result.num_samples,
            "separation_score": result.separation_score,
        }

    async def retrieve(
        self,
        query: str,
        limit: int = 20,
        focus_paths: list[Path] | None = None,
        use_intent_routing: bool = True,
        use_multihop: bool = True,
        **kwargs,
    ) -> EnhancedRetrievalResult:
        """
        Perform enhanced retrieval with all novel components.

        Args:
            query: Natural language query.
            limit: Maximum results.
            focus_paths: Paths to prioritize.
            use_intent_routing: Use query intent for weight adjustment.
            use_multihop: Use multi-hop graph expansion.

        Returns:
            EnhancedRetrievalResult with full metadata.
        """
        # Step 1: Classify query intent
        intent = "general"
        intent_confidence = 0.5
        strategy = None

        if use_intent_routing and self.query_router is not None:
            classification, strategy = self.query_router.route(query)
            intent = classification.primary_intent.value
            intent_confidence = classification.confidence

            logger.debug(
                "Query intent classified",
                intent=intent,
                confidence=intent_confidence,
                entities=classification.extracted_entities[:3],
            )

        # Step 2: Adjust retrieval based on intent
        if strategy is not None and use_intent_routing:
            # Create intent-adjusted retriever or adjust weights dynamically
            # For now, we'll pass the strategy to multihop
            pass

        # Step 3: Decide retrieval path
        graph_paths_found = 0

        # Check if multi-hop should be enabled for this intent
        should_use_graph = (
            use_multihop
            and self.multihop_retriever is not None
            and strategy is not None
            and strategy.enable_graph_expansion
        )

        if should_use_graph:
            # Use multi-hop retrieval
            multihop_result = await self.multihop_retriever.retrieve_multihop(
                query=query,
                intent=intent,
                limit=limit,
                **kwargs,
            )
            chunks = multihop_result.chunks
            scores = multihop_result.scores
            graph_paths_found = len(multihop_result.paths)

            # Compute entropy
            from icd.retrieval.entropy import EntropyCalculator
            entropy_calc = EntropyCalculator(
                temperature=self.config.retrieval.entropy_temperature
            )
            entropy = entropy_calc.compute_entropy(scores)

        else:
            # Use base retrieval
            result = await self.base_retriever.retrieve(
                query=query,
                limit=limit,
                focus_paths=focus_paths,
                **kwargs,
            )
            chunks = result.chunks
            scores = result.scores
            entropy = result.entropy

        # Step 4: Get calibrated threshold
        calibrated_threshold = None
        if self.entropy_gate is not None:
            calibrated_threshold = self.entropy_gate.get_threshold()

        return EnhancedRetrievalResult(
            chunks=chunks,
            scores=scores,
            entropy=entropy,
            query=query,
            intent=intent,
            intent_confidence=intent_confidence,
            graph_paths_found=graph_paths_found,
            calibrated_threshold=calibrated_threshold,
            metadata={
                "use_intent_routing": use_intent_routing,
                "use_multihop": should_use_graph,
                "strategy_used": strategy is not None,
            },
        )

    def should_trigger_rlm(self, entropy: float) -> bool:
        """
        Check if RLM should be triggered using calibrated threshold.
        """
        if self.entropy_gate is not None:
            return self.entropy_gate.should_trigger_rlm(entropy)
        return entropy >= self.config.rlm.entropy_threshold

    async def compile_pack(
        self,
        chunks: list["Chunk"],
        scores: list[float],
        budget_tokens: int | None = None,
        query: str | None = None,
        use_dependency_packing: bool = True,
    ) -> Any:
        """
        Compile context pack using dependency-aware packing.

        Args:
            chunks: Chunks to pack.
            scores: Corresponding scores.
            budget_tokens: Token budget.
            query: Original query.
            use_dependency_packing: Use DAC-Pack algorithm.

        Returns:
            Pack result (DACPackResult or PackResult).
        """
        if use_dependency_packing and self.graph_builder is not None:
            from icd.pack.dependency_packer import DependencyAwarePacker
            packer = DependencyAwarePacker(self.config, self.graph_builder)
            return await packer.compile(
                chunks=chunks,
                scores=scores,
                budget_tokens=budget_tokens,
                query=query,
            )
        else:
            from icd.pack.compiler import PackCompiler
            packer = PackCompiler(self.config)
            return await packer.compile(
                chunks=chunks,
                scores=scores,
                budget_tokens=budget_tokens,
                query=query,
            )


def create_enhanced_retriever(
    config: "Config",
    base_retriever: "HybridRetriever",
    graph_builder: "CodeGraphBuilder | None" = None,
    sqlite_store: Any = None,
) -> EnhancedRetriever:
    """
    Create an enhanced retriever with all novel components.

    Args:
        config: ICD configuration.
        base_retriever: Base hybrid retriever.
        graph_builder: Optional code graph builder.
        sqlite_store: Optional SQLite store for calibration.

    Returns:
        EnhancedRetriever instance.
    """
    return EnhancedRetriever(
        config=config,
        base_retriever=base_retriever,
        graph_builder=graph_builder,
        sqlite_store=sqlite_store,
    )
