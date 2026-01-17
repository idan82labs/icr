"""
Multi-Hop Graph Retrieval (MHGR).

Extends basic graph traversal with query-guided multi-hop reasoning.
Different query intents require different traversal strategies:

- "What is X?" → Follow CONTAINS, INHERITS backward to find definitions
- "How is X used?" → Follow CALLS, REFERENCES forward to find usage
- "What depends on X?" → Transitive forward closure for impact analysis
- "What does X depend on?" → Transitive backward closure for dependencies

This is a novel contribution: intent-aware multi-hop traversal for code
graphs, with configurable hop decay and relevance propagation.

Reference: Inspired by GraphRAG (Microsoft, 2024) but adapted for
code-specific relationship types and query intents.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from icd.config import Config
    from icd.graph.builder import CodeGraphBuilder, EdgeType, GraphNode
    from icd.retrieval.hybrid import Chunk, RetrievalResult

logger = structlog.get_logger(__name__)


class TraversalDirection(str, Enum):
    """Direction of graph traversal."""
    FORWARD = "forward"   # Follow outgoing edges
    BACKWARD = "backward"  # Follow incoming edges
    BOTH = "both"  # Follow both directions


@dataclass
class HopConfig:
    """Configuration for a single hop in multi-hop traversal."""

    edge_types: list[str]  # Edge types to follow
    direction: TraversalDirection
    max_neighbors: int = 10  # Max neighbors per node
    weight_decay: float = 0.7  # Relevance decay per hop


@dataclass
class MultiHopPath:
    """A path discovered during multi-hop traversal."""

    nodes: list[str]  # Node IDs in path
    edge_types: list[str]  # Edge types traversed
    total_weight: float  # Cumulative relevance weight
    depth: int


@dataclass
class MultiHopResult:
    """Result of multi-hop retrieval."""

    chunks: list["Chunk"]
    scores: list[float]
    paths: list[MultiHopPath]  # Reasoning paths
    metadata: dict[str, Any] = field(default_factory=dict)


# Intent-to-traversal-strategy mapping
TRAVERSAL_STRATEGIES: dict[str, list[HopConfig]] = {
    "definition": [
        # Hop 1: Find what contains/defines the symbol
        HopConfig(
            edge_types=["CONTAINS"],
            direction=TraversalDirection.BACKWARD,
            max_neighbors=5,
            weight_decay=0.8,
        ),
        # Hop 2: Find parent classes/interfaces
        HopConfig(
            edge_types=["INHERITS", "IMPLEMENTS"],
            direction=TraversalDirection.FORWARD,
            max_neighbors=3,
            weight_decay=0.6,
        ),
    ],

    "implementation": [
        # Hop 1: Find what this calls
        HopConfig(
            edge_types=["CALLS", "IMPORTS"],
            direction=TraversalDirection.FORWARD,
            max_neighbors=8,
            weight_decay=0.7,
        ),
        # Hop 2: Find implementations of called functions
        HopConfig(
            edge_types=["CONTAINS", "CALLS"],
            direction=TraversalDirection.FORWARD,
            max_neighbors=5,
            weight_decay=0.5,
        ),
    ],

    "usage": [
        # Hop 1: Find callers
        HopConfig(
            edge_types=["CALLS"],
            direction=TraversalDirection.BACKWARD,
            max_neighbors=10,
            weight_decay=0.8,
        ),
        # Hop 2: Find callers of callers
        HopConfig(
            edge_types=["CALLS"],
            direction=TraversalDirection.BACKWARD,
            max_neighbors=5,
            weight_decay=0.6,
        ),
        # Hop 3: Find references
        HopConfig(
            edge_types=["REFERENCES", "IMPORTS"],
            direction=TraversalDirection.BACKWARD,
            max_neighbors=5,
            weight_decay=0.4,
        ),
    ],

    "impact": [
        # Transitive forward closure for impact analysis
        HopConfig(
            edge_types=["CALLS", "IMPORTS", "REFERENCES"],
            direction=TraversalDirection.BACKWARD,
            max_neighbors=15,
            weight_decay=0.8,
        ),
        HopConfig(
            edge_types=["CALLS", "IMPORTS"],
            direction=TraversalDirection.BACKWARD,
            max_neighbors=10,
            weight_decay=0.6,
        ),
        HopConfig(
            edge_types=["CALLS"],
            direction=TraversalDirection.BACKWARD,
            max_neighbors=5,
            weight_decay=0.4,
        ),
    ],

    "dependencies": [
        # Transitive backward closure for dependency analysis
        HopConfig(
            edge_types=["IMPORTS", "CALLS", "USES_TYPE"],
            direction=TraversalDirection.FORWARD,
            max_neighbors=10,
            weight_decay=0.8,
        ),
        HopConfig(
            edge_types=["IMPORTS", "INHERITS"],
            direction=TraversalDirection.FORWARD,
            max_neighbors=8,
            weight_decay=0.6,
        ),
        HopConfig(
            edge_types=["IMPORTS"],
            direction=TraversalDirection.FORWARD,
            max_neighbors=5,
            weight_decay=0.4,
        ),
    ],

    "default": [
        HopConfig(
            edge_types=["CALLS", "CONTAINS", "IMPORTS"],
            direction=TraversalDirection.BOTH,
            max_neighbors=8,
            weight_decay=0.7,
        ),
        HopConfig(
            edge_types=["CALLS", "INHERITS"],
            direction=TraversalDirection.BOTH,
            max_neighbors=5,
            weight_decay=0.5,
        ),
    ],
}


class MultiHopRetriever:
    """
    Multi-hop graph retrieval with query-guided traversal.

    Key innovations:
    1. Intent-aware traversal strategies
    2. Configurable per-hop behavior (edge types, direction, decay)
    3. Path tracking for explainability
    4. Relevance propagation with decay
    """

    def __init__(
        self,
        config: "Config",
        graph_builder: "CodeGraphBuilder",
        base_retriever: Any,
    ) -> None:
        self.config = config
        self.graph = graph_builder
        self.base_retriever = base_retriever

        # Multi-hop parameters
        self.max_total_nodes = 30  # Max nodes to expand
        self.min_relevance_threshold = 0.1  # Skip nodes below this
        self.strategies = TRAVERSAL_STRATEGIES

    async def retrieve_multihop(
        self,
        query: str,
        intent: str = "default",
        limit: int = 20,
        seed_chunks: list["Chunk"] | None = None,
        **kwargs,
    ) -> MultiHopResult:
        """
        Perform multi-hop retrieval based on query intent.

        Args:
            query: Natural language query.
            intent: Query intent (definition, usage, impact, etc.)
            limit: Maximum results.
            seed_chunks: Optional seed chunks (otherwise uses base retriever)

        Returns:
            MultiHopResult with chunks, scores, and paths.
        """
        # Step 1: Get seed chunks from base retrieval if not provided
        if seed_chunks is None:
            initial_result = await self.base_retriever.retrieve(
                query=query,
                limit=min(limit, 10),  # Fewer seeds for multi-hop
                **kwargs,
            )
            seed_chunks = initial_result.chunks
            seed_scores = initial_result.scores
        else:
            seed_scores = [1.0] * len(seed_chunks)

        if not seed_chunks:
            return MultiHopResult(
                chunks=[],
                scores=[],
                paths=[],
                metadata={"reason": "no_seeds"},
            )

        # Step 2: Find graph nodes for seed chunks
        seed_nodes = self._find_nodes_for_chunks(seed_chunks)

        if not seed_nodes:
            logger.debug("No graph nodes found for seed chunks")
            return MultiHopResult(
                chunks=seed_chunks,
                scores=seed_scores,
                paths=[],
                metadata={"reason": "no_graph_nodes"},
            )

        # Step 3: Get traversal strategy
        strategy = self.strategies.get(intent, self.strategies["default"])

        # Step 4: Multi-hop traversal
        expanded_nodes, paths = self._traverse_multihop(
            seed_nodes=seed_nodes,
            seed_scores=dict(zip(seed_nodes, seed_scores[:len(seed_nodes)])),
            strategy=strategy,
        )

        # Step 5: Get chunks for expanded nodes
        expanded_chunks = await self._get_chunks_for_nodes(expanded_nodes)

        # Step 6: Merge with seeds and compute final scores
        final_chunks, final_scores = self._merge_and_score(
            seed_chunks=seed_chunks,
            seed_scores=seed_scores,
            expanded_chunks=expanded_chunks,
            expanded_scores=expanded_nodes,
            limit=limit,
        )

        logger.debug(
            "Multi-hop retrieval complete",
            intent=intent,
            seeds=len(seed_chunks),
            expanded=len(expanded_nodes),
            final=len(final_chunks),
            paths=len(paths),
        )

        return MultiHopResult(
            chunks=final_chunks,
            scores=final_scores,
            paths=paths,
            metadata={
                "intent": intent,
                "seed_count": len(seed_chunks),
                "expanded_count": len(expanded_nodes),
                "path_count": len(paths),
                "strategy_hops": len(strategy),
            },
        )

    def _find_nodes_for_chunks(self, chunks: list["Chunk"]) -> list[str]:
        """Find graph node IDs for chunks."""
        node_ids = []

        for chunk in chunks:
            # Try by chunk_id first
            node = self.graph.find_node_by_chunk(chunk.chunk_id)
            if node:
                node_ids.append(node.node_id)
                continue

            # Try by file + symbol
            if chunk.symbol_name:
                for graph_node in self.graph.get_nodes().values():
                    if (graph_node.file_path == chunk.file_path and
                        graph_node.name == chunk.symbol_name):
                        node_ids.append(graph_node.node_id)
                        break

        return node_ids

    def _traverse_multihop(
        self,
        seed_nodes: list[str],
        seed_scores: dict[str, float],
        strategy: list[HopConfig],
    ) -> tuple[dict[str, float], list[MultiHopPath]]:
        """
        Execute multi-hop traversal with strategy.

        Returns:
            Tuple of (node_id -> relevance_score, discovered_paths)
        """
        # Node relevance scores (accumulated across hops)
        node_scores: dict[str, float] = {}
        for node_id in seed_nodes:
            node_scores[node_id] = seed_scores.get(node_id, 1.0)

        # Track paths for explainability
        all_paths: list[MultiHopPath] = []

        # Current frontier for BFS
        current_frontier = set(seed_nodes)

        # Execute each hop in strategy
        for hop_idx, hop_config in enumerate(strategy):
            if len(node_scores) >= self.max_total_nodes:
                break

            next_frontier: set[str] = set()

            for node_id in current_frontier:
                current_score = node_scores.get(node_id, 0.0)

                # Skip low-relevance nodes
                if current_score < self.min_relevance_threshold:
                    continue

                # Get neighbors based on hop config
                neighbors = self._get_neighbors(
                    node_id=node_id,
                    edge_types=hop_config.edge_types,
                    direction=hop_config.direction,
                    max_count=hop_config.max_neighbors,
                )

                for neighbor_id, edge_type in neighbors:
                    # Compute decayed relevance
                    new_score = current_score * hop_config.weight_decay

                    # Track path
                    path = MultiHopPath(
                        nodes=[node_id, neighbor_id],
                        edge_types=[edge_type],
                        total_weight=new_score,
                        depth=hop_idx + 1,
                    )
                    all_paths.append(path)

                    # Update node score (max of existing and new)
                    if neighbor_id not in node_scores:
                        node_scores[neighbor_id] = new_score
                        next_frontier.add(neighbor_id)
                    else:
                        node_scores[neighbor_id] = max(
                            node_scores[neighbor_id],
                            new_score
                        )

                    # Check expansion limit
                    if len(node_scores) >= self.max_total_nodes:
                        break

                if len(node_scores) >= self.max_total_nodes:
                    break

            current_frontier = next_frontier

        # Remove seed nodes from expanded (they're already in results)
        for seed in seed_nodes:
            if seed in node_scores:
                del node_scores[seed]

        return node_scores, all_paths

    def _get_neighbors(
        self,
        node_id: str,
        edge_types: list[str],
        direction: TraversalDirection,
        max_count: int,
    ) -> list[tuple[str, str]]:
        """
        Get neighbors of a node filtered by edge types and direction.

        Returns:
            List of (neighbor_id, edge_type) tuples.
        """
        neighbors: list[tuple[str, str]] = []

        for edge in self.graph.get_edges():
            edge_type_str = edge.edge_type.value.upper()

            if edge_type_str not in [et.upper() for et in edge_types]:
                continue

            if direction == TraversalDirection.FORWARD:
                if edge.source_id == node_id:
                    neighbors.append((edge.target_id, edge_type_str))
            elif direction == TraversalDirection.BACKWARD:
                if edge.target_id == node_id:
                    neighbors.append((edge.source_id, edge_type_str))
            else:  # BOTH
                if edge.source_id == node_id:
                    neighbors.append((edge.target_id, edge_type_str))
                elif edge.target_id == node_id:
                    neighbors.append((edge.source_id, edge_type_str))

            if len(neighbors) >= max_count:
                break

        return neighbors

    async def _get_chunks_for_nodes(
        self,
        node_scores: dict[str, float]
    ) -> dict[str, tuple["Chunk", float]]:
        """Get chunks for expanded nodes with their scores."""
        result: dict[str, tuple["Chunk", float]] = {}

        for node_id, score in node_scores.items():
            node = self.graph.get_nodes().get(node_id)
            if not node or not node.chunk_id:
                continue

            # Retrieve chunk
            chunks = await self.base_retriever.retrieve_by_ids([node.chunk_id])
            if chunks:
                result[node.chunk_id] = (chunks[0], score)

        return result

    def _merge_and_score(
        self,
        seed_chunks: list["Chunk"],
        seed_scores: list[float],
        expanded_chunks: dict[str, tuple["Chunk", float]],
        expanded_scores: dict[str, float],
        limit: int,
    ) -> tuple[list["Chunk"], list[float]]:
        """Merge seed and expanded chunks, compute final scores."""
        seen_ids: set[str] = set()
        combined: list[tuple["Chunk", float]] = []

        # Add seeds first
        for chunk, score in zip(seed_chunks, seed_scores):
            if chunk.chunk_id not in seen_ids:
                seen_ids.add(chunk.chunk_id)
                combined.append((chunk, score))

        # Add expanded
        for chunk_id, (chunk, score) in expanded_chunks.items():
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                combined.append((chunk, score))

        # Sort by score
        combined.sort(key=lambda x: x[1], reverse=True)

        # Take top limit
        combined = combined[:limit]

        return [c for c, _ in combined], [s for _, s in combined]

    def get_impact_analysis(
        self,
        chunk: "Chunk",
        max_depth: int = 3,
    ) -> dict[str, Any]:
        """
        Analyze impact of changes to a chunk.

        Returns nodes that would be affected if this chunk changes.
        """
        # Find node for chunk
        node_id = None
        if chunk.symbol_name:
            for node in self.graph.get_nodes().values():
                if (node.file_path == chunk.file_path and
                    node.name == chunk.symbol_name):
                    node_id = node.node_id
                    break

        if not node_id:
            return {"affected_nodes": [], "depth_distribution": {}}

        # BFS for affected nodes
        affected: dict[str, int] = {}  # node_id -> depth
        queue = deque([(node_id, 0)])
        visited = {node_id}

        while queue:
            current_id, depth = queue.popleft()

            if depth >= max_depth:
                continue

            # Find nodes that depend on current (backward edges)
            for edge in self.graph.get_edges():
                if edge.target_id == current_id:
                    if edge.edge_type.value.upper() in ["CALLS", "IMPORTS", "USES_TYPE"]:
                        if edge.source_id not in visited:
                            visited.add(edge.source_id)
                            affected[edge.source_id] = depth + 1
                            queue.append((edge.source_id, depth + 1))

        # Compute depth distribution
        depth_dist: dict[int, int] = defaultdict(int)
        for node_id, depth in affected.items():
            depth_dist[depth] += 1

        return {
            "affected_nodes": list(affected.keys()),
            "affected_count": len(affected),
            "depth_distribution": dict(depth_dist),
            "max_depth_reached": max(affected.values()) if affected else 0,
        }


def create_multihop_retriever(
    config: "Config",
    graph_builder: "CodeGraphBuilder",
    base_retriever: Any,
) -> MultiHopRetriever:
    """Create a multi-hop retriever instance."""
    return MultiHopRetriever(config, graph_builder, base_retriever)
