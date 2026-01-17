"""
Graph-aware retrieval for multi-hop context expansion.

Implements graph traversal strategies for retrieving related code:
- Follow imports/exports to find dependencies
- Follow call graphs to find callers/callees
- Follow inheritance chains
- Community detection for cohesive modules

Reference: "GraphRAG" (Microsoft, 2024), "Knowledge Graphs for RAG"
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog

from icd.graph.builder import CodeGraphBuilder, EdgeType, GraphNode

if TYPE_CHECKING:
    from icd.config import Config
    from icd.retrieval.hybrid import Chunk, RetrievalResult

logger = structlog.get_logger(__name__)


@dataclass
class GraphExpansionResult:
    """Result of graph-based context expansion."""

    original_chunks: list["Chunk"]
    expanded_chunks: list["Chunk"]
    paths: list[list[str]]  # Paths taken during expansion
    metadata: dict[str, Any]


class GraphRetriever:
    """
    Graph-aware retrieval system.

    Enhances retrieval by following structural relationships:
    1. Start with initial retrieval results
    2. Find graph nodes corresponding to retrieved chunks
    3. Traverse graph to find related code
    4. Retrieve chunks for related nodes
    5. Merge and re-rank
    """

    def __init__(
        self,
        config: "Config",
        graph_builder: CodeGraphBuilder,
        base_retriever: Any,  # HybridRetriever
    ) -> None:
        self.config = config
        self.graph = graph_builder
        self.base_retriever = base_retriever

        # Expansion parameters
        self.max_hops = 2  # Maximum traversal depth
        self.max_expanded_chunks = 10  # Maximum additional chunks from expansion
        self.expansion_weight = 0.5  # Weight for expanded chunks vs original

        # Edge type priorities for different query types
        self.edge_priorities = {
            "implementation": [EdgeType.CALLS, EdgeType.CONTAINS, EdgeType.IMPORTS],
            "interface": [EdgeType.IMPLEMENTS, EdgeType.INHERITS, EdgeType.USES_TYPE],
            "usage": [EdgeType.CALLS, EdgeType.IMPORTS, EdgeType.REFERENCES],
            "default": [EdgeType.CALLS, EdgeType.INHERITS, EdgeType.IMPORTS],
        }

    async def retrieve_with_expansion(
        self,
        query: str,
        limit: int | None = None,
        query_type: str = "default",
        **kwargs,
    ) -> "RetrievalResult":
        """
        Retrieve with graph-based context expansion.

        Args:
            query: Natural language query.
            limit: Maximum results.
            query_type: Type hint for edge prioritization.
            **kwargs: Additional retrieval params.

        Returns:
            Enhanced retrieval result with graph-expanded context.
        """
        from icd.retrieval.hybrid import RetrievalResult

        # Step 1: Initial retrieval
        initial_result = await self.base_retriever.retrieve(
            query=query,
            limit=limit,
            **kwargs,
        )

        if not initial_result.chunks:
            return initial_result

        # Step 2: Find graph nodes for retrieved chunks
        seed_nodes = self._find_nodes_for_chunks(initial_result.chunks)

        if not seed_nodes:
            logger.debug("No graph nodes found for retrieved chunks")
            return initial_result

        # Step 3: Expand via graph traversal
        expanded_node_ids = self._expand_via_graph(
            seed_nodes=seed_nodes,
            query_type=query_type,
        )

        if not expanded_node_ids:
            return initial_result

        # Step 4: Get chunks for expanded nodes
        expanded_chunks = await self._get_chunks_for_nodes(expanded_node_ids)

        if not expanded_chunks:
            return initial_result

        # Step 5: Merge and re-rank
        merged_result = self._merge_results(
            original=initial_result,
            expanded=expanded_chunks,
            limit=limit or 20,
        )

        merged_result.metadata["graph_expansion"] = {
            "seed_nodes": len(seed_nodes),
            "expanded_nodes": len(expanded_node_ids),
            "expanded_chunks": len(expanded_chunks),
        }

        return merged_result

    def _find_nodes_for_chunks(self, chunks: list["Chunk"]) -> list[str]:
        """Find graph nodes corresponding to retrieved chunks."""
        node_ids = []

        for chunk in chunks:
            # Try to find by chunk ID
            node = self.graph.find_node_by_chunk(chunk.chunk_id)
            if node:
                node_ids.append(node.node_id)
                continue

            # Try to find by file + symbol name
            if chunk.symbol_name:
                for graph_node in self.graph.get_nodes().values():
                    if (
                        graph_node.file_path == chunk.file_path
                        and graph_node.name == chunk.symbol_name
                    ):
                        node_ids.append(graph_node.node_id)
                        break

        return node_ids

    def _expand_via_graph(
        self,
        seed_nodes: list[str],
        query_type: str,
    ) -> list[str]:
        """
        Expand seed nodes via BFS traversal.

        Uses edge type priorities based on query type.
        """
        edge_types = self.edge_priorities.get(query_type, self.edge_priorities["default"])

        visited: set[str] = set(seed_nodes)
        expanded: list[str] = []

        # BFS with depth limit
        queue: deque[tuple[str, int]] = deque((n, 0) for n in seed_nodes)

        while queue and len(expanded) < self.max_expanded_chunks:
            node_id, depth = queue.popleft()

            if depth >= self.max_hops:
                continue

            # Get neighbors via prioritized edge types
            neighbors = self.graph.get_neighbors(node_id, edge_types)

            for neighbor_id in neighbors:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    expanded.append(neighbor_id)
                    queue.append((neighbor_id, depth + 1))

                    if len(expanded) >= self.max_expanded_chunks:
                        break

        logger.debug(
            "Graph expansion complete",
            seeds=len(seed_nodes),
            expanded=len(expanded),
            max_depth=self.max_hops,
        )

        return expanded

    async def _get_chunks_for_nodes(self, node_ids: list[str]) -> list["Chunk"]:
        """Get chunks corresponding to graph nodes."""
        chunks = []
        seen_chunk_ids: set[str] = set()

        for node_id in node_ids:
            node = self.graph.get_nodes().get(node_id)
            if not node or not node.chunk_id:
                continue

            if node.chunk_id in seen_chunk_ids:
                continue

            # Retrieve the chunk
            chunk_list = await self.base_retriever.retrieve_by_ids([node.chunk_id])
            if chunk_list:
                chunks.extend(chunk_list)
                seen_chunk_ids.add(node.chunk_id)

        return chunks

    def _merge_results(
        self,
        original: "RetrievalResult",
        expanded: list["Chunk"],
        limit: int,
    ) -> "RetrievalResult":
        """Merge original and expanded results."""
        from icd.retrieval.hybrid import RetrievalResult

        # Combine, deduplicating by chunk_id
        seen_ids: set[str] = set()
        combined_chunks: list["Chunk"] = []
        combined_scores: list[float] = []

        # Add original chunks with their scores
        for chunk, score in zip(original.chunks, original.scores):
            if chunk.chunk_id not in seen_ids:
                seen_ids.add(chunk.chunk_id)
                combined_chunks.append(chunk)
                combined_scores.append(score)

        # Add expanded chunks with discounted scores
        for chunk in expanded:
            if chunk.chunk_id not in seen_ids:
                seen_ids.add(chunk.chunk_id)
                combined_chunks.append(chunk)
                # Use expansion weight as score
                combined_scores.append(self.expansion_weight)

        # Sort by score
        sorted_pairs = sorted(
            zip(combined_chunks, combined_scores),
            key=lambda x: x[1],
            reverse=True,
        )

        # Take top limit
        sorted_pairs = sorted_pairs[:limit]

        return RetrievalResult(
            chunks=[c for c, _ in sorted_pairs],
            scores=[s for _, s in sorted_pairs],
            entropy=original.entropy,  # Keep original entropy
            query=original.query,
            metadata=original.metadata.copy(),
        )

    def get_dependency_chain(
        self,
        start_chunk: "Chunk",
        direction: str = "both",  # "callers", "callees", "both"
        max_depth: int = 3,
    ) -> list[list[str]]:
        """
        Get dependency chains for a chunk.

        Useful for understanding impact of changes.

        Args:
            start_chunk: Starting chunk.
            direction: Which dependencies to follow.
            max_depth: Maximum chain length.

        Returns:
            List of dependency paths.
        """
        # Find starting node
        start_node = None
        if start_chunk.symbol_name:
            for node in self.graph.get_nodes().values():
                if (
                    node.file_path == start_chunk.file_path
                    and node.name == start_chunk.symbol_name
                ):
                    start_node = node
                    break

        if not start_node:
            return []

        paths: list[list[str]] = []

        # DFS for paths
        def dfs(node_id: str, path: list[str], depth: int) -> None:
            if depth >= max_depth:
                if len(path) > 1:
                    paths.append(path.copy())
                return

            neighbors = []

            if direction in ("callees", "both"):
                # Outgoing CALLS edges
                for edge in self.graph.get_edges():
                    if edge.source_id == node_id and edge.edge_type == EdgeType.CALLS:
                        neighbors.append(edge.target_id)

            if direction in ("callers", "both"):
                # Incoming CALLS edges
                for edge in self.graph.get_edges():
                    if edge.target_id == node_id and edge.edge_type == EdgeType.CALLS:
                        neighbors.append(edge.source_id)

            if not neighbors:
                if len(path) > 1:
                    paths.append(path.copy())
                return

            for neighbor_id in neighbors:
                if neighbor_id not in path:  # Avoid cycles
                    path.append(neighbor_id)
                    dfs(neighbor_id, path, depth + 1)
                    path.pop()

        dfs(start_node.node_id, [start_node.node_id], 0)

        return paths

    def find_related_interfaces(self, chunk: "Chunk") -> list[GraphNode]:
        """Find interfaces/types related to a chunk."""
        related = []

        # Find node for chunk
        node = None
        if chunk.symbol_name:
            for n in self.graph.get_nodes().values():
                if n.file_path == chunk.file_path and n.name == chunk.symbol_name:
                    node = n
                    break

        if not node:
            return []

        # Find interfaces via IMPLEMENTS, USES_TYPE edges
        for edge in self.graph.get_edges():
            if edge.source_id == node.node_id:
                if edge.edge_type in (EdgeType.IMPLEMENTS, EdgeType.USES_TYPE):
                    target_node = self.graph.get_nodes().get(edge.target_id)
                    if target_node:
                        related.append(target_node)

        return related

    def compute_pagerank(self) -> dict[str, float]:
        """
        Compute PageRank scores for code graph nodes.

        Higher scores indicate more "important" code (referenced by many).
        """
        try:
            import networkx as nx
        except ImportError:
            logger.warning("networkx not available for PageRank")
            return {}

        G = self.graph.to_networkx()

        if G.number_of_nodes() == 0:
            return {}

        try:
            scores = nx.pagerank(G, alpha=0.85)
            return scores
        except Exception as e:
            logger.debug("PageRank failed", error=str(e))
            return {}

    def find_communities(self) -> list[set[str]]:
        """
        Find cohesive communities in the code graph.

        Useful for understanding module boundaries.
        """
        try:
            import networkx as nx
            from networkx.algorithms import community
        except ImportError:
            logger.warning("networkx not available for community detection")
            return []

        G = self.graph.to_networkx()

        if G.number_of_nodes() == 0:
            return []

        try:
            # Use Louvain algorithm for community detection
            communities = community.louvain_communities(G.to_undirected())
            return [set(c) for c in communities]
        except Exception as e:
            logger.debug("Community detection failed", error=str(e))
            return []
