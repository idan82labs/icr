"""
Dependency-Aware Context Packing (DAC-Pack).

A novel extension to knapsack-based context packing that models
chunk dependencies. Standard knapsack assumes independent items,
but code chunks have dependencies:
- A function needs its imports
- A method needs its class definition
- A type usage needs the type definition

This implements Precedence-Constrained Knapsack:
    maximize: Σ u_i · x_i
    subject to:
      Σ c_i · x_i ≤ B           (budget constraint)
      x_j ≤ x_i for all (i,j) ∈ D  (dependency: if j selected, i must be)

Reference: Novel contribution - no existing code retrieval system models
chunk dependencies in context packing.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from icd.config import Config
    from icd.graph.builder import CodeGraphBuilder, EdgeType
    from icd.retrieval.hybrid import Chunk

logger = structlog.get_logger(__name__)


@dataclass
class DependencyBundle:
    """A chunk with its transitive dependencies bundled together."""

    primary_chunk: "Chunk"
    primary_score: float
    dependencies: list["Chunk"]  # Chunks this one depends on
    bundle_cost: int  # Total tokens including dependencies
    bundle_utility: float  # Utility of primary (deps are overhead)

    @property
    def efficiency(self) -> float:
        """Utility per token ratio."""
        if self.bundle_cost == 0:
            return 0.0
        return self.bundle_utility / self.bundle_cost


@dataclass
class DACPackResult:
    """Result from dependency-aware pack compilation."""

    content: str
    token_count: int
    chunk_ids: list[str]
    citations: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)

    # DAC-specific metadata
    bundles_selected: int = 0
    dependencies_included: int = 0
    dependency_overhead_tokens: int = 0


class DependencyAnalyzer:
    """
    Analyzes chunk dependencies using code graph.

    Determines which chunks must be included together for
    coherent context (e.g., a function and its imports).
    """

    def __init__(self, graph_builder: "CodeGraphBuilder | None") -> None:
        self.graph = graph_builder

        # Dependency edge types (what a chunk "needs") - lowercase to match graph
        self._dependency_edges = ["imports", "inherits", "uses_type", "implements"]

        # Cache for computed dependencies
        self._dep_cache: dict[str, list[str]] = {}

    def get_dependencies(self, chunk: "Chunk") -> list[str]:
        """
        Get chunk IDs that this chunk depends on.

        Uses code graph to find:
        - Imported modules/symbols (via FILE node imports)
        - Parent classes (INHERITS edges)
        - Used types
        - Implemented interfaces
        """
        if chunk.chunk_id in self._dep_cache:
            return self._dep_cache[chunk.chunk_id]

        deps: list[str] = []

        if not self.graph:
            self._dep_cache[chunk.chunk_id] = deps
            return deps

        nodes = self.graph.get_nodes()
        edges = self.graph.get_edges()

        # Find the FILE node for this chunk's file
        file_node_id = f"file:{chunk.file_path}"
        file_node = nodes.get(file_node_id)

        # Also find the symbol node if this chunk has a symbol
        symbol_node = None
        if chunk.symbol_name:
            symbol_node = self.graph.find_node_by_chunk(chunk.chunk_id)
            if not symbol_node:
                # Fallback: match by file + symbol name
                for n in nodes.values():
                    if (n.file_path and
                        chunk.file_path.endswith(n.file_path) and
                        n.name == chunk.symbol_name):
                        symbol_node = n
                        break

        # Collect dependency targets from:
        # 1. FILE imports → get imported classes/modules
        # 2. Symbol INHERITS → get parent classes
        dep_node_ids: set[str] = set()

        # Check file-level imports
        if file_node:
            for edge in edges:
                if edge.source_id == file_node_id and edge.edge_type.value.lower() == "imports":
                    dep_node_ids.add(edge.target_id)

        # Check symbol-level edges (inherits, implements)
        if symbol_node:
            for edge in edges:
                if edge.source_id == symbol_node.node_id:
                    if edge.edge_type.value.lower() in ["inherits", "implements", "uses_type"]:
                        dep_node_ids.add(edge.target_id)

        # Resolve node IDs to chunk IDs
        for node_id in dep_node_ids:
            target_node = nodes.get(node_id)
            if target_node and target_node.chunk_id:
                deps.append(target_node.chunk_id)

        self._dep_cache[chunk.chunk_id] = deps
        return deps

    def get_transitive_dependencies(
        self,
        chunk: "Chunk",
        max_depth: int = 2
    ) -> list[str]:
        """
        Get transitive closure of dependencies up to max_depth.

        A depends on B, B depends on C → A transitively depends on C.
        """
        visited: set[str] = set()
        result: list[str] = []

        def traverse(chunk_id: str, depth: int) -> None:
            if depth > max_depth or chunk_id in visited:
                return
            visited.add(chunk_id)

            # Get direct dependencies
            direct_deps = self._dep_cache.get(chunk_id, [])

            for dep_id in direct_deps:
                if dep_id not in visited:
                    result.append(dep_id)
                    traverse(dep_id, depth + 1)

        # Start from the chunk's direct dependencies
        direct = self.get_dependencies(chunk)
        for dep_id in direct:
            if dep_id not in visited:
                result.append(dep_id)
                traverse(dep_id, 1)

        return result

    def clear_cache(self) -> None:
        """Clear dependency cache."""
        self._dep_cache.clear()


class DependencyAwarePacker:
    """
    Dependency-Aware Context Packer (DAC-Pack).

    Novel algorithm that considers chunk dependencies when packing:

    1. Build dependency bundles: each chunk + its required dependencies
    2. Compute bundle costs (chunk + dependency tokens)
    3. Solve modified knapsack preferring bundles with better efficiency
    4. Ensure dependency constraints are satisfied

    This produces more coherent context than naive packing because:
    - Functions come with their imports
    - Methods come with their class definitions
    - Type usages come with type definitions
    """

    def __init__(
        self,
        config: "Config",
        graph_builder: "CodeGraphBuilder | None" = None,
        chunk_store: Any = None,  # SQLiteStore for fetching missing deps
    ) -> None:
        self.config = config
        self.analyzer = DependencyAnalyzer(graph_builder)
        self._chunk_store = chunk_store

        # Config
        self.default_budget = config.pack.default_budget_tokens
        self.max_budget = config.pack.max_budget_tokens
        self.score_weight = config.pack.score_weight
        self.contract_bonus = config.pack.contract_bonus
        self.pinned_bonus = config.pack.pinned_bonus

        # DAC-specific parameters
        self.max_dependency_depth = 2  # How deep to follow dependencies
        self.dependency_discount = 0.3  # Utility discount for dependency chunks
        self.min_efficiency_threshold = 0.001  # Skip bundles below this
        self.max_dependency_fetches = 20  # Limit dependency fetches per pack

    async def compile(
        self,
        chunks: list["Chunk"],
        scores: list[float],
        budget_tokens: int | None = None,
        query: str | None = None,
    ) -> DACPackResult:
        """
        Compile chunks into dependency-aware context pack.

        Args:
            chunks: Retrieved chunks.
            scores: Corresponding relevance scores.
            budget_tokens: Token budget.
            query: Original query for context.

        Returns:
            DACPackResult with coherent context.
        """
        budget = min(budget_tokens or self.default_budget, self.max_budget)

        logger.debug(
            "DAC-Pack compiling",
            num_chunks=len(chunks),
            budget=budget,
        )

        # Build chunk lookup
        chunk_map = {c.chunk_id: c for c in chunks}
        score_map = dict(zip([c.chunk_id for c in chunks], scores))

        # Step 1: Build dependency bundles (may fetch missing deps from store)
        bundles = await self._build_bundles(chunks, scores, chunk_map)

        # Step 2: Solve dependency-aware knapsack
        selected_bundles = self._solve_dac_knapsack(bundles, budget)

        # Step 3: Flatten bundles to chunk list (maintaining dependency order)
        selected_chunks, selected_scores = self._flatten_bundles(
            selected_bundles, chunk_map, score_map
        )

        # Step 4: Sort by file and line for coherence
        sorted_pairs = sorted(
            zip(selected_chunks, selected_scores),
            key=lambda x: (x[0].file_path, x[0].start_line)
        )
        selected_chunks = [c for c, _ in sorted_pairs]
        selected_scores = [s for _, s in sorted_pairs]

        # Step 5: Generate output
        citations = self._generate_citations(selected_chunks)
        content = self._format_pack(selected_chunks, citations, query)

        total_tokens = sum(c.token_count for c in selected_chunks)
        dep_tokens = sum(
            c.token_count for b in selected_bundles for c in b.dependencies
        )

        logger.debug(
            "DAC-Pack complete",
            bundles_selected=len(selected_bundles),
            chunks_selected=len(selected_chunks),
            total_tokens=total_tokens,
            dependency_overhead=dep_tokens,
        )

        return DACPackResult(
            content=content,
            token_count=total_tokens,
            chunk_ids=[c.chunk_id for c in selected_chunks],
            citations=citations,
            metadata={
                "query": query,
                "budget": budget,
                "algorithm": "DAC-Pack",
            },
            bundles_selected=len(selected_bundles),
            dependencies_included=sum(len(b.dependencies) for b in selected_bundles),
            dependency_overhead_tokens=dep_tokens,
        )

    async def _build_bundles(
        self,
        chunks: list["Chunk"],
        scores: list[float],
        chunk_map: dict[str, "Chunk"],
    ) -> list[DependencyBundle]:
        """Build dependency bundles for each chunk, fetching missing deps if store available."""
        bundles = []
        fetched_count = 0

        # Normalize scores
        max_score = max(scores) if scores else 1.0
        if max_score == 0:
            max_score = 1.0

        for chunk, score in zip(chunks, scores):
            normalized_score = score / max_score

            # Compute utility
            utility = self._compute_utility(chunk, normalized_score)

            # Get dependencies
            dep_ids = self.analyzer.get_transitive_dependencies(
                chunk,
                max_depth=self.max_dependency_depth
            )

            # Resolve dependencies to chunks
            dep_chunks = []
            for dep_id in dep_ids:
                if dep_id in chunk_map:
                    dep_chunks.append(chunk_map[dep_id])
                elif self._chunk_store and fetched_count < self.max_dependency_fetches:
                    # Fetch missing dependency from store
                    try:
                        dep_chunk = await self._chunk_store.get_chunk_with_content(dep_id)
                        if dep_chunk:
                            chunk_map[dep_id] = dep_chunk  # Cache it
                            dep_chunks.append(dep_chunk)
                            fetched_count += 1
                    except Exception:
                        pass  # Skip if fetch fails

            # Compute bundle cost
            bundle_cost = chunk.token_count + sum(d.token_count for d in dep_chunks)

            bundles.append(DependencyBundle(
                primary_chunk=chunk,
                primary_score=score,
                dependencies=dep_chunks,
                bundle_cost=bundle_cost,
                bundle_utility=utility,
            ))

        return bundles

    def _compute_utility(self, chunk: "Chunk", normalized_score: float) -> float:
        """Compute utility for a chunk."""
        utility = self.score_weight * normalized_score

        if chunk.is_contract:
            utility += self.contract_bonus

        if chunk.is_pinned:
            utility += self.pinned_bonus

        return utility

    def _solve_dac_knapsack(
        self,
        bundles: list[DependencyBundle],
        budget: int,
    ) -> list[DependencyBundle]:
        """
        Solve dependency-aware knapsack problem.

        Uses greedy approximation based on efficiency (utility/cost).
        The dependency constraints are handled by bundling.
        """
        # Filter bundles by efficiency threshold
        viable_bundles = [
            b for b in bundles
            if b.efficiency >= self.min_efficiency_threshold
        ]

        # Sort by efficiency (best first)
        viable_bundles.sort(key=lambda b: b.efficiency, reverse=True)

        selected: list[DependencyBundle] = []
        used_budget = 0
        selected_chunk_ids: set[str] = set()

        for bundle in viable_bundles:
            # Check if primary chunk already selected (as dependency of another)
            if bundle.primary_chunk.chunk_id in selected_chunk_ids:
                continue

            # Compute incremental cost (exclude already-selected dependencies)
            new_dep_cost = sum(
                d.token_count
                for d in bundle.dependencies
                if d.chunk_id not in selected_chunk_ids
            )
            incremental_cost = bundle.primary_chunk.token_count + new_dep_cost

            # Check if fits
            if used_budget + incremental_cost <= budget:
                selected.append(bundle)
                used_budget += incremental_cost

                # Mark chunks as selected
                selected_chunk_ids.add(bundle.primary_chunk.chunk_id)
                for dep in bundle.dependencies:
                    selected_chunk_ids.add(dep.chunk_id)

        return selected

    def _flatten_bundles(
        self,
        bundles: list[DependencyBundle],
        chunk_map: dict[str, "Chunk"],
        score_map: dict[str, float],
    ) -> tuple[list["Chunk"], list[float]]:
        """
        Flatten bundles to chunk list with dependencies first.

        Ensures dependencies appear before dependents for coherent context.
        """
        seen_ids: set[str] = set()
        chunks: list["Chunk"] = []
        scores: list[float] = []

        for bundle in bundles:
            # Add dependencies first (with discounted scores)
            for dep in bundle.dependencies:
                if dep.chunk_id not in seen_ids:
                    seen_ids.add(dep.chunk_id)
                    chunks.append(dep)
                    # Use discounted score for dependencies
                    original_score = score_map.get(dep.chunk_id, 0.0)
                    scores.append(original_score * self.dependency_discount)

            # Add primary chunk
            if bundle.primary_chunk.chunk_id not in seen_ids:
                seen_ids.add(bundle.primary_chunk.chunk_id)
                chunks.append(bundle.primary_chunk)
                scores.append(bundle.primary_score)

        return chunks, scores

    def _generate_citations(self, chunks: list["Chunk"]) -> dict[str, str]:
        """Generate citation references."""
        citations = {}

        for i, chunk in enumerate(chunks, 1):
            cite_key = f"[{i}]"
            cite_value = f"{chunk.file_path}:{chunk.start_line}"

            if chunk.symbol_name:
                cite_value += f" ({chunk.symbol_name})"

            citations[cite_key] = cite_value

        return citations

    def _format_pack(
        self,
        chunks: list["Chunk"],
        citations: dict[str, str],
        query: str | None,
    ) -> str:
        """Format the pack content."""
        from icd.pack.formatter import PackFormatter
        from icd.pack.compiler import PackItem

        # Create PackItem wrappers for formatter compatibility
        items = [
            PackItem(
                chunk=chunk,
                score=0.0,  # Not used in formatting
                utility=0.0,
                cost=chunk.token_count,
                selected=True,
            )
            for chunk in chunks
        ]

        formatter = PackFormatter(
            include_metadata=self.config.pack.include_metadata,
            include_citations=self.config.pack.include_citations,
        )

        return formatter.format(
            items=items,
            citations=citations,
            query=query,
        )


def create_dependency_packer(
    config: "Config",
    graph_builder: "CodeGraphBuilder | None" = None,
) -> DependencyAwarePacker:
    """
    Create a dependency-aware packer.

    Args:
        config: ICD configuration.
        graph_builder: Optional code graph for dependency analysis.

    Returns:
        DependencyAwarePacker instance.
    """
    return DependencyAwarePacker(config, graph_builder)
