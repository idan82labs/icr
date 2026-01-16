"""
Knapsack-based pack compiler.

Compiles retrieved chunks into a context pack that fits within
a token budget, maximizing utility.

Implements: max Σ u_i  s.t. Σ c_i ≤ B
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from icd.config import Config
    from icd.retrieval.hybrid import Chunk

logger = structlog.get_logger(__name__)


@dataclass
class PackResult:
    """Result from pack compilation."""

    content: str
    token_count: int
    chunk_ids: list[str]
    citations: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PackItem:
    """An item for the knapsack problem."""

    chunk: "Chunk"
    score: float
    utility: float
    cost: int  # token count
    selected: bool = False


class PackCompiler:
    """
    Knapsack-based pack compiler.

    Features:
    - Dynamic programming knapsack for optimal selection
    - Utility computation considering relevance, contracts, pinned
    - Token-based cost model
    - Citation generation
    """

    def __init__(self, config: "Config") -> None:
        """
        Initialize the pack compiler.

        Args:
            config: ICD configuration.
        """
        self.config = config
        self.default_budget = config.pack.default_budget_tokens
        self.max_budget = config.pack.max_budget_tokens
        self.include_metadata = config.pack.include_metadata
        self.include_citations = config.pack.include_citations

        # Utility weights
        self.score_weight = 0.6
        self.contract_bonus = 0.2
        self.pinned_bonus = 0.2

    async def compile(
        self,
        chunks: list["Chunk"],
        scores: list[float],
        budget_tokens: int | None = None,
        query: str | None = None,
    ) -> PackResult:
        """
        Compile chunks into a context pack.

        Args:
            chunks: Retrieved chunks.
            scores: Corresponding scores.
            budget_tokens: Token budget for the pack.
            query: Original query (for context).

        Returns:
            PackResult with compiled content.
        """
        budget = min(budget_tokens or self.default_budget, self.max_budget)

        logger.debug(
            "Compiling pack",
            num_chunks=len(chunks),
            budget_tokens=budget,
        )

        # Create pack items with utility
        items = self._create_items(chunks, scores)

        # Solve knapsack
        selected_items = self._knapsack(items, budget)

        # Sort selected by file and line for coherence
        selected_items.sort(
            key=lambda x: (x.chunk.file_path, x.chunk.start_line)
        )

        # Generate citations
        citations = self._generate_citations(selected_items)

        # Format pack
        from icd.pack.formatter import PackFormatter

        formatter = PackFormatter(
            include_metadata=self.include_metadata,
            include_citations=self.include_citations,
        )

        content = formatter.format(
            items=selected_items,
            citations=citations,
            query=query,
        )

        # Compute total tokens
        total_tokens = sum(item.cost for item in selected_items)

        # Metadata
        metadata = {
            "query": query,
            "budget": budget,
            "total_utility": sum(item.utility for item in selected_items),
            "num_chunks": len(selected_items),
            "num_contracts": sum(
                1 for item in selected_items if item.chunk.is_contract
            ),
            "num_pinned": sum(
                1 for item in selected_items if item.chunk.is_pinned
            ),
        }

        logger.debug(
            "Pack compiled",
            num_selected=len(selected_items),
            total_tokens=total_tokens,
        )

        return PackResult(
            content=content,
            token_count=total_tokens,
            chunk_ids=[item.chunk.chunk_id for item in selected_items],
            citations=citations,
            metadata=metadata,
        )

    def _create_items(
        self,
        chunks: list["Chunk"],
        scores: list[float],
    ) -> list[PackItem]:
        """Create pack items with utility scores."""
        items = []

        # Normalize scores
        max_score = max(scores) if scores else 1.0
        if max_score == 0:
            max_score = 1.0

        for chunk, score in zip(chunks, scores):
            normalized_score = score / max_score

            # Compute utility
            utility = self._compute_utility(chunk, normalized_score)

            items.append(
                PackItem(
                    chunk=chunk,
                    score=score,
                    utility=utility,
                    cost=chunk.token_count,
                )
            )

        return items

    def _compute_utility(
        self,
        chunk: "Chunk",
        normalized_score: float,
    ) -> float:
        """
        Compute utility for a chunk.

        utility = w_s·score + w_c·is_contract + w_p·is_pinned
        """
        utility = self.score_weight * normalized_score

        if chunk.is_contract:
            utility += self.contract_bonus

        if chunk.is_pinned:
            utility += self.pinned_bonus

        return utility

    def _knapsack(
        self,
        items: list[PackItem],
        budget: int,
    ) -> list[PackItem]:
        """
        Solve the 0/1 knapsack problem using dynamic programming.

        max Σ u_i  s.t. Σ c_i ≤ B

        Args:
            items: Pack items with utility and cost.
            budget: Token budget.

        Returns:
            Selected items.
        """
        n = len(items)
        if n == 0:
            return []

        # For efficiency, scale costs to reduce DP table size
        # We use a resolution of 10 tokens
        resolution = 10
        scaled_budget = budget // resolution

        # Scale costs
        scaled_costs = [max(1, item.cost // resolution) for item in items]

        # DP table: dp[i][w] = max utility using items 0..i-1 with capacity w
        # Use space-optimized version (only keep previous row)
        prev = [0.0] * (scaled_budget + 1)
        curr = [0.0] * (scaled_budget + 1)

        # Track selections
        selection_table = [[False] * (scaled_budget + 1) for _ in range(n)]

        for i in range(n):
            for w in range(scaled_budget + 1):
                # Don't take item i
                curr[w] = prev[w]

                # Take item i if it fits
                if scaled_costs[i] <= w:
                    take_value = items[i].utility + prev[w - scaled_costs[i]]
                    if take_value > curr[w]:
                        curr[w] = take_value
                        selection_table[i][w] = True

            # Swap rows
            prev, curr = curr, prev

        # Backtrack to find selected items
        selected = []
        w = scaled_budget

        for i in range(n - 1, -1, -1):
            if selection_table[i][w]:
                selected.append(items[i])
                items[i].selected = True
                w -= scaled_costs[i]

        # Verify budget constraint with actual costs
        total_cost = sum(item.cost for item in selected)
        if total_cost > budget:
            # Remove lowest utility items until within budget
            selected.sort(key=lambda x: x.utility)
            while total_cost > budget and selected:
                removed = selected.pop(0)
                removed.selected = False
                total_cost -= removed.cost

        return selected

    def _generate_citations(
        self,
        items: list[PackItem],
    ) -> dict[str, str]:
        """Generate citation references for items."""
        citations = {}

        for i, item in enumerate(items, 1):
            cite_key = f"[{i}]"
            cite_value = f"{item.chunk.file_path}:{item.chunk.start_line}"

            if item.chunk.symbol_name:
                cite_value += f" ({item.chunk.symbol_name})"

            citations[cite_key] = cite_value

        return citations

    def estimate_pack_size(
        self,
        chunks: list["Chunk"],
    ) -> int:
        """Estimate the token size of a pack."""
        # Base overhead
        overhead = 100

        # Chunk content
        content_tokens = sum(c.token_count for c in chunks)

        # Metadata overhead per chunk
        metadata_overhead = 20 * len(chunks)

        return overhead + content_tokens + metadata_overhead


class IncrementalPackCompiler:
    """
    Incremental pack compiler for streaming scenarios.

    Builds packs incrementally as chunks arrive.
    """

    def __init__(self, config: "Config") -> None:
        """
        Initialize the incremental compiler.

        Args:
            config: ICD configuration.
        """
        self.config = config
        self.budget = config.pack.default_budget_tokens
        self._items: list[PackItem] = []
        self._selected: list[PackItem] = []
        self._used_budget = 0

    def add_chunk(
        self,
        chunk: "Chunk",
        score: float,
    ) -> bool:
        """
        Add a chunk to the pack.

        Returns True if the chunk was added.

        Args:
            chunk: Chunk to add.
            score: Chunk score.

        Returns:
            True if chunk was added to pack.
        """
        # Create item
        utility = 0.6 * score
        if chunk.is_contract:
            utility += 0.2
        if chunk.is_pinned:
            utility += 0.2

        item = PackItem(
            chunk=chunk,
            score=score,
            utility=utility,
            cost=chunk.token_count,
        )

        self._items.append(item)

        # Check if it fits
        if self._used_budget + item.cost <= self.budget:
            self._selected.append(item)
            item.selected = True
            self._used_budget += item.cost
            return True

        # Check if it's better than existing items
        # Find lowest utility selected item
        if self._selected:
            min_item = min(self._selected, key=lambda x: x.utility)
            if item.utility > min_item.utility:
                # Try to make room
                if self._used_budget - min_item.cost + item.cost <= self.budget:
                    self._selected.remove(min_item)
                    min_item.selected = False
                    self._used_budget -= min_item.cost

                    self._selected.append(item)
                    item.selected = True
                    self._used_budget += item.cost
                    return True

        return False

    def get_pack(self, query: str | None = None) -> PackResult:
        """
        Get the current pack.

        Args:
            query: Original query.

        Returns:
            PackResult with current pack.
        """
        from icd.pack.formatter import PackFormatter

        # Sort selected
        selected = sorted(
            self._selected,
            key=lambda x: (x.chunk.file_path, x.chunk.start_line),
        )

        # Generate citations
        citations = {}
        for i, item in enumerate(selected, 1):
            cite_key = f"[{i}]"
            cite_value = f"{item.chunk.file_path}:{item.chunk.start_line}"
            if item.chunk.symbol_name:
                cite_value += f" ({item.chunk.symbol_name})"
            citations[cite_key] = cite_value

        # Format
        formatter = PackFormatter(
            include_metadata=self.config.pack.include_metadata,
            include_citations=self.config.pack.include_citations,
        )

        content = formatter.format(
            items=selected,
            citations=citations,
            query=query,
        )

        return PackResult(
            content=content,
            token_count=self._used_budget,
            chunk_ids=[item.chunk.chunk_id for item in selected],
            citations=citations,
            metadata={
                "query": query,
                "total_utility": sum(item.utility for item in selected),
                "num_chunks": len(selected),
            },
        )

    def reset(self) -> None:
        """Reset the compiler state."""
        self._items.clear()
        self._selected.clear()
        self._used_budget = 0
