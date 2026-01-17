"""
RLM plan generation for iterative retrieval.

Generates retrieval plans when initial results have high entropy,
decomposing complex queries into focused sub-queries.

Supports two decomposition modes:
1. LLM-based (uses Claude to intelligently decompose queries)
2. Heuristic-based (rule-based decomposition, no API calls)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from icd.config import Config
    from icd.retrieval.hybrid import Chunk, RetrievalResult

logger = structlog.get_logger(__name__)


def is_llm_decomposition_available() -> bool:
    """Check if LLM decomposition is available (dynamic check)."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


class QueryType(str, Enum):
    """Types of sub-queries in a plan."""

    DEFINITION = "definition"  # Find definitions
    USAGE = "usage"  # Find usages/callers
    IMPLEMENTATION = "implementation"  # Find implementations
    RELATED = "related"  # Find related code
    CONTRACT = "contract"  # Find contracts/interfaces
    TEST = "test"  # Find tests
    DOCUMENTATION = "documentation"  # Find docs


@dataclass
class SubQuery:
    """A sub-query in the retrieval plan."""

    query: str
    query_type: QueryType
    priority: int
    focus_paths: list[str] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    completed: bool = False
    results: list["Chunk"] = field(default_factory=list)


@dataclass
class RetrievalPlan:
    """A plan for iterative retrieval."""

    original_query: str
    sub_queries: list[SubQuery]
    max_iterations: int
    current_iteration: int = 0
    completed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class RLMPlanner:
    """
    Planner for RLM iterative retrieval.

    Features:
    - Query decomposition
    - Priority-based sub-query ordering
    - Focus path inference
    - Iteration management
    """

    def __init__(self, config: "Config") -> None:
        """
        Initialize the RLM planner.

        Args:
            config: ICD configuration.
        """
        self.config = config
        self.max_iterations = config.rlm.max_iterations
        self.budget_per_iteration = config.rlm.budget_per_iteration

    def create_plan(
        self,
        query: str,
        initial_result: "RetrievalResult",
    ) -> RetrievalPlan:
        """
        Create a retrieval plan for a query.

        Args:
            query: Original user query.
            initial_result: Initial retrieval results.

        Returns:
            RetrievalPlan with decomposed sub-queries.
        """
        logger.debug("Creating RLM plan", query=query[:100])

        # Analyze query to determine decomposition strategy
        query_analysis = self._analyze_query(query)

        # Generate sub-queries based on analysis
        sub_queries = self._generate_sub_queries(
            query, query_analysis, initial_result
        )

        # Sort by priority
        sub_queries.sort(key=lambda sq: sq.priority, reverse=True)

        plan = RetrievalPlan(
            original_query=query,
            sub_queries=sub_queries,
            max_iterations=self.max_iterations,
            metadata={
                "query_analysis": query_analysis,
                "initial_entropy": initial_result.entropy,
                "initial_results": len(initial_result.chunks),
            },
        )

        logger.info(
            "Created RLM plan",
            num_sub_queries=len(sub_queries),
            max_iterations=self.max_iterations,
        )

        return plan

    async def create_plan_with_llm(
        self,
        query: str,
        initial_result: "RetrievalResult",
        use_llm: bool = True,
    ) -> RetrievalPlan:
        """
        Create a retrieval plan using LLM-based decomposition.

        This is the preferred method when an Anthropic API key is available.
        Falls back to heuristic decomposition if LLM is not available.

        Args:
            query: Original user query.
            initial_result: Initial retrieval results.
            use_llm: Whether to use LLM (if available).

        Returns:
            RetrievalPlan with decomposed sub-queries.
        """
        if use_llm and is_llm_decomposition_available():
            try:
                from icd.rlm.llm_decomposer import LLMDecomposer

                decomposer = LLMDecomposer()
                decomposition = await decomposer.decompose(query)

                # Convert LLM decomposition to sub-queries
                sub_queries = []
                focus_paths = self._extract_focus_paths(initial_result)

                for sq in decomposition.sub_queries:
                    sub_queries.append(SubQuery(
                        query=sq.query,
                        query_type=QueryType(sq.query_type.value),
                        priority=sq.priority,
                        focus_paths=focus_paths,
                    ))

                plan = RetrievalPlan(
                    original_query=query,
                    sub_queries=sub_queries,
                    max_iterations=self.max_iterations,
                    metadata={
                        "decomposition_method": "llm",
                        "llm_reasoning": decomposition.reasoning,
                        "used_llm": decomposition.used_llm,
                        "initial_entropy": initial_result.entropy,
                        "initial_results": len(initial_result.chunks),
                    },
                )

                logger.info(
                    "Created RLM plan with LLM",
                    num_sub_queries=len(sub_queries),
                    used_llm=decomposition.used_llm,
                    reasoning=decomposition.reasoning[:100],
                )

                return plan

            except Exception as e:
                logger.warning(f"LLM decomposition failed, using heuristics: {e}")

        # Fall back to heuristic-based plan
        return self.create_plan(query, initial_result)

    def _analyze_query(self, query: str) -> dict[str, Any]:
        """Analyze query to determine its characteristics."""
        analysis: dict[str, Any] = {
            "has_function_ref": False,
            "has_class_ref": False,
            "has_file_ref": False,
            "is_how_question": False,
            "is_what_question": False,
            "is_where_question": False,
            "has_implementation": False,
            "has_usage": False,
            "has_test": False,
            "entities": [],
        }

        query_lower = query.lower()

        # Detect question types
        analysis["is_how_question"] = query_lower.startswith("how")
        analysis["is_what_question"] = query_lower.startswith("what")
        analysis["is_where_question"] = query_lower.startswith("where")

        # Detect implementation/usage intent
        analysis["has_implementation"] = any(
            word in query_lower
            for word in ["implement", "implementation", "implements", "defined"]
        )
        analysis["has_usage"] = any(
            word in query_lower
            for word in ["use", "uses", "using", "called", "calls", "caller"]
        )
        analysis["has_test"] = any(
            word in query_lower for word in ["test", "tests", "testing", "spec"]
        )

        # Extract potential entity references
        # CamelCase patterns
        camel_case = re.findall(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b", query)
        analysis["entities"].extend(camel_case)
        if camel_case:
            analysis["has_class_ref"] = True

        # snake_case patterns
        snake_case = re.findall(r"\b[a-z]+(?:_[a-z]+)+\b", query)
        analysis["entities"].extend(snake_case)
        if snake_case:
            analysis["has_function_ref"] = True

        # File path patterns
        file_refs = re.findall(r"[\w/]+\.\w+", query)
        if file_refs:
            analysis["has_file_ref"] = True
            analysis["entities"].extend(file_refs)

        # Quoted strings
        quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', query)
        for match in quoted:
            entity = match[0] or match[1]
            analysis["entities"].append(entity)

        # Deduplicate entities
        analysis["entities"] = list(set(analysis["entities"]))

        return analysis

    def _generate_sub_queries(
        self,
        query: str,
        analysis: dict[str, Any],
        initial_result: "RetrievalResult",
    ) -> list[SubQuery]:
        """Generate sub-queries based on query analysis."""
        sub_queries: list[SubQuery] = []

        # Extract focus paths from initial results
        focus_paths = self._extract_focus_paths(initial_result)

        # Base priority
        priority = 10

        # If we have specific entities, search for their definitions first
        for entity in analysis.get("entities", [])[:5]:
            sub_queries.append(
                SubQuery(
                    query=f"definition of {entity}",
                    query_type=QueryType.DEFINITION,
                    priority=priority,
                    focus_paths=focus_paths,
                    constraints={"symbol_name": entity},
                )
            )
            priority -= 1

        # If implementation is requested
        if analysis.get("has_implementation"):
            for entity in analysis.get("entities", [])[:3]:
                sub_queries.append(
                    SubQuery(
                        query=f"implementation of {entity}",
                        query_type=QueryType.IMPLEMENTATION,
                        priority=priority,
                        focus_paths=focus_paths,
                    )
                )
                priority -= 1

        # If usage is requested
        if analysis.get("has_usage"):
            for entity in analysis.get("entities", [])[:3]:
                sub_queries.append(
                    SubQuery(
                        query=f"usages of {entity}",
                        query_type=QueryType.USAGE,
                        priority=priority,
                        focus_paths=focus_paths,
                    )
                )
                priority -= 1

        # If tests are requested
        if analysis.get("has_test"):
            sub_queries.append(
                SubQuery(
                    query=f"tests for {query}",
                    query_type=QueryType.TEST,
                    priority=priority,
                    constraints={"file_pattern": "*test*"},
                )
            )
            priority -= 1

        # Always look for related contracts
        sub_queries.append(
            SubQuery(
                query=f"interfaces and types related to {query}",
                query_type=QueryType.CONTRACT,
                priority=priority - 5,
                constraints={"is_contract": True},
            )
        )

        # "How" questions benefit from implementation details
        if analysis.get("is_how_question") and not analysis.get("has_implementation"):
            sub_queries.append(
                SubQuery(
                    query=query.replace("how", "implementation"),
                    query_type=QueryType.IMPLEMENTATION,
                    priority=priority,
                    focus_paths=focus_paths,
                )
            )
            priority -= 1

        # "Where" questions benefit from usage search
        if analysis.get("is_where_question") and not analysis.get("has_usage"):
            sub_queries.append(
                SubQuery(
                    query=query.replace("where", "files containing"),
                    query_type=QueryType.USAGE,
                    priority=priority,
                    focus_paths=focus_paths,
                )
            )
            priority -= 1

        # Add a refined version of the original query
        sub_queries.append(
            SubQuery(
                query=query,
                query_type=QueryType.RELATED,
                priority=1,
                focus_paths=focus_paths,
            )
        )

        return sub_queries

    def _extract_focus_paths(
        self,
        result: "RetrievalResult",
    ) -> list[str]:
        """Extract focus paths from initial results."""
        if not result.chunks:
            return []

        # Get unique directories from top results
        directories: dict[str, int] = {}
        for chunk in result.chunks[:10]:
            parts = chunk.file_path.split("/")
            if len(parts) > 1:
                directory = "/".join(parts[:-1])
                directories[directory] = directories.get(directory, 0) + 1

        # Sort by frequency and return top directories
        sorted_dirs = sorted(
            directories.items(), key=lambda x: x[1], reverse=True
        )

        return [d[0] for d in sorted_dirs[:3]]

    def get_next_sub_query(self, plan: RetrievalPlan) -> SubQuery | None:
        """
        Get the next sub-query to execute.

        Args:
            plan: Current retrieval plan.

        Returns:
            Next sub-query or None if plan is complete.
        """
        if plan.completed or plan.current_iteration >= plan.max_iterations:
            return None

        # Find highest priority incomplete sub-query
        for sq in plan.sub_queries:
            if not sq.completed:
                return sq

        # All sub-queries completed
        plan.completed = True
        return None

    def update_plan(
        self,
        plan: RetrievalPlan,
        sub_query: SubQuery,
        results: list["Chunk"],
    ) -> None:
        """
        Update plan after executing a sub-query.

        Args:
            plan: Current retrieval plan.
            sub_query: Completed sub-query.
            results: Results from the sub-query.
        """
        sub_query.completed = True
        sub_query.results = results
        plan.current_iteration += 1

        # Check if we should add more sub-queries based on results
        if results:
            # Extract new entities from results
            new_entities = set()
            for chunk in results[:5]:
                if chunk.symbol_name:
                    new_entities.add(chunk.symbol_name)

            # Add sub-queries for new entities not already covered
            existing_entities = set()
            for sq in plan.sub_queries:
                for entity in sq.constraints.get("symbol_name", []):
                    existing_entities.add(entity)

            new_entities -= existing_entities

            for entity in list(new_entities)[:2]:
                if len(plan.sub_queries) < 10:  # Limit total sub-queries
                    plan.sub_queries.append(
                        SubQuery(
                            query=f"related to {entity}",
                            query_type=QueryType.RELATED,
                            priority=0,
                            constraints={"symbol_name": entity},
                        )
                    )

        logger.debug(
            "Updated RLM plan",
            iteration=plan.current_iteration,
            remaining=sum(1 for sq in plan.sub_queries if not sq.completed),
        )

    def should_continue(
        self,
        plan: RetrievalPlan,
        aggregated_entropy: float,
    ) -> bool:
        """
        Determine if RLM should continue iterating.

        Args:
            plan: Current retrieval plan.
            aggregated_entropy: Entropy of aggregated results.

        Returns:
            True if should continue.
        """
        # Stop if max iterations reached
        if plan.current_iteration >= plan.max_iterations:
            return False

        # Stop if all sub-queries completed
        if all(sq.completed for sq in plan.sub_queries):
            return False

        # Stop if entropy is low enough (confident results)
        if aggregated_entropy < 0.3:
            return False

        return True
