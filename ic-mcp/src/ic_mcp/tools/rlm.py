"""
Recursive Lookup Management (RLM) tools for IC-MCP.

These tools provide advanced multi-step analysis capabilities:
- rlm_plan: Generate an execution plan for complex tasks
- rlm_map_reduce: Execute map-reduce operations across sources
"""

import logging
from typing import Any
from uuid import UUID, uuid4

from ic_mcp.schemas.inputs import (
    RlmMapReduceInput,
    RlmPlanInput,
)
from ic_mcp.schemas.outputs import (
    Artifact,
    Evidence,
    MapResult,
    PlanStep,
    RlmMapReduceOutput,
    RlmPlanOutput,
    StopCondition,
)
from ic_mcp.schemas.validation import count_tokens

logger = logging.getLogger(__name__)


class RlmTools:
    """
    Recursive Lookup Management tools.

    These tools provide advanced multi-step analysis for complex tasks
    that require iterative exploration and aggregation.
    """

    def __init__(self) -> None:
        """Initialize RLM tools."""
        pass

    def _analyze_task(self, task: str) -> dict[str, Any]:
        """
        Analyze a task to understand what operations are needed.

        Returns analysis including:
        - task_type: The type of task (exploration, aggregation, impact, etc.)
        - keywords: Extracted keywords for search
        - complexity: Estimated complexity (low, medium, high)
        """
        task_lower = task.lower()

        # Detect task type
        task_type = "exploration"
        if any(kw in task_lower for kw in ["find all", "list all", "every", "across"]):
            task_type = "aggregation"
        elif any(kw in task_lower for kw in ["impact", "affect", "dependency", "trace"]):
            task_type = "impact"
        elif any(kw in task_lower for kw in ["compare", "diff", "difference"]):
            task_type = "comparison"
        elif any(kw in task_lower for kw in ["summarize", "overview", "understand"]):
            task_type = "summarization"

        # Extract keywords (simplified)
        import re

        words = re.findall(r"\b[a-zA-Z][a-zA-Z_]+\b", task)
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "dare", "ought", "used", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into",
            "through", "during", "before", "after", "above", "below",
            "between", "under", "again", "further", "then", "once",
            "all", "any", "both", "each", "few", "more", "most", "other",
            "some", "such", "only", "own", "same", "so", "than", "too",
            "very", "just", "also", "now", "find", "list", "show", "get",
        }
        keywords = [w for w in words if w.lower() not in stopwords][:10]

        # Estimate complexity
        task_tokens = count_tokens(task)
        if task_tokens > 200 or "comprehensive" in task_lower:
            complexity = "high"
        elif task_tokens > 50:
            complexity = "medium"
        else:
            complexity = "low"

        return {
            "task_type": task_type,
            "keywords": keywords,
            "complexity": complexity,
            "task_tokens": task_tokens,
        }

    async def rlm_plan(
        self,
        input_data: RlmPlanInput,
        request_id: UUID,
    ) -> RlmPlanOutput:
        """
        Generate an execution plan for a complex task.

        Analyzes the task and creates a sequence of steps to achieve it,
        respecting budget constraints.

        Args:
            input_data: Validated input parameters
            request_id: Unique request identifier

        Returns:
            RlmPlanOutput with execution plan
        """
        logger.info(f"rlm_plan: task={input_data.task[:100]}, scope={input_data.scope}")

        analysis = self._analyze_task(input_data.task)
        steps: list[PlanStep] = []
        artifacts: list[Artifact] = []
        stop_conditions: list[StopCondition] = []

        budget = input_data.budget
        max_steps = budget.get("max_steps", 10)
        max_peek_lines = budget.get("max_peek_lines", 1000)
        max_candidates = budget.get("max_candidates", 100)

        # Generate steps based on task type
        if analysis["task_type"] == "exploration":
            steps = self._generate_exploration_plan(
                input_data.task,
                input_data.scope,
                analysis,
                max_steps,
                max_candidates,
            )
        elif analysis["task_type"] == "aggregation":
            steps = self._generate_aggregation_plan(
                input_data.task,
                input_data.scope,
                analysis,
                max_steps,
                max_candidates,
            )
        elif analysis["task_type"] == "impact":
            steps = self._generate_impact_plan(
                input_data.task,
                input_data.scope,
                analysis,
                max_steps,
            )
        elif analysis["task_type"] == "comparison":
            steps = self._generate_comparison_plan(
                input_data.task,
                input_data.scope,
                analysis,
                max_steps,
            )
        else:  # summarization or default
            steps = self._generate_summarization_plan(
                input_data.task,
                input_data.scope,
                analysis,
                max_steps,
                max_peek_lines,
            )

        # Add stop conditions based on complexity
        if analysis["complexity"] == "high":
            stop_conditions = [
                StopCondition(metric="confidence", threshold=0.8, comparison="ge"),
                StopCondition(metric="coverage", threshold=0.9, comparison="ge"),
                StopCondition(metric="steps_executed", threshold=float(max_steps), comparison="ge"),
            ]
        else:
            stop_conditions = [
                StopCondition(metric="confidence", threshold=0.7, comparison="ge"),
                StopCondition(metric="steps_executed", threshold=float(max_steps // 2), comparison="ge"),
            ]

        # Define artifacts
        artifacts = [
            Artifact(
                name="results",
                schema_description="List of findings from the analysis",
            ),
            Artifact(
                name="evidence",
                schema_description="Evidence objects supporting findings",
            ),
        ]

        if analysis["task_type"] == "impact":
            artifacts.append(
                Artifact(
                    name="impact_graph",
                    schema_description="Graph of nodes and edges showing impact",
                )
            )

        # Estimate tokens
        estimated_tokens = sum(
            count_tokens(str(s.model_dump()))
            for s in steps
        ) + 500  # Buffer for results

        return RlmPlanOutput(
            request_id=request_id,
            steps=steps,
            stop_conditions=stop_conditions,
            artifacts=artifacts,
            estimated_tokens=estimated_tokens,
            estimated_steps=len(steps),
        )

    def _generate_exploration_plan(
        self,
        task: str,
        scope: str,
        analysis: dict[str, Any],
        max_steps: int,
        max_candidates: int,
    ) -> list[PlanStep]:
        """Generate a plan for exploration tasks."""
        steps: list[PlanStep] = []
        keywords = analysis["keywords"]

        # Step 1: Initial search
        steps.append(
            PlanStep(
                id=f"step_{len(steps) + 1}",
                type="search",
                tool="env_search",
                args={
                    "query": " ".join(keywords[:3]),
                    "scope": scope,
                    "limit": min(50, max_candidates),
                },
                rationale=f"Search for relevant files using keywords: {', '.join(keywords[:3])}",
                expected_output_shape="List of file paths and snippets matching the query",
            )
        )

        # Step 2: Peek at top results
        steps.append(
            PlanStep(
                id=f"step_{len(steps) + 1}",
                type="peek",
                tool="env_peek",
                args={
                    "path": "{{step_1.results[0].path}}",
                    "start_line": 1,
                    "end_line": 100,
                },
                rationale="Examine the most relevant file in detail",
                expected_output_shape="File content with line numbers",
                depends_on=["step_1"],
            )
        )

        # Step 3: Extract specific content if needed
        if analysis["complexity"] != "low":
            steps.append(
                PlanStep(
                    id=f"step_{len(steps) + 1}",
                    type="slice",
                    tool="env_slice",
                    args={
                        "path": "{{step_1.results[0].path}}",
                        "symbol": keywords[0] if keywords else None,
                    },
                    rationale="Extract specific symbol or section for detailed analysis",
                    expected_output_shape="Sliced content with context",
                    depends_on=["step_1"],
                )
            )

        # Step 4: Aggregate findings
        steps.append(
            PlanStep(
                id=f"step_{len(steps) + 1}",
                type="aggregate",
                tool="env_aggregate",
                args={
                    "op": "unique",
                    "inputs": "{{previous_findings}}",
                },
                rationale="Deduplicate and organize findings",
                expected_output_shape="Unique list of relevant items",
                depends_on=[f"step_{i + 1}" for i in range(len(steps))],
            )
        )

        # Step 5: Pack final context
        steps.append(
            PlanStep(
                id=f"step_{len(steps) + 1}",
                type="pack",
                tool="memory_pack",
                args={
                    "prompt": task,
                    "k": min(20, max_candidates // 5),
                },
                rationale="Pack the most relevant context for the task",
                expected_output_shape="Packed markdown with evidence",
                depends_on=[f"step_{len(steps)}"],
            )
        )

        # Step 6: Stop
        steps.append(
            PlanStep(
                id=f"step_{len(steps) + 1}",
                type="stop",
                tool="stop",
                args={"reason": "exploration_complete"},
                rationale="All exploration steps completed",
                expected_output_shape="Final results",
                depends_on=[f"step_{len(steps)}"],
            )
        )

        return steps[:max_steps]

    def _generate_aggregation_plan(
        self,
        task: str,
        scope: str,
        analysis: dict[str, Any],
        max_steps: int,
        max_candidates: int,
    ) -> list[PlanStep]:
        """Generate a plan for aggregation tasks."""
        steps: list[PlanStep] = []
        keywords = analysis["keywords"]

        # Step 1: Broad search
        steps.append(
            PlanStep(
                id=f"step_{len(steps) + 1}",
                type="search",
                tool="env_search",
                args={
                    "query": keywords[0] if keywords else task[:50],
                    "scope": scope,
                    "limit": max_candidates,
                },
                rationale="Broad search to find all relevant items",
                expected_output_shape="List of all matching files/items",
            )
        )

        # Step 2: Group results
        steps.append(
            PlanStep(
                id=f"step_{len(steps) + 1}",
                type="aggregate",
                tool="env_aggregate",
                args={
                    "op": "group_by",
                    "inputs": "{{step_1.results}}",
                    "params": {"key_pattern": r"^([^/]+)"},
                },
                rationale="Group results by directory/category",
                expected_output_shape="Grouped results with counts",
                depends_on=["step_1"],
            )
        )

        # Step 3: Count items
        steps.append(
            PlanStep(
                id=f"step_{len(steps) + 1}",
                type="aggregate",
                tool="env_aggregate",
                args={
                    "op": "count",
                    "inputs": "{{step_1.results}}",
                },
                rationale="Count occurrences for statistics",
                expected_output_shape="Count statistics",
                depends_on=["step_1"],
            )
        )

        # Step 4: Get top items
        steps.append(
            PlanStep(
                id=f"step_{len(steps) + 1}",
                type="aggregate",
                tool="env_aggregate",
                args={
                    "op": "top_k",
                    "inputs": "{{step_1.results}}",
                    "params": {"k": 10},
                },
                rationale="Get the most relevant items",
                expected_output_shape="Top k items by score",
                depends_on=["step_1"],
            )
        )

        # Step 5: Stop
        steps.append(
            PlanStep(
                id=f"step_{len(steps) + 1}",
                type="stop",
                tool="stop",
                args={"reason": "aggregation_complete"},
                rationale="Aggregation complete",
                expected_output_shape="Aggregated results",
                depends_on=[f"step_{len(steps)}"],
            )
        )

        return steps[:max_steps]

    def _generate_impact_plan(
        self,
        task: str,
        scope: str,
        analysis: dict[str, Any],
        max_steps: int,
    ) -> list[PlanStep]:
        """Generate a plan for impact analysis tasks."""
        steps: list[PlanStep] = []
        keywords = analysis["keywords"]

        # Step 1: Search for the item to analyze
        steps.append(
            PlanStep(
                id=f"step_{len(steps) + 1}",
                type="search",
                tool="env_search",
                args={
                    "query": keywords[0] if keywords else task[:50],
                    "scope": "repo",
                    "limit": 10,
                },
                rationale="Find the files/symbols to analyze for impact",
                expected_output_shape="List of target files",
            )
        )

        # Step 2: Impact analysis
        steps.append(
            PlanStep(
                id=f"step_{len(steps) + 1}",
                type="impact",
                tool="project_impact",
                args={
                    "changed_paths": "{{step_1.results.map(r => r.path)}}",
                    "max_nodes": 100,
                    "max_edges": 500,
                },
                rationale="Analyze the impact of changes to identified files",
                expected_output_shape="Impact graph with nodes and edges",
                depends_on=["step_1"],
            )
        )

        # Step 3: Peek at affected files
        steps.append(
            PlanStep(
                id=f"step_{len(steps) + 1}",
                type="peek",
                tool="env_peek",
                args={
                    "path": "{{step_2.nodes[0].path}}",
                    "start_line": 1,
                    "end_line": 50,
                },
                rationale="Examine affected file content",
                expected_output_shape="File content preview",
                depends_on=["step_2"],
            )
        )

        # Step 4: Stop
        steps.append(
            PlanStep(
                id=f"step_{len(steps) + 1}",
                type="stop",
                tool="stop",
                args={"reason": "impact_analysis_complete"},
                rationale="Impact analysis complete",
                expected_output_shape="Impact graph and analysis",
                depends_on=[f"step_{len(steps)}"],
            )
        )

        return steps[:max_steps]

    def _generate_comparison_plan(
        self,
        task: str,
        scope: str,
        analysis: dict[str, Any],
        max_steps: int,
    ) -> list[PlanStep]:
        """Generate a plan for comparison tasks."""
        steps: list[PlanStep] = []
        keywords = analysis["keywords"]

        # Step 1: Search for first item
        steps.append(
            PlanStep(
                id=f"step_{len(steps) + 1}",
                type="search",
                tool="env_search",
                args={
                    "query": keywords[0] if len(keywords) > 0 else task[:30],
                    "scope": scope,
                    "limit": 20,
                },
                rationale="Find the first set of items to compare",
                expected_output_shape="First item set",
            )
        )

        # Step 2: Search for second item
        steps.append(
            PlanStep(
                id=f"step_{len(steps) + 1}",
                type="search",
                tool="env_search",
                args={
                    "query": keywords[1] if len(keywords) > 1 else task[30:60],
                    "scope": scope,
                    "limit": 20,
                },
                rationale="Find the second set of items to compare",
                expected_output_shape="Second item set",
            )
        )

        # Step 3: Diff the sets
        steps.append(
            PlanStep(
                id=f"step_{len(steps) + 1}",
                type="aggregate",
                tool="env_aggregate",
                args={
                    "op": "diff_sets",
                    "inputs": "{{[...step_1.results, ...step_2.results]}}",
                    "params": {
                        "set_a": "{{range(0, step_1.results.length)}}",
                        "set_b": "{{range(step_1.results.length, step_1.results.length + step_2.results.length)}}",
                    },
                },
                rationale="Compare the two sets to find differences",
                expected_output_shape="Diff results showing only_a, only_b, both",
                depends_on=["step_1", "step_2"],
            )
        )

        # Step 4: Stop
        steps.append(
            PlanStep(
                id=f"step_{len(steps) + 1}",
                type="stop",
                tool="stop",
                args={"reason": "comparison_complete"},
                rationale="Comparison complete",
                expected_output_shape="Comparison results",
                depends_on=["step_3"],
            )
        )

        return steps[:max_steps]

    def _generate_summarization_plan(
        self,
        task: str,
        scope: str,
        analysis: dict[str, Any],
        max_steps: int,
        max_peek_lines: int,
    ) -> list[PlanStep]:
        """Generate a plan for summarization tasks."""
        steps: list[PlanStep] = []
        keywords = analysis["keywords"]

        # Step 1: Search for relevant content
        steps.append(
            PlanStep(
                id=f"step_{len(steps) + 1}",
                type="search",
                tool="env_search",
                args={
                    "query": " ".join(keywords[:3]) if keywords else task[:50],
                    "scope": scope,
                    "limit": 30,
                },
                rationale="Find content to summarize",
                expected_output_shape="List of relevant files/sections",
            )
        )

        # Step 2-4: Peek at top results (multiple)
        for i in range(3):
            steps.append(
                PlanStep(
                    id=f"step_{len(steps) + 1}",
                    type="peek",
                    tool="env_peek",
                    args={
                        "path": f"{{{{step_1.results[{i}].path}}}}",
                        "start_line": 1,
                        "end_line": min(100, max_peek_lines // 3),
                    },
                    rationale=f"Examine result #{i + 1} for summarization",
                    expected_output_shape="File content to summarize",
                    depends_on=["step_1"],
                )
            )

        # Step 5: Pack for summary
        steps.append(
            PlanStep(
                id=f"step_{len(steps) + 1}",
                type="pack",
                tool="memory_pack",
                args={
                    "prompt": f"Summarize: {task}",
                    "k": 10,
                },
                rationale="Pack content for summarization",
                expected_output_shape="Packed context for summary generation",
                depends_on=[f"step_{i + 1}" for i in range(len(steps))],
            )
        )

        # Step 6: Stop
        steps.append(
            PlanStep(
                id=f"step_{len(steps) + 1}",
                type="stop",
                tool="stop",
                args={"reason": "summarization_complete"},
                rationale="Summarization data gathered",
                expected_output_shape="Summary context",
                depends_on=[f"step_{len(steps)}"],
            )
        )

        return steps[:max_steps]

    async def rlm_map_reduce(
        self,
        input_data: RlmMapReduceInput,
        request_id: UUID,
    ) -> RlmMapReduceOutput:
        """
        Execute a map-reduce operation across sources.

        Maps a prompt template across multiple sources and then reduces
        the results using a reduce prompt.

        Args:
            input_data: Validated input parameters
            request_id: Unique request identifier

        Returns:
            RlmMapReduceOutput with map and reduce results
        """
        logger.info(
            f"rlm_map_reduce: task={input_data.task[:50]}, "
            f"sources={len(input_data.sources)}"
        )

        map_results: list[MapResult] = []
        evidence: list[Evidence] = []
        total_tokens = 0
        failed_count = 0

        # Map phase: Process each source
        for i, source in enumerate(input_data.sources):
            try:
                # In a real implementation, this would:
                # 1. Fetch the source content
                # 2. Apply the map_prompt template
                # 3. Process with an LLM or analysis function

                # Simplified simulation for the tool interface
                map_prompt_filled = input_data.map_prompt.replace(
                    "{{source}}", source
                )
                map_tokens = count_tokens(map_prompt_filled)
                total_tokens += map_tokens

                # Simulate map result (in production, this would be actual processing)
                result = {
                    "source": source,
                    "processed": True,
                    "summary": f"Processed {source} with map prompt",
                }

                map_results.append(
                    MapResult(
                        source_id=f"map_{i}",
                        result=result,
                        tokens_used=map_tokens,
                        error=None,
                    )
                )

                evidence.append(
                    Evidence(
                        source_id=f"map_{i}",
                        source_type="index",
                        path=source,
                        repo_rev="working-tree",
                        content=f"Map result for {source}",
                    )
                )

            except Exception as e:
                logger.warning(f"Map failed for source {source}: {e}")
                failed_count += 1
                map_results.append(
                    MapResult(
                        source_id=f"map_{i}",
                        result=None,
                        tokens_used=0,
                        error=str(e),
                    )
                )

        # Reduce phase: Combine results
        successful_results = [r for r in map_results if r.error is None]

        # In production, this would apply the reduce_prompt to combine results
        reduce_prompt_filled = input_data.reduce_prompt.replace(
            "{{results}}", str([r.result for r in successful_results])
        )
        reduce_tokens = count_tokens(reduce_prompt_filled)
        total_tokens += reduce_tokens

        # Simulated reduce result
        reduce_result = {
            "task": input_data.task,
            "sources_processed": len(successful_results),
            "sources_failed": failed_count,
            "combined_summary": f"Reduced {len(successful_results)} results for task: {input_data.task[:50]}",
            "findings": [
                r.result.get("summary") if isinstance(r.result, dict) else str(r.result)
                for r in successful_results[:10]
            ],
        }

        return RlmMapReduceOutput(
            request_id=request_id,
            map_results=map_results,
            reduce_result=reduce_result,
            total_tokens_used=total_tokens,
            sources_processed=len(input_data.sources),
            sources_failed=failed_count,
            evidence=evidence,
        )
