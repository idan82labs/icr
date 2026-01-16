"""
Pydantic models for all tool inputs.

These models provide strict validation for tool parameters with
appropriate bounds and constraints as specified in the PRD.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class MemoryPackInput(BaseModel):
    """Input schema for memory_pack tool."""

    mode: Literal["auto", "pack", "rlm"] = Field(
        default="auto",
        description="Resolution mode: auto selects based on entropy, pack uses direct packing, rlm uses recursive lookup",
    )
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The query or task prompt to pack context for",
    )
    repo_root: str = Field(
        ...,
        min_length=1,
        description="Absolute path to the repository root",
    )
    budget_tokens: int = Field(
        default=4000,
        ge=512,
        le=12000,
        description="Token budget for the context pack (512-12000)",
    )
    k: int = Field(
        default=20,
        ge=5,
        le=50,
        description="Number of top sources to consider (5-50)",
    )
    focus_paths: list[str] = Field(
        default_factory=list,
        max_length=100,
        description="Optional list of paths to focus on",
    )
    pinned_only: bool = Field(
        default=False,
        description="If true, only include pinned memory items",
    )


class MemoryPinInput(BaseModel):
    """Input schema for memory_pin tool."""

    source_id: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Unique identifier for the memory source",
    )
    path: str = Field(
        ...,
        min_length=1,
        description="Path to the file or resource to pin",
    )
    label: str | None = Field(
        default=None,
        max_length=256,
        description="Optional human-readable label for the pin",
    )
    ttl_seconds: int | None = Field(
        default=None,
        ge=60,
        le=86400,
        description="Optional time-to-live in seconds (60-86400)",
    )


class MemoryUnpinInput(BaseModel):
    """Input schema for memory_unpin tool."""

    source_id: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Unique identifier of the memory source to unpin",
    )


class MemoryListInput(BaseModel):
    """Input schema for memory_list tool."""

    filter_type: Literal["all", "pinned", "recent", "stale"] = Field(
        default="all",
        description="Filter type for listing memory items",
    )
    limit: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Maximum number of items to return (1-200)",
    )
    cursor: str | None = Field(
        default=None,
        description="Pagination cursor from previous response",
    )


class MemoryGetInput(BaseModel):
    """Input schema for memory_get tool."""

    source_id: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Unique identifier of the memory source to retrieve",
    )
    include_content: bool = Field(
        default=True,
        description="Whether to include the full content",
    )


class MemoryStatsInput(BaseModel):
    """Input schema for memory_stats tool."""

    include_breakdown: bool = Field(
        default=False,
        description="Include detailed breakdown by source type",
    )


class EnvSearchInput(BaseModel):
    """Input schema for env_search tool."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Search query string",
    )
    scope: Literal["repo", "transcript", "diffs", "contracts", "all"] = Field(
        ...,
        description="Search scope",
    )
    path_prefix: str | None = Field(
        default=None,
        description="Optional path prefix to filter results",
    )
    language: str | None = Field(
        default=None,
        max_length=50,
        description="Optional language filter (e.g., 'python', 'typescript')",
    )
    time_window_seconds: int | None = Field(
        default=None,
        ge=1,
        le=604800,
        description="Optional time window in seconds (max 7 days)",
    )
    limit: int = Field(
        default=20,
        ge=1,
        le=50,
        description="Maximum number of results (1-50)",
    )
    cursor: str | None = Field(
        default=None,
        description="Pagination cursor from previous response",
    )
    explain: bool = Field(
        default=False,
        description="Include explanation of search strategy",
    )


class EnvPeekInput(BaseModel):
    """Input schema for env_peek tool."""

    path: str = Field(
        ...,
        min_length=1,
        description="Path to the file to peek",
    )
    start_line: int = Field(
        ...,
        ge=1,
        description="Starting line number (1-indexed)",
    )
    end_line: int = Field(
        ...,
        ge=1,
        description="Ending line number (1-indexed, inclusive)",
    )
    max_lines: int = Field(
        default=200,
        ge=1,
        le=400,
        description="Maximum lines to return (1-400)",
    )

    @field_validator("end_line")
    @classmethod
    def end_line_gte_start(cls, v: int, info: Any) -> int:
        """Validate that end_line is >= start_line."""
        if "start_line" in info.data and v < info.data["start_line"]:
            raise ValueError("end_line must be >= start_line")
        return v


class EnvSliceInput(BaseModel):
    """Input schema for env_slice tool."""

    path: str = Field(
        ...,
        min_length=1,
        description="Path to the file to slice",
    )
    symbol: str | None = Field(
        default=None,
        max_length=256,
        description="Symbol name to extract (function, class, etc.)",
    )
    start_line: int | None = Field(
        default=None,
        ge=1,
        description="Starting line number for range slice",
    )
    end_line: int | None = Field(
        default=None,
        ge=1,
        description="Ending line number for range slice",
    )
    context_lines: int = Field(
        default=3,
        ge=0,
        le=20,
        description="Number of context lines around the slice",
    )

    @field_validator("end_line")
    @classmethod
    def end_line_gte_start(cls, v: int | None, info: Any) -> int | None:
        """Validate that end_line is >= start_line if both provided."""
        if v is not None and "start_line" in info.data:
            start = info.data.get("start_line")
            if start is not None and v < start:
                raise ValueError("end_line must be >= start_line")
        return v


class EnvAggregateInput(BaseModel):
    """Input schema for env_aggregate tool."""

    op: Literal[
        "extract_regex",
        "unique",
        "sort",
        "group_by",
        "count",
        "top_k",
        "join_on",
        "diff_sets",
    ] = Field(
        ...,
        description="Aggregation operation to perform",
    )
    inputs: list[str] = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Input strings to aggregate (max 200)",
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Operation-specific parameters",
    )
    limit: int = Field(
        default=100,
        ge=1,
        le=200,
        description="Maximum number of results (1-200)",
    )


class ProjectMapInput(BaseModel):
    """Input schema for project_map tool."""

    repo_root: str = Field(
        ...,
        min_length=1,
        description="Absolute path to the repository root",
    )
    depth: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Directory depth to map (1-10)",
    )
    include_patterns: list[str] = Field(
        default_factory=list,
        max_length=50,
        description="Glob patterns to include",
    )
    exclude_patterns: list[str] = Field(
        default_factory=list,
        max_length=50,
        description="Glob patterns to exclude",
    )
    include_stats: bool = Field(
        default=False,
        description="Include file statistics (size, lines, etc.)",
    )


class ProjectSymbolSearchInput(BaseModel):
    """Input schema for project_symbol_search tool."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Symbol search query",
    )
    repo_root: str = Field(
        ...,
        min_length=1,
        description="Absolute path to the repository root",
    )
    symbol_types: list[Literal["function", "class", "method", "variable", "type", "interface"]] = (
        Field(
            default_factory=list,
            description="Filter by symbol types",
        )
    )
    languages: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Filter by programming languages",
    )
    limit: int = Field(
        default=30,
        ge=1,
        le=100,
        description="Maximum number of results (1-100)",
    )
    cursor: str | None = Field(
        default=None,
        description="Pagination cursor from previous response",
    )


class ProjectImpactInput(BaseModel):
    """Input schema for project_impact tool."""

    changed_paths: list[str] = Field(
        ...,
        min_length=1,
        max_length=200,
        description="List of changed file paths (max 200)",
    )
    query: str | None = Field(
        default=None,
        max_length=1000,
        description="Optional query to focus the impact analysis",
    )
    max_nodes: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Maximum number of nodes in impact graph (10-500)",
    )
    max_edges: int = Field(
        default=500,
        ge=10,
        le=2000,
        description="Maximum number of edges in impact graph (10-2000)",
    )


class ProjectCommandsInput(BaseModel):
    """Input schema for project_commands tool."""

    repo_root: str = Field(
        ...,
        min_length=1,
        description="Absolute path to the repository root",
    )
    command_type: Literal["build", "test", "lint", "format", "run", "all"] = Field(
        default="all",
        description="Type of commands to discover",
    )


class RlmPlanInput(BaseModel):
    """Input schema for rlm_plan tool."""

    task: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Task description to plan for",
    )
    scope: Literal["repo", "contracts", "diffs", "all"] = Field(
        ...,
        description="Scope of the planning",
    )
    budget: dict[str, int] = Field(
        default_factory=lambda: {
            "max_steps": 10,
            "max_peek_lines": 1000,
            "max_candidates": 100,
        },
        description="Budget constraints for the plan",
    )

    @field_validator("budget")
    @classmethod
    def validate_budget(cls, v: dict[str, int]) -> dict[str, int]:
        """Validate budget constraints."""
        defaults = {
            "max_steps": 10,
            "max_peek_lines": 1000,
            "max_candidates": 100,
        }
        result = {**defaults, **v}

        if not 1 <= result["max_steps"] <= 20:
            raise ValueError("max_steps must be between 1 and 20")
        if not 100 <= result["max_peek_lines"] <= 2000:
            raise ValueError("max_peek_lines must be between 100 and 2000")
        if not 10 <= result["max_candidates"] <= 200:
            raise ValueError("max_candidates must be between 10 and 200")

        return result


class RlmMapReduceInput(BaseModel):
    """Input schema for rlm_map_reduce tool."""

    task: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Task to execute via map-reduce",
    )
    sources: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Source identifiers to process",
    )
    map_prompt: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Prompt template for map phase",
    )
    reduce_prompt: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Prompt template for reduce phase",
    )
    max_parallel: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum parallel map operations",
    )


class AdminPingInput(BaseModel):
    """Input schema for admin_ping tool."""

    echo: str | None = Field(
        default=None,
        max_length=256,
        description="Optional string to echo back",
    )
    include_diagnostics: bool = Field(
        default=False,
        description="Include server diagnostics in response",
    )
