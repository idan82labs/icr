"""
Pydantic models for all tool outputs.

These models define the structure of tool responses including
success envelopes, error envelopes, and domain-specific output types.
"""

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


# =============================================================================
# Common Types
# =============================================================================


class Span(BaseModel):
    """Line span within a file."""

    start_line: int = Field(..., ge=1, description="Starting line number (1-indexed)")
    end_line: int = Field(..., ge=1, description="Ending line number (1-indexed)")


class Evidence(BaseModel):
    """Evidence object providing provenance for retrieved content."""

    source_id: str = Field(..., description="Unique identifier for this source")
    source_type: Literal["file", "diff", "transcript", "contract", "index"] = Field(
        ..., description="Type of the source"
    )
    path: str = Field(..., description="Path to the source file or resource")
    repo_rev: str = Field(
        default="working-tree",
        description="Git SHA or 'working-tree' for uncommitted changes",
    )
    span: Span | None = Field(default=None, description="Line span within the source")
    mtime: datetime | None = Field(default=None, description="Last modification time (RFC3339)")
    indexed_at: datetime | None = Field(default=None, description="When this was indexed (RFC3339)")
    content: str | None = Field(default=None, description="Extracted content snippet")


class SourceInfo(BaseModel):
    """Information about a source in results."""

    source_id: str = Field(..., description="Unique source identifier")
    path: str = Field(..., description="Path to the source")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score (0-1)")


class ErrorInfo(BaseModel):
    """Structured error information."""

    code: str = Field(..., description="Error code for programmatic handling")
    message: str = Field(..., description="Human-readable error message")
    retryable: bool = Field(default=False, description="Whether the operation can be retried")
    details: dict[str, Any] = Field(
        default_factory=dict, description="Additional error details"
    )


class PaginationInfo(BaseModel):
    """Pagination metadata for paginated responses."""

    cursor: str | None = Field(default=None, description="Cursor for next page, null if last page")
    has_more: bool = Field(default=False, description="Whether more results are available")
    total_count: int | None = Field(default=None, description="Total count if known")


# =============================================================================
# Response Envelopes
# =============================================================================


class SuccessEnvelope(BaseModel):
    """Base success response envelope."""

    ok: Literal[True] = Field(default=True, description="Indicates success")
    request_id: UUID = Field(..., description="Unique request identifier for tracing")


class ErrorEnvelope(BaseModel):
    """Error response envelope."""

    ok: Literal[False] = Field(default=False, description="Indicates failure")
    error: ErrorInfo = Field(..., description="Error details")
    request_id: UUID = Field(..., description="Unique request identifier for tracing")


# =============================================================================
# Memory Tool Outputs
# =============================================================================


class MemoryPackOutput(SuccessEnvelope):
    """Output schema for memory_pack tool."""

    mode_resolved: Literal["pack", "rlm"] = Field(
        ..., description="The mode that was actually used"
    )
    pack_markdown: str = Field(..., description="The packed context as markdown")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score for the pack (0-1)"
    )
    budget_used_tokens: int = Field(..., ge=0, description="Tokens used from the budget")
    entropy: float = Field(..., ge=0.0, description="Information entropy of the pack")
    gating_reason_codes: list[str] = Field(
        default_factory=list, description="Reason codes for mode selection"
    )
    top_sources: list[SourceInfo] = Field(
        default_factory=list, description="Top contributing sources"
    )
    evidence: list[Evidence] = Field(
        default_factory=list, description="Evidence for the packed content"
    )
    warnings: list[str] = Field(default_factory=list, description="Any warnings generated")


class MemoryPinOutput(SuccessEnvelope):
    """Output schema for memory_pin tool."""

    source_id: str = Field(..., description="The pinned source identifier")
    path: str = Field(..., description="Path that was pinned")
    label: str | None = Field(default=None, description="Label for the pin")
    pinned_at: datetime = Field(..., description="When the pin was created")
    expires_at: datetime | None = Field(default=None, description="When the pin expires")


class MemoryUnpinOutput(SuccessEnvelope):
    """Output schema for memory_unpin tool."""

    source_id: str = Field(..., description="The unpinned source identifier")
    was_pinned: bool = Field(..., description="Whether the source was actually pinned")


class MemoryItem(BaseModel):
    """A memory item in the list."""

    source_id: str = Field(..., description="Unique source identifier")
    path: str = Field(..., description="Path to the source")
    source_type: Literal["file", "diff", "transcript", "contract", "index"] = Field(
        ..., description="Type of source"
    )
    pinned: bool = Field(default=False, description="Whether this item is pinned")
    label: str | None = Field(default=None, description="Optional label")
    last_accessed: datetime | None = Field(default=None, description="Last access time")
    indexed_at: datetime | None = Field(default=None, description="When indexed")
    size_bytes: int | None = Field(default=None, ge=0, description="Size in bytes")


class MemoryListOutput(SuccessEnvelope):
    """Output schema for memory_list tool."""

    items: list[MemoryItem] = Field(default_factory=list, description="Memory items")
    pagination: PaginationInfo = Field(..., description="Pagination information")


class MemoryGetOutput(SuccessEnvelope):
    """Output schema for memory_get tool."""

    source_id: str = Field(..., description="Source identifier")
    path: str = Field(..., description="Path to the source")
    source_type: Literal["file", "diff", "transcript", "contract", "index"] = Field(
        ..., description="Type of source"
    )
    content: str | None = Field(default=None, description="Full content if requested")
    evidence: Evidence | None = Field(default=None, description="Evidence metadata")
    pinned: bool = Field(default=False, description="Whether pinned")
    label: str | None = Field(default=None, description="Pin label if any")


class MemoryStatsOutput(SuccessEnvelope):
    """Output schema for memory_stats tool."""

    total_items: int = Field(..., ge=0, description="Total number of memory items")
    pinned_count: int = Field(..., ge=0, description="Number of pinned items")
    total_size_bytes: int = Field(..., ge=0, description="Total size in bytes")
    index_freshness: datetime | None = Field(default=None, description="Last index update time")
    breakdown: dict[str, int] | None = Field(
        default=None, description="Breakdown by source type"
    )


# =============================================================================
# Env Tool Outputs
# =============================================================================


class SearchResult(BaseModel):
    """A single search result."""

    source_id: str = Field(..., description="Source identifier")
    path: str = Field(..., description="File path")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    snippet: str = Field(..., description="Matching snippet")
    span: Span | None = Field(default=None, description="Line span of match")
    highlights: list[tuple[int, int]] = Field(
        default_factory=list, description="Character offsets of highlights in snippet"
    )


class EnvSearchOutput(SuccessEnvelope):
    """Output schema for env_search tool."""

    results: list[SearchResult] = Field(default_factory=list, description="Search results")
    pagination: PaginationInfo = Field(..., description="Pagination information")
    explanation: str | None = Field(default=None, description="Search strategy explanation")
    evidence: list[Evidence] = Field(
        default_factory=list, description="Evidence for results"
    )


class EnvPeekOutput(SuccessEnvelope):
    """Output schema for env_peek tool."""

    path: str = Field(..., description="File path")
    content: str = Field(..., description="File content for the requested range")
    start_line: int = Field(..., ge=1, description="Actual start line returned")
    end_line: int = Field(..., ge=1, description="Actual end line returned")
    total_lines: int = Field(..., ge=0, description="Total lines in file")
    truncated: bool = Field(default=False, description="Whether content was truncated")
    evidence: Evidence = Field(..., description="Evidence for the peeked content")


class EnvSliceOutput(SuccessEnvelope):
    """Output schema for env_slice tool."""

    path: str = Field(..., description="File path")
    content: str = Field(..., description="Sliced content")
    symbol: str | None = Field(default=None, description="Symbol name if symbol-based slice")
    span: Span = Field(..., description="Line span of the slice")
    context_before: int = Field(default=0, ge=0, description="Context lines before")
    context_after: int = Field(default=0, ge=0, description="Context lines after")
    evidence: Evidence = Field(..., description="Evidence for the slice")


class AggregateResult(BaseModel):
    """Result of an aggregation operation."""

    key: str | None = Field(default=None, description="Group key if applicable")
    value: Any = Field(..., description="Aggregation result value")
    count: int | None = Field(default=None, ge=0, description="Count if applicable")


class EnvAggregateOutput(SuccessEnvelope):
    """Output schema for env_aggregate tool."""

    op: str = Field(..., description="Operation that was performed")
    results: list[AggregateResult] = Field(
        default_factory=list, description="Aggregation results"
    )
    total_inputs: int = Field(..., ge=0, description="Number of inputs processed")
    truncated: bool = Field(default=False, description="Whether results were truncated")


# =============================================================================
# Project Tool Outputs
# =============================================================================


class FileNode(BaseModel):
    """A node in the project map."""

    path: str = Field(..., description="Relative path from repo root")
    type: Literal["file", "directory"] = Field(..., description="Node type")
    name: str = Field(..., description="File or directory name")
    children: list["FileNode"] = Field(default_factory=list, description="Child nodes")
    size_bytes: int | None = Field(default=None, ge=0, description="File size")
    line_count: int | None = Field(default=None, ge=0, description="Number of lines")
    language: str | None = Field(default=None, description="Detected language")


class ProjectMapOutput(SuccessEnvelope):
    """Output schema for project_map tool."""

    repo_root: str = Field(..., description="Repository root path")
    tree: FileNode = Field(..., description="Project tree structure")
    total_files: int = Field(..., ge=0, description="Total number of files")
    total_directories: int = Field(..., ge=0, description="Total number of directories")
    languages: dict[str, int] = Field(
        default_factory=dict, description="File count by language"
    )


class SymbolResult(BaseModel):
    """A symbol search result."""

    name: str = Field(..., description="Symbol name")
    qualified_name: str = Field(..., description="Fully qualified name")
    symbol_type: Literal["function", "class", "method", "variable", "type", "interface"] = Field(
        ..., description="Type of symbol"
    )
    path: str = Field(..., description="File path containing the symbol")
    span: Span = Field(..., description="Line span of the symbol definition")
    signature: str | None = Field(default=None, description="Symbol signature")
    docstring: str | None = Field(default=None, description="Symbol documentation")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")


class ProjectSymbolSearchOutput(SuccessEnvelope):
    """Output schema for project_symbol_search tool."""

    results: list[SymbolResult] = Field(default_factory=list, description="Symbol results")
    pagination: PaginationInfo = Field(..., description="Pagination information")


class ImpactNode(BaseModel):
    """A node in the impact graph."""

    id: str = Field(..., description="Unique node identifier")
    type: Literal[
        "contract",
        "endpoint",
        "shared_type",
        "fe_client",
        "fe_component",
        "be_handler",
        "test",
        "file",
    ] = Field(..., description="Node type")
    label: str = Field(..., description="Human-readable label")
    path: str = Field(..., description="File path")


class ImpactEdge(BaseModel):
    """An edge in the impact graph."""

    from_node: str = Field(..., alias="from", description="Source node ID")
    to_node: str = Field(..., alias="to", description="Target node ID")
    type: Literal["uses", "imports", "serializes", "deserializes", "routes_to", "tests"] = Field(
        ..., description="Edge type"
    )
    evidence_source_id: str | None = Field(
        default=None, description="Source ID providing evidence for this edge"
    )

    class Config:
        """Pydantic config."""

        populate_by_name = True


class ProjectImpactOutput(SuccessEnvelope):
    """Output schema for project_impact tool."""

    nodes: list[ImpactNode] = Field(default_factory=list, description="Impact graph nodes")
    edges: list[ImpactEdge] = Field(default_factory=list, description="Impact graph edges")
    root_paths: list[str] = Field(default_factory=list, description="Changed paths that were roots")
    depth_reached: int = Field(..., ge=0, description="Maximum depth reached in analysis")
    truncated: bool = Field(default=False, description="Whether graph was truncated")
    evidence: list[Evidence] = Field(
        default_factory=list, description="Evidence for impact relationships"
    )


class ProjectCommand(BaseModel):
    """A discovered project command."""

    name: str = Field(..., description="Command name")
    command: str = Field(..., description="Full command string")
    type: Literal["build", "test", "lint", "format", "run", "other"] = Field(
        ..., description="Command type"
    )
    source: str = Field(..., description="Source file where discovered")
    description: str | None = Field(default=None, description="Command description")


class ProjectCommandsOutput(SuccessEnvelope):
    """Output schema for project_commands tool."""

    commands: list[ProjectCommand] = Field(
        default_factory=list, description="Discovered commands"
    )
    config_files: list[str] = Field(
        default_factory=list, description="Config files examined"
    )


# =============================================================================
# RLM Tool Outputs
# =============================================================================


class PlanStep(BaseModel):
    """A step in an RLM execution plan."""

    id: str = Field(..., description="Unique step identifier")
    type: Literal["search", "peek", "slice", "aggregate", "impact", "pack", "stop"] = Field(
        ..., description="Step type"
    )
    tool: str = Field(..., description="Tool to invoke")
    args: dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
    rationale: str = Field(..., description="Why this step is needed")
    expected_output_shape: str = Field(
        ..., description="Description of expected output"
    )
    depends_on: list[str] = Field(
        default_factory=list, description="Step IDs this depends on"
    )


class StopCondition(BaseModel):
    """A condition for stopping RLM execution."""

    metric: str = Field(..., description="Metric to evaluate")
    threshold: float = Field(..., description="Threshold value")
    comparison: Literal["lt", "le", "eq", "ge", "gt"] = Field(
        default="ge", description="Comparison operator"
    )


class Artifact(BaseModel):
    """An artifact produced by RLM execution."""

    name: str = Field(..., description="Artifact name")
    schema_description: str = Field(..., description="Description of the artifact schema")


class RlmPlanOutput(SuccessEnvelope):
    """Output schema for rlm_plan tool."""

    steps: list[PlanStep] = Field(default_factory=list, description="Execution steps")
    stop_conditions: list[StopCondition] = Field(
        default_factory=list, description="Stop conditions"
    )
    artifacts: list[Artifact] = Field(
        default_factory=list, description="Expected artifacts"
    )
    estimated_tokens: int = Field(..., ge=0, description="Estimated token usage")
    estimated_steps: int = Field(..., ge=0, description="Estimated number of steps")


class MapResult(BaseModel):
    """Result from a map operation."""

    source_id: str = Field(..., description="Source that was mapped")
    result: Any = Field(..., description="Map result")
    tokens_used: int = Field(..., ge=0, description="Tokens used for this map")
    error: str | None = Field(default=None, description="Error if map failed")


class RlmMapReduceOutput(SuccessEnvelope):
    """Output schema for rlm_map_reduce tool."""

    map_results: list[MapResult] = Field(
        default_factory=list, description="Results from map phase"
    )
    reduce_result: Any = Field(..., description="Final reduced result")
    total_tokens_used: int = Field(..., ge=0, description="Total tokens used")
    sources_processed: int = Field(..., ge=0, description="Number of sources processed")
    sources_failed: int = Field(default=0, ge=0, description="Number of failed sources")
    evidence: list[Evidence] = Field(
        default_factory=list, description="Evidence from processing"
    )


# =============================================================================
# Admin Tool Outputs
# =============================================================================


class AdminPingOutput(SuccessEnvelope):
    """Output schema for admin_ping tool."""

    echo: str | None = Field(default=None, description="Echoed string")
    server_version: str = Field(..., description="Server version")
    server_name: str = Field(default="icr", description="Server name")
    uptime_seconds: float = Field(..., ge=0.0, description="Server uptime in seconds")
    diagnostics: dict[str, Any] | None = Field(
        default=None, description="Server diagnostics if requested"
    )


# Enable forward references
FileNode.model_rebuild()
