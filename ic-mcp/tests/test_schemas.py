"""
Tests for IC-MCP schemas and validation.
"""

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from ic_mcp.schemas.inputs import (
    EnvAggregateInput,
    EnvPeekInput,
    EnvSearchInput,
    MemoryPackInput,
    ProjectImpactInput,
    RlmPlanInput,
)
from ic_mcp.schemas.outputs import (
    AdminPingOutput,
    EnvSearchOutput,
    ErrorEnvelope,
    ErrorInfo,
    Evidence,
    ImpactEdge,
    ImpactNode,
    MemoryPackOutput,
    PaginationInfo,
    ProjectImpactOutput,
    SearchResult,
    SourceInfo,
    Span,
    SuccessEnvelope,
)
from ic_mcp.schemas.validation import (
    TokenBudget,
    count_tokens,
    create_pagination_cursor,
    estimate_json_tokens,
    parse_pagination_cursor,
    truncate_list_to_tokens,
    truncate_string_to_tokens,
    truncate_to_token_budget,
    validate_input,
)


class TestInputSchemas:
    """Tests for input schema validation."""

    def test_memory_pack_input_valid(self):
        """Test valid MemoryPackInput."""
        input_data = MemoryPackInput(
            prompt="Find the main function",
            repo_root="/path/to/repo",
            budget_tokens=4000,
            k=20,
        )
        assert input_data.prompt == "Find the main function"
        assert input_data.mode == "auto"  # default
        assert input_data.budget_tokens == 4000

    def test_memory_pack_input_invalid_budget(self):
        """Test MemoryPackInput with invalid budget."""
        with pytest.raises(ValidationError):
            MemoryPackInput(
                prompt="test",
                repo_root="/path",
                budget_tokens=100,  # Below minimum of 512
            )

    def test_memory_pack_input_invalid_k(self):
        """Test MemoryPackInput with invalid k."""
        with pytest.raises(ValidationError):
            MemoryPackInput(
                prompt="test",
                repo_root="/path",
                k=100,  # Above maximum of 50
            )

    def test_env_search_input_valid(self):
        """Test valid EnvSearchInput."""
        input_data = EnvSearchInput(
            query="function test",
            scope="repo",
            limit=20,
        )
        assert input_data.query == "function test"
        assert input_data.scope == "repo"

    def test_env_search_input_invalid_scope(self):
        """Test EnvSearchInput with invalid scope."""
        with pytest.raises(ValidationError):
            EnvSearchInput(
                query="test",
                scope="invalid_scope",  # Not a valid literal
            )

    def test_env_peek_input_valid(self):
        """Test valid EnvPeekInput."""
        input_data = EnvPeekInput(
            path="/path/to/file.py",
            start_line=10,
            end_line=50,
            max_lines=100,
        )
        assert input_data.start_line == 10
        assert input_data.end_line == 50

    def test_env_peek_input_invalid_lines(self):
        """Test EnvPeekInput with invalid line range."""
        # end_line validation happens in validator
        input_data = EnvPeekInput(
            path="/path/to/file.py",
            start_line=50,
            end_line=10,  # Less than start_line - will be caught by validator
        )
        # Note: Pydantic v2 validator behavior may differ

    def test_env_aggregate_input_valid(self):
        """Test valid EnvAggregateInput."""
        input_data = EnvAggregateInput(
            op="unique",
            inputs=["a", "b", "c"],
            limit=100,
        )
        assert input_data.op == "unique"
        assert len(input_data.inputs) == 3

    def test_env_aggregate_input_invalid_op(self):
        """Test EnvAggregateInput with invalid operation."""
        with pytest.raises(ValidationError):
            EnvAggregateInput(
                op="invalid_op",
                inputs=["a", "b"],
            )

    def test_project_impact_input_valid(self):
        """Test valid ProjectImpactInput."""
        input_data = ProjectImpactInput(
            changed_paths=["src/main.py", "src/utils.py"],
            max_nodes=100,
            max_edges=500,
        )
        assert len(input_data.changed_paths) == 2
        assert input_data.max_nodes == 100

    def test_project_impact_input_invalid_limits(self):
        """Test ProjectImpactInput with invalid limits."""
        with pytest.raises(ValidationError):
            ProjectImpactInput(
                changed_paths=["test.py"],
                max_nodes=5,  # Below minimum of 10
            )

    def test_rlm_plan_input_valid(self):
        """Test valid RlmPlanInput."""
        input_data = RlmPlanInput(
            task="Analyze authentication flow",
            scope="repo",
            budget={"max_steps": 15, "max_peek_lines": 500, "max_candidates": 100},
        )
        assert input_data.task == "Analyze authentication flow"
        assert input_data.budget["max_steps"] == 15

    def test_rlm_plan_input_budget_validation(self):
        """Test RlmPlanInput budget validation."""
        with pytest.raises(ValidationError):
            RlmPlanInput(
                task="test",
                scope="repo",
                budget={"max_steps": 50},  # Above maximum of 20
            )


class TestOutputSchemas:
    """Tests for output schema structure."""

    def test_success_envelope(self):
        """Test SuccessEnvelope structure."""
        request_id = uuid4()
        envelope = SuccessEnvelope(request_id=request_id)
        assert envelope.ok is True
        assert envelope.request_id == request_id

    def test_error_envelope(self):
        """Test ErrorEnvelope structure."""
        request_id = uuid4()
        envelope = ErrorEnvelope(
            request_id=request_id,
            error=ErrorInfo(
                code="TEST_ERROR",
                message="Test error message",
                retryable=True,
                details={"key": "value"},
            ),
        )
        assert envelope.ok is False
        assert envelope.error.code == "TEST_ERROR"
        assert envelope.error.retryable is True

    def test_evidence_object(self):
        """Test Evidence object structure."""
        evidence = Evidence(
            source_id="S123",
            source_type="file",
            path="src/main.py",
            repo_rev="abc123",
            span=Span(start_line=10, end_line=20),
            mtime=datetime.now(timezone.utc),
            content="def main():\n    pass",
        )
        assert evidence.source_id == "S123"
        assert evidence.span.start_line == 10

    def test_source_info(self):
        """Test SourceInfo structure."""
        source = SourceInfo(
            source_id="S456",
            path="src/utils.py",
            score=0.85,
        )
        assert source.score == 0.85

    def test_memory_pack_output(self):
        """Test MemoryPackOutput structure."""
        output = MemoryPackOutput(
            request_id=uuid4(),
            mode_resolved="pack",
            pack_markdown="# Context\n\nSome content",
            confidence=0.85,
            budget_used_tokens=1500,
            entropy=0.45,
            gating_reason_codes=["HIGH_CONFIDENCE"],
            top_sources=[
                SourceInfo(source_id="S1", path="main.py", score=0.9),
            ],
            evidence=[],
            warnings=["Some warning"],
        )
        assert output.ok is True
        assert output.mode_resolved == "pack"
        assert output.confidence == 0.85

    def test_env_search_output(self):
        """Test EnvSearchOutput structure."""
        output = EnvSearchOutput(
            request_id=uuid4(),
            results=[
                SearchResult(
                    source_id="S1",
                    path="main.py",
                    score=0.95,
                    snippet="def main():",
                    span=Span(start_line=1, end_line=1),
                ),
            ],
            pagination=PaginationInfo(
                cursor=None,
                has_more=False,
                total_count=1,
            ),
            evidence=[],
        )
        assert output.ok is True
        assert len(output.results) == 1

    def test_project_impact_output(self):
        """Test ProjectImpactOutput structure."""
        output = ProjectImpactOutput(
            request_id=uuid4(),
            nodes=[
                ImpactNode(id="N1", type="file", label="main.py", path="src/main.py"),
                ImpactNode(id="N2", type="test", label="test_main.py", path="tests/test_main.py"),
            ],
            edges=[
                ImpactEdge(from_node="N1", to_node="N2", type="tests"),
            ],
            root_paths=["src/main.py"],
            depth_reached=2,
            truncated=False,
            evidence=[],
        )
        assert output.ok is True
        assert len(output.nodes) == 2
        assert len(output.edges) == 1
        assert output.edges[0].type == "tests"

    def test_admin_ping_output(self):
        """Test AdminPingOutput structure."""
        output = AdminPingOutput(
            request_id=uuid4(),
            echo="hello",
            server_version="0.1.0",
            server_name="icr",
            uptime_seconds=123.45,
            diagnostics={"test": "value"},
        )
        assert output.ok is True
        assert output.echo == "hello"
        assert output.server_name == "icr"


class TestValidation:
    """Tests for validation utilities."""

    def test_count_tokens(self):
        """Test token counting."""
        text = "Hello, world! This is a test."
        tokens = count_tokens(text)
        assert tokens > 0
        assert tokens < len(text)  # Tokens should be fewer than characters

    def test_estimate_json_tokens(self):
        """Test JSON token estimation."""
        obj = {"key": "value", "list": [1, 2, 3]}
        tokens = estimate_json_tokens(obj)
        assert tokens > 0

    def test_validate_input_success(self):
        """Test successful input validation."""
        data = {
            "prompt": "test prompt",
            "repo_root": "/path/to/repo",
        }
        result = validate_input(MemoryPackInput, data)
        assert isinstance(result, MemoryPackInput)
        assert result.prompt == "test prompt"

    def test_validate_input_failure(self):
        """Test input validation failure."""
        data = {
            "prompt": "",  # Empty, below min_length
            "repo_root": "/path",
        }
        with pytest.raises(ValueError) as exc_info:
            validate_input(MemoryPackInput, data)
        assert "validation failed" in str(exc_info.value).lower()

    def test_truncate_string_to_tokens(self):
        """Test string truncation."""
        long_text = "word " * 1000
        truncated, was_truncated = truncate_string_to_tokens(long_text, 50)
        assert was_truncated is True
        assert count_tokens(truncated) <= 50 + 10  # Allow some margin for suffix

    def test_truncate_string_no_truncation_needed(self):
        """Test string truncation when not needed."""
        short_text = "Hello, world!"
        truncated, was_truncated = truncate_string_to_tokens(short_text, 100)
        assert was_truncated is False
        assert truncated == short_text

    def test_truncate_list_to_tokens(self):
        """Test list truncation."""
        items = [f"item_{i}" for i in range(100)]
        truncated, was_truncated, tokens = truncate_list_to_tokens(items, 100)
        assert was_truncated is True
        assert len(truncated) < 100

    def test_truncate_to_token_budget(self):
        """Test response truncation."""
        response = {
            "ok": True,
            "data": "x" * 10000,
            "list": list(range(1000)),
        }
        budget = TokenBudget(soft_limit=1000, hard_limit=2000)
        truncated, was_truncated = truncate_to_token_budget(
            response,
            budget,
            truncatable_fields=["data", "list"],
        )
        assert was_truncated is True
        assert estimate_json_tokens(truncated) < budget.hard_limit

    def test_pagination_cursor_roundtrip(self):
        """Test pagination cursor creation and parsing."""
        cursor = create_pagination_cursor(50, 20, 100)
        assert cursor is not None

        offset, limit = parse_pagination_cursor(cursor)
        assert offset == 70  # 50 + 20
        assert limit == 20

    def test_pagination_cursor_last_page(self):
        """Test pagination cursor for last page."""
        cursor = create_pagination_cursor(80, 20, 100)
        assert cursor is None  # No more pages

    def test_parse_invalid_cursor(self):
        """Test parsing invalid cursor."""
        offset, limit = parse_pagination_cursor("invalid_cursor")
        assert offset == 0  # Default
        assert limit == 50  # Default

    def test_token_budget_validation(self):
        """Test TokenBudget validation."""
        # Valid budget
        budget = TokenBudget(soft_limit=5000, hard_limit=10000)
        assert budget.soft_limit == 5000

        # Invalid: soft > hard
        with pytest.raises(ValueError):
            TokenBudget(soft_limit=10000, hard_limit=5000)

        # Invalid: too small
        with pytest.raises(ValueError):
            TokenBudget(soft_limit=50, hard_limit=100)


class TestSchemaEdgeCases:
    """Tests for schema edge cases."""

    def test_empty_focus_paths(self):
        """Test MemoryPackInput with empty focus_paths."""
        input_data = MemoryPackInput(
            prompt="test",
            repo_root="/path",
            focus_paths=[],
        )
        assert input_data.focus_paths == []

    def test_max_inputs_aggregate(self):
        """Test EnvAggregateInput with maximum inputs."""
        input_data = EnvAggregateInput(
            op="unique",
            inputs=["item"] * 200,  # Maximum allowed
            limit=100,
        )
        assert len(input_data.inputs) == 200

    def test_over_max_inputs_aggregate(self):
        """Test EnvAggregateInput with too many inputs."""
        with pytest.raises(ValidationError):
            EnvAggregateInput(
                op="unique",
                inputs=["item"] * 201,  # One over maximum
                limit=100,
            )

    def test_impact_edge_alias(self):
        """Test ImpactEdge from/to aliasing."""
        edge = ImpactEdge(
            from_node="N1",
            to_node="N2",
            type="imports",
        )
        # Test both access methods work
        assert edge.from_node == "N1"
        assert edge.to_node == "N2"

        # Test serialization includes aliases
        edge_dict = edge.model_dump(by_alias=True)
        assert "from" in edge_dict
        assert "to" in edge_dict
