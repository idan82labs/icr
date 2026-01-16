"""
Integration tests for MCP tool integration.

Tests cover:
- MCP tool schemas
- Tool input validation
- Tool response formatting
- Tool execution simulation
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal
from unittest.mock import AsyncMock, MagicMock

import pytest


# ==============================================================================
# MCP Tool Definitions (Simulated)
# ==============================================================================

MCP_TOOLS = {
    "memory_pack": {
        "name": "memory_pack",
        "description": "Pack relevant context for a query",
        "parameters": {
            "mode": {"type": "string", "enum": ["auto", "pack", "rlm"], "default": "auto"},
            "prompt": {"type": "string", "required": True},
            "repo_root": {"type": "string", "required": True},
            "budget_tokens": {"type": "integer", "default": 4000, "min": 512, "max": 12000},
            "k": {"type": "integer", "default": 20, "min": 5, "max": 50},
            "focus_paths": {"type": "array", "items": {"type": "string"}, "default": []},
            "pinned_only": {"type": "boolean", "default": False},
        },
    },
    "env_search": {
        "name": "env_search",
        "description": "Search the context environment",
        "parameters": {
            "query": {"type": "string", "required": True},
            "scope": {"type": "string", "enum": ["repo", "transcript", "diffs", "contracts", "all"], "required": True},
            "path_prefix": {"type": "string", "default": None},
            "limit": {"type": "integer", "default": 20, "min": 1, "max": 50},
        },
    },
    "env_peek": {
        "name": "env_peek",
        "description": "Peek at file contents",
        "parameters": {
            "path": {"type": "string", "required": True},
            "start_line": {"type": "integer", "required": True, "min": 1},
            "end_line": {"type": "integer", "required": True, "min": 1},
            "max_lines": {"type": "integer", "default": 200, "min": 1, "max": 400},
        },
    },
    "project_map": {
        "name": "project_map",
        "description": "Get project structure map",
        "parameters": {
            "repo_root": {"type": "string", "required": True},
            "depth": {"type": "integer", "default": 3, "min": 1, "max": 10},
            "include_patterns": {"type": "array", "items": {"type": "string"}, "default": []},
            "exclude_patterns": {"type": "array", "items": {"type": "string"}, "default": []},
        },
    },
    "project_symbol_search": {
        "name": "project_symbol_search",
        "description": "Search for code symbols",
        "parameters": {
            "query": {"type": "string", "required": True},
            "repo_root": {"type": "string", "required": True},
            "symbol_types": {"type": "array", "items": {"type": "string"}, "default": []},
            "limit": {"type": "integer", "default": 30, "min": 1, "max": 100},
        },
    },
    "project_impact": {
        "name": "project_impact",
        "description": "Analyze impact of changes",
        "parameters": {
            "changed_paths": {"type": "array", "items": {"type": "string"}, "required": True, "min_length": 1},
            "query": {"type": "string", "default": None},
            "max_nodes": {"type": "integer", "default": 100, "min": 10, "max": 500},
        },
    },
}


def validate_tool_input(tool_name: str, params: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate tool input parameters.

    Returns:
        Tuple of (is_valid, error_messages)
    """
    if tool_name not in MCP_TOOLS:
        return False, [f"Unknown tool: {tool_name}"]

    tool_def = MCP_TOOLS[tool_name]
    param_defs = tool_def["parameters"]
    errors = []

    # Check required parameters
    for param_name, param_def in param_defs.items():
        if param_def.get("required") and param_name not in params:
            errors.append(f"Missing required parameter: {param_name}")

    # Check parameter types and constraints
    for param_name, value in params.items():
        if param_name not in param_defs:
            errors.append(f"Unknown parameter: {param_name}")
            continue

        param_def = param_defs[param_name]

        # Type checking
        expected_type = param_def.get("type")
        if expected_type == "string" and not isinstance(value, str):
            errors.append(f"Parameter {param_name} must be string")
        elif expected_type == "integer" and not isinstance(value, int):
            errors.append(f"Parameter {param_name} must be integer")
        elif expected_type == "boolean" and not isinstance(value, bool):
            errors.append(f"Parameter {param_name} must be boolean")
        elif expected_type == "array" and not isinstance(value, list):
            errors.append(f"Parameter {param_name} must be array")

        # Enum checking
        if "enum" in param_def and value not in param_def["enum"]:
            errors.append(f"Parameter {param_name} must be one of {param_def['enum']}")

        # Range checking
        if isinstance(value, int):
            if "min" in param_def and value < param_def["min"]:
                errors.append(f"Parameter {param_name} below minimum {param_def['min']}")
            if "max" in param_def and value > param_def["max"]:
                errors.append(f"Parameter {param_name} above maximum {param_def['max']}")

    return len(errors) == 0, errors


# ==============================================================================
# Tool Schema Tests
# ==============================================================================

@pytest.mark.integration
class TestMCPToolSchemas:
    """Tests for MCP tool schemas."""

    def test_all_tools_have_required_fields(self):
        """Test all tools have required schema fields."""
        for tool_name, tool_def in MCP_TOOLS.items():
            assert "name" in tool_def
            assert "description" in tool_def
            assert "parameters" in tool_def

    def test_memory_pack_schema(self):
        """Test memory_pack tool schema."""
        tool = MCP_TOOLS["memory_pack"]

        assert tool["name"] == "memory_pack"
        assert "prompt" in tool["parameters"]
        assert tool["parameters"]["prompt"]["required"] is True

        # Mode enum values
        assert tool["parameters"]["mode"]["enum"] == ["auto", "pack", "rlm"]

        # Budget bounds
        assert tool["parameters"]["budget_tokens"]["min"] == 512
        assert tool["parameters"]["budget_tokens"]["max"] == 12000

    def test_env_search_schema(self):
        """Test env_search tool schema."""
        tool = MCP_TOOLS["env_search"]

        assert tool["parameters"]["query"]["required"] is True
        assert tool["parameters"]["scope"]["required"] is True
        assert "repo" in tool["parameters"]["scope"]["enum"]

    def test_env_peek_schema(self):
        """Test env_peek tool schema."""
        tool = MCP_TOOLS["env_peek"]

        assert tool["parameters"]["path"]["required"] is True
        assert tool["parameters"]["start_line"]["min"] == 1
        assert tool["parameters"]["max_lines"]["max"] == 400


# ==============================================================================
# Input Validation Tests
# ==============================================================================

@pytest.mark.integration
class TestToolInputValidation:
    """Tests for tool input validation."""

    def test_valid_memory_pack_input(self):
        """Test valid memory_pack input."""
        params = {
            "prompt": "Where is auth token validated?",
            "repo_root": "/path/to/repo",
            "budget_tokens": 4000,
            "mode": "auto",
        }

        is_valid, errors = validate_tool_input("memory_pack", params)

        assert is_valid is True
        assert len(errors) == 0

    def test_missing_required_parameter(self):
        """Test missing required parameter."""
        params = {
            "budget_tokens": 4000,
            # Missing prompt and repo_root
        }

        is_valid, errors = validate_tool_input("memory_pack", params)

        assert is_valid is False
        assert any("prompt" in e for e in errors)
        assert any("repo_root" in e for e in errors)

    def test_invalid_enum_value(self):
        """Test invalid enum value."""
        params = {
            "prompt": "test",
            "repo_root": "/path",
            "mode": "invalid_mode",
        }

        is_valid, errors = validate_tool_input("memory_pack", params)

        assert is_valid is False
        assert any("mode" in e for e in errors)

    def test_value_below_minimum(self):
        """Test value below minimum."""
        params = {
            "prompt": "test",
            "repo_root": "/path",
            "budget_tokens": 100,  # Below 512 minimum
        }

        is_valid, errors = validate_tool_input("memory_pack", params)

        assert is_valid is False
        assert any("minimum" in e for e in errors)

    def test_value_above_maximum(self):
        """Test value above maximum."""
        params = {
            "prompt": "test",
            "repo_root": "/path",
            "budget_tokens": 50000,  # Above 12000 maximum
        }

        is_valid, errors = validate_tool_input("memory_pack", params)

        assert is_valid is False
        assert any("maximum" in e for e in errors)

    def test_wrong_type(self):
        """Test wrong parameter type."""
        params = {
            "prompt": 123,  # Should be string
            "repo_root": "/path",
        }

        is_valid, errors = validate_tool_input("memory_pack", params)

        assert is_valid is False
        assert any("string" in e for e in errors)

    def test_unknown_parameter(self):
        """Test unknown parameter."""
        params = {
            "prompt": "test",
            "repo_root": "/path",
            "unknown_param": "value",
        }

        is_valid, errors = validate_tool_input("memory_pack", params)

        assert is_valid is False
        assert any("unknown" in e.lower() for e in errors)


# ==============================================================================
# Tool Response Formatting Tests
# ==============================================================================

@pytest.mark.integration
class TestToolResponseFormatting:
    """Tests for tool response formatting."""

    def test_memory_pack_response_format(self):
        """Test memory_pack response format."""
        response = {
            "content": "# Context Pack\n\n```typescript\ncode here\n```",
            "token_count": 500,
            "chunk_count": 3,
            "entropy": 1.5,
            "mode": "pack",
            "citations": {"[1]": "src/auth.ts:10-20"},
        }

        # Verify required fields
        assert "content" in response
        assert "token_count" in response
        assert "mode" in response

        # Verify content is string
        assert isinstance(response["content"], str)

    def test_env_search_response_format(self):
        """Test env_search response format."""
        response = {
            "results": [
                {
                    "id": "chunk_123",
                    "file": "src/auth.ts",
                    "line": 10,
                    "content": "function validateToken",
                    "score": 0.95,
                }
            ],
            "total": 15,
            "query": "validateToken",
            "next_cursor": "abc123",
        }

        assert "results" in response
        assert isinstance(response["results"], list)
        assert "total" in response

    def test_env_peek_response_format(self):
        """Test env_peek response format."""
        response = {
            "content": "function example() {\n  return true;\n}",
            "path": "/src/example.ts",
            "start_line": 10,
            "end_line": 12,
            "total_lines": 3,
        }

        assert "content" in response
        assert "path" in response
        assert "start_line" in response
        assert "end_line" in response

    def test_project_symbol_search_response_format(self):
        """Test project_symbol_search response format."""
        response = {
            "symbols": [
                {
                    "name": "handleAuth",
                    "type": "function",
                    "file": "src/auth/handler.ts",
                    "line": 9,
                    "signature": "async function handleAuth(token: AuthToken): Promise<User | null>",
                }
            ],
            "total": 5,
            "query": "handleAuth",
        }

        assert "symbols" in response
        assert isinstance(response["symbols"], list)
        if response["symbols"]:
            symbol = response["symbols"][0]
            assert "name" in symbol
            assert "type" in symbol
            assert "file" in symbol

    def test_project_impact_response_format(self):
        """Test project_impact response format."""
        response = {
            "impacts": [
                {
                    "file": "src/api/endpoints.ts",
                    "type": "usage",
                    "line": 15,
                    "description": "Calls handleAuth function",
                }
            ],
            "changed_files": ["src/auth/handler.ts"],
            "total_impacts": 3,
        }

        assert "impacts" in response
        assert "changed_files" in response


# ==============================================================================
# Tool Execution Simulation Tests
# ==============================================================================

@pytest.mark.integration
class TestToolExecutionSimulation:
    """Tests for simulated tool execution."""

    @pytest.fixture
    def mock_icd_service(self):
        """Create mock ICD service."""
        service = MagicMock()
        service.retrieve = AsyncMock(return_value=MagicMock(
            chunks=[],
            scores=[],
            entropy=1.0,
        ))
        service.compile_pack = AsyncMock(return_value=MagicMock(
            content="# Pack",
            token_count=100,
            chunk_ids=[],
        ))
        return service

    @pytest.mark.asyncio
    async def test_memory_pack_execution(self, mock_icd_service):
        """Test memory_pack tool execution."""
        params = {
            "prompt": "Where is auth token validated?",
            "repo_root": "/path/to/repo",
            "budget_tokens": 4000,
            "mode": "auto",
        }

        # Validate input
        is_valid, errors = validate_tool_input("memory_pack", params)
        assert is_valid

        # Simulate execution
        result = await mock_icd_service.compile_pack(
            query=params["prompt"],
            budget_tokens=params["budget_tokens"],
        )

        assert result is not None

    @pytest.mark.asyncio
    async def test_env_search_execution(self, mock_icd_service):
        """Test env_search tool execution."""
        params = {
            "query": "validateToken",
            "scope": "repo",
            "limit": 20,
        }

        is_valid, errors = validate_tool_input("env_search", params)
        assert is_valid

        result = await mock_icd_service.retrieve(
            query=params["query"],
            limit=params["limit"],
        )

        assert result is not None


# ==============================================================================
# Error Response Tests
# ==============================================================================

@pytest.mark.integration
class TestToolErrorResponses:
    """Tests for tool error responses."""

    def test_validation_error_format(self):
        """Test validation error response format."""
        params = {
            "prompt": "",  # Empty prompt
            "repo_root": "/path",
        }

        is_valid, errors = validate_tool_input("memory_pack", params)

        # Should have structured error
        assert isinstance(errors, list)
        for error in errors:
            assert isinstance(error, str)

    def test_unknown_tool_error(self):
        """Test unknown tool error."""
        is_valid, errors = validate_tool_input("nonexistent_tool", {})

        assert is_valid is False
        assert any("Unknown tool" in e for e in errors)


# ==============================================================================
# Tool Integration Tests
# ==============================================================================

@pytest.mark.integration
class TestToolIntegration:
    """Integration tests for tool interactions."""

    def test_tool_chaining_validation(self):
        """Test validation for tool chaining scenario."""
        # First tool: search
        search_params = {
            "query": "handleAuth",
            "scope": "repo",
            "limit": 10,
        }
        is_valid1, _ = validate_tool_input("env_search", search_params)
        assert is_valid1

        # Second tool: peek based on search results
        peek_params = {
            "path": "/src/auth/handler.ts",
            "start_line": 5,
            "end_line": 20,
        }
        is_valid2, _ = validate_tool_input("env_peek", peek_params)
        assert is_valid2

    def test_all_tools_validate(self):
        """Test all tools can be validated with valid inputs."""
        valid_inputs = {
            "memory_pack": {
                "prompt": "test",
                "repo_root": "/path",
                "mode": "auto",
            },
            "env_search": {
                "query": "test",
                "scope": "repo",
            },
            "env_peek": {
                "path": "/file.ts",
                "start_line": 1,
                "end_line": 10,
            },
            "project_map": {
                "repo_root": "/path",
                "depth": 3,
            },
            "project_symbol_search": {
                "query": "function",
                "repo_root": "/path",
            },
            "project_impact": {
                "changed_paths": ["/file.ts"],
            },
        }

        for tool_name, params in valid_inputs.items():
            is_valid, errors = validate_tool_input(tool_name, params)
            assert is_valid, f"Tool {tool_name} validation failed: {errors}"
