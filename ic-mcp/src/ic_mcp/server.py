"""
MCP Server main entry point for IC-MCP.

This module implements the MCP (Model Context Protocol) server that exposes
ICR tools for Claude Code integration. The server handles tool registration,
request routing, and response formatting.

Server Configuration:
- Default transport: local stdio
- Server name: icr
- Tools exposed as: mcp__icr__<tool_name>

Output Limits:
- Soft limit: 8,000 tokens per response
- Hard limit: 25,000 tokens per response
"""

import asyncio
import json
import logging
import sys
from typing import Any, Callable, Coroutine
from uuid import UUID, uuid4

from pydantic import BaseModel

from ic_mcp.schemas.inputs import (
    AdminPingInput,
    EnvAggregateInput,
    EnvPeekInput,
    EnvSearchInput,
    EnvSliceInput,
    MemoryGetInput,
    MemoryListInput,
    MemoryPackInput,
    MemoryPinInput,
    MemoryStatsInput,
    MemoryUnpinInput,
    ProjectCommandsInput,
    ProjectImpactInput,
    ProjectMapInput,
    ProjectSymbolSearchInput,
    RlmMapReduceInput,
    RlmPlanInput,
)
from ic_mcp.schemas.outputs import ErrorEnvelope, ErrorInfo
from ic_mcp.schemas.validation import get_json_schema, validate_input
from ic_mcp.tools.admin import AdminTools, set_server_start_time
from ic_mcp.tools.env import EnvTools
from ic_mcp.tools.memory import MemoryTools
from ic_mcp.tools.project import ProjectTools
from ic_mcp.tools.rlm import RlmTools

logger = logging.getLogger(__name__)

# Server configuration
SERVER_NAME = "icr"
SERVER_VERSION = "0.1.0"
PROTOCOL_VERSION = "2024-11-05"


# Type alias for tool handlers
ToolHandler = Callable[[dict[str, Any], UUID], Coroutine[Any, Any, BaseModel]]


class ToolDefinition:
    """Definition of an MCP tool."""

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: type[BaseModel],
        handler: ToolHandler,
    ) -> None:
        """
        Initialize a tool definition.

        Args:
            name: Tool name (without namespace prefix)
            description: Human-readable tool description
            input_schema: Pydantic model for input validation
            handler: Async function to handle tool calls
        """
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.handler = handler

    @property
    def full_name(self) -> str:
        """Get the full namespaced tool name."""
        return f"{SERVER_NAME}__{self.name}"

    def get_json_schema(self) -> dict[str, Any]:
        """Get JSON Schema for the tool's input."""
        return get_json_schema(self.input_schema)


class ICRMCPServer:
    """
    IC-MCP Server implementation.

    Handles MCP protocol messages, tool registration, and request routing.
    """

    def __init__(self, repo_root: str | None = None) -> None:
        """
        Initialize the MCP server.

        Args:
            repo_root: Optional default repository root for tools
        """
        self.repo_root = repo_root
        self._tools: dict[str, ToolDefinition] = {}
        self._initialized = False

        # Initialize tool instances
        self._memory_tools = MemoryTools()
        self._env_tools = EnvTools(repo_root)
        self._project_tools = ProjectTools()
        self._rlm_tools = RlmTools()
        self._admin_tools = AdminTools()

        # Register all tools
        self._register_tools()

        # Set server start time
        set_server_start_time()

    def _register_tools(self) -> None:
        """Register all available tools."""
        # Memory tools
        self._register_tool(
            ToolDefinition(
                name="memory_pack",
                description=(
                    "Pack relevant context for a prompt. Analyzes the repository and "
                    "returns the most relevant code context within a token budget. "
                    "Use this as the primary tool for context retrieval."
                ),
                input_schema=MemoryPackInput,
                handler=self._memory_tools.memory_pack,
            )
        )

        self._register_tool(
            ToolDefinition(
                name="memory_pin",
                description=(
                    "Pin a source to always include in context packs. "
                    "Use this to ensure important files are always considered."
                ),
                input_schema=MemoryPinInput,
                handler=self._memory_tools.memory_pin,
            )
        )

        self._register_tool(
            ToolDefinition(
                name="memory_unpin",
                description="Remove a pin from a source.",
                input_schema=MemoryUnpinInput,
                handler=self._memory_tools.memory_unpin,
            )
        )

        self._register_tool(
            ToolDefinition(
                name="memory_list",
                description=(
                    "List memory items with optional filtering. "
                    "Shows pinned, recent, or all tracked sources."
                ),
                input_schema=MemoryListInput,
                handler=self._memory_tools.memory_list,
            )
        )

        self._register_tool(
            ToolDefinition(
                name="memory_get",
                description="Get details of a specific memory item by source ID.",
                input_schema=MemoryGetInput,
                handler=self._memory_tools.memory_get,
            )
        )

        self._register_tool(
            ToolDefinition(
                name="memory_stats",
                description="Get memory statistics including item counts and sizes.",
                input_schema=MemoryStatsInput,
                handler=self._memory_tools.memory_stats,
            )
        )

        # Env tools
        self._register_tool(
            ToolDefinition(
                name="env_search",
                description=(
                    "Search across the environment (repository, transcripts, diffs, contracts). "
                    "Returns relevant files and snippets matching the query."
                ),
                input_schema=EnvSearchInput,
                handler=self._env_tools.env_search,
            )
        )

        self._register_tool(
            ToolDefinition(
                name="env_peek",
                description=(
                    "View specific lines of a file. "
                    "Provides bounded access to file content with line numbers."
                ),
                input_schema=EnvPeekInput,
                handler=self._env_tools.env_peek,
            )
        )

        self._register_tool(
            ToolDefinition(
                name="env_slice",
                description=(
                    "Extract a slice from a file by symbol name or line range. "
                    "Useful for getting function or class definitions with context."
                ),
                input_schema=EnvSliceInput,
                handler=self._env_tools.env_slice,
            )
        )

        self._register_tool(
            ToolDefinition(
                name="env_aggregate",
                description=(
                    "Perform aggregation operations on data. "
                    "Supports: extract_regex, unique, sort, group_by, count, top_k, join_on, diff_sets."
                ),
                input_schema=EnvAggregateInput,
                handler=self._env_tools.env_aggregate,
            )
        )

        # Project tools
        self._register_tool(
            ToolDefinition(
                name="project_map",
                description=(
                    "Generate a project structure map. "
                    "Creates a hierarchical view of the project with optional file statistics."
                ),
                input_schema=ProjectMapInput,
                handler=self._project_tools.project_map,
            )
        )

        self._register_tool(
            ToolDefinition(
                name="project_symbol_search",
                description=(
                    "Search for symbols (functions, classes, etc.) across the project. "
                    "Useful for finding definitions and their locations."
                ),
                input_schema=ProjectSymbolSearchInput,
                handler=self._project_tools.project_symbol_search,
            )
        )

        self._register_tool(
            ToolDefinition(
                name="project_impact",
                description=(
                    "Analyze the impact of file changes. "
                    "Builds a dependency graph showing what might be affected by changes."
                ),
                input_schema=ProjectImpactInput,
                handler=self._project_tools.project_impact,
            )
        )

        self._register_tool(
            ToolDefinition(
                name="project_commands",
                description=(
                    "Discover project commands from config files. "
                    "Examines package.json, Makefile, pyproject.toml, etc."
                ),
                input_schema=ProjectCommandsInput,
                handler=self._project_tools.project_commands,
            )
        )

        # RLM tools
        self._register_tool(
            ToolDefinition(
                name="rlm_plan",
                description=(
                    "Generate an execution plan for a complex task. "
                    "Creates a sequence of steps to achieve the task within budget constraints."
                ),
                input_schema=RlmPlanInput,
                handler=self._rlm_tools.rlm_plan,
            )
        )

        self._register_tool(
            ToolDefinition(
                name="rlm_map_reduce",
                description=(
                    "Execute a map-reduce operation across multiple sources. "
                    "Maps a prompt template across sources and reduces the results."
                ),
                input_schema=RlmMapReduceInput,
                handler=self._rlm_tools.rlm_map_reduce,
            )
        )

        # Admin tools
        self._register_tool(
            ToolDefinition(
                name="admin_ping",
                description=(
                    "Health check and diagnostics. "
                    "Returns server status and optionally detailed diagnostics."
                ),
                input_schema=AdminPingInput,
                handler=self._admin_tools.admin_ping,
            )
        )

    def _register_tool(self, tool: ToolDefinition) -> None:
        """Register a tool with the server."""
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.full_name}")

    def get_tool_list(self) -> list[dict[str, Any]]:
        """Get the list of available tools in MCP format."""
        return [
            {
                "name": tool.full_name,
                "description": tool.description,
                "inputSchema": tool.get_json_schema(),
            }
            for tool in self._tools.values()
        ]

    async def handle_message(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """
        Handle an incoming MCP message.

        Args:
            message: The JSON-RPC message

        Returns:
            Response message or None for notifications
        """
        method = message.get("method")
        msg_id = message.get("id")
        params = message.get("params", {})

        logger.debug(f"Handling message: method={method}, id={msg_id}")

        try:
            if method == "initialize":
                return self._handle_initialize(msg_id, params)

            elif method == "initialized":
                # Notification, no response needed
                self._initialized = True
                return None

            elif method == "tools/list":
                return self._handle_tools_list(msg_id)

            elif method == "tools/call":
                return await self._handle_tool_call(msg_id, params)

            elif method == "ping":
                return self._make_response(msg_id, {})

            elif method == "shutdown":
                return self._make_response(msg_id, {})

            else:
                return self._make_error_response(
                    msg_id,
                    -32601,
                    f"Method not found: {method}",
                )

        except Exception as e:
            logger.exception(f"Error handling message: {e}")
            return self._make_error_response(
                msg_id,
                -32603,
                f"Internal error: {e}",
            )

    def _handle_initialize(
        self,
        msg_id: Any,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle initialize request."""
        client_info = params.get("clientInfo", {})
        logger.info(
            f"Initialize from client: {client_info.get('name', 'unknown')} "
            f"v{client_info.get('version', '?')}"
        )

        return self._make_response(
            msg_id,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {
                    "tools": {"listChanged": False},
                },
                "serverInfo": {
                    "name": SERVER_NAME,
                    "version": SERVER_VERSION,
                },
            },
        )

    def _handle_tools_list(self, msg_id: Any) -> dict[str, Any]:
        """Handle tools/list request."""
        return self._make_response(
            msg_id,
            {"tools": self.get_tool_list()},
        )

    async def _handle_tool_call(
        self,
        msg_id: Any,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle tools/call request."""
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        # Strip server prefix if present
        if tool_name.startswith(f"{SERVER_NAME}__"):
            tool_name = tool_name[len(SERVER_NAME) + 2:]

        if tool_name not in self._tools:
            return self._make_error_response(
                msg_id,
                -32602,
                f"Unknown tool: {tool_name}",
            )

        tool = self._tools[tool_name]
        request_id = uuid4()

        try:
            # Validate input
            validated_input = validate_input(tool.input_schema, arguments)

            # Call handler
            result = await tool.handler(validated_input, request_id)

            # Serialize result
            result_dict = result.model_dump(mode="json")

            return self._make_response(
                msg_id,
                {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result_dict, indent=2, default=str),
                        }
                    ],
                    "isError": False,
                },
            )

        except ValueError as e:
            # Validation error
            error_response = ErrorEnvelope(
                request_id=request_id,
                error=ErrorInfo(
                    code="VALIDATION_ERROR",
                    message=str(e),
                    retryable=False,
                ),
            )
            return self._make_response(
                msg_id,
                {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                error_response.model_dump(mode="json"),
                                indent=2,
                            ),
                        }
                    ],
                    "isError": True,
                },
            )

        except Exception as e:
            logger.exception(f"Error in tool {tool_name}: {e}")
            error_response = ErrorEnvelope(
                request_id=request_id,
                error=ErrorInfo(
                    code="INTERNAL_ERROR",
                    message=str(e),
                    retryable=True,
                ),
            )
            return self._make_response(
                msg_id,
                {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                error_response.model_dump(mode="json"),
                                indent=2,
                            ),
                        }
                    ],
                    "isError": True,
                },
            )

    def _make_response(
        self,
        msg_id: Any,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Create a JSON-RPC response."""
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": result,
        }

    def _make_error_response(
        self,
        msg_id: Any,
        code: int,
        message: str,
    ) -> dict[str, Any]:
        """Create a JSON-RPC error response."""
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {
                "code": code,
                "message": message,
            },
        }


def create_server(repo_root: str | None = None) -> ICRMCPServer:
    """
    Create an MCP server instance.

    Args:
        repo_root: Optional default repository root

    Returns:
        Configured ICRMCPServer instance
    """
    return ICRMCPServer(repo_root)


async def run_server(server: ICRMCPServer) -> None:
    """
    Run the MCP server with stdio transport.

    Args:
        server: The server instance to run
    """
    from ic_mcp.transport.stdio import run_stdio_server

    await run_stdio_server(server)


def main() -> None:
    """Main entry point for the IC-MCP server."""
    import argparse

    parser = argparse.ArgumentParser(
        description="IC-MCP Server - MCP tools for intelligent code retrieval"
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=None,
        help="Default repository root path",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Log file path (logs to stderr if not specified)",
    )

    args = parser.parse_args()

    # Configure logging
    log_handlers: list[logging.Handler] = []

    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        log_handlers.append(file_handler)
    else:
        # Log to stderr to avoid interfering with stdio transport
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        log_handlers.append(stderr_handler)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        handlers=log_handlers,
    )

    logger.info(f"Starting IC-MCP server v{SERVER_VERSION}")
    logger.info(f"Protocol version: {PROTOCOL_VERSION}")

    if args.repo_root:
        logger.info(f"Default repo root: {args.repo_root}")

    # Create and run server
    server = create_server(args.repo_root)

    try:
        asyncio.run(run_server(server))
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.exception(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
