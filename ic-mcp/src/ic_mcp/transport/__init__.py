"""
Transport layer implementations for IC-MCP.

This module provides transport implementations for the MCP server,
primarily stdio for local Claude Code integration.
"""

from ic_mcp.transport.stdio import StdioTransport, run_stdio_server

__all__ = ["StdioTransport", "run_stdio_server"]
