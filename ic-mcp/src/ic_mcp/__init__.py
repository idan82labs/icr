"""
IC-MCP: MCP Server for ICR

This module provides the MCP (Model Context Protocol) server implementation
for the ICR (Intelligent Code Research) system. It exposes safe, bounded tools
for Claude Code integration.

The server exposes tools under the `icr` namespace, accessible as:
    mcp__icr__<tool_name>

Example:
    mcp__icr__memory_pack
    mcp__icr__env_search
    mcp__icr__project_map
"""

__version__ = "0.1.0"
__author__ = "ICR Team"

from ic_mcp.server import create_server, main

__all__ = ["create_server", "main", "__version__"]
