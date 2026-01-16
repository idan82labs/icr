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

import json
import logging
import urllib.request
from typing import Optional

__version__ = "0.1.0"
__author__ = "ICR Team"

logger = logging.getLogger(__name__)


def check_for_updates(timeout: int = 5) -> Optional[dict]:
    """
    Check GitHub for a newer version of ICR.

    This function is non-blocking and fails silently if the check fails.
    It's intended to be called optionally on server startup.

    Args:
        timeout: Request timeout in seconds (default: 5)

    Returns:
        dict with keys {current, latest, url} if update available, else None
    """
    try:
        url = "https://api.github.com/repos/idan82labs/icr/releases/latest"
        request = urllib.request.Request(
            url,
            headers={"Accept": "application/vnd.github.v3+json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode())
            latest = data.get("tag_name", "").lstrip("v")
            if latest and latest != __version__:
                return {
                    "current": __version__,
                    "latest": latest,
                    "url": data.get("html_url", ""),
                    "name": data.get("name", f"Version {latest}"),
                }
    except urllib.error.URLError as e:
        logger.debug(f"Update check failed (network): {e}")
    except json.JSONDecodeError as e:
        logger.debug(f"Update check failed (parse): {e}")
    except Exception as e:
        logger.debug(f"Update check failed: {e}")

    return None


def get_version_info() -> dict:
    """
    Get version information for the IC-MCP package.

    Returns:
        dict with version, author, and optional update info
    """
    info = {
        "version": __version__,
        "author": __author__,
        "package": "ic-mcp",
    }

    # Optionally check for updates (non-blocking, silent failure)
    update = check_for_updates()
    if update:
        info["update_available"] = update

    return info


from ic_mcp.server import create_server, main

__all__ = [
    "create_server",
    "main",
    "__version__",
    "check_for_updates",
    "get_version_info",
]
