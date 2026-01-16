"""
Administrative tools for IC-MCP.

These tools provide server management and diagnostics:
- admin_ping: Health check and diagnostics
"""

import logging
import os
import platform
import sys
import time
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from ic_mcp.schemas.inputs import AdminPingInput
from ic_mcp.schemas.outputs import AdminPingOutput

logger = logging.getLogger(__name__)

# Server start time for uptime calculation
_server_start_time: float | None = None


def set_server_start_time() -> None:
    """Set the server start time (called when server starts)."""
    global _server_start_time
    _server_start_time = time.time()


def get_uptime() -> float:
    """Get server uptime in seconds."""
    if _server_start_time is None:
        return 0.0
    return time.time() - _server_start_time


class AdminTools:
    """
    Administrative tools for server management.

    Provides health checks, diagnostics, and server information.
    """

    def __init__(self) -> None:
        """Initialize admin tools."""
        pass

    async def admin_ping(
        self,
        input_data: AdminPingInput,
        request_id: UUID,
    ) -> AdminPingOutput:
        """
        Health check and diagnostics.

        Returns server status and optionally detailed diagnostics.

        Args:
            input_data: Validated input parameters
            request_id: Unique request identifier

        Returns:
            AdminPingOutput with server status
        """
        logger.info(f"admin_ping: echo={input_data.echo}, diagnostics={input_data.include_diagnostics}")

        # Import version
        from ic_mcp import __version__

        diagnostics: dict[str, Any] | None = None

        if input_data.include_diagnostics:
            diagnostics = self._collect_diagnostics()

        return AdminPingOutput(
            request_id=request_id,
            echo=input_data.echo,
            server_version=__version__,
            server_name="icr",
            uptime_seconds=get_uptime(),
            diagnostics=diagnostics,
        )

    def _collect_diagnostics(self) -> dict[str, Any]:
        """Collect server diagnostics."""
        import gc

        # Memory info
        try:
            import resource

            mem_usage = resource.getrusage(resource.RUSAGE_SELF)
            memory_info = {
                "max_rss_mb": mem_usage.ru_maxrss / (1024 * 1024),
                "user_time_seconds": mem_usage.ru_utime,
                "system_time_seconds": mem_usage.ru_stime,
            }
        except ImportError:
            # Windows doesn't have resource module
            memory_info = {"available": False}

        # Python info
        python_info = {
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        }

        # Platform info
        platform_info = {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        }

        # GC stats
        gc_stats = {
            "collections": gc.get_count(),
            "threshold": gc.get_threshold(),
            "objects_tracked": len(gc.get_objects()),
        }

        # Environment (safe subset)
        safe_env_vars = [
            "PATH",
            "PYTHONPATH",
            "VIRTUAL_ENV",
            "HOME",
            "USER",
            "SHELL",
            "TERM",
        ]
        environment = {
            k: os.environ.get(k, "")[:200]  # Truncate for safety
            for k in safe_env_vars
            if k in os.environ
        }

        # Loaded modules count
        modules_info = {
            "loaded_count": len(sys.modules),
            "ic_mcp_modules": [
                name for name in sys.modules.keys()
                if name.startswith("ic_mcp")
            ],
        }

        # Current time
        timestamp = datetime.now(timezone.utc).isoformat()

        return {
            "timestamp": timestamp,
            "python": python_info,
            "platform": platform_info,
            "memory": memory_info,
            "gc": gc_stats,
            "environment": environment,
            "modules": modules_info,
        }
