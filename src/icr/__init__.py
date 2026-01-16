"""
ICR - Intelligent Code Retrieval

A unified tool for semantic code search, context generation, and Claude Code integration.
"""

__version__ = "0.1.0"
__author__ = "ICR Contributors"

from icr.core import (
    ICRConfig,
    RepoConfig,
    get_icr_root,
    get_repo_id,
    get_repo_root,
    ensure_initialized,
    load_config,
    ICRError,
    NotInitializedError,
    ConfigurationError,
)

__all__ = [
    "__version__",
    "__author__",
    "ICRConfig",
    "RepoConfig",
    "get_icr_root",
    "get_repo_id",
    "get_repo_root",
    "ensure_initialized",
    "load_config",
    "ICRError",
    "NotInitializedError",
    "ConfigurationError",
]
