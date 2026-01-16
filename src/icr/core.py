"""
ICR Core Utilities

Shared utilities for the ICR ecosystem including configuration management,
path resolution, and common operations.
"""

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Optional, Any

import yaml
from pydantic import BaseModel, Field


class ICRError(Exception):
    """Base exception for ICR errors."""

    pass


class NotInitializedError(ICRError):
    """Raised when ICR has not been initialized."""

    pass


class ConfigurationError(ICRError):
    """Raised when there's a configuration problem."""

    pass


class EmbeddingConfig(BaseModel):
    """Configuration for the embedding backend."""

    backend: str = Field(default="local-onnx", description="Embedding backend type")
    model: str = Field(
        default="all-MiniLM-L6-v2", description="Model name for embeddings"
    )
    dimension: int = Field(default=384, description="Embedding dimension")
    batch_size: int = Field(default=32, description="Batch size for embedding")


class IndexConfig(BaseModel):
    """Configuration for the vector index."""

    ef_construction: int = Field(default=200, description="HNSW ef_construction")
    m: int = Field(default=16, description="HNSW M parameter")
    ef_search: int = Field(default=100, description="HNSW ef for search")


class HooksConfig(BaseModel):
    """Configuration for Claude Code hooks."""

    enabled: bool = Field(default=True, description="Whether hooks are enabled")
    prompt_submit: bool = Field(default=True, description="Handle UserPromptSubmit")
    stop: bool = Field(default=True, description="Handle Stop hook")
    precompact: bool = Field(default=True, description="Handle PreCompact hook")


class MCPConfig(BaseModel):
    """Configuration for MCP server."""

    enabled: bool = Field(default=True, description="Whether MCP server is enabled")
    transport: str = Field(default="stdio", description="MCP transport type")


class RepoConfig(BaseModel):
    """Configuration for a specific repository."""

    repo_id: str = Field(description="Unique identifier for the repository")
    path: str = Field(description="Absolute path to the repository")
    name: str = Field(description="Human-readable name")
    indexed_at: Optional[str] = Field(default=None, description="Last index timestamp")
    chunk_count: int = Field(default=0, description="Number of indexed chunks")
    file_count: int = Field(default=0, description="Number of indexed files")


class InvariantConfig(BaseModel):
    """Configuration for a pinned invariant."""

    id: str = Field(description="Unique identifier for the invariant")
    content: str = Field(description="The invariant content")
    repo_id: Optional[str] = Field(
        default=None, description="Associated repository (if any)"
    )
    created_at: str = Field(description="Creation timestamp")


class ICRConfig(BaseModel):
    """Main ICR configuration."""

    version: str = Field(default="1", description="Configuration version")
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    hooks: HooksConfig = Field(default_factory=HooksConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    repositories: dict[str, RepoConfig] = Field(
        default_factory=dict, description="Indexed repositories"
    )
    invariants: list[InvariantConfig] = Field(
        default_factory=list, description="Pinned invariants"
    )


def get_icr_root() -> Path:
    """
    Get the ICR root directory path.

    Returns:
        Path to ~/.icr directory

    Example:
        >>> root = get_icr_root()
        >>> print(root)
        /home/user/.icr
    """
    return Path.home() / ".icr"


def get_repo_id(path: str | Path) -> str:
    """
    Compute a stable repository ID from a path.

    The ID is based on the absolute path to ensure consistency
    across different working directories.

    Args:
        path: Path to the repository

    Returns:
        A stable hash-based identifier for the repository

    Example:
        >>> repo_id = get_repo_id("/path/to/repo")
        >>> print(repo_id)
        repo_a1b2c3d4
    """
    abs_path = Path(path).resolve()
    path_hash = hashlib.sha256(str(abs_path).encode()).hexdigest()[:8]
    return f"repo_{path_hash}"


def get_repo_root(path: str | Path) -> Path:
    """
    Find the git root directory or use the provided path.

    Searches upward from the given path for a .git directory.
    If not found, returns the provided path as-is.

    Args:
        path: Starting path for the search

    Returns:
        Path to the repository root

    Example:
        >>> root = get_repo_root("/path/to/repo/src/file.py")
        >>> print(root)
        /path/to/repo
    """
    abs_path = Path(path).resolve()

    # If it's a file, start from its parent
    if abs_path.is_file():
        abs_path = abs_path.parent

    # Try to find git root using git command
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(abs_path),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return Path(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Fallback: search for .git directory manually
    current = abs_path
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent

    # No git root found, return the original path
    return abs_path


def ensure_initialized() -> Path:
    """
    Ensure ICR is properly initialized.

    Checks that the ICR root directory exists and contains
    a valid configuration file.

    Returns:
        Path to the ICR root directory

    Raises:
        NotInitializedError: If ICR has not been initialized

    Example:
        >>> try:
        ...     root = ensure_initialized()
        ... except NotInitializedError:
        ...     print("Run 'icr init' first")
    """
    icr_root = get_icr_root()

    if not icr_root.exists():
        raise NotInitializedError(
            f"ICR not initialized. Run 'icr init' to set up ICR at {icr_root}"
        )

    config_path = icr_root / "config.yaml"
    if not config_path.exists():
        raise NotInitializedError(
            f"ICR configuration not found at {config_path}. Run 'icr init' to initialize."
        )

    return icr_root


def load_config() -> ICRConfig:
    """
    Load the ICR configuration.

    Returns:
        The ICR configuration object

    Raises:
        NotInitializedError: If ICR has not been initialized
        ConfigurationError: If the configuration is invalid

    Example:
        >>> config = load_config()
        >>> print(config.embedding.model)
        all-MiniLM-L6-v2
    """
    icr_root = ensure_initialized()
    config_path = icr_root / "config.yaml"

    try:
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
        return ICRConfig(**data)
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid YAML in configuration: {e}")
    except Exception as e:
        raise ConfigurationError(f"Failed to load configuration: {e}")


def save_config(config: ICRConfig) -> None:
    """
    Save the ICR configuration.

    Args:
        config: The configuration to save

    Raises:
        NotInitializedError: If ICR has not been initialized
        ConfigurationError: If saving fails

    Example:
        >>> config = load_config()
        >>> config.hooks.enabled = False
        >>> save_config(config)
    """
    icr_root = ensure_initialized()
    config_path = icr_root / "config.yaml"

    try:
        with open(config_path, "w") as f:
            yaml.dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)
    except Exception as e:
        raise ConfigurationError(f"Failed to save configuration: {e}")


def initialize_icr() -> Path:
    """
    Initialize ICR for the current user.

    Creates the ICR root directory and default configuration.

    Returns:
        Path to the created ICR root directory

    Example:
        >>> root = initialize_icr()
        >>> print(f"ICR initialized at {root}")
    """
    icr_root = get_icr_root()

    # Create directory structure
    icr_root.mkdir(parents=True, exist_ok=True)
    (icr_root / "repos").mkdir(exist_ok=True)
    (icr_root / "cache").mkdir(exist_ok=True)
    (icr_root / "logs").mkdir(exist_ok=True)

    # Create default configuration
    config_path = icr_root / "config.yaml"
    if not config_path.exists():
        config = ICRConfig()
        with open(config_path, "w") as f:
            yaml.dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)

    # Create empty database placeholder
    db_path = icr_root / "icr.db"
    if not db_path.exists():
        db_path.touch()

    return icr_root


def get_repo_data_path(repo_id: str) -> Path:
    """
    Get the data directory path for a specific repository.

    Args:
        repo_id: The repository identifier

    Returns:
        Path to the repository's data directory

    Example:
        >>> path = get_repo_data_path("repo_a1b2c3d4")
        >>> print(path)
        /home/user/.icr/repos/repo_a1b2c3d4
    """
    return get_icr_root() / "repos" / repo_id


def format_file_size(size_bytes: int) -> str:
    """
    Format a file size in bytes to a human-readable string.

    Args:
        size_bytes: Size in bytes

    Returns:
        Human-readable size string

    Example:
        >>> print(format_file_size(1536))
        1.5 KB
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds to a human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Human-readable duration string

    Example:
        >>> print(format_duration(125.5))
        2m 5.5s
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    remaining = seconds % 60
    if minutes < 60:
        return f"{minutes}m {remaining:.1f}s"
    hours = minutes // 60
    remaining_mins = minutes % 60
    return f"{hours}h {remaining_mins}m"
