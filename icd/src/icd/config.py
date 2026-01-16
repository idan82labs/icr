"""
Configuration module for ICD.

Provides strongly-typed configuration with pydantic, supporting both
file-based and environment variable configuration.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingBackend(str, Enum):
    """Supported embedding backends."""

    LOCAL_ONNX = "local_onnx"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    CUSTOM = "custom"


class VectorDtype(str, Enum):
    """Vector storage data types."""

    FLOAT16 = "float16"
    FLOAT32 = "float32"


class StorageConfig(BaseModel):
    """Storage layer configuration."""

    # SQLite settings
    db_path: Path = Field(
        default=Path(".icd/index.db"),
        description="Path to SQLite database file",
    )
    wal_mode: bool = Field(
        default=True,
        description="Enable WAL mode for concurrent reads",
    )
    cache_size_mb: int = Field(
        default=64,
        ge=8,
        le=512,
        description="SQLite cache size in MB",
    )

    # Vector index settings
    max_vectors_per_repo: int = Field(
        default=250000,
        ge=1000,
        le=10000000,
        description="Maximum vectors per repository",
    )
    vector_dtype: VectorDtype = Field(
        default=VectorDtype.FLOAT16,
        description="Vector storage data type (float16 saves 50% memory)",
    )
    hnsw_m: int = Field(
        default=16,
        ge=4,
        le=64,
        description="HNSW M parameter (connections per node)",
    )
    hnsw_ef_construction: int = Field(
        default=200,
        ge=50,
        le=500,
        description="HNSW ef_construction (index build quality)",
    )
    hnsw_ef_search: int = Field(
        default=100,
        ge=10,
        le=500,
        description="HNSW ef_search (query quality vs speed)",
    )


class EmbeddingConfig(BaseModel):
    """Embedding generation configuration."""

    backend: EmbeddingBackend = Field(
        default=EmbeddingBackend.LOCAL_ONNX,
        description="Embedding backend to use",
    )
    model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="Embedding model name",
    )
    model_path: Path | None = Field(
        default=None,
        description="Path to local ONNX model (auto-downloads if None)",
    )
    dimension: int = Field(
        default=384,
        ge=64,
        le=4096,
        description="Embedding dimension",
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        le=256,
        description="Batch size for embedding generation",
    )
    max_tokens: int = Field(
        default=512,
        ge=128,
        le=8192,
        description="Maximum tokens per chunk for embedding",
    )
    normalize: bool = Field(
        default=True,
        description="L2 normalize embeddings",
    )


class ChunkingConfig(BaseModel):
    """Code chunking configuration."""

    min_tokens: int = Field(
        default=200,
        ge=50,
        le=500,
        description="Minimum tokens per chunk",
    )
    target_tokens: int = Field(
        default=500,
        ge=200,
        le=1000,
        description="Target tokens per chunk",
    )
    max_tokens: int = Field(
        default=1200,
        ge=500,
        le=4000,
        description="Maximum tokens per chunk",
    )
    overlap_tokens: int = Field(
        default=50,
        ge=0,
        le=200,
        description="Token overlap between chunks",
    )
    preserve_symbols: bool = Field(
        default=True,
        description="Preserve symbol boundaries (functions, classes)",
    )


class RetrievalConfig(BaseModel):
    """Retrieval and ranking configuration."""

    # Hybrid scoring weights
    weight_embedding: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight for semantic similarity (w_e)",
    )
    weight_bm25: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for BM25 lexical score (w_b)",
    )
    weight_recency: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Weight for recency boost (w_r)",
    )
    weight_contract: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Weight for contract indicator (w_c)",
    )
    weight_focus: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Weight for focus scope indicator (w_f)",
    )
    weight_pinned: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Weight for pinned indicator (w_p)",
    )

    # Recency decay
    recency_tau_days: float = Field(
        default=30.0,
        ge=1.0,
        le=365.0,
        description="Recency decay time constant (tau) in days",
    )

    # MMR settings
    mmr_lambda: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="MMR lambda (relevance vs diversity)",
    )

    # Retrieval limits
    initial_candidates: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Initial candidates for re-ranking",
    )
    final_results: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Final results after MMR",
    )

    # Entropy settings
    entropy_temperature: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Temperature for entropy computation",
    )


class PackConfig(BaseModel):
    """Pack compilation configuration."""

    default_budget_tokens: int = Field(
        default=8000,
        ge=1000,
        le=128000,
        description="Default token budget for packs",
    )
    max_budget_tokens: int = Field(
        default=32000,
        ge=4000,
        le=200000,
        description="Maximum token budget for packs",
    )
    include_metadata: bool = Field(
        default=True,
        description="Include chunk metadata in packs",
    )
    include_citations: bool = Field(
        default=True,
        description="Include citation markers in packs",
    )


class WatcherConfig(BaseModel):
    """File system watcher configuration."""

    debounce_ms: int = Field(
        default=500,
        ge=100,
        le=5000,
        description="Debounce delay in milliseconds",
    )
    ignore_patterns: list[str] = Field(
        default_factory=lambda: [
            "**/.git/**",
            "**/.hg/**",
            "**/.svn/**",
            "**/node_modules/**",
            "**/__pycache__/**",
            "**/*.pyc",
            "**/venv/**",
            "**/.venv/**",
            "**/dist/**",
            "**/build/**",
            "**/.tox/**",
            "**/coverage/**",
            "**/.coverage",
            "**/*.egg-info/**",
            "**/target/**",  # Rust
            "**/vendor/**",  # Go
        ],
        description="Glob patterns to ignore",
    )
    watch_extensions: list[str] = Field(
        default_factory=lambda: [
            ".py",
            ".js",
            ".ts",
            ".tsx",
            ".jsx",
            ".go",
            ".rs",
            ".java",
            ".c",
            ".cpp",
            ".h",
            ".hpp",
            ".cs",
            ".rb",
            ".php",
            ".swift",
            ".kt",
            ".scala",
            ".md",
            ".rst",
            ".txt",
            ".json",
            ".yaml",
            ".yml",
            ".toml",
        ],
        description="File extensions to watch",
    )
    max_file_size_kb: int = Field(
        default=500,
        ge=10,
        le=10000,
        description="Maximum file size to index in KB",
    )


class RLMConfig(BaseModel):
    """Retrieval-augmented LM (RLM) configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable RLM fallback for high-entropy queries",
    )
    entropy_threshold: float = Field(
        default=2.5,
        ge=0.0,
        le=10.0,
        description="Entropy threshold to trigger RLM",
    )
    max_iterations: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum RLM iterations",
    )
    budget_per_iteration: int = Field(
        default=2000,
        ge=500,
        le=10000,
        description="Token budget per RLM iteration",
    )


class ContractConfig(BaseModel):
    """Contract detection configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable contract detection",
    )
    patterns: list[str] = Field(
        default_factory=lambda: [
            "interface",
            "abstract",
            "protocol",
            "trait",
            "type.*=",  # TypeScript type aliases
            "@dataclass",
            "schema",
            "model",
            "struct",
            "enum",
        ],
        description="Patterns indicating contracts",
    )
    boost_factor: float = Field(
        default=1.5,
        ge=1.0,
        le=5.0,
        description="Boost factor for contracts in retrieval",
    )


class NetworkConfig(BaseModel):
    """Network and API configuration."""

    enabled: bool = Field(
        default=False,
        description="Enable network access (for remote embeddings)",
    )
    api_base_url: str | None = Field(
        default=None,
        description="Base URL for remote API",
    )
    api_key: str | None = Field(
        default=None,
        description="API key for remote services",
    )
    timeout_seconds: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Request timeout in seconds",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts",
    )


class TelemetryConfig(BaseModel):
    """Telemetry and metrics configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable local telemetry collection",
    )
    metrics_path: Path = Field(
        default=Path(".icd/metrics.db"),
        description="Path to metrics database",
    )
    retention_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Metrics retention period in days",
    )


class Config(BaseSettings):
    """
    Main ICD configuration.

    Can be configured via:
    1. Configuration file (icd.toml or icd.yaml)
    2. Environment variables with ICD_ prefix
    3. Programmatic overrides
    """

    model_config = SettingsConfigDict(
        env_prefix="ICD_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    # Core settings
    project_root: Path = Field(
        default_factory=lambda: Path.cwd(),
        description="Project root directory",
    )
    data_dir: Path = Field(
        default=Path(".icd"),
        description="ICD data directory (relative to project_root)",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )

    # Sub-configurations
    storage: StorageConfig = Field(default_factory=StorageConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    pack: PackConfig = Field(default_factory=PackConfig)
    watcher: WatcherConfig = Field(default_factory=WatcherConfig)
    rlm: RLMConfig = Field(default_factory=RLMConfig)
    contract: ContractConfig = Field(default_factory=ContractConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)

    @field_validator("project_root", mode="before")
    @classmethod
    def resolve_project_root(cls, v: Path | str) -> Path:
        """Resolve project root to absolute path."""
        path = Path(v) if isinstance(v, str) else v
        return path.resolve()

    @property
    def absolute_data_dir(self) -> Path:
        """Get absolute path to data directory."""
        if self.data_dir.is_absolute():
            return self.data_dir
        return self.project_root / self.data_dir

    @property
    def db_path(self) -> Path:
        """Get absolute path to SQLite database."""
        return self.absolute_data_dir / "index.db"

    @property
    def vector_index_path(self) -> Path:
        """Get absolute path to vector index."""
        return self.absolute_data_dir / "vectors.hnsw"

    @property
    def metrics_path(self) -> Path:
        """Get absolute path to metrics database."""
        return self.absolute_data_dir / "metrics.db"

    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        self.absolute_data_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_file(cls, path: Path) -> "Config":
        """Load configuration from a TOML or YAML file."""
        import json

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        suffix = path.suffix.lower()
        content = path.read_text()

        if suffix == ".toml":
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib  # type: ignore
            data = tomllib.loads(content)
        elif suffix in (".yaml", ".yml"):
            try:
                import yaml

                data = yaml.safe_load(content)
            except ImportError:
                raise ImportError("PyYAML required for YAML config files")
        elif suffix == ".json":
            data = json.loads(content)
        else:
            raise ValueError(f"Unsupported config format: {suffix}")

        return cls(**data)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return self.model_dump()


def get_default_config() -> Config:
    """Get default configuration instance."""
    return Config()


def load_config(
    config_path: Path | None = None,
    project_root: Path | None = None,
) -> Config:
    """
    Load configuration with automatic discovery.

    Priority:
    1. Explicit config_path if provided
    2. icd.toml in project_root
    3. .icd/config.toml in project_root
    4. Default configuration
    """
    root = project_root or Path.cwd()

    if config_path and config_path.exists():
        config = Config.from_file(config_path)
        config = config.model_copy(update={"project_root": root})
        return config

    # Auto-discover config
    candidates = [
        root / "icd.toml",
        root / ".icd" / "config.toml",
        root / "icd.yaml",
        root / ".icd" / "config.yaml",
    ]

    for candidate in candidates:
        if candidate.exists():
            config = Config.from_file(candidate)
            config = config.model_copy(update={"project_root": root})
            return config

    # Return default with project root set
    return Config(project_root=root)
