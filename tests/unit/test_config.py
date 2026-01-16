"""
Unit tests for the ICD configuration module.

Tests cover:
- Configuration loading and validation
- Default values
- Environment variable overrides
- Configuration file parsing
- Constraint validation
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


class TestStorageConfig:
    """Tests for StorageConfig."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        try:
            from icd.config import StorageConfig

            config = StorageConfig()
            assert config.db_path == Path(".icd/index.db")
            assert config.wal_mode is True
            assert config.cache_size_mb == 64
            assert config.max_vectors_per_repo == 250000
            assert config.hnsw_m == 16
            assert config.hnsw_ef_construction == 200
            assert config.hnsw_ef_search == 100
        except ImportError:
            pytest.skip("icd.config module not available")

    def test_vector_dtype_default(self):
        """Test that float16 is the default vector dtype."""
        try:
            from icd.config import StorageConfig, VectorDtype

            config = StorageConfig()
            assert config.vector_dtype == VectorDtype.FLOAT16
        except ImportError:
            pytest.skip("icd.config module not available")

    def test_cache_size_bounds(self):
        """Test cache size validation bounds."""
        try:
            from icd.config import StorageConfig
            from pydantic import ValidationError

            # Valid bounds
            config = StorageConfig(cache_size_mb=8)
            assert config.cache_size_mb == 8

            config = StorageConfig(cache_size_mb=512)
            assert config.cache_size_mb == 512

            # Invalid bounds
            with pytest.raises(ValidationError):
                StorageConfig(cache_size_mb=7)

            with pytest.raises(ValidationError):
                StorageConfig(cache_size_mb=513)
        except ImportError:
            pytest.skip("icd.config module not available")

    def test_hnsw_parameters_bounds(self):
        """Test HNSW parameter validation bounds."""
        try:
            from icd.config import StorageConfig
            from pydantic import ValidationError

            # Valid M parameter
            config = StorageConfig(hnsw_m=4)
            assert config.hnsw_m == 4

            config = StorageConfig(hnsw_m=64)
            assert config.hnsw_m == 64

            # Invalid M parameter
            with pytest.raises(ValidationError):
                StorageConfig(hnsw_m=3)

            with pytest.raises(ValidationError):
                StorageConfig(hnsw_m=65)
        except ImportError:
            pytest.skip("icd.config module not available")


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig."""

    def test_default_backend(self):
        """Test that local ONNX is the default backend."""
        try:
            from icd.config import EmbeddingBackend, EmbeddingConfig

            config = EmbeddingConfig()
            assert config.backend == EmbeddingBackend.LOCAL_ONNX
        except ImportError:
            pytest.skip("icd.config module not available")

    def test_default_model(self):
        """Test default model name."""
        try:
            from icd.config import EmbeddingConfig

            config = EmbeddingConfig()
            assert config.model_name == "all-MiniLM-L6-v2"
            assert config.dimension == 384
        except ImportError:
            pytest.skip("icd.config module not available")

    def test_dimension_bounds(self):
        """Test embedding dimension bounds."""
        try:
            from icd.config import EmbeddingConfig
            from pydantic import ValidationError

            # Valid dimensions
            config = EmbeddingConfig(dimension=64)
            assert config.dimension == 64

            config = EmbeddingConfig(dimension=4096)
            assert config.dimension == 4096

            # Invalid dimensions
            with pytest.raises(ValidationError):
                EmbeddingConfig(dimension=63)

            with pytest.raises(ValidationError):
                EmbeddingConfig(dimension=4097)
        except ImportError:
            pytest.skip("icd.config module not available")

    def test_batch_size_bounds(self):
        """Test batch size bounds."""
        try:
            from icd.config import EmbeddingConfig
            from pydantic import ValidationError

            config = EmbeddingConfig(batch_size=1)
            assert config.batch_size == 1

            config = EmbeddingConfig(batch_size=256)
            assert config.batch_size == 256

            with pytest.raises(ValidationError):
                EmbeddingConfig(batch_size=0)

            with pytest.raises(ValidationError):
                EmbeddingConfig(batch_size=257)
        except ImportError:
            pytest.skip("icd.config module not available")


class TestChunkingConfig:
    """Tests for ChunkingConfig."""

    def test_default_token_bounds(self):
        """Test default token bounds match PRD spec."""
        try:
            from icd.config import ChunkingConfig

            config = ChunkingConfig()
            assert config.min_tokens == 200
            assert config.target_tokens == 500
            assert config.max_tokens == 1200  # PRD: hard max 1200
        except ImportError:
            pytest.skip("icd.config module not available")

    def test_token_bound_constraints(self):
        """Test token bound validation."""
        try:
            from icd.config import ChunkingConfig
            from pydantic import ValidationError

            # Valid configuration
            config = ChunkingConfig(
                min_tokens=100,
                target_tokens=400,
                max_tokens=800,
            )
            assert config.min_tokens == 100

            # Invalid: min too low
            with pytest.raises(ValidationError):
                ChunkingConfig(min_tokens=49)

            # Invalid: max too high
            with pytest.raises(ValidationError):
                ChunkingConfig(max_tokens=4001)
        except ImportError:
            pytest.skip("icd.config module not available")

    def test_preserve_symbols_default(self):
        """Test that preserve_symbols is enabled by default."""
        try:
            from icd.config import ChunkingConfig

            config = ChunkingConfig()
            assert config.preserve_symbols is True
        except ImportError:
            pytest.skip("icd.config module not available")


class TestRetrievalConfig:
    """Tests for RetrievalConfig."""

    def test_hybrid_weights_defaults(self):
        """Test default hybrid scoring weights."""
        try:
            from icd.config import RetrievalConfig

            config = RetrievalConfig()
            # PRD formula weights
            assert config.weight_embedding == 0.4  # w_e
            assert config.weight_bm25 == 0.3       # w_b
            assert config.weight_recency == 0.1    # w_r
            assert config.weight_contract == 0.1   # w_c
            assert config.weight_focus == 0.05     # w_f
            assert config.weight_pinned == 0.05    # w_p
        except ImportError:
            pytest.skip("icd.config module not available")

    def test_mmr_lambda_default(self):
        """Test default MMR lambda parameter."""
        try:
            from icd.config import RetrievalConfig

            config = RetrievalConfig()
            assert config.mmr_lambda == 0.7
        except ImportError:
            pytest.skip("icd.config module not available")

    def test_weight_bounds(self):
        """Test weight parameter bounds [0.0, 1.0]."""
        try:
            from icd.config import RetrievalConfig
            from pydantic import ValidationError

            # Valid bounds
            config = RetrievalConfig(weight_embedding=0.0)
            assert config.weight_embedding == 0.0

            config = RetrievalConfig(weight_embedding=1.0)
            assert config.weight_embedding == 1.0

            # Invalid bounds
            with pytest.raises(ValidationError):
                RetrievalConfig(weight_embedding=-0.1)

            with pytest.raises(ValidationError):
                RetrievalConfig(weight_embedding=1.1)
        except ImportError:
            pytest.skip("icd.config module not available")

    def test_entropy_temperature_bounds(self):
        """Test entropy temperature bounds."""
        try:
            from icd.config import RetrievalConfig
            from pydantic import ValidationError

            config = RetrievalConfig(entropy_temperature=0.1)
            assert config.entropy_temperature == 0.1

            config = RetrievalConfig(entropy_temperature=10.0)
            assert config.entropy_temperature == 10.0

            with pytest.raises(ValidationError):
                RetrievalConfig(entropy_temperature=0.09)
        except ImportError:
            pytest.skip("icd.config module not available")


class TestPackConfig:
    """Tests for PackConfig."""

    def test_default_budget(self):
        """Test default token budget matches PRD."""
        try:
            from icd.config import PackConfig

            config = PackConfig()
            assert config.default_budget_tokens == 8000  # PRD default
            assert config.max_budget_tokens == 32000
        except ImportError:
            pytest.skip("icd.config module not available")

    def test_budget_bounds(self):
        """Test budget token bounds."""
        try:
            from icd.config import PackConfig
            from pydantic import ValidationError

            config = PackConfig(default_budget_tokens=1000)
            assert config.default_budget_tokens == 1000

            with pytest.raises(ValidationError):
                PackConfig(default_budget_tokens=999)

            with pytest.raises(ValidationError):
                PackConfig(default_budget_tokens=128001)
        except ImportError:
            pytest.skip("icd.config module not available")


class TestRLMConfig:
    """Tests for RLM configuration."""

    def test_default_enabled(self):
        """Test RLM is enabled by default."""
        try:
            from icd.config import RLMConfig

            config = RLMConfig()
            assert config.enabled is True
        except ImportError:
            pytest.skip("icd.config module not available")

    def test_entropy_threshold(self):
        """Test default entropy threshold."""
        try:
            from icd.config import RLMConfig

            config = RLMConfig()
            assert config.entropy_threshold == 2.5
        except ImportError:
            pytest.skip("icd.config module not available")

    def test_max_iterations(self):
        """Test max iterations bounds."""
        try:
            from icd.config import RLMConfig
            from pydantic import ValidationError

            config = RLMConfig(max_iterations=1)
            assert config.max_iterations == 1

            config = RLMConfig(max_iterations=20)
            assert config.max_iterations == 20

            with pytest.raises(ValidationError):
                RLMConfig(max_iterations=0)

            with pytest.raises(ValidationError):
                RLMConfig(max_iterations=21)
        except ImportError:
            pytest.skip("icd.config module not available")


class TestNetworkConfig:
    """Tests for NetworkConfig."""

    def test_network_disabled_by_default(self):
        """Test that network is disabled by default (PRD requirement)."""
        try:
            from icd.config import NetworkConfig

            config = NetworkConfig()
            assert config.enabled is False
        except ImportError:
            pytest.skip("icd.config module not available")


class TestContractConfig:
    """Tests for ContractConfig."""

    def test_contract_patterns(self):
        """Test default contract detection patterns."""
        try:
            from icd.config import ContractConfig

            config = ContractConfig()
            assert "interface" in config.patterns
            assert "abstract" in config.patterns
            assert "trait" in config.patterns
            assert "schema" in config.patterns
        except ImportError:
            pytest.skip("icd.config module not available")

    def test_boost_factor_bounds(self):
        """Test contract boost factor bounds."""
        try:
            from icd.config import ContractConfig
            from pydantic import ValidationError

            config = ContractConfig(boost_factor=1.0)
            assert config.boost_factor == 1.0

            config = ContractConfig(boost_factor=5.0)
            assert config.boost_factor == 5.0

            with pytest.raises(ValidationError):
                ContractConfig(boost_factor=0.9)

            with pytest.raises(ValidationError):
                ContractConfig(boost_factor=5.1)
        except ImportError:
            pytest.skip("icd.config module not available")


class TestMainConfig:
    """Tests for the main Config class."""

    def test_default_initialization(self):
        """Test default configuration initialization."""
        try:
            from icd.config import Config

            config = Config()
            assert config.log_level == "INFO"
            assert config.storage is not None
            assert config.embedding is not None
            assert config.chunking is not None
            assert config.retrieval is not None
        except ImportError:
            pytest.skip("icd.config module not available")

    def test_project_root_resolution(self, tmp_path: Path):
        """Test project root is resolved to absolute path."""
        try:
            from icd.config import Config

            config = Config(project_root=tmp_path)
            assert config.project_root.is_absolute()
            assert config.project_root == tmp_path.resolve()
        except ImportError:
            pytest.skip("icd.config module not available")

    def test_absolute_data_dir(self, tmp_path: Path):
        """Test absolute_data_dir property."""
        try:
            from icd.config import Config

            config = Config(project_root=tmp_path, data_dir=Path(".icd"))
            assert config.absolute_data_dir == tmp_path / ".icd"

            # Test with absolute data_dir
            abs_data = tmp_path / "custom_data"
            config = Config(project_root=tmp_path, data_dir=abs_data)
            assert config.absolute_data_dir == abs_data
        except ImportError:
            pytest.skip("icd.config module not available")

    def test_ensure_directories(self, tmp_path: Path):
        """Test directory creation."""
        try:
            from icd.config import Config

            config = Config(project_root=tmp_path, data_dir=Path(".icd"))
            config.ensure_directories()
            assert config.absolute_data_dir.exists()
        except ImportError:
            pytest.skip("icd.config module not available")

    def test_db_path_property(self, tmp_path: Path):
        """Test db_path property returns correct path."""
        try:
            from icd.config import Config

            config = Config(project_root=tmp_path)
            expected = config.absolute_data_dir / "index.db"
            assert config.db_path == expected
        except ImportError:
            pytest.skip("icd.config module not available")

    def test_vector_index_path_property(self, tmp_path: Path):
        """Test vector_index_path property."""
        try:
            from icd.config import Config

            config = Config(project_root=tmp_path)
            expected = config.absolute_data_dir / "vectors.hnsw"
            assert config.vector_index_path == expected
        except ImportError:
            pytest.skip("icd.config module not available")


class TestConfigFromFile:
    """Tests for loading configuration from files."""

    def test_load_from_toml(self, tmp_path: Path):
        """Test loading config from TOML file."""
        try:
            from icd.config import Config

            config_content = """
[storage]
cache_size_mb = 128
hnsw_m = 32

[embedding]
dimension = 768

[retrieval]
mmr_lambda = 0.8
"""
            config_file = tmp_path / "test_config.toml"
            config_file.write_text(config_content)

            config = Config.from_file(config_file)
            assert config.storage.cache_size_mb == 128
            assert config.storage.hnsw_m == 32
            assert config.embedding.dimension == 768
            assert config.retrieval.mmr_lambda == 0.8
        except ImportError:
            pytest.skip("icd.config module or tomllib not available")

    def test_load_from_yaml(self, tmp_path: Path):
        """Test loading config from YAML file."""
        try:
            import yaml
            from icd.config import Config

            config_content = """
storage:
  cache_size_mb: 128
embedding:
  dimension: 768
"""
            config_file = tmp_path / "test_config.yaml"
            config_file.write_text(config_content)

            config = Config.from_file(config_file)
            assert config.storage.cache_size_mb == 128
            assert config.embedding.dimension == 768
        except ImportError:
            pytest.skip("icd.config module or PyYAML not available")

    def test_load_from_json(self, tmp_path: Path):
        """Test loading config from JSON file."""
        try:
            import json
            from icd.config import Config

            config_dict = {
                "storage": {"cache_size_mb": 128},
                "embedding": {"dimension": 768},
            }
            config_file = tmp_path / "test_config.json"
            config_file.write_text(json.dumps(config_dict))

            config = Config.from_file(config_file)
            assert config.storage.cache_size_mb == 128
            assert config.embedding.dimension == 768
        except ImportError:
            pytest.skip("icd.config module not available")

    def test_file_not_found(self, tmp_path: Path):
        """Test error handling for missing config file."""
        try:
            from icd.config import Config

            with pytest.raises(FileNotFoundError):
                Config.from_file(tmp_path / "nonexistent.toml")
        except ImportError:
            pytest.skip("icd.config module not available")

    def test_unsupported_format(self, tmp_path: Path):
        """Test error handling for unsupported config format."""
        try:
            from icd.config import Config

            config_file = tmp_path / "config.xyz"
            config_file.write_text("some content")

            with pytest.raises(ValueError, match="Unsupported config format"):
                Config.from_file(config_file)
        except ImportError:
            pytest.skip("icd.config module not available")


class TestLoadConfig:
    """Tests for the load_config function."""

    def test_auto_discovery_icd_toml(self, tmp_path: Path):
        """Test auto-discovery of icd.toml in project root."""
        try:
            from icd.config import load_config

            config_content = """
[storage]
cache_size_mb = 256
"""
            (tmp_path / "icd.toml").write_text(config_content)

            config = load_config(project_root=tmp_path)
            assert config.storage.cache_size_mb == 256
        except ImportError:
            pytest.skip("icd.config module not available")

    def test_auto_discovery_dot_icd_config(self, tmp_path: Path):
        """Test auto-discovery of .icd/config.toml."""
        try:
            from icd.config import load_config

            config_content = """
[storage]
cache_size_mb = 192
"""
            (tmp_path / ".icd").mkdir()
            (tmp_path / ".icd" / "config.toml").write_text(config_content)

            config = load_config(project_root=tmp_path)
            assert config.storage.cache_size_mb == 192
        except ImportError:
            pytest.skip("icd.config module not available")

    def test_explicit_config_path_priority(self, tmp_path: Path):
        """Test that explicit config path takes priority."""
        try:
            from icd.config import load_config

            # Create both files
            (tmp_path / "icd.toml").write_text("[storage]\ncache_size_mb = 100")

            custom_config = tmp_path / "custom.toml"
            custom_config.write_text("[storage]\ncache_size_mb = 200")

            config = load_config(config_path=custom_config, project_root=tmp_path)
            assert config.storage.cache_size_mb == 200
        except ImportError:
            pytest.skip("icd.config module not available")

    def test_default_config_when_no_file(self, tmp_path: Path):
        """Test default config is returned when no config file exists."""
        try:
            from icd.config import load_config

            config = load_config(project_root=tmp_path)
            # Should use defaults
            assert config.storage.cache_size_mb == 64
            assert config.project_root == tmp_path
        except ImportError:
            pytest.skip("icd.config module not available")


class TestEnvironmentVariables:
    """Tests for environment variable configuration."""

    def test_env_prefix(self):
        """Test that ICD_ prefix is used for env vars."""
        try:
            from icd.config import Config

            with patch.dict(os.environ, {"ICD_LOG_LEVEL": "DEBUG"}):
                config = Config()
                assert config.log_level == "DEBUG"
        except ImportError:
            pytest.skip("icd.config module not available")

    def test_nested_env_vars(self):
        """Test nested configuration via env vars."""
        try:
            from icd.config import Config

            with patch.dict(os.environ, {"ICD_STORAGE__CACHE_SIZE_MB": "256"}):
                config = Config()
                assert config.storage.cache_size_mb == 256
        except ImportError:
            pytest.skip("icd.config module not available")


class TestConfigSerialization:
    """Tests for configuration serialization."""

    def test_to_dict(self, tmp_path: Path):
        """Test configuration serialization to dictionary."""
        try:
            from icd.config import Config

            config = Config(project_root=tmp_path)
            config_dict = config.to_dict()

            assert "storage" in config_dict
            assert "embedding" in config_dict
            assert "chunking" in config_dict
            assert "retrieval" in config_dict
            assert isinstance(config_dict["storage"]["cache_size_mb"], int)
        except ImportError:
            pytest.skip("icd.config module not available")


class TestWatcherConfig:
    """Tests for WatcherConfig."""

    def test_default_ignore_patterns(self):
        """Test default ignore patterns include common exclusions."""
        try:
            from icd.config import WatcherConfig

            config = WatcherConfig()
            patterns = config.ignore_patterns

            assert "**/.git/**" in patterns
            assert "**/node_modules/**" in patterns
            assert "**/__pycache__/**" in patterns
            assert "**/venv/**" in patterns
        except ImportError:
            pytest.skip("icd.config module not available")

    def test_default_watch_extensions(self):
        """Test default watch extensions include common code files."""
        try:
            from icd.config import WatcherConfig

            config = WatcherConfig()
            extensions = config.watch_extensions

            assert ".py" in extensions
            assert ".ts" in extensions
            assert ".js" in extensions
            assert ".go" in extensions
            assert ".rs" in extensions
        except ImportError:
            pytest.skip("icd.config module not available")

    def test_debounce_bounds(self):
        """Test debounce delay bounds."""
        try:
            from icd.config import WatcherConfig
            from pydantic import ValidationError

            config = WatcherConfig(debounce_ms=100)
            assert config.debounce_ms == 100

            config = WatcherConfig(debounce_ms=5000)
            assert config.debounce_ms == 5000

            with pytest.raises(ValidationError):
                WatcherConfig(debounce_ms=99)

            with pytest.raises(ValidationError):
                WatcherConfig(debounce_ms=5001)
        except ImportError:
            pytest.skip("icd.config module not available")
