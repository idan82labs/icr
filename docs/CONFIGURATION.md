# ICR Configuration Reference

Complete reference for all ICR configuration options, including YAML configuration, environment variables, and performance tuning.

---

## Table of Contents

- [Configuration Overview](#configuration-overview)
- [Full config.yaml Reference](#full-configyaml-reference)
- [Environment Variables](#environment-variables)
- [Ignore Patterns](#ignore-patterns)
- [Performance Tuning](#performance-tuning)
- [Security Settings](#security-settings)
- [Embedding Backend Options](#embedding-backend-options)
- [Per-Project Configuration](#per-project-configuration)

---

## Configuration Overview

### Configuration Hierarchy

ICR loads configuration in the following order (later overrides earlier):

1. **Built-in defaults**: Hard-coded sensible defaults
2. **User config**: `~/.icr/config.yaml`
3. **Project config**: `<repo>/.icr/config.yaml`
4. **Environment variables**: `ICD_*` prefixed variables

### Configuration File Locations

| Location | Purpose |
|----------|---------|
| `~/.icr/config.yaml` | User-level global settings |
| `<repo>/.icr/config.yaml` | Project-specific overrides |
| `<repo>/icd.yaml` | Alternative project config location |

### Creating Configuration

```bash
# Initialize with defaults
icr init

# This creates ~/.icr/config.yaml with commented defaults
```

---

## Full config.yaml Reference

```yaml
# ICR Configuration File
# All settings shown with their default values

icr:
  #############################################################################
  # STORAGE SETTINGS
  #############################################################################
  storage:
    # Path to SQLite database (relative to project or absolute)
    db_path: .icd/index.db

    # Enable WAL mode for better concurrent read performance
    wal_mode: true

    # SQLite cache size in megabytes (8-512)
    cache_size_mb: 64

    # Maximum vectors per repository (1000-10000000)
    max_vectors_per_repo: 250000

    # Vector storage data type: float16 (saves 50% memory) or float32
    # Policy: Store as float16, compute as float32
    vector_dtype: float16

    # HNSW index parameters
    # M: connections per node (4-64), higher = more accurate but slower
    hnsw_m: 16

    # ef_construction: build-time accuracy (50-500), higher = better index
    hnsw_ef_construction: 200

    # ef_search: query-time accuracy (10-500), higher = more accurate
    hnsw_ef_search: 100

  #############################################################################
  # EMBEDDING SETTINGS
  #############################################################################
  embedding:
    # Backend: local_onnx (default), local_transformers, openai, anthropic, custom
    backend: local_onnx

    # Model name (for local backends, this downloads automatically)
    model_name: all-MiniLM-L6-v2

    # Path to local ONNX model file (null = auto-download)
    model_path: null

    # Embedding dimension (must match model output)
    dimension: 384

    # Batch size for embedding generation (1-256)
    batch_size: 32

    # Maximum tokens per chunk for embedding (128-8192)
    max_tokens: 512

    # L2 normalize embeddings (recommended for cosine similarity)
    normalize: true

    # Remote API settings (only used if backend is openai/anthropic/custom)
    # api_key: ${ICR_EMBEDDING_API_KEY}  # Use environment variable
    # api_base_url: https://api.openai.com/v1

  #############################################################################
  # CHUNKING SETTINGS
  #############################################################################
  chunking:
    # Minimum tokens per chunk (50-500)
    min_tokens: 200

    # Target tokens per chunk (200-1000)
    target_tokens: 500

    # Maximum tokens per chunk (500-4000, hard limit)
    max_tokens: 1200

    # Token overlap between adjacent chunks (0-200)
    overlap_tokens: 50

    # Preserve symbol boundaries (functions, classes, methods)
    preserve_symbols: true

  #############################################################################
  # RETRIEVAL SETTINGS
  #############################################################################
  retrieval:
    # Hybrid scoring weights (should sum to ~1.0)
    #
    # score = w_e * embedding_sim + w_b * bm25 + w_r * recency
    #       + w_c * is_contract + w_f * in_focus + w_p * is_pinned

    # Embedding similarity weight (0.0-1.0)
    weight_embedding: 0.4

    # BM25 lexical score weight (0.0-1.0)
    weight_bm25: 0.3

    # Recency boost weight (0.0-1.0)
    weight_recency: 0.1

    # Contract indicator weight (0.0-1.0)
    weight_contract: 0.1

    # Focus scope indicator weight (0.0-1.0)
    weight_focus: 0.05

    # Pinned indicator weight (0.0-1.0)
    weight_pinned: 0.05

    # Recency decay time constant in days (1-365)
    # Half-life ≈ 0.693 * tau_days
    recency_tau_days: 30.0

    # MMR lambda: trade-off between relevance (1.0) and diversity (0.0)
    mmr_lambda: 0.7

    # Initial candidates for re-ranking (10-1000)
    initial_candidates: 100

    # Final results after MMR selection (1-100)
    final_results: 20

    # Temperature for entropy computation (0.1-10.0)
    entropy_temperature: 1.0

  #############################################################################
  # PACK COMPILATION SETTINGS
  #############################################################################
  pack:
    # Default token budget for context packs (1000-128000)
    default_budget_tokens: 8000

    # Maximum allowed token budget (4000-200000)
    max_budget_tokens: 32000

    # Include metadata in pack output
    include_metadata: true

    # Include citation markers in pack
    include_citations: true

  #############################################################################
  # FILE WATCHER SETTINGS
  #############################################################################
  watcher:
    # Debounce delay in milliseconds (100-5000)
    # Groups rapid changes into single update
    debounce_ms: 500

    # Glob patterns to ignore (always excludes these)
    ignore_patterns:
      # Version control
      - "**/.git/**"
      - "**/.hg/**"
      - "**/.svn/**"

      # Dependencies
      - "**/node_modules/**"
      - "**/vendor/**"
      - "**/.venv/**"
      - "**/venv/**"

      # Build outputs
      - "**/dist/**"
      - "**/build/**"
      - "**/target/**"
      - "**/__pycache__/**"
      - "**/*.pyc"
      - "**/*.egg-info/**"

      # IDE/Editor
      - "**/.idea/**"
      - "**/.vscode/**"

      # Test coverage
      - "**/coverage/**"
      - "**/.coverage"
      - "**/.tox/**"

    # File extensions to index (only these are processed)
    watch_extensions:
      - ".py"
      - ".js"
      - ".ts"
      - ".tsx"
      - ".jsx"
      - ".go"
      - ".rs"
      - ".java"
      - ".c"
      - ".cpp"
      - ".h"
      - ".hpp"
      - ".cs"
      - ".rb"
      - ".php"
      - ".swift"
      - ".kt"
      - ".scala"
      - ".md"
      - ".rst"
      - ".txt"
      - ".json"
      - ".yaml"
      - ".yml"
      - ".toml"

    # Maximum file size to index in KB (10-10000)
    max_file_size_kb: 500

  #############################################################################
  # RLM (RECURSIVE LANGUAGE MODEL) SETTINGS
  #############################################################################
  rlm:
    # Enable RLM fallback for high-entropy queries
    enabled: true

    # Entropy threshold to trigger RLM mode (0.0-10.0)
    # Low entropy (< threshold): Pack mode
    # High entropy (>= threshold): RLM mode
    entropy_threshold: 2.5

    # Maximum RLM iterations (1-20)
    max_iterations: 5

    # Token budget per RLM iteration (500-10000)
    budget_per_iteration: 2000

  #############################################################################
  # CONTRACT DETECTION SETTINGS
  #############################################################################
  contract:
    # Enable automatic contract detection
    enabled: true

    # Patterns indicating contract definitions
    patterns:
      - "interface"
      - "abstract"
      - "protocol"
      - "trait"
      - "type.*="
      - "@dataclass"
      - "schema"
      - "model"
      - "struct"
      - "enum"

    # Boost factor for contracts in retrieval (1.0-5.0)
    boost_factor: 1.5

  #############################################################################
  # NETWORK SETTINGS
  #############################################################################
  network:
    # Enable network access (required for remote embedding backends)
    # Default: false (local-only mode)
    enabled: false

    # Base URL for remote API (if using remote backend)
    api_base_url: null

    # API key for remote services (use environment variable)
    # api_key: ${ICR_API_KEY}

    # Request timeout in seconds (5-300)
    timeout_seconds: 30

    # Maximum retry attempts (0-10)
    max_retries: 3

  #############################################################################
  # TELEMETRY SETTINGS
  #############################################################################
  telemetry:
    # Enable local telemetry collection (metrics stored locally only)
    enabled: true

    # Path to metrics database
    metrics_path: .icd/metrics.db

    # Metrics retention period in days (1-365)
    retention_days: 30

  #############################################################################
  # LOGGING SETTINGS
  #############################################################################
  logging:
    # Log level: DEBUG, INFO, WARNING, ERROR
    level: INFO

    # Log format: json, text
    format: text

    # Log file path (null = stdout only)
    file_path: null
```

---

## Environment Variables

All configuration options can be overridden with environment variables using the `ICD_` prefix and `__` for nesting.

### Pattern

```
ICD_<SECTION>__<OPTION>=<value>
```

### Examples

```bash
# Override embedding backend
export ICD_EMBEDDING__BACKEND=openai
export ICD_EMBEDDING__API_KEY=sk-your-api-key

# Override token budget
export ICD_PACK__DEFAULT_BUDGET_TOKENS=10000

# Override retrieval weights
export ICD_RETRIEVAL__WEIGHT_EMBEDDING=0.5
export ICD_RETRIEVAL__WEIGHT_BM25=0.3

# Enable network for remote embeddings
export ICD_NETWORK__ENABLED=true

# Change log level
export ICD_LOGGING__LEVEL=DEBUG
```

### Precedence

Environment variables take highest precedence:

1. Built-in defaults (lowest)
2. User config file
3. Project config file
4. Environment variables (highest)

---

## Ignore Patterns

### Default Ignore Patterns

ICR always ignores these patterns by default:

```yaml
# Security-sensitive files
- ".env"
- ".env.*"
- "*.pem"
- "*.key"
- "*.p12"
- "*.pfx"
- "id_rsa"
- "id_dsa"
- "id_ecdsa"
- "id_ed25519"
- "credentials.json"
- "service-account*.json"
- ".aws/**"
- ".ssh/**"
- "**/secrets/**"
- "**/credentials/**"
- ".netrc"
- ".npmrc"
- ".pypirc"
- "*.keystore"

# Version control
- ".git/**"
- ".hg/**"
- ".svn/**"

# Large binary files
- "*.zip"
- "*.tar*"
- "*.jar"
- "*.war"
- "*.exe"
- "*.dll"
- "*.so"
- "*.dylib"

# Dependencies (large, reconstructable)
- "node_modules/**"
- "vendor/**"
- ".venv/**"
- "venv/**"

# Build outputs
- "dist/**"
- "build/**"
- "target/**"
- "__pycache__/**"
- "*.pyc"
- "*.egg-info/**"
```

### Adding Custom Ignore Patterns

In your config file:

```yaml
watcher:
  ignore_patterns:
    # Keep defaults by not overriding entirely
    # Add project-specific patterns
    - "**/generated/**"
    - "**/fixtures/**"
    - "**/*.generated.ts"
```

### Per-Project Ignore

Create `.icr/ignore` in project root:

```gitignore
# This file works like .gitignore
# These patterns are added to ignore_patterns

# Large data files
data/raw/**
*.csv
*.parquet

# Generated code
*_pb2.py
*.generated.*
```

---

## Performance Tuning

### Scalability Tiers

#### Tier 1: Standard (Guaranteed)

For projects up to 10,000 files:

```yaml
storage:
  max_vectors_per_repo: 100000
  cache_size_mb: 64
  hnsw_m: 16
  hnsw_ef_construction: 200
  hnsw_ef_search: 100

embedding:
  batch_size: 32
```

**Resource Requirements:**
- RAM: < 2GB
- Disk: < 1GB index size
- Cold start: < 5 seconds

#### Tier 2: Large (Stretch)

For projects up to 100,000 files:

```yaml
storage:
  max_vectors_per_repo: 1000000
  cache_size_mb: 256
  hnsw_m: 24
  hnsw_ef_construction: 400
  hnsw_ef_search: 150

embedding:
  batch_size: 64

chunking:
  # More aggressive chunking for large repos
  target_tokens: 400
  max_tokens: 800
```

**Resource Requirements:**
- RAM: < 10GB
- Disk: < 5GB index size
- Cold start: < 10 seconds
- Hardware: SSD required, 16GB+ system RAM, 4+ CPU cores

### Performance Targets

| Operation | P50 Target | P95 Target | Notes |
|-----------|------------|------------|-------|
| ANN lookup (given embedding) | 15ms | 40ms | HNSW float16 |
| Query embedding (local) | 20ms | 50ms | ONNX optimized |
| Query embedding (remote) | 80ms | 200ms | Network dependent |
| End-to-end semantic | 40ms | 100ms | Local embedding |
| Hybrid search (full) | 70ms | 150ms | All components |
| Pack compilation | 200ms | 500ms | Knapsack + format |
| RLM plan generation | 300ms | 800ms | Template-based |
| Map-reduce aggregation | 2s | 8s | Hard abort at 20s |

### Tuning for Speed

```yaml
# Prioritize speed over accuracy
storage:
  hnsw_ef_search: 50  # Lower = faster, less accurate

retrieval:
  initial_candidates: 50  # Fewer candidates to rank
  final_results: 10  # Fewer final results

embedding:
  batch_size: 64  # Larger batches for throughput

pack:
  default_budget_tokens: 4000  # Smaller packs compile faster
```

### Tuning for Accuracy

```yaml
# Prioritize accuracy over speed
storage:
  hnsw_ef_search: 200  # Higher = slower, more accurate

retrieval:
  initial_candidates: 200  # More candidates to consider
  final_results: 30  # More diverse results

embedding:
  batch_size: 16  # Smaller batches, more consistent

pack:
  default_budget_tokens: 12000  # Larger packs, more context
```

### Memory Optimization

```yaml
# For memory-constrained environments
storage:
  cache_size_mb: 32  # Smaller cache
  vector_dtype: float16  # Already default, 50% savings
  max_vectors_per_repo: 100000  # Limit vectors

embedding:
  batch_size: 16  # Smaller batches use less memory

watcher:
  max_file_size_kb: 200  # Skip larger files
```

---

## Security Settings

### Network Security

```yaml
# Strict local-only mode (default)
network:
  enabled: false  # No network access whatsoever

# If you need remote embeddings
network:
  enabled: true
  api_base_url: https://api.openai.com/v1
  # Never put API keys in config files!
  # Use environment variables: ICD_NETWORK__API_KEY
  timeout_seconds: 30
  max_retries: 3
```

### Secret Detection

ICR automatically skips files matching these patterns:

```yaml
# Built-in (cannot be disabled)
security:
  always_ignore:
    - ".env"
    - ".env.*"
    - "*.pem"
    - "*.key"
    - "id_rsa*"
    - "credentials.json"
    - ".aws/**"
    - ".ssh/**"
```

### Custom Security Patterns

```yaml
# Add your own sensitive patterns
watcher:
  ignore_patterns:
    - "**/secrets.yaml"
    - "**/*.secret"
    - "**/api_keys/**"
```

### File Size Limits

```yaml
watcher:
  # Prevent indexing large files that might contain dumps
  max_file_size_kb: 500

  # Be more restrictive for sensitive projects
  # max_file_size_kb: 100
```

---

## Embedding Backend Options

### Local ONNX (Default)

Fastest local option, uses ONNX Runtime for CPU inference.

```yaml
embedding:
  backend: local_onnx
  model_name: all-MiniLM-L6-v2
  dimension: 384
```

**Supported Models:**
- `all-MiniLM-L6-v2` (384 dim, default)
- `all-MiniLM-L12-v2` (384 dim, slightly better)
- `all-mpnet-base-v2` (768 dim, best quality)

### Local Transformers

Uses HuggingFace Transformers, slower but more flexible.

```yaml
embedding:
  backend: local_transformers
  model_name: sentence-transformers/all-MiniLM-L6-v2
  dimension: 384
```

### OpenAI

Requires network access and API key.

```yaml
embedding:
  backend: openai
  model_name: text-embedding-3-small
  dimension: 1536

network:
  enabled: true
  api_base_url: https://api.openai.com/v1
```

Set API key via environment:
```bash
export ICD_EMBEDDING__API_KEY=sk-your-key
```

### Anthropic

(When available)

```yaml
embedding:
  backend: anthropic
  model_name: voyage-code-2  # Example
  dimension: 1024

network:
  enabled: true
```

### Custom Backend

For self-hosted or other providers:

```yaml
embedding:
  backend: custom
  model_name: your-model
  dimension: 768

network:
  enabled: true
  api_base_url: https://your-embedding-server.com/v1
```

---

## Per-Project Configuration

### Creating Project Config

```bash
cd /path/to/project
mkdir -p .icr
cat > .icr/config.yaml << 'EOF'
icr:
  # Project-specific overrides
  pack:
    default_budget_tokens: 6000

  watcher:
    ignore_patterns:
      - "**/generated/**"
      - "**/migrations/**"

  retrieval:
    # Boost contracts more for this API-heavy project
    weight_contract: 0.2
EOF
```

### Common Project Configurations

#### Python Project

```yaml
icr:
  watcher:
    watch_extensions:
      - ".py"
      - ".pyi"
      - ".md"
      - ".rst"
      - ".yaml"
      - ".toml"
    ignore_patterns:
      - "**/__pycache__/**"
      - "**/.pytest_cache/**"
      - "**/htmlcov/**"
```

#### TypeScript/JavaScript Project

```yaml
icr:
  watcher:
    watch_extensions:
      - ".ts"
      - ".tsx"
      - ".js"
      - ".jsx"
      - ".json"
      - ".md"
    ignore_patterns:
      - "**/node_modules/**"
      - "**/dist/**"
      - "**/.next/**"
      - "**/coverage/**"
```

#### Monorepo

```yaml
icr:
  # Higher limits for larger codebase
  storage:
    max_vectors_per_repo: 500000

  pack:
    default_budget_tokens: 10000

  watcher:
    ignore_patterns:
      - "**/node_modules/**"
      - "**/dist/**"
      - "**/build/**"
      # Ignore other packages when working in one
      # - "packages/other-package/**"
```

---

## Next Steps

- [TROUBLESHOOTING.md](TROUBLESHOOTING.md): Common issues
- [API_REFERENCE.md](API_REFERENCE.md): Tool documentation
- [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md): Contributing
