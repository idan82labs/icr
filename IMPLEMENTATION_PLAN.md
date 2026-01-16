# ICR Implementation Plan - Master Document

**Version:** 1.0
**Date:** 2026-01-16
**Status:** Implementation Ready
**Classification:** Research-Grade Implementation

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Critical Clarifications and Corrections](#2-critical-clarifications-and-corrections)
3. [Architecture Overview](#3-architecture-overview)
4. [Performance Targets (Revised)](#4-performance-targets-revised)
5. [Scalability Constraints](#5-scalability-constraints)
6. [Implementation Phases](#6-implementation-phases)
7. [Component Specifications](#7-component-specifications)
8. [Development Checklist](#8-development-checklist)
9. [Testing Strategy](#9-testing-strategy)
10. [Deployment and Installation](#10-deployment-and-installation)

---

## 1. Executive Summary

ICR (Infinite Context Runtime) implements an **RLM-inspired runtime** that provides an "unlimited context feel" in Claude Code by treating long context as an external environment that the model inspects via bounded tools.

**Key Insight:** We are NOT implementing a new model. We are implementing an RLM-style runtime/REPL-like interaction pattern around an existing model. The long context is treated as an external environment that the model inspects via bounded tools.

### Core Deliverables

1. **icd (daemon)** - Data plane for indexing, storage, and retrieval
2. **ic-mcp (MCP server)** - Tool plane exposing safe, bounded operations
3. **ic-claude (Claude Code plugin)** - Behavior plane for hooks and commands

### Research Foundation

This implementation is based on the Recursive Language Models (RLM) research paradigm, which proposes treating prompts as external environments rather than forcing all information into model context. Key principles:

- Symbolic inspection over wholesale context inclusion
- Bounded, typed operations with deterministic behavior
- Variance-controlled cost/latency budgets
- Non-generative aggregation for trustworthy results

---

## 2. Critical Clarifications and Corrections

### 2.1 Performance Target Decomposition

The PRD specifies aggregate latency targets. For implementation clarity, we decompose these into measurable sub-components:

| Operation | Component | P50 Target | P95 Target | Notes |
|-----------|-----------|------------|------------|-------|
| **Semantic Search** | Query embedding compute (local) | 20ms | 50ms | Using ONNX-optimized model |
| **Semantic Search** | Query embedding compute (remote) | 80ms | 200ms | Network-dependent |
| **Semantic Search** | ANN lookup (given embedding) | 15ms | 40ms | HNSW float16 storage |
| **Semantic Search** | End-to-end (local embedding) | 40ms | 100ms | Sum of above |
| **Hybrid Search** | Lexical (BM25/FTS5) | 10ms | 30ms | SQLite FTS5 |
| **Hybrid Search** | Merge + rerank | 10ms | 25ms | MMR diversity |
| **Hybrid Search** | End-to-end (local embedding) | 70ms | 150ms | All components |
| **Pack Mode** | Total compilation | 200ms | 500ms | Knapsack + formatting |
| **RLM Plan** | Generation | 300ms | 800ms | Template-based |
| **Map-Reduce** | Aggregation | 2s | 8s | Hard abort at 20s |

**Critical:** These targets assume local embedding generation. Remote embedding adds 60-150ms latency variance.

### 2.2 Vector Storage Policy

**Policy:** Float16 storage, Float32 compute

Implementation details:
- Store vectors in float16 format (2× storage savings)
- Upcast to float32 for distance computation at query time
- Optional: maintain float32 cache for frequently-accessed "hot" vectors
- Do NOT expect float16 end-to-end computation

Code implication:
```python
# Storage
vectors_f16 = vectors.astype(np.float16)

# Query-time computation
def compute_distance(query_f32: np.ndarray, stored_f16: np.ndarray) -> float:
    stored_f32 = stored_f16.astype(np.float32)
    return np.dot(query_f32, stored_f32)
```

### 2.3 Scalability Tiers

The PRD targets are achievable but require specific implementation constraints:

#### Tier 1 (Guaranteed - Default Configuration)
| Metric | Target | Constraints |
|--------|--------|-------------|
| File count | 10,000 | Standard projects |
| Chunk count | 100,000 | Symbol-level chunking |
| RAM usage | < 2GB | Lazy loading enabled |
| Index size | < 1GB | Float16 + dedup |
| Cold start | < 5s | Incremental loading |

#### Tier 2 (Stretch - With Optimizations)
| Metric | Target | Required Optimizations |
|--------|--------|------------------------|
| File count | 100,000+ | Aggressive dedup, lazy loading |
| Chunk count | 1,000,000+ | Memory-mapped HNSW, stable chunk IDs |
| RAM usage | < 10GB | Incremental HNSW, metadata packing |
| Index size | < 5GB | Float16 storage, content-hash dedup |
| Cold start | < 10s | Background indexing, partial loading |

**Hardware assumptions for Tier 2:**
- SSD storage (not HDD)
- 16GB+ system RAM
- Multi-core CPU (4+ cores)

### 2.4 Embedding Backend Abstraction

The system MUST support pluggable embedding backends with local-first default:

```
EmbeddingBackend (abstract)
├── LocalONNXBackend (default)
│   └── Uses sentence-transformers/all-MiniLM-L6-v2 or similar
├── LocalTransformersBackend
│   └── HuggingFace transformers with local model
└── RemoteBackend (explicit opt-in only)
    ├── OpenAIBackend
    ├── CohereBackend
    └── AnthropicBackend (if/when available)
```

**Policy:** Remote embedding backends require explicit user opt-in via configuration. Default behavior MUST be local-only (no network egress).

---

## 3. Architecture Overview

### 3.1 System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                     Claude Code Session                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    User Prompt                           │    │
│  └────────────────────────┬────────────────────────────────┘    │
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              UserPromptSubmit Hook                       │    │
│  │  • Invokes ic-hook-userpromptsubmit                      │    │
│  │  • Returns additionalContext (pack/RLM header)           │    │
│  └────────────────────────┬────────────────────────────────┘    │
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                ic-mcp (MCP Server)                       │    │
│  │  • memory_pack, env_search, project_impact, etc.         │    │
│  │  • Local stdio transport                                  │    │
│  └────────────────────────┬────────────────────────────────┘    │
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   icd (Daemon)                           │    │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐        │    │
│  │  │ Indexer │ │ Storage │ │Retrieval│ │ Memory  │        │    │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘        │    │
│  └────────────────────────┬────────────────────────────────┘    │
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │            Context Environment (E)                       │    │
│  │  • ~/.icr/repos/<repo_id>/                               │    │
│  │  • SQLite + FTS5 + HNSW vectors + contracts + memory    │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Data Flow

1. **Ingestion Flow:**
   ```
   File System Changes → Watcher → Chunker → Embedder → Indexer → Storage
   ```

2. **Query Flow:**
   ```
   User Prompt → Hook → Pack Request → Hybrid Retrieval → MMR Diversity → Knapsack → Pack
   ```

3. **Memory Flow:**
   ```
   Claude Response → Stop Hook → Ledger Parser → Memory Store → Prior Update
   ```

### 3.3 RLM-Inspired Runtime Model

```
┌────────────────────────────────────────────────────────────────┐
│                    RLM-Lite Runtime                             │
│                                                                 │
│   ┌──────────┐     ┌──────────┐     ┌──────────┐               │
│   │ rlm_plan │ ──► │env_search│ ──► │ env_peek │               │
│   └──────────┘     └──────────┘     └──────────┘               │
│        │                │                │                      │
│        ▼                ▼                ▼                      │
│   ┌──────────────────────────────────────────────────┐         │
│   │              Stop Conditions                      │         │
│   │  • Max steps: 12                                  │         │
│   │  • Max peek lines: 1200                          │         │
│   │  • Max candidates: 50                            │         │
│   │  • Wall clock: 8s (pack+plan), 20s (map-reduce)  │         │
│   └──────────────────────────────────────────────────┘         │
│        │                                                        │
│        ▼                                                        │
│   ┌──────────────────────────────────────────────────┐         │
│   │  Fallback: If exceeded → Pack mode + warning     │         │
│   └──────────────────────────────────────────────────┘         │
└────────────────────────────────────────────────────────────────┘
```

---

## 4. Performance Targets (Revised)

### 4.1 Latency Budget Breakdown

For a typical `memory_pack` call with mode=auto:

```
Total Budget: 300ms (P50), 1.5s (P95)

├── Query Analysis: 5ms
├── Embedding Computation (local): 20ms
├── Hybrid Retrieval
│   ├── ANN Lookup: 15ms
│   ├── FTS5 Query: 10ms
│   └── Score Merge: 5ms
├── MMR Diversity Selection: 10ms
├── Gating Decision: 5ms
├── Knapsack Packing: 20ms
├── Markdown Formatting: 10ms
└── I/O + Overhead: ~50ms

Total Estimated P50: ~150ms (well within budget)
P95 headroom for: disk I/O variance, large result sets, cold cache
```

### 4.2 Embedding Latency Awareness

| Scenario | Embedding Latency | Notes |
|----------|-------------------|-------|
| Local ONNX (CPU) | 15-30ms | Default, recommended |
| Local ONNX (GPU) | 5-10ms | If CUDA available |
| Local Transformers | 30-80ms | Fallback option |
| Remote API | 80-300ms | Explicit opt-in only |

**Implementation:** Expose embedding latency as a separate telemetry metric. SLO violations due to embedding latency should be distinguished from system bugs.

---

## 5. Scalability Constraints

### 5.1 Chunking Discipline

To achieve scalability targets, chunking MUST follow these rules:

1. **Symbol-level, not line-level:** Chunk by functions, classes, methods - not individual lines
2. **Semantic boundaries:** Respect logical boundaries (docstrings attached to functions)
3. **Stable chunk IDs:** Content-hash based IDs for deduplication
4. **Size bounds:** 200-800 tokens (soft), 1200 tokens (hard max)

### 5.2 Deduplication Strategy

```python
class ChunkDeduplicator:
    def compute_chunk_id(self, content: str, metadata: dict) -> str:
        """
        Stable chunk ID based on:
        1. Content hash (primary)
        2. Structural path (file + symbol hierarchy)
        """
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        struct_path = f"{metadata['file']}:{metadata['symbol_path']}"
        return f"{content_hash}:{hashlib.md5(struct_path.encode()).hexdigest()[:8]}"

    def should_update(self, old_id: str, new_content: str) -> bool:
        """Only re-embed if content actually changed."""
        new_hash = hashlib.sha256(new_content.encode()).hexdigest()[:16]
        old_hash = old_id.split(':')[0]
        return new_hash != old_hash
```

### 5.3 Memory Management

```python
class IndexManager:
    """Lazy loading and memory-mapped access for large indexes."""

    def __init__(self, repo_path: Path):
        self.hnsw_index = None  # Lazy loaded
        self.mmap_vectors = None  # Memory-mapped

    def ensure_loaded(self, tier: str = "hot"):
        """
        Load index tiers on demand:
        - hot: frequently accessed vectors (in-memory)
        - warm: recent vectors (memory-mapped)
        - cold: archived vectors (disk-only, load on demand)
        """
        pass
```

---

## 6. Implementation Phases

### Phase 0: Foundation (Weeks 1-2)

**Goal:** Core infrastructure and basic functionality

| Ticket | Description | Priority |
|--------|-------------|----------|
| P0-01 | Define repo_id computation + storage layout | Critical |
| P0-02 | Create icr.sqlite schema (files, chunks, metadata) | Critical |
| P0-03 | Implement file watcher with debouncing | Critical |
| P0-04 | Implement symbol-based chunker (tree-sitter) | Critical |
| P0-05 | Implement lexical index (FTS5) + BM25 | Critical |
| P0-06 | Implement embedding backend abstraction | Critical |
| P0-07 | Implement local ONNX embedding backend | Critical |
| P0-08 | Implement float16 vector storage | Critical |
| P0-09 | Implement HNSW index wrapper | Critical |
| P0-10 | Basic hybrid retrieval (no MMR yet) | Critical |

**Deliverable:** Can index a repository and perform basic hybrid search.

### Phase 1: Memory Compiler (Weeks 3-4)

**Goal:** Pack mode fully functional

| Ticket | Description | Priority |
|--------|-------------|----------|
| P1-01 | Implement MMR diversity selection | Critical |
| P1-02 | Implement knapsack pack compiler | Critical |
| P1-03 | Implement pack markdown formatter | Critical |
| P1-04 | Implement project_map tool | High |
| P1-05 | Implement project_symbol_search tool | High |
| P1-06 | Implement memory_pack tool (mode=pack) | Critical |
| P1-07 | Implement contract detector (heuristics) | High |
| P1-08 | Implement contract index | High |
| P1-09 | Implement project_impact (basic) | High |
| P1-10 | Implement retrieval entropy computation | High |

**Deliverable:** `memory_pack(mode=pack)` returns useful context packs.

### Phase 2: RLM-Lite (Weeks 5-6)

**Goal:** Bounded inspection and aggregation

| Ticket | Description | Priority |
|--------|-------------|----------|
| P2-01 | Implement env_search tool | Critical |
| P2-02 | Implement env_peek tool | Critical |
| P2-03 | Implement env_slice tool | High |
| P2-04 | Implement env_aggregate tool | High |
| P2-05 | Implement rlm_plan tool | Critical |
| P2-06 | Implement rlm_map_reduce tool | High |
| P2-07 | Implement stop condition enforcement | Critical |
| P2-08 | Implement variance-aware throttling | High |
| P2-09 | Implement mode gating (entropy-based) | Critical |
| P2-10 | Implement Beta prior tracking | High |

**Deliverable:** Full RLM-lite loop with bounded execution.

### Phase 3: Integration (Weeks 7-8)

**Goal:** Full Claude Code integration

| Ticket | Description | Priority |
|--------|-------------|----------|
| P3-01 | Implement UserPromptSubmit hook | Critical |
| P3-02 | Implement Stop hook (ledger parsing) | Critical |
| P3-03 | Implement PreCompact hook | High |
| P3-04 | Implement /ic pack command | Critical |
| P3-05 | Implement /ic search command | Critical |
| P3-06 | Implement /ic impact command | High |
| P3-07 | Implement installer/deployer | Critical |
| P3-08 | Implement config layer (ignore patterns) | Critical |
| P3-09 | Implement policy layer (no-network toggle) | High |
| P3-10 | Integration testing | Critical |

**Deliverable:** Fully functional Claude Code plugin.

### Phase 4: Polish (Week 9+)

**Goal:** Production readiness

| Ticket | Description | Priority |
|--------|-------------|----------|
| P4-01 | Implement local telemetry/observability | High |
| P4-02 | Performance profiling and optimization | High |
| P4-03 | Error handling hardening | High |
| P4-04 | Documentation completion | High |
| P4-05 | Acceptance test suite | Critical |
| P4-06 | Example repository testing | High |

---

## 7. Component Specifications

### 7.1 icd (Daemon) Specification

```
icd/
├── src/
│   ├── __init__.py
│   ├── main.py                 # Entry point
│   ├── config.py               # Configuration management
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── sqlite_store.py     # SQLite + FTS5
│   │   ├── vector_store.py     # HNSW + float16
│   │   ├── contract_store.py   # Contract index
│   │   └── memory_store.py     # Derived memory
│   ├── indexing/
│   │   ├── __init__.py
│   │   ├── watcher.py          # File system watcher
│   │   ├── chunker.py          # Symbol-based chunking
│   │   ├── embedder.py         # Embedding backend
│   │   └── contract_detector.py# Contract detection
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── hybrid.py           # Hybrid retrieval
│   │   ├── mmr.py              # MMR diversity
│   │   └── entropy.py          # Retrieval entropy
│   ├── pack/
│   │   ├── __init__.py
│   │   ├── compiler.py         # Knapsack packer
│   │   ├── formatter.py        # Markdown formatter
│   │   └── gating.py           # Mode selection
│   ├── rlm/
│   │   ├── __init__.py
│   │   ├── planner.py          # RLM plan generation
│   │   ├── aggregator.py       # Non-generative aggregation
│   │   └── budget.py           # Stop conditions
│   └── metrics/
│       ├── __init__.py
│       ├── ewr.py              # Exploration waste ratio
│       ├── imr.py              # Impact miss rate
│       └── telemetry.py        # Local telemetry
├── pyproject.toml
└── tests/
```

### 7.2 ic-mcp (MCP Server) Specification

```
ic-mcp/
├── src/
│   ├── __init__.py
│   ├── server.py               # MCP server main
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── memory.py           # memory_* tools
│   │   ├── env.py              # env_* tools
│   │   ├── project.py          # project_* tools
│   │   ├── rlm.py              # rlm_* tools
│   │   └── admin.py            # admin_* tools
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── validation.py       # JSON Schema validation
│   └── transport/
│       ├── __init__.py
│       └── stdio.py            # Stdio transport
├── pyproject.toml
└── tests/
```

### 7.3 ic-claude (Plugin) Specification

```
ic-claude-plugin/
├── plugin.json                  # Plugin manifest
├── commands/
│   └── ic.md                    # /ic command definition
├── hooks/
│   └── hooks.json               # Hook configuration
├── scripts/
│   ├── ic-hook-userpromptsubmit.py
│   ├── ic-hook-stop.py
│   ├── ic-hook-precompact.py
│   └── ic-cli.py                # CLI fallback
├── installer/
│   ├── install.py               # Installation script
│   └── uninstall.py             # Uninstallation script
└── tests/
```

---

## 8. Development Checklist

### 8.1 Critical Items (Must Have Before Lock)

- [ ] **Installer/Deployer for Hooks**
  - User-level settings first
  - Project-level fallback
  - Plugin-level optional

- [ ] **Config + Policy Layer**
  - Ignore patterns (`.env`, secrets, keys)
  - Max file sizes
  - "No network by default" policy toggle

- [ ] **Embedding Backend Abstraction**
  - Local embeddings default
  - Optional remote via explicit opt-in

- [ ] **Observability/Telemetry (Local-Only)**
  - Per-tool timing
  - EWR/IMR proxy counters
  - Gating reason codes
  - Entropy values
  - Budget usage

### 8.2 Configuration Schema

```yaml
# ~/.icr/config.yaml
icr:
  # Storage settings
  storage:
    root: ~/.icr
    max_vectors_per_repo: 250000
    vector_dtype: float16

  # Embedding settings
  embedding:
    backend: local-onnx  # local-onnx | local-transformers | remote-*
    model: all-MiniLM-L6-v2
    # remote_* settings only used if backend is remote-*
    # remote_api_key: <env:ICR_EMBEDDING_API_KEY>
    # remote_endpoint: <url>

  # Security settings
  security:
    ignore_patterns:
      - ".env"
      - "**/secrets/**"
      - "id_rsa"
      - "*.pem"
      - "*.key"
      - ".aws/**"
      - ".ssh/**"
      - "*.p12"
      - "credentials.json"
    max_file_size_kb: 1024
    network_enabled: false  # Explicit opt-in required

  # Performance settings
  performance:
    max_pack_tokens: 8000
    hard_max_tokens: 25000
    rlm_max_steps: 12
    rlm_max_peek_lines: 1200
    rlm_max_candidates: 50
    pack_timeout_ms: 8000
    mapreduce_timeout_ms: 20000

  # Telemetry (local only)
  telemetry:
    enabled: true
    retention_days: 30
```

### 8.3 Default Ignore Patterns

```python
DEFAULT_IGNORE_PATTERNS = [
    # Secrets and credentials
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "credentials.json",
    "service-account*.json",

    # Directories
    ".aws/**",
    ".ssh/**",
    "**/secrets/**",
    "**/credentials/**",

    # Common secret files
    ".netrc",
    ".npmrc",
    ".pypirc",
    "*.keystore",

    # Git internals
    ".git/**",

    # Large binary files (by extension)
    "*.zip",
    "*.tar*",
    "*.jar",
    "*.war",
    "*.exe",
    "*.dll",
    "*.so",
    "*.dylib",

    # Node modules (large, reconstructable)
    "node_modules/**",

    # Build outputs
    "dist/**",
    "build/**",
    "target/**",
    "__pycache__/**",
    "*.pyc",
]
```

---

## 9. Testing Strategy

### 9.1 Unit Tests

Each component must have >80% test coverage:

```python
# Example: test_hybrid_retrieval.py
class TestHybridRetrieval:
    def test_semantic_only(self):
        """Test pure semantic search."""
        pass

    def test_lexical_only(self):
        """Test pure BM25 search."""
        pass

    def test_hybrid_merge(self):
        """Test score merging with weights."""
        pass

    def test_contract_boost(self):
        """Test contract detection boost."""
        pass

    def test_recency_decay(self):
        """Test time-based score decay."""
        pass
```

### 9.2 Integration Tests

```python
# Example: test_pack_flow.py
class TestPackFlow:
    def test_full_pack_generation(self):
        """Index repo → query → pack generation."""
        pass

    def test_hook_injection(self):
        """Simulate UserPromptSubmit hook."""
        pass

    def test_stop_hook_ledger_parsing(self):
        """Verify ledger extraction from response."""
        pass
```

### 9.3 Acceptance Tests (from PRD)

1. **Exploration Waste Test**
   - Prompt: "Where is auth token validated?"
   - Pass: ≤1 manual grep; uses project_symbol_search; returns correct location

2. **Impact Correctness Test**
   - Change: rename endpoint field
   - Pass: project_impact returns ≥1 FE usage candidate

3. **RLM Gating Test**
   - Prompt: "Audit all usages of endpoint X"
   - Pass: memory_pack(auto) → RLM mode; bounded tool usage

4. **Compaction Survival Test**
   - After /compact
   - Pass: pinned invariants persist

5. **Hook Fallback Test**
   - Disable hooks
   - Pass: /ic commands still work

---

## 10. Deployment and Installation

### 10.1 Installation Flow

```bash
# 1. Install ICR package
pip install icr  # or: pipx install icr

# 2. Initialize ICR for current user
icr init

# 3. Configure Claude Code integration
icr configure claude-code

# 4. Verify installation
icr doctor
```

### 10.2 Claude Code Configuration

The installer will create/update:

1. **User-level hooks** (`~/.claude/settings.json`):
```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "matcher": ".*",
        "hooks": [
          {
            "type": "command",
            "command": "icr hook prompt-submit"
          }
        ]
      }
    ],
    "Stop": [
      {
        "matcher": ".*",
        "hooks": [
          {
            "type": "command",
            "command": "icr hook stop"
          }
        ]
      }
    ]
  }
}
```

2. **MCP server configuration** (`~/.claude.json`):
```json
{
  "mcpServers": {
    "icr": {
      "command": "icr",
      "args": ["mcp-serve"],
      "env": {}
    }
  }
}
```

### 10.3 Health Check

```bash
$ icr doctor
ICR Health Check
================

✓ Configuration found at ~/.icr/config.yaml
✓ SQLite database accessible
✓ Vector index initialized
✓ Embedding backend: local-onnx (all-MiniLM-L6-v2)
✓ Claude Code hooks configured (user-level)
✓ MCP server configuration found

Status: HEALTHY
```

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **E** | Context Environment - local persistent state |
| **Pack** | Bounded markdown payload with citations |
| **RLM-lite** | Bounded inspect/decompose/aggregate loop |
| **EWR** | Exploration Waste Ratio |
| **IMR** | Impact Miss Rate |
| **MMR** | Maximal Marginal Relevance (diversity algorithm) |
| **HNSW** | Hierarchical Navigable Small World (ANN algorithm) |
| **FTS5** | SQLite Full-Text Search extension |

---

## Appendix B: References

1. Recursive Language Models (RLM) - arXiv paper on "prompt as environment" paradigm
2. Claude Code Documentation - hooks, plugins, MCP integration
3. MCP Specification - tool protocols and sampling capabilities
4. HNSW Paper - Malkov & Yashunin, 2018
5. MMR Paper - Carbonell & Goldstein, 1998

---

*This document serves as the authoritative implementation reference for ICR. All implementation decisions should align with this specification.*
