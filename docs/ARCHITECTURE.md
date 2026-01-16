# ICR Architecture

This document provides a comprehensive overview of the ICR (Infinite Context Runtime) system architecture, including component responsibilities, data flows, and design decisions.

---

## Table of Contents

- [Critical Clarification](#critical-clarification)
- [System Overview](#system-overview)
- [Component Architecture](#component-architecture)
- [Data Flow Diagrams](#data-flow-diagrams)
- [Storage Layout](#storage-layout)
- [Index Types](#index-types)
- [RLM Runtime Model](#rlm-runtime-model)
- [Design Decisions](#design-decisions)

---

## Critical Clarification

> **ICR implements an RLM-inspired runtime: the long context is treated as an external environment that the model inspects via bounded tools.**

We are **NOT** implementing a new model. We are implementing an RLM-style runtime/REPL-like interaction pattern around an existing model. The key insight is that long context should be treated as an external environment that the model can inspect through bounded, typed operations rather than being forced entirely into the context window.

---

## System Overview

### High-Level Architecture

```
+------------------------------------------------------------------+
|                      Claude Code Session                          |
|                                                                   |
|  +------------------------------------------------------------+  |
|  |                      User Prompt                            |  |
|  +---------------------------+--------------------------------+  |
|                              |                                    |
|                              v                                    |
|  +------------------------------------------------------------+  |
|  |               UserPromptSubmit Hook                         |  |
|  |   - Intercepts user prompts before processing               |  |
|  |   - Invokes ic-hook-userpromptsubmit                        |  |
|  |   - Returns additionalContext (pack/RLM header)             |  |
|  +---------------------------+--------------------------------+  |
|                              |                                    |
|                              v                                    |
|  +------------------------------------------------------------+  |
|  |                 ic-mcp (MCP Server)                         |  |
|  |   +----------+  +----------+  +----------+  +----------+    |  |
|  |   | memory_* |  |  env_*   |  |project_* |  |  rlm_*   |    |  |
|  |   |  tools   |  |  tools   |  |  tools   |  |  tools   |    |  |
|  |   +----------+  +----------+  +----------+  +----------+    |  |
|  |                Local stdio transport                        |  |
|  +---------------------------+--------------------------------+  |
|                              |                                    |
|                              v                                    |
|  +------------------------------------------------------------+  |
|  |                    icd (Daemon)                             |  |
|  |                                                             |  |
|  |   +-------------+  +-------------+  +-------------+         |  |
|  |   |   Indexer   |  |   Storage   |  |  Retrieval  |         |  |
|  |   |  - Watcher  |  |  - SQLite   |  |  - Hybrid   |         |  |
|  |   |  - Chunker  |  |  - FTS5     |  |  - MMR      |         |  |
|  |   |  - Embedder |  |  - HNSW     |  |  - Entropy  |         |  |
|  |   +-------------+  +-------------+  +-------------+         |  |
|  |                                                             |  |
|  |   +-------------+  +-------------+  +-------------+         |  |
|  |   |    Pack     |  |     RLM     |  |   Memory    |         |  |
|  |   |  - Knapsack |  |  - Planner  |  |  - Derived  |         |  |
|  |   |  - Format   |  |  - Budget   |  |  - Pinned   |         |  |
|  |   |  - Gating   |  |  - Aggregate|  |  - Prior    |         |  |
|  |   +-------------+  +-------------+  +-------------+         |  |
|  +---------------------------+--------------------------------+  |
|                              |                                    |
|                              v                                    |
|  +------------------------------------------------------------+  |
|  |              Context Environment (E)                        |  |
|  |                                                             |  |
|  |   ~/.icr/repos/<repo_id>/                                   |  |
|  |   +--------+  +--------+  +--------+  +--------+            |  |
|  |   | index  |  | chunks |  |vectors |  |contracts|           |  |
|  |   |  .db   |  |  .db   |  | .hnsw  |  |  .db    |           |  |
|  |   +--------+  +--------+  +--------+  +--------+            |  |
|  |                                                             |  |
|  +------------------------------------------------------------+  |
+------------------------------------------------------------------+
```

### Three-Plane Architecture

ICR follows a clean separation of concerns across three planes:

| Plane | Component | Responsibility |
|-------|-----------|----------------|
| **Data Plane** | icd (daemon) | Indexing, storage, retrieval, pack compilation |
| **Tool Plane** | ic-mcp (MCP server) | Safe, bounded tool exposure via MCP protocol |
| **Behavior Plane** | ic-claude (plugin) | Claude Code hooks, commands, and integration |

---

## Component Architecture

### icd (Daemon) - Data Plane

The daemon is the core engine responsible for all data operations.

```
icd/
+-- src/
    +-- main.py                 # Entry point, service orchestration
    +-- config.py               # Configuration management
    |
    +-- storage/
    |   +-- sqlite_store.py     # SQLite + FTS5 for metadata & lexical
    |   +-- vector_store.py     # HNSW index with float16 storage
    |   +-- contract_store.py   # Contract/interface index
    |   +-- memory_store.py     # Derived memory & pinned items
    |
    +-- indexing/
    |   +-- watcher.py          # File system watcher with debouncing
    |   +-- chunker.py          # Symbol-based chunking (tree-sitter)
    |   +-- embedder.py         # Embedding backend abstraction
    |   +-- contract_detector.py# Heuristic contract detection
    |
    +-- retrieval/
    |   +-- hybrid.py           # Combined semantic + lexical retrieval
    |   +-- mmr.py              # Maximal Marginal Relevance diversity
    |   +-- entropy.py          # Retrieval entropy computation
    |
    +-- pack/
    |   +-- compiler.py         # Knapsack-based pack compilation
    |   +-- formatter.py        # Markdown output with citations
    |   +-- gating.py           # Mode selection (pack vs RLM)
    |
    +-- rlm/
    |   +-- planner.py          # RLM plan generation
    |   +-- aggregator.py       # Non-generative aggregation
    |   +-- budget.py           # Stop condition enforcement
    |
    +-- metrics/
        +-- ewr.py              # Exploration Waste Ratio
        +-- imr.py              # Impact Miss Rate
        +-- telemetry.py        # Local-only telemetry
```

#### Key Responsibilities

1. **Indexing**: Watch file system changes, parse code into chunks, generate embeddings
2. **Storage**: Persist metadata, vectors, and derived state efficiently
3. **Retrieval**: Execute hybrid searches with diversity-aware ranking
4. **Pack Compilation**: Assemble context packs within token budgets
5. **RLM Execution**: Run bounded inspection loops when needed

### ic-mcp (MCP Server) - Tool Plane

The MCP server exposes safe, bounded operations to Claude Code.

```
ic-mcp/
+-- src/
    +-- server.py               # MCP server main loop
    |
    +-- tools/
    |   +-- memory.py           # memory_pack, memory_pin, etc.
    |   +-- env.py              # env_search, env_peek, env_slice, etc.
    |   +-- project.py          # project_map, project_impact, etc.
    |   +-- rlm.py              # rlm_plan, rlm_map_reduce
    |   +-- admin.py            # admin_ping, diagnostics
    |
    +-- schemas/
    |   +-- inputs.py           # Pydantic input models
    |   +-- outputs.py          # Pydantic output models
    |   +-- validation.py       # JSON Schema utilities
    |
    +-- transport/
        +-- stdio.py            # Stdio transport for local communication
```

#### Design Principles

- **Bounded Operations**: All tools have explicit limits (tokens, results, time)
- **Typed Interfaces**: Strict input/output validation via Pydantic
- **Deterministic Behavior**: Same inputs produce same outputs
- **Safe Defaults**: Conservative limits to prevent runaway operations

### ic-claude (Plugin) - Behavior Plane

The plugin integrates ICR into Claude Code's hook and command system.

```
ic-claude/
+-- plugin.json                 # Plugin manifest
|
+-- commands/
|   +-- ic.md                   # /ic command definition
|
+-- hooks/
|   +-- hooks.json              # Hook configuration
|
+-- scripts/
    +-- ic-hook-userpromptsubmit.py   # Prompt interception
    +-- ic-hook-stop.py               # Response processing
    +-- ic-hook-precompact.py         # Pre-compaction handling
    +-- ic-cli.py                     # CLI fallback
```

#### Hook Integration

| Hook | Purpose |
|------|---------|
| `UserPromptSubmit` | Inject context pack into prompt |
| `Stop` | Extract ledger/invariants from response |
| `PreCompact` | Preserve pinned items during compaction |

---

## Data Flow Diagrams

### Ingestion Flow

```
+-------------+     +-------------+     +-------------+
|  File       |     |   Watcher   |     |   Chunker   |
|  System     | --> |  (debounce) | --> | (tree-sitter)|
|  Changes    |     |   500ms     |     |   symbols   |
+-------------+     +-------------+     +------+------+
                                               |
                                               v
+-------------+     +-------------+     +-------------+
|   Storage   | <-- |   Indexer   | <-- |  Embedder   |
|  SQLite +   |     |  (dedup,    |     |  (ONNX or   |
|  HNSW       |     |   update)   |     |   remote)   |
+-------------+     +-------------+     +-------------+
```

**Flow Description:**

1. **File System Changes**: Detected by watchdog with 500ms debouncing
2. **Watcher**: Filters by extension and ignore patterns, batches changes
3. **Chunker**: Parses with tree-sitter, extracts symbol-level chunks
4. **Embedder**: Generates embeddings (local ONNX by default)
5. **Indexer**: Deduplicates chunks, updates only changed content
6. **Storage**: Persists to SQLite (metadata) and HNSW (vectors)

### Query Flow

```
+-------------+     +-------------+     +-------------+
|    User     |     |    Hook     |     |   memory    |
|   Prompt    | --> |  intercept  | --> |   _pack     |
+-------------+     +-------------+     +------+------+
                                               |
                                               v
                                        +-------------+
                                        |   Gating    |
                                        |  Decision   |
                                        +------+------+
                                               |
                         +---------------------+---------------------+
                         |                                           |
                         v                                           v
                  +-------------+                             +-------------+
                  | Pack Mode   |                             | RLM Mode    |
                  | (low entropy)|                            |(high entropy)|
                  +------+------+                             +------+------+
                         |                                           |
                         v                                           v
                  +-------------+                             +-------------+
                  |   Hybrid    |                             |  Bounded    |
                  |  Retrieval  |                             |  Inspection |
                  +------+------+                             +------+------+
                         |                                           |
                         v                                           v
                  +-------------+                             +-------------+
                  |     MMR     |                             | Aggregation |
                  |  Diversity  |                             |   Loop      |
                  +------+------+                             +------+------+
                         |                                           |
                         v                                           v
                  +-------------+                             +-------------+
                  |  Knapsack   |                             |   Results   |
                  |   Packing   |                             |  Synthesis  |
                  +------+------+                             +------+------+
                         |                                           |
                         +---------------------+---------------------+
                                               |
                                               v
                                        +-------------+
                                        |   Context   |
                                        |    Pack     |
                                        +-------------+
```

**Flow Description:**

1. **User Prompt**: Intercepted by UserPromptSubmit hook
2. **Hook**: Invokes memory_pack tool via MCP
3. **Gating**: Computes retrieval entropy, selects mode
4. **Pack Mode**: Direct retrieval + MMR + knapsack packing
5. **RLM Mode**: Bounded inspection loop with aggregation
6. **Context Pack**: Formatted markdown with citations

### Memory Flow

```
+-------------+     +-------------+     +-------------+
|   Claude    |     |    Stop     |     |   Ledger    |
|  Response   | --> |    Hook     | --> |   Parser    |
+-------------+     +-------------+     +------+------+
                                               |
                                               v
                                        +-------------+
                                        |   Memory    |
                                        |   Store     |
                                        +------+------+
                                               |
                                               v
                                        +-------------+
                                        |    Prior    |
                                        |   Update    |
                                        +-------------+
```

**Flow Description:**

1. **Claude Response**: Processed by Stop hook
2. **Ledger Parser**: Extracts invariants, decisions, key facts
3. **Memory Store**: Persists derived memory items
4. **Prior Update**: Updates Beta prior for future gating decisions

---

## Storage Layout

### Directory Structure

```
~/.icr/
+-- config.yaml                 # Global configuration
+-- repos/
|   +-- <repo_id_1>/
|   |   +-- index.db            # SQLite: files, chunks, metadata
|   |   +-- vectors.hnsw        # HNSW: float16 vector index
|   |   +-- contracts.db        # Contract/interface index
|   |   +-- memory.db           # Derived memory, pins
|   |   +-- metrics.db          # Local telemetry
|   |   +-- config.yaml         # Repo-specific overrides
|   |
|   +-- <repo_id_2>/
|       +-- ...
|
+-- models/
|   +-- all-MiniLM-L6-v2.onnx   # Local embedding model
|
+-- logs/
    +-- icd.log                 # Daemon logs
```

### Repo ID Computation

```python
import hashlib
from pathlib import Path

def compute_repo_id(repo_root: Path) -> str:
    """
    Compute stable repository identifier.

    Uses content-addressed hashing for stability across moves.
    Falls back to path-based hash if no git history.
    """
    git_dir = repo_root / ".git"

    if git_dir.exists():
        # Use first commit hash for stability
        head_file = git_dir / "HEAD"
        if head_file.exists():
            ref = head_file.read_text().strip()
            if ref.startswith("ref: "):
                ref_path = git_dir / ref[5:]
                if ref_path.exists():
                    commit = ref_path.read_text().strip()
                    return hashlib.sha256(commit.encode()).hexdigest()[:16]

    # Fallback: path-based hash
    return hashlib.sha256(str(repo_root.resolve()).encode()).hexdigest()[:16]
```

### SQLite Schema

```sql
-- files table: track indexed files
CREATE TABLE files (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    content_hash TEXT NOT NULL,
    mtime_ns INTEGER NOT NULL,
    size_bytes INTEGER NOT NULL,
    language TEXT,
    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- chunks table: symbol-level chunks
CREATE TABLE chunks (
    id TEXT PRIMARY KEY,  -- content-hash:struct-hash
    file_id INTEGER REFERENCES files(id),
    symbol_path TEXT,     -- e.g., "module.Class.method"
    symbol_type TEXT,     -- function, class, method, etc.
    start_line INTEGER,
    end_line INTEGER,
    content TEXT NOT NULL,
    token_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- FTS5 virtual table for lexical search
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    content,
    symbol_path,
    content='chunks',
    content_rowid='rowid'
);

-- contracts table: interfaces, types, schemas
CREATE TABLE contracts (
    id INTEGER PRIMARY KEY,
    chunk_id TEXT REFERENCES chunks(id),
    contract_type TEXT,   -- interface, type, schema, etc.
    name TEXT,
    signature TEXT,
    dependencies TEXT     -- JSON array of referenced types
);

-- memory table: derived knowledge
CREATE TABLE memory (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,   -- invariant, decision, fact
    content TEXT NOT NULL,
    source_chunks TEXT,   -- JSON array of chunk IDs
    pinned BOOLEAN DEFAULT FALSE,
    ttl_seconds INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP
);

-- Indexes for common queries
CREATE INDEX idx_files_path ON files(path);
CREATE INDEX idx_chunks_file ON chunks(file_id);
CREATE INDEX idx_chunks_symbol ON chunks(symbol_path);
CREATE INDEX idx_memory_pinned ON memory(pinned);
```

---

## Index Types

### 1. Lexical Index (FTS5 + BM25)

**Purpose**: Fast keyword-based search with relevance ranking.

**Implementation**: SQLite FTS5 with BM25 scoring.

```sql
-- Search example
SELECT c.*, bm25(chunks_fts) as score
FROM chunks_fts
JOIN chunks c ON chunks_fts.rowid = c.rowid
WHERE chunks_fts MATCH 'authenticate user'
ORDER BY score
LIMIT 20;
```

**Characteristics**:
- P50 latency: 10ms
- Exact keyword matching
- Good for specific identifiers and error messages

### 2. Vector Index (HNSW)

**Purpose**: Semantic similarity search via embeddings.

**Implementation**: hnswlib with float16 storage, float32 compute.

```python
import hnswlib
import numpy as np

class VectorIndex:
    def __init__(self, dim: int = 384, max_elements: int = 250000):
        self.index = hnswlib.Index(space='cosine', dim=dim)
        self.index.init_index(
            max_elements=max_elements,
            ef_construction=200,
            M=16
        )
        self.index.set_ef(100)  # Query-time accuracy

    def add(self, ids: list[str], vectors: np.ndarray):
        """Add vectors with float16 storage."""
        vectors_f16 = vectors.astype(np.float16)
        # Store as float16, convert to float32 for index
        self.index.add_items(vectors_f16.astype(np.float32), ids)

    def search(self, query: np.ndarray, k: int = 20) -> tuple[list[str], list[float]]:
        """Search with float32 computation."""
        query_f32 = query.astype(np.float32)
        ids, distances = self.index.knn_query(query_f32, k=k)
        return ids[0].tolist(), distances[0].tolist()
```

**Characteristics**:
- P50 latency: 15ms (given embedding)
- Semantic similarity matching
- Good for conceptual queries

### 3. Symbol Index

**Purpose**: Navigate code structure via symbol hierarchy.

**Implementation**: Tree-sitter parsed symbol table.

```python
# Example symbol hierarchy
{
    "auth.py": {
        "symbols": [
            {
                "name": "AuthService",
                "type": "class",
                "line_start": 10,
                "line_end": 150,
                "children": [
                    {
                        "name": "authenticate",
                        "type": "method",
                        "line_start": 25,
                        "line_end": 60,
                        "signature": "def authenticate(self, token: str) -> User"
                    },
                    {
                        "name": "validate_token",
                        "type": "method",
                        "line_start": 62,
                        "line_end": 90
                    }
                ]
            }
        ]
    }
}
```

**Characteristics**:
- Structured navigation
- Type-aware search
- Supports: functions, classes, methods, interfaces, types

### 4. Contract Index

**Purpose**: Prioritize interfaces, types, schemas in retrieval.

**Implementation**: Heuristic detection + dedicated index.

```python
CONTRACT_PATTERNS = [
    r'\binterface\s+\w+',      # TypeScript/Java interfaces
    r'\babstract\s+class\s+',  # Abstract classes
    r'\bprotocol\s+\w+',       # Swift protocols
    r'\btrait\s+\w+',          # Rust traits
    r'\btype\s+\w+\s*=',       # Type aliases
    r'@dataclass',             # Python dataclasses
    r'\bschema\b',             # Schema definitions
    r'\bmodel\b',              # ORM models
    r'\bstruct\s+\w+',         # Go/Rust structs
    r'\benum\s+\w+',           # Enumerations
]
```

**Characteristics**:
- Automatic detection via patterns
- Boosted ranking (1.5x default)
- Cross-language support

---

## RLM Runtime Model

### Bounded Execution Model

```
+------------------------------------------------------------------+
|                       RLM-Lite Runtime                            |
|                                                                   |
|   +----------+     +----------+     +----------+                  |
|   | rlm_plan | --> |env_search| --> | env_peek |                  |
|   +----------+     +----------+     +----------+                  |
|        |                |                |                        |
|        v                v                v                        |
|   +----------------------------------------------------------+   |
|   |                  Stop Conditions                          |   |
|   |                                                           |   |
|   |   - Max steps: 12                                         |   |
|   |   - Max peek lines: 1200                                  |   |
|   |   - Max candidates: 50                                    |   |
|   |   - Wall clock: 8s (pack+plan), 20s (map-reduce)          |   |
|   |                                                           |   |
|   +----------------------------------------------------------+   |
|        |                                                          |
|        v                                                          |
|   +----------------------------------------------------------+   |
|   |  Fallback: If exceeded --> Pack mode + warning            |   |
|   +----------------------------------------------------------+   |
|                                                                   |
+------------------------------------------------------------------+
```

### Gating Decision Logic

```python
def select_mode(query: str, context: QueryContext) -> str:
    """
    Select between pack and RLM modes based on retrieval entropy.

    Low entropy (< 2.5): Pack mode - clear retrieval signal
    High entropy (>= 2.5): RLM mode - needs exploration
    """
    # Compute initial retrieval
    candidates = hybrid_retrieve(query, k=50)

    # Compute entropy of score distribution
    scores = [c.score for c in candidates]
    entropy = compute_entropy(scores, temperature=1.0)

    if entropy < context.config.rlm.entropy_threshold:
        return "pack"
    else:
        return "rlm"

def compute_entropy(scores: list[float], temperature: float = 1.0) -> float:
    """
    Compute Shannon entropy of score distribution.

    High entropy = flat distribution = uncertain retrieval
    Low entropy = peaked distribution = confident retrieval
    """
    import numpy as np

    # Convert to probabilities via softmax
    scores = np.array(scores) / temperature
    exp_scores = np.exp(scores - np.max(scores))
    probs = exp_scores / exp_scores.sum()

    # Shannon entropy
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    return float(entropy)
```

---

## Design Decisions

### 1. Float16 Storage, Float32 Compute

**Rationale**: 50% storage savings with minimal accuracy loss.

```python
# Storage: float16 for memory efficiency
vectors_f16 = vectors.astype(np.float16)  # 2 bytes per dim

# Compute: float32 for numerical stability
def cosine_similarity(query: np.ndarray, stored: np.ndarray) -> float:
    query_f32 = query.astype(np.float32)
    stored_f32 = stored.astype(np.float32)
    return np.dot(query_f32, stored_f32) / (
        np.linalg.norm(query_f32) * np.linalg.norm(stored_f32)
    )
```

### 2. Local-First Embedding

**Rationale**: No network egress by default for privacy and latency.

**Default**: ONNX-optimized `all-MiniLM-L6-v2` (384 dimensions)

**Latency Impact**:
- Local ONNX: 20ms P50
- Remote API: 80-300ms P50 (when explicitly enabled)

### 3. Symbol-Level Chunking

**Rationale**: Preserve semantic boundaries for better retrieval.

**Rules**:
- Chunk by functions, classes, methods - not lines
- Keep docstrings attached to their symbols
- Stable chunk IDs via content hashing
- Size bounds: 200-800 tokens (soft), 1200 tokens (hard)

### 4. Hybrid Retrieval with MMR

**Rationale**: Combine precision of lexical search with recall of semantic search, while ensuring diversity.

**Scoring Formula**:
```
score = w_e * sim_embedding
      + w_b * score_bm25
      + w_r * recency_decay
      + w_c * is_contract
      + w_f * in_focus
      + w_p * is_pinned
```

**Default Weights**:
- w_e = 0.4 (embedding similarity)
- w_b = 0.3 (BM25 lexical)
- w_r = 0.1 (recency)
- w_c = 0.1 (contract boost)
- w_f = 0.05 (focus path)
- w_p = 0.05 (pinned)

### 5. Bounded Operations

**Rationale**: Prevent runaway costs and ensure predictable latency.

**All tools enforce**:
- Maximum result counts
- Token budgets
- Timeout limits
- Memory caps

---

## Next Steps

- [RESEARCH_FOUNDATION.md](RESEARCH_FOUNDATION.md): Theoretical basis for ICR
- [API_REFERENCE.md](API_REFERENCE.md): Complete tool documentation
- [CONFIGURATION.md](CONFIGURATION.md): Configuration options
