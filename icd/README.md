# ICD: Intelligent Context Daemon

Data plane for ICR. Handles indexing, storage, retrieval, and context packing.

## Features

### Indexing
- Tree-sitter code chunking (language-aware)
- ONNX embeddings (local, no API calls)
- Contract detection (interfaces, types, schemas)

### Storage
- SQLite with FTS5 for BM25 search
- HNSW vector index (float16 for memory efficiency)
- Chunk metadata and content storage

### Retrieval
- Hybrid scoring: semantic + BM25 + recency + contract boost
- MMR diversity selection
- Entropy-based confidence measurement

### Packing
- Knapsack compiler (budget-aware optimization)
- Citation generation
- RLM query refinement for scattered results

## Installation

```bash
pip install icd
```

## CLI Usage

```bash
# Index a directory
icd -p /path/to/project index

# Search
icd -p /path/to/project search "how does auth work"

# Stats
icd -p /path/to/project stats
```

## Scoring Formula

```
score(d,q) = w_e·cos(E_d, E_q)     # Semantic similarity
           + w_b·BM25(d,q)          # Keyword match
           + w_r·exp(-dt/τ)         # Recency boost
           + w_c·𝟙_contract         # Contract indicator
           + w_f·𝟙_focus            # Focus scope boost
           + w_p·𝟙_pinned           # Pinned indicator
```

Default weights: `w_e=0.4, w_b=0.3, w_r=0.1, w_c=0.1, w_f=0.05, w_p=0.05`

## RLM Mode

When initial retrieval has high entropy (scattered results), ICD can refine:

1. Decompose query into sub-queries (definition, usage, implementation)
2. Execute sub-queries
3. Aggregate with score fusion
4. Re-pack with combined results

This is **query expansion**, not recursive LLM calls.

## Configuration

```yaml
embedding:
  backend: local_onnx
  model_name: all-MiniLM-L6-v2
  dimension: 384

retrieval:
  weight_embedding: 0.4
  weight_bm25: 0.3
  entropy_threshold: 2.5  # Trigger RLM above this

pack:
  default_budget_tokens: 8000
```

## License

MIT
