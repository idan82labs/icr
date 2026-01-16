# CLAUDE.md - ICR Project Instructions

## Project Overview

**ICR (Intelligent Context Retrieval)** is a Claude Code plugin that provides:
- Semantic code search (finding code by meaning, not just keywords)
- Budget-aware context packing (knapsack optimization)
- RLM boost (query expansion when results are scattered)

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      ICR Stack                          │
├─────────────────────────────────────────────────────────┤
│  ic-claude    │  Claude Code integration (skill/hooks)  │
├───────────────┼─────────────────────────────────────────┤
│  ic-mcp       │  MCP server (tools for Claude)          │
├───────────────┼─────────────────────────────────────────┤
│  icd          │  Indexing, storage, retrieval engine    │
└───────────────┴─────────────────────────────────────────┘
```

### Key Directories

- `icd/` - Data plane: embeddings, vector store, BM25, knapsack compiler
- `ic-mcp/` - Tool plane: MCP server exposing ICR tools
- `ic-claude/` - Behavior plane: skill definition, hooks
- `docs/` - Documentation and research notes

## Development Commands

```bash
# Install dependencies
pip install -e icd/
pip install -e ic-mcp/

# Index a project
.venv/bin/icd -p /path/to/project index

# Search
.venv/bin/icd -p /path/to/project search "query"

# Run MCP server
.venv/bin/ic-mcp --repo-root /path/to/project

# Run tests
pytest icd/tests/
pytest ic-mcp/tests/
```

## Key Concepts

### Hybrid Scoring
```
score(d,q) = w_e·cos(E_d, E_q)     # Semantic similarity
           + w_b·BM25(d,q)          # Keyword match
           + w_r·exp(-dt/τ)         # Recency boost
           + w_c·𝟙_contract         # Contract indicator
```

### RLM (Query Expansion)
NOT recursive LLM calls. It's:
1. Initial retrieval → check entropy
2. If high entropy (scattered) → decompose into sub-queries
3. Execute sub-queries → aggregate results
4. Pack with knapsack compiler

### When to Use ICR vs Native Tools
- **ICR**: Conceptual questions, unfamiliar codebases, pattern discovery
- **Native grep/glob**: Known symbols, file patterns, exact matches

## Configuration

`.icr/config.yaml`:
```yaml
embedding:
  model_name: all-MiniLM-L6-v2

retrieval:
  weight_embedding: 0.4
  weight_bm25: 0.3
  entropy_threshold: 2.5  # Trigger RLM above this

pack:
  default_budget_tokens: 8000
```

## Honest Limitations

- Small embedding model (384 dimensions) - Claude's native understanding is often better for simple queries
- Requires re-indexing after major codebase changes
- ~2GB RAM for embedding model
- RLM is query expansion, not true recursive LLM calls
- Entropy-based gating uses magic thresholds (not learned)

## Testing

When making changes:
1. Run unit tests: `pytest icd/tests/ -v`
2. Test on real codebase: Index this repo and search
3. Test MCP integration: Run server and call tools

## Code Style

- Python 3.10+
- Type hints required
- Async where appropriate (especially in retrieval)
- Structured logging with structlog
