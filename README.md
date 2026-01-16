# ICR - Intelligent Context Retrieval

Smart context packing for Claude Code. Semantic search + budget-aware compilation.

## What ICR Actually Does

ICR helps Claude find relevant code **by meaning, not just keywords**. It:

1. **Indexes** your codebase locally (no data leaves your machine)
2. **Searches** using hybrid retrieval (semantic vectors + BM25 keywords)
3. **Packs** the most relevant code into Claude's context window efficiently
4. **Boosts** with query refinement when initial results are scattered (RLM mode)

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/idan82labs/icr/main/install.sh | bash
```

Then restart Claude Code.

## When to Use ICR vs Native Tools

**Use ICR when:**
- Exploring unfamiliar codebases
- Asking "how does X work?" conceptually
- You don't know the right keywords to grep for
- Results from grep are overwhelming/unranked

**Use native Grep/Glob when:**
- You know the exact symbol name
- You need literal string matching
- Simple file pattern searches
- Small targeted edits

| Query Type | ICR | Native Tools |
|------------|-----|--------------|
| "How does auth work?" | Better | - |
| Find `solve_dependencies` | - | Better |
| "Trace request flow" | Better | - |
| Find `*.test.py` files | - | Better |

**ICR complements native tools, it doesn't replace them.**

---

## How It Works

```
Query ──► Hybrid Search ──► Entropy Check
          (semantic+BM25)        │
                                 ▼
                    ┌────────────────────┐
                    │  Low entropy?      │
                    │  (confident match) │
                    └─────────┬──────────┘
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
         Direct Pack                   RLM Boost
         (fast path)              (query refinement)
               │                             │
               └──────────────┬──────────────┘
                              ▼
                     Knapsack Compiler
                  (budget-aware packing)
                              │
                              ▼
                       Context Pack
                   (with file citations)
```

### Key Components

| Component | What It Does |
|-----------|--------------|
| **Hybrid Search** | Combines semantic similarity + keyword matching + recency |
| **Contract Awareness** | Prioritizes interfaces, types, schemas |
| **Knapsack Packer** | Optimizes which code fits in token budget |
| **RLM Boost** | Query refinement when results are scattered |

---

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

- **icd**: Data plane - ONNX embeddings, SQLite storage, HNSW vectors
- **ic-mcp**: Tool plane - MCP server exposing `icr__memory_pack`, `icr__env_search`
- **ic-claude**: Behavior plane - Skill definition, hooks for context preservation

---

## Configuration

Edit `.icr/config.yaml`:

```yaml
embedding:
  model_name: all-MiniLM-L6-v2  # Local ONNX model

retrieval:
  weight_embedding: 0.4  # Semantic similarity
  weight_bm25: 0.3       # Keyword matching
  weight_recency: 0.1    # Recent files boost
  weight_contract: 0.1   # Interface/type boost

pack:
  default_budget_tokens: 8000  # Context per query

rlm:
  enabled: true
  entropy_threshold: 2.5  # Trigger RLM when results scattered
```

## Re-index After Major Changes

```bash
.icr/venv/bin/icd -p . index
```

## Ignore Files

Create `.icrignore` (same syntax as `.gitignore`):

```
# Skip test fixtures
tests/fixtures/
*.min.js
```

---

## Honest Limitations

- **Not magic**: ICR uses a small embedding model (384 dimensions). Claude's native understanding is often better for simple queries.
- **Index maintenance**: You need to re-index after major codebase changes.
- **Memory usage**: ~2GB RAM for the embedding model.
- **RLM is query expansion**: Not true recursive LLM calls - it's deterministic query refinement based on entropy.

## When ICR Shines

1. **Large unfamiliar codebases** - Onboarding to 5000+ file projects
2. **Conceptual questions** - "How does the payment flow work?"
3. **Pattern discovery** - "How do other endpoints handle errors?"
4. **Pre-compaction preservation** - Maintaining context across long sessions

## Requirements

- Python 3.10+
- Claude Code
- ~2GB RAM

## Uninstall

```bash
rm -rf .icr .mcp.json
```

---

## License

MIT License. See [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup.
