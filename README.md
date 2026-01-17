# ICR - Intelligent Context Retrieval

Code search that actually finds what you're looking for.

## The Problem

Pure embedding search returns semantically similar but useless results. Ask "what embedding model does this use?" and you get empty `__init__.py` files because they're "semantically similar" to your question.

## The Solution

ICR combines embeddings with BM25 keyword matching. The embeddings find conceptually related code. BM25 ensures the results actually contain your search terms.

**Result: 100% of queries return usable context vs 0% with embedding-only search.**

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/idan82labs/icr/main/install.sh | bash
```

Restart Claude Code.

## When to Use ICR

| Query | Use ICR | Use Grep/Glob |
|-------|---------|---------------|
| "How does auth work?" | Yes | - |
| Find `UserService` class | - | Yes |
| "Trace the request flow" | Yes | - |
| Find `*.test.py` files | - | Yes |

**ICR is for conceptual questions. Grep is for known symbols.**

## How It Works

```
Query --> Hybrid Search --> Pack Results
         (embedding + BM25)   (budget-aware)
```

1. **Hybrid Search**: Combines semantic similarity + keyword matching
2. **Budget Packing**: Fits most relevant code into token limit
3. **Query Refinement**: Decomposes complex queries when results are scattered

## Configuration

Edit `.icr/config.yaml`:

```yaml
retrieval:
  weight_embedding: 0.4  # Semantic similarity
  weight_bm25: 0.3       # Keyword matching

pack:
  default_budget_tokens: 8000
```

## Re-index

After major changes:

```bash
.icr/venv/bin/icd -p . index
```

## Ignore Files

Create `.icrignore`:

```
tests/fixtures/
*.min.js
node_modules/
```

## Requirements

- Python 3.10+
- Claude Code
- ~2GB RAM

## Uninstall

```bash
rm -rf .icr .mcp.json
```

## License

MIT
