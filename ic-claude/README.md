# ICR for Claude Code

Intelligent Context Retrieval integration for Claude Code. Semantic search + smart context packing.

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/idan82labs/icr/main/install.sh | bash
```

Then restart Claude Code.

## What ICR Does

ICR helps Claude find relevant code **by meaning, not just keywords**:

| You ask | Native tools | With ICR |
|---------|-------------|----------|
| "auth code" | Searches for "auth" | Also finds `verifyIdentity()`, `TokenValidator` |
| "database setup" | Searches for "database" | Also finds `ConnectionPool`, ORM configs |

## When to Use ICR

**Good for:**
- Large unfamiliar codebases (1000+ files)
- Conceptual questions ("how does X work?")
- When you don't know what to grep for
- Pattern discovery

**Use native tools for:**
- Known symbol names (grep is faster)
- File path patterns (glob is better)
- Exact string matches

**ICR complements native tools - it doesn't replace them.**

## How It Works

1. **Index** - Code is chunked and embedded locally (no API calls)
2. **Search** - Hybrid retrieval: semantic vectors + BM25 keywords
3. **Pack** - Knapsack compiler fits best code in token budget
4. **Boost** - RLM refines queries when results are scattered

## Re-index

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

## Uninstall

```bash
rm -rf .icr .mcp.json
```

## Honest Limitations

- Uses a small embedding model (384 dimensions) - Claude's native understanding is often better for simple queries
- Requires re-indexing after major codebase changes
- ~2GB RAM for the embedding model
- RLM is query expansion, not true recursive LLM calls
