# ICR - Infinite Context for Claude Code

Give Claude perfect memory of your codebase with one command.

## Install (30 seconds)

```bash
curl -fsSL https://raw.githubusercontent.com/idan82labs/icr/main/install.sh | bash
```

Then restart Claude Code. **That's it.**

## Just Ask Questions

After installation, Claude automatically uses ICR when you ask about code:

```
You: "How does the auth system work?"
Claude: [searches codebase] → [reads relevant files] → [explains with file citations]

You: "Where is the database configured?"
Claude: [finds config] → [shows you the code] → [explains the setup]

You: "Find all API endpoints"
Claude: [scans codebase] → [lists endpoints with file:line references]
```

No special commands. No syntax to learn. Just ask.

---

## How It Works

1. **Indexing**: Your code is chunked and embedded locally (no API calls, no data leaves your machine)
2. **Retrieval**: Questions trigger hybrid search (semantic vectors + BM25 keywords)
3. **Context**: Relevant code is packed into Claude's context window using knapsack optimization
4. **Answer**: Claude responds with accurate info citing specific files and line numbers

---

## Why ICR vs Native Claude Code Tools?

Claude Code already has Glob, Grep, Read, and an Explore agent. So why use ICR?

| Feature | Native Claude Code | With ICR |
|---------|-------------------|----------|
| **Search type** | Keyword/regex only | Semantic + keyword hybrid |
| **"Find auth code"** | Must guess filenames | Finds by *meaning* |
| **Caching** | Re-scans every query | Pre-indexed, instant |
| **Context packing** | Manual file selection | Auto-optimized for token budget |
| **Interface priority** | Treats all code equally | Prioritizes types/interfaces |
| **Large codebases** | Slower on 10k+ files | Handles 100k+ files |

**The key difference:** Native tools search for *words*. ICR searches for *meaning*.

Example: "How does authentication work?"
- **Native:** Greps for "auth", "login", "password" → misses OAuth handler named `verifyIdentity()`
- **ICR:** Finds semantically related code → includes `verifyIdentity()`, `TokenValidator`, `SessionManager`

ICR is most valuable for:
- Large codebases (1000+ files)
- Unfamiliar projects (onboarding)
- Complex questions ("how does X flow through the system?")
- When you don't know the right keywords

---

## The Technical Details

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

ICR implements an RLM-inspired runtime where the codebase is treated as an external environment that Claude inspects via bounded tools, rather than forcing everything into the context window.

### Architecture

| Component | Role | Description |
|-----------|------|-------------|
| **icd** | Data Plane | Indexing, storage, and retrieval |
| **ic-mcp** | Tool Plane | MCP server exposing search and read operations |
| **ic-claude** | Behavior Plane | Claude Code integration (auto-invoked skill) |

### Key Features

- **Hybrid Retrieval**: Semantic (HNSW vectors) + lexical (BM25) + MMR diversity
- **Smart Packing**: Knapsack-optimized context compilation with citations
- **Contract Awareness**: Prioritizes interfaces, types, and schemas
- **Local-First**: ONNX embedding model runs entirely on your machine

---

## Configuration

Edit `.icr/config.yaml` to tune:

```yaml
embedding:
  model_name: all-MiniLM-L6-v2  # or all-mpnet-base-v2 for better quality

retrieval:
  weight_embedding: 0.4  # Semantic similarity weight
  weight_bm25: 0.3       # Keyword match weight
  weight_recency: 0.1    # Recent files boost

pack:
  default_budget_tokens: 8000  # Context size per query
```

## Re-index After Changes

```bash
.icr/venv/bin/icd index --repo-root .
```

## Requirements

- Python 3.10+
- Claude Code
- 2GB RAM

## Uninstall

```bash
rm -rf .icr .mcp.json
```

---

## License

MIT License. See [LICENSE](LICENSE).
