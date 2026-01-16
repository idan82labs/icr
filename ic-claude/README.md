# ICR Plugin for Claude Code

Give Claude perfect memory of your codebase with one command.

## Quick Start

### 1. Install the plugin

```bash
# From Claude Code
/plugin install icr@idan82labs/icr
```

Or manually:
```bash
git clone https://github.com/idan82labs/icr.git ~/.claude/plugins/icr
```

### 2. Set up in your project

```bash
/icr:setup
```

This creates `.icr/` with a Python venv and installs ICR packages.

### 3. Index your codebase

```bash
/icr:index
```

### 4. Just ask questions!

Now Claude automatically uses ICR to find relevant code:

- "How does the auth system work?"
- "Where is the database connection configured?"
- "Find all usages of UserService"
- "Explain the payment flow"

## How It Works

1. **Indexing**: ICR chunks your code, embeds it with a local ONNX model, and stores vectors in HNSW
2. **Retrieval**: When you ask a question, ICR finds semantically relevant code using hybrid search (vectors + BM25)
3. **Context**: Results are compiled into a context pack that fits Claude's token budget
4. **Answer**: Claude answers with accurate, grounded information citing specific files

## Commands

| Command | Description |
|---------|-------------|
| `/icr:setup` | One-time setup for a project |
| `/icr:index` | Index/re-index the codebase |

## No API Keys Required

ICR runs entirely locally:
- Embeddings: ONNX model (all-MiniLM-L6-v2)
- Vector search: HNSW (hnswlib)
- Lexical search: SQLite FTS5

## Project Structure

After setup, your project has:

```
.icr/
  venv/           # Python environment
  index.db        # SQLite database
  vectors.hnsw    # Vector index
  config.yaml     # Configuration
  mcp.log         # Server logs
```

## Configuration

Edit `.icr/config.yaml` to customize:

```yaml
embedding:
  model_name: all-MiniLM-L6-v2  # or all-mpnet-base-v2 for higher quality

retrieval:
  weight_embedding: 0.4  # Semantic similarity weight
  weight_bm25: 0.3       # Keyword match weight
  weight_recency: 0.1    # Recent files boost

pack:
  default_budget_tokens: 8000  # Context size
```
