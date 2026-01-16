# ICR - Infinite Context for Claude Code

Give Claude perfect memory of your codebase with one command.

## Install (30 seconds)

In your project directory, run:

```bash
curl -fsSL https://raw.githubusercontent.com/idan82labs/icr/main/install.sh | bash
```

Then restart Claude Code.

**That's it.** Just ask questions normally.

## How It Works

After installation, Claude automatically uses ICR when you ask about code:

| You ask | Claude does |
|---------|-------------|
| "How does auth work?" | Searches code, reads relevant files, explains |
| "Where is the DB configured?" | Finds config files, shows you the code |
| "Find all API endpoints" | Scans codebase, lists them with locations |

No special commands. No syntax to learn. Just ask.

## What's Happening Behind the Scenes

1. **Indexing**: Your code is chunked and embedded locally (no API calls)
2. **Retrieval**: Questions trigger hybrid search (semantic + keyword)
3. **Context**: Relevant code is packed into Claude's context window
4. **Answer**: Claude responds with accurate info citing specific files

All processing happens locally. Your code never leaves your machine.

## Requirements

- Python 3.10+
- Claude Code

## Re-index After Major Changes

```bash
.icr/venv/bin/icd index --repo-root .
```

## Configuration (Optional)

Edit `.icr/config.yaml`:

```yaml
embedding:
  model_name: all-MiniLM-L6-v2  # or all-mpnet-base-v2 for better quality

retrieval:
  weight_embedding: 0.4  # Semantic similarity
  weight_bm25: 0.3       # Keyword matching
  weight_recency: 0.1    # Recent files boost

pack:
  default_budget_tokens: 8000  # Context size per query
```

## Uninstall

```bash
rm -rf .icr .mcp.json
```
