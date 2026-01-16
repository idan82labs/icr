---
description: Set up ICR for this project (one-time setup)
allowed-tools: Bash, Read, Write
---

# ICR Setup

Set up Infinite Context Runtime for this project. This command:
1. Creates a Python virtual environment
2. Installs ICR packages (icd, ic-mcp)
3. Configures the MCP server for Claude Code

## Steps

1. Check if ICR is already set up by looking for `.icr/` directory
2. If not set up, create venv and install packages:

```bash
# Create ICR directory
mkdir -p .icr

# Create venv (use python3.10+ if available)
python3 -m venv .icr/venv

# Install packages
.icr/venv/bin/pip install --upgrade pip
.icr/venv/bin/pip install icd ic-mcp
```

3. Create `.icr/config.yaml` with default settings:

```yaml
embedding:
  model_name: all-MiniLM-L6-v2
  dimension: 384

retrieval:
  weight_embedding: 0.4
  weight_bm25: 0.3
  weight_recency: 0.1
  weight_contract: 0.1

pack:
  default_budget_tokens: 8000
```

4. Add MCP server to project settings (`.claude/settings.local.json`):

```json
{
  "mcpServers": {
    "icr": {
      "type": "stdio",
      "command": ".icr/venv/bin/ic-mcp",
      "args": ["--repo-root", ".", "--log-file", ".icr/mcp.log"]
    }
  }
}
```

5. Tell user to restart Claude Code to activate ICR.

## Success Message

After setup, tell the user:
"ICR is set up! Restart Claude Code, then use `/icr:index` to index your codebase."
