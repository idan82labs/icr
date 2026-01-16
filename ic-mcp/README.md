# IC-MCP: MCP Server for Intelligent Context Retrieval

MCP (Model Context Protocol) server that exposes ICR tools to Claude Code.

## What It Does

IC-MCP provides Claude with tools to:
- Search code semantically (not just keywords)
- Read specific code sections with line numbers
- Pack relevant context within token budgets
- Find symbols, commands, and project structure

## Installation

```bash
pip install ic-mcp
```

## Run

```bash
ic-mcp --repo-root /path/to/project
```

Or configure in `.mcp.json` for Claude Code integration.

## Available Tools

### Core Search
| Tool | Purpose |
|------|---------|
| `icr__env_search` | Hybrid search (semantic + BM25) |
| `icr__env_peek` | Read specific lines from a file |
| `icr__project_symbol_search` | Find functions/classes by name |

### Context Packing
| Tool | Purpose |
|------|---------|
| `icr__memory_pack` | Compile relevant context for a query |
| `icr__memory_pin` | Pin important files to always include |
| `icr__memory_stats` | Show index statistics |

### Project Info
| Tool | Purpose |
|------|---------|
| `icr__project_map` | Show directory structure |
| `icr__project_commands` | Find build/test commands |
| `icr__project_impact` | Analyze change impact |

## How memory_pack Works

```
Query → Hybrid Search → Entropy Check
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
        Low Entropy                 High Entropy
        (confident)                 (scattered)
              │                           │
              ▼                           ▼
        Direct Pack               RLM Query Refinement
              │                           │
              └─────────────┬─────────────┘
                            ▼
                   Knapsack Compiler
                   (budget-aware)
                            │
                            ▼
                     Context Pack
```

**RLM Boost**: When initial retrieval has high entropy (scattered results), ICR refines the query into sub-queries to gather better context.

## Configuration

Environment variables:
- `ICR_LOG_LEVEL`: DEBUG, INFO, WARNING, ERROR
- `ICR_LOG_FILE`: Path to log file

## License

MIT
