# ICR - Infinite Context Runtime

[![Research Grade](https://img.shields.io/badge/status-research--grade-blue.svg)](https://github.com/icr/icr)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**ICR implements an RLM-inspired runtime: the long context is treated as an external environment that the model inspects via bounded tools.**

ICR (Infinite Context Runtime) provides an "unlimited context feel" in Claude Code by implementing a Recursive Language Model (RLM) style runtime. Rather than forcing all information into the model's context window, ICR treats the codebase as an external environment that can be efficiently searched, inspected, and aggregated through bounded, typed operations.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Architecture](#architecture)
- [Documentation](#documentation)
- [Performance](#performance)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

ICR addresses the fundamental challenge of working with large codebases in AI-assisted development: **context window limitations**. Traditional approaches either truncate context (losing important information) or expand context windows (increasing cost and latency).

ICR takes a different approach inspired by the RLM research paradigm:

1. **Prompt as Environment**: Treat the codebase as an external environment rather than forcing it into context
2. **Bounded Tools**: Provide safe, typed operations with deterministic bounds
3. **Variance Control**: Engineer for predictable latency and cost
4. **Non-Generative Aggregation**: Use deterministic aggregation for trustworthy results

### Core Components

| Component | Role | Description |
|-----------|------|-------------|
| **icd** | Data Plane | Daemon for indexing, storage, and retrieval |
| **ic-mcp** | Tool Plane | MCP server exposing safe, bounded operations |
| **ic-claude** | Behavior Plane | Claude Code plugin for hooks and commands |

---

## Key Features

- **Hybrid Retrieval**: Combines semantic (vector) and lexical (BM25) search with MMR diversity
- **Smart Context Packs**: Knapsack-optimized context compilation with citations
- **Contract Awareness**: Automatically identifies and prioritizes interfaces, types, and schemas
- **Impact Analysis**: Trace dependencies and understand change propagation
- **Memory Persistence**: Pin important context that survives conversation compaction
- **Local-First**: Default local embedding generation with no network egress
- **Research-Grade**: Built on solid theoretical foundations from RLM research

---

## Quick Start

```bash
# Install ICR
pip install icr

# Initialize for your user
icr init

# Configure Claude Code integration
icr configure claude-code

# Verify installation
icr doctor
```

Once configured, ICR automatically enhances your Claude Code sessions with intelligent context retrieval.

---

## Installation

### Prerequisites

- Python 3.10 or higher
- Claude Code CLI installed and configured
- 2GB+ available RAM (recommended: 4GB+)

### Install from PyPI

```bash
pip install icr
```

### Install from Source

```bash
git clone https://github.com/icr/icr.git
cd icr
pip install -e ".[dev]"
```

### Post-Installation Setup

```bash
# Initialize ICR configuration
icr init

# Configure Claude Code hooks
icr configure claude-code

# Verify everything is working
icr doctor
```

### Verify Installation

```bash
$ icr doctor
ICR Health Check
================

[OK] Configuration found at ~/.icr/config.yaml
[OK] SQLite database accessible
[OK] Vector index initialized
[OK] Embedding backend: local-onnx (all-MiniLM-L6-v2)
[OK] Claude Code hooks configured (user-level)
[OK] MCP server configuration found

Status: HEALTHY
```

---

## Basic Usage

### Automatic Context Injection

Once configured, ICR automatically injects relevant context on each prompt via the `UserPromptSubmit` hook. No manual intervention required.

### Manual Commands

Use `/ic` commands within Claude Code for explicit control:

```bash
# Get a context pack for a specific query
/ic pack "Where is authentication implemented?"

# Search the codebase
/ic search "database connection" --scope=repo

# Analyze impact of changes
/ic impact src/api/auth.py

# Pin important context
/ic pin src/types/index.ts --label="Core types"

# View memory statistics
/ic stats
```

### MCP Tools

ICR exposes tools through the MCP protocol, accessible as `mcp__icr__<tool_name>`:

```python
# Example: memory_pack tool
mcp__icr__memory_pack(
    prompt="How does the caching layer work?",
    repo_root="/path/to/project",
    budget_tokens=4000
)
```

---

## Architecture

```
+----------------------------------------------------------+
|                   Claude Code Session                      |
|  +------------------------------------------------------+  |
|  |                    User Prompt                        |  |
|  +---------------------------+--------------------------+  |
|                              |                             |
|                              v                             |
|  +------------------------------------------------------+  |
|  |              UserPromptSubmit Hook                    |  |
|  |  - Invokes ic-hook-userpromptsubmit                   |  |
|  |  - Returns additionalContext (pack/RLM header)        |  |
|  +---------------------------+--------------------------+  |
|                              |                             |
|                              v                             |
|  +------------------------------------------------------+  |
|  |                ic-mcp (MCP Server)                    |  |
|  |  - memory_pack, env_search, project_impact, etc.      |  |
|  |  - Local stdio transport                              |  |
|  +---------------------------+--------------------------+  |
|                              |                             |
|                              v                             |
|  +------------------------------------------------------+  |
|  |                   icd (Daemon)                        |  |
|  |  +--------+ +--------+ +----------+ +---------+       |  |
|  |  |Indexer | |Storage | |Retrieval | | Memory  |       |  |
|  |  +--------+ +--------+ +----------+ +---------+       |  |
|  +---------------------------+--------------------------+  |
|                              |                             |
|                              v                             |
|  +------------------------------------------------------+  |
|  |            Context Environment (E)                    |  |
|  |  - ~/.icr/repos/<repo_id>/                            |  |
|  |  - SQLite + FTS5 + HNSW vectors + contracts + memory  |  |
|  +------------------------------------------------------+  |
+----------------------------------------------------------+
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture documentation.

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture deep dive |
| [docs/RESEARCH_FOUNDATION.md](docs/RESEARCH_FOUNDATION.md) | RLM theory and research basis |
| [docs/USER_GUIDE.md](docs/USER_GUIDE.md) | End-user documentation |
| [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) | Developer documentation |
| [docs/API_REFERENCE.md](docs/API_REFERENCE.md) | MCP tools API reference |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | Configuration reference |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Common issues and solutions |
| [docs/CHANGELOG.md](docs/CHANGELOG.md) | Version history |

---

## Performance

### Latency Targets

| Operation | P50 | P95 | Notes |
|-----------|-----|-----|-------|
| ANN lookup (given embedding) | 15ms | 40ms | HNSW float16 storage |
| Query embedding (local) | 20ms | 50ms | ONNX-optimized model |
| End-to-end semantic | 40ms | 100ms | Sum of above |
| Hybrid search (full) | 70ms | 150ms | All components |
| Pack compilation | 200ms | 500ms | Knapsack + formatting |

### Scalability Tiers

| Tier | Files | Chunks | RAM | Index Size |
|------|-------|--------|-----|------------|
| **Tier 1 (Guaranteed)** | 10,000 | 100,000 | <2GB | <1GB |
| **Tier 2 (Stretch)** | 100,000 | 1,000,000 | <10GB | <5GB |

---

## Contributing

We welcome contributions! Please see [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) for:

- Development environment setup
- Code style requirements
- Testing guidelines
- Pull request process

---

## License

ICR is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

ICR is built on research from the Recursive Language Models (RLM) paradigm. We thank the researchers who developed the theoretical foundations for treating prompts as external environments.

---

*ICR is a research-grade implementation. For production use, please thoroughly test in your environment.*
