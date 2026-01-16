# ICR User Guide

This guide covers everything you need to know to effectively use ICR (Infinite Context Runtime) in your daily development workflow with Claude Code.

---

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Configuration](#configuration)
- [Using /ic Commands](#using-ic-commands)
- [Understanding Context Packs](#understanding-context-packs)
- [Working with Pinned Invariants](#working-with-pinned-invariants)
- [Interpreting Impact Analysis](#interpreting-impact-analysis)
- [Best Practices](#best-practices)
- [Tips and Tricks](#tips-and-tricks)

---

## Introduction

ICR enhances Claude Code by providing intelligent context retrieval. Instead of manually searching for relevant code and pasting it into your prompts, ICR automatically identifies and injects the most relevant context for your queries.

### What ICR Does For You

1. **Automatic Context Injection**: Relevant code snippets are added to every prompt
2. **Hybrid Search**: Combines semantic understanding with keyword matching
3. **Impact Analysis**: Understand how changes propagate through your codebase
4. **Memory Persistence**: Important context survives conversation compaction
5. **Contract Awareness**: Interfaces and types are prioritized in results

### How It Works

When you send a prompt to Claude Code:

1. ICR intercepts your prompt via hooks
2. Analyzes your query to understand intent
3. Searches the indexed codebase for relevant code
4. Compiles a "context pack" within token limits
5. Injects this context into your prompt
6. Claude sees your question plus relevant code

---

## Installation

### Prerequisites

Before installing ICR, ensure you have:

- **Python 3.10+**: Check with `python --version`
- **Claude Code CLI**: Installed and configured
- **2GB+ RAM**: Recommended 4GB+ for larger projects

### Step 1: Install ICR

```bash
# Using pip
pip install icr

# Or using pipx (recommended for CLI tools)
pipx install icr
```

### Step 2: Initialize ICR

```bash
# Initialize ICR configuration
icr init
```

This creates the configuration directory at `~/.icr/` with default settings.

### Step 3: Configure Claude Code Integration

```bash
# Set up hooks and MCP server
icr configure claude-code
```

This modifies your Claude Code configuration to:
- Register the ICR MCP server
- Set up UserPromptSubmit hooks
- Enable context injection

### Step 4: Verify Installation

```bash
icr doctor
```

Expected output:
```
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

### Step 5: Index Your First Repository

```bash
# Navigate to your project
cd /path/to/your/project

# Start indexing (runs in background)
icr index --watch
```

ICR will:
1. Scan all source files
2. Parse into symbol-level chunks
3. Generate embeddings
4. Build search indexes

Initial indexing time depends on project size:
- Small project (1k files): ~30 seconds
- Medium project (10k files): ~5 minutes
- Large project (100k files): ~30 minutes

---

## Configuration

### Configuration File Location

ICR uses a YAML configuration file at `~/.icr/config.yaml`.

### Key Configuration Options

```yaml
# ~/.icr/config.yaml

icr:
  # Embedding settings
  embedding:
    backend: local_onnx      # local_onnx | openai | anthropic
    model_name: all-MiniLM-L6-v2
    batch_size: 32

  # Retrieval settings
  retrieval:
    weight_embedding: 0.4    # Semantic similarity weight
    weight_bm25: 0.3         # Lexical matching weight
    mmr_lambda: 0.7          # Diversity vs relevance trade-off

  # Pack settings
  pack:
    default_budget_tokens: 8000
    max_budget_tokens: 32000

  # File watching
  watcher:
    ignore_patterns:
      - "**/.git/**"
      - "**/node_modules/**"
      - "**/__pycache__/**"
    watch_extensions:
      - ".py"
      - ".js"
      - ".ts"
      - ".tsx"
```

### Per-Project Configuration

Create `.icr/config.yaml` in your project root for project-specific settings:

```yaml
# /path/to/project/.icr/config.yaml

icr:
  # Override default budget for this project
  pack:
    default_budget_tokens: 6000

  # Project-specific ignore patterns
  watcher:
    ignore_patterns:
      - "**/generated/**"
      - "**/vendor/**"
```

### Environment Variables

Override any setting with environment variables:

```bash
# Override embedding backend
export ICD_EMBEDDING__BACKEND=openai
export ICD_EMBEDDING__API_KEY=sk-...

# Override token budget
export ICD_PACK__DEFAULT_BUDGET_TOKENS=10000
```

---

## Using /ic Commands

ICR provides the `/ic` command family within Claude Code.

### /ic pack

Get a context pack for a specific query:

```bash
/ic pack "How does authentication work?"
```

**Options:**
- `--budget`: Token budget (default: 4000)
- `--k`: Number of sources to consider (default: 20)
- `--focus`: Focus on specific paths

**Examples:**
```bash
# Basic query
/ic pack "Where is the database connection configured?"

# With custom budget
/ic pack "Explain the API routing" --budget 8000

# Focus on specific directory
/ic pack "How do tests work?" --focus tests/
```

### /ic search

Search the codebase for specific content:

```bash
/ic search "error handling" --scope repo
```

**Options:**
- `--scope`: Search scope (repo, transcript, diffs, contracts, all)
- `--limit`: Maximum results (default: 20)
- `--language`: Filter by language
- `--explain`: Show search strategy

**Examples:**
```bash
# Search repository
/ic search "validateToken" --scope repo

# Search contracts only
/ic search "UserInput" --scope contracts

# Filter by language
/ic search "async function" --language typescript
```

### /ic impact

Analyze impact of changes:

```bash
/ic impact src/api/auth.py
```

**Options:**
- `--query`: Focus the analysis
- `--max-nodes`: Maximum graph nodes (default: 100)

**Examples:**
```bash
# Basic impact analysis
/ic impact src/models/user.py

# Focused analysis
/ic impact src/types/index.ts --query "Who uses UserType?"

# Multiple files
/ic impact src/api/*.py
```

### /ic pin

Pin important context for persistence:

```bash
/ic pin src/types/index.ts --label "Core types"
```

**Options:**
- `--label`: Human-readable label
- `--ttl`: Time-to-live in seconds

**Examples:**
```bash
# Pin indefinitely
/ic pin src/config/schema.ts --label "Config schema"

# Pin for 1 hour
/ic pin src/api/routes.py --ttl 3600

# Pin multiple files
/ic pin src/types/*.ts --label "Type definitions"
```

### /ic unpin

Remove a pinned item:

```bash
/ic unpin src/types/index.ts
```

### /ic list

List memory items:

```bash
/ic list --filter pinned
```

**Options:**
- `--filter`: Filter type (all, pinned, recent, stale)
- `--limit`: Maximum items (default: 50)

### /ic stats

View memory statistics:

```bash
/ic stats
```

**Output:**
```
ICR Memory Statistics
=====================

Repository: /path/to/project
Repo ID: a1b2c3d4e5f6

Files indexed: 1,234
Chunks stored: 15,678
Vectors stored: 15,678
Contracts detected: 456

Memory usage:
  Index size: 145 MB
  Vector size: 89 MB
  Total: 234 MB

Pinned items: 12
Recent queries: 45
```

---

## Understanding Context Packs

### What is a Context Pack?

A context pack is a curated selection of code snippets relevant to your query, formatted as markdown with citations.

### Pack Structure

```markdown
# Context Pack

## Query Analysis
Your query about "authentication flow" matches:
- 5 semantic matches (embedding similarity)
- 8 lexical matches (keyword matching)
- 2 contract matches (interfaces/types)

## Sources

### [1] src/auth/service.py:AuthService.authenticate (lines 25-60)

```python
def authenticate(self, token: str) -> User:
    """
    Authenticate a user by validating their token.

    Args:
        token: JWT token from the request

    Returns:
        User object if valid, raises AuthError otherwise
    """
    payload = self.jwt_service.decode(token)
    user = self.user_repo.find_by_id(payload['user_id'])
    if not user:
        raise AuthError("User not found")
    return user
```

### [2] src/auth/types.py:AuthConfig (lines 5-15)

```python
@dataclass
class AuthConfig:
    """Configuration for authentication service."""
    jwt_secret: str
    token_expiry_hours: int = 24
    refresh_enabled: bool = True
```

## Citations
- [1]: src/auth/service.py (score: 0.89)
- [2]: src/auth/types.py (score: 0.85)

## Metadata
- Token budget: 4000
- Tokens used: 2345
- Sources considered: 50
- Retrieval entropy: 1.8 (low - confident)
```

### Reading Citations

Citations link snippets to their source files:
- `[1]`: Reference number in pack
- `src/auth/service.py`: File path
- `AuthService.authenticate`: Symbol name
- `lines 25-60`: Line numbers
- `score: 0.89`: Relevance score (0-1)

### Token Budget

The token budget controls pack size:
- **Smaller budget (2000-4000)**: Focused, essential context
- **Medium budget (4000-8000)**: Balanced coverage
- **Larger budget (8000-12000)**: Comprehensive context

---

## Working with Pinned Invariants

### What are Pinned Invariants?

Pinned items are pieces of context you mark as important. They:
- Are always included in context packs
- Survive conversation compaction
- Persist across sessions

### When to Pin

Pin content that:
- Defines core types/interfaces
- Contains project conventions
- Documents critical constraints
- Needs to be remembered across conversations

### Examples

```bash
# Pin core type definitions
/ic pin src/types/index.ts --label "Core types"

# Pin API contracts
/ic pin src/api/schema.graphql --label "GraphQL schema"

# Pin configuration schemas
/ic pin src/config/config.schema.json --label "Config schema"

# Pin architecture documentation
/ic pin docs/ARCHITECTURE.md --label "Architecture reference"
```

### Managing Pins

```bash
# List all pinned items
/ic list --filter pinned

# Remove a pin
/ic unpin src/types/index.ts

# View pin details
/ic get src/types/index.ts
```

### Pin TTL (Time-To-Live)

For temporary pins, use TTL:

```bash
# Pin for debugging session (1 hour)
/ic pin src/debug/trace.py --ttl 3600 --label "Debug trace"

# Pin for feature work (24 hours)
/ic pin src/features/new-feature.ts --ttl 86400
```

---

## Interpreting Impact Analysis

### What is Impact Analysis?

Impact analysis shows how changes to a file affect other parts of the codebase.

### Running Impact Analysis

```bash
/ic impact src/models/user.py
```

### Understanding the Output

```
Impact Analysis: src/models/user.py
===================================

Direct Dependencies (files that import this):
  [HIGH] src/services/user_service.py (12 references)
  [HIGH] src/api/users.py (8 references)
  [MED]  src/services/auth_service.py (3 references)
  [LOW]  tests/test_user.py (15 references)

Transitive Impact (downstream effects):
  src/services/user_service.py
    -> src/api/users.py
    -> src/api/auth.py
  src/services/auth_service.py
    -> src/middleware/auth.py
    -> src/api/protected.py

Contract Impact:
  [WARN] Modifies UserModel interface
         Used by: 8 files
         Breaking change potential: HIGH

Suggested Review:
  1. src/services/user_service.py - Primary consumer
  2. src/api/users.py - API surface
  3. tests/test_user.py - Update tests

Total impact score: 0.78 (significant)
```

### Impact Levels

| Level | Meaning | Action |
|-------|---------|--------|
| HIGH | Critical dependency | Review carefully |
| MED | Moderate coupling | Check for issues |
| LOW | Loose coupling | Likely safe |

### Using Impact for Refactoring

Before refactoring, run impact analysis:

```bash
# Check what depends on the module
/ic impact src/utils/helpers.py

# With focused query
/ic impact src/types/index.ts --query "Who uses UserType?"
```

---

## Best Practices

### 1. Start Sessions with Context

Begin coding sessions by establishing context:

```
You: I'm working on the authentication system. Let me get context.
/ic pack "authentication system overview"
```

### 2. Pin Important Files Early

When starting on a feature:

```bash
# Pin the relevant files
/ic pin src/features/auth/types.ts --label "Auth types"
/ic pin src/features/auth/service.ts --label "Auth service"
```

### 3. Use Focused Queries

Be specific in your queries:

```bash
# Too broad
/ic pack "how does it work"

# Better
/ic pack "How does JWT token validation work in AuthService?"
```

### 4. Check Impact Before Changes

Before modifying shared code:

```bash
/ic impact src/types/shared.ts
```

### 5. Review Contracts

When learning a codebase:

```bash
/ic search "interface" --scope contracts
/ic search "type.*=" --scope contracts --language typescript
```

### 6. Adjust Budget for Task Complexity

- Quick questions: `/ic pack "..." --budget 2000`
- Normal tasks: `/ic pack "..."` (default 4000)
- Complex analysis: `/ic pack "..." --budget 8000`

### 7. Use Focus Paths

When working in specific areas:

```bash
/ic pack "test utilities" --focus tests/
/ic search "mock" --focus tests/mocks/
```

---

## Tips and Tricks

### Tip 1: Quick Contract Reference

Get all interfaces quickly:

```bash
/ic search "^interface" --scope contracts --limit 50
```

### Tip 2: Find Recent Changes

Search recently modified code:

```bash
/ic search "your feature" --scope diffs
```

### Tip 3: Understand Dependencies

Map import relationships:

```bash
/ic impact src/index.ts --max-nodes 200
```

### Tip 4: Debug Index Issues

If results seem wrong:

```bash
# Check index stats
/ic stats

# Force re-index
icr index --force
```

### Tip 5: Export Pack for Sharing

Save a context pack:

```bash
/ic pack "authentication overview" > auth-context.md
```

### Tip 6: Session Memory

View what ICR remembers:

```bash
/ic list --filter recent --limit 20
```

### Tip 7: Performance Monitoring

Check response times:

```bash
icr metrics --last 1h
```

---

## Next Steps

- [API_REFERENCE.md](API_REFERENCE.md): Complete tool documentation
- [CONFIGURATION.md](CONFIGURATION.md): Full configuration reference
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md): Common issues and solutions
