# /ic - Infinite Context Runtime Commands

## Overview

The `/ic` command provides access to ICR (Infinite Context Runtime) functionality within Claude Code. ICR implements RLM-inspired unlimited context management, allowing Claude to maintain awareness of large codebases through intelligent context packing, prior retrieval, and ledger-based state tracking.

## Usage

```
/ic <subcommand> [arguments]
```

## Subcommands

### pack

Generate and display the current context pack.

```
/ic pack [--verbose] [--tokens <max>]
```

**Options:**
- `--verbose`: Show detailed pack composition including sources
- `--tokens <max>`: Override maximum token budget (default: 4000)

**Example:**
```
/ic pack --verbose
```

**Output:**
Returns a context pack containing:
- Active priors (recently relevant context)
- Pinned invariants (always-included context)
- Environment summary
- Active file relevance scores

---

### search

Search the ICR environment for relevant context.

```
/ic search <query> [--limit <n>] [--type <type>]
```

**Arguments:**
- `query`: Natural language or keyword search query

**Options:**
- `--limit <n>`: Maximum results to return (default: 10)
- `--type <type>`: Filter by type (file, chunk, memory, decision)

**Example:**
```
/ic search "authentication middleware"
/ic search "database schema" --type file --limit 5
```

**Output:**
Returns ranked search results with:
- Source path or identifier
- Relevance score
- Preview snippet
- Last accessed timestamp

---

### impact

Analyze the impact of changes to specified files.

```
/ic impact <paths...> [--depth <n>]
```

**Arguments:**
- `paths`: One or more file paths to analyze

**Options:**
- `--depth <n>`: Dependency traversal depth (default: 2)

**Example:**
```
/ic impact src/auth/login.py src/auth/session.py
/ic impact ./api/ --depth 3
```

**Output:**
Returns impact analysis including:
- Direct dependents
- Transitive dependents (up to depth)
- Test files affected
- Configuration files affected
- Risk assessment

---

### pin

Pin an invariant to ensure it's always included in context.

```
/ic pin <invariant> [--priority <n>] [--expires <duration>]
```

**Arguments:**
- `invariant`: Text content to pin as invariant

**Options:**
- `--priority <n>`: Priority level 1-10 (default: 5)
- `--expires <duration>`: Auto-expire after duration (e.g., "1h", "7d")

**Example:**
```
/ic pin "Always use TypeScript strict mode"
/ic pin "Database migrations must be backwards compatible" --priority 10
/ic pin "Sprint focus: authentication refactor" --expires 7d
```

**Output:**
Returns confirmation with:
- Assigned invariant ID
- Current priority
- Expiration (if set)
- Total pinned invariants count

---

### unpin

Remove a pinned invariant.

```
/ic unpin <id>
```

**Arguments:**
- `id`: Invariant ID to remove (from `/ic pin` or `/ic status`)

**Example:**
```
/ic unpin inv_a1b2c3d4
```

**Output:**
Confirmation of removal with remaining pinned count.

---

### status

Display ICR system status and statistics.

```
/ic status [--verbose]
```

**Options:**
- `--verbose`: Show detailed statistics

**Example:**
```
/ic status
/ic status --verbose
```

**Output:**
Returns status including:
- ICR version and health
- Database statistics (files indexed, chunks, memories)
- Vector index status
- Active session information
- Pinned invariants list
- Recent ledger entries
- Hook status (if verbose)

---

### ledger

View or manage the session ledger.

```
/ic ledger [--last <n>] [--type <type>]
```

**Options:**
- `--last <n>`: Show last n entries (default: 10)
- `--type <type>`: Filter by type (decision, todo, question, file)

**Example:**
```
/ic ledger --last 20
/ic ledger --type decision
```

**Output:**
Returns ledger entries with:
- Timestamp
- Entry type
- Content
- Associated files

---

### compact

Manually trigger context compaction.

```
/ic compact [--preserve-invariants] [--dry-run]
```

**Options:**
- `--preserve-invariants`: Ensure all pinned invariants survive compaction
- `--dry-run`: Show what would be compacted without executing

**Example:**
```
/ic compact --dry-run
/ic compact --preserve-invariants
```

**Output:**
Returns compaction summary:
- Tokens before/after
- Preserved items
- Discarded items (with reasons)

---

### config

View or modify ICR configuration.

```
/ic config [<key>] [<value>]
```

**Arguments:**
- `key`: Configuration key (optional, lists all if omitted)
- `value`: New value (optional, shows current if omitted)

**Example:**
```
/ic config
/ic config max_context_tokens
/ic config max_context_tokens 6000
```

---

### sync

Manually trigger environment synchronization.

```
/ic sync [--full] [--path <path>]
```

**Options:**
- `--full`: Full re-index instead of incremental
- `--path <path>`: Sync specific path only

**Example:**
```
/ic sync
/ic sync --path ./src/
/ic sync --full
```

---

### clear

Clear ICR state (use with caution).

```
/ic clear <target>
```

**Arguments:**
- `target`: What to clear (priors, ledger, invariants, all)

**Example:**
```
/ic clear priors
/ic clear ledger
```

---

## Integration Notes

### Automatic Injection

When ICR hooks are properly installed, context packs are automatically injected on each prompt via the `UserPromptSubmit` hook. Use `/ic pack` to see what would be injected.

### Ledger Extraction

The `Stop` hook automatically extracts ledger entries from Claude's responses when they follow the structured format:

```
Ledger:
- Decisions: [list]
- Todos: [list]
- Open Questions: [list]
- Files touched: [list]
```

### Fallback Mode

If hooks fail to install, all `/ic` commands remain functional as explicit commands. Use `/ic status` to check hook status.

## Environment Variables

- `ICR_CONFIG_PATH`: Override config location (default: ~/.icr/config.yaml)
- `ICR_DB_PATH`: Override database location (default: ~/.icr/icr.db)
- `ICR_LOG_LEVEL`: Set logging level (DEBUG, INFO, WARNING, ERROR)
- `ICR_DISABLE_HOOKS`: Disable automatic hook injection

## Troubleshooting

### Hooks not working

Run `/ic status --verbose` to check hook installation status. If hooks show as not installed:

```bash
icr install --hooks
```

### Context pack empty

Ensure the environment has been indexed:

```
/ic sync --full
```

### Search returning no results

Check that embeddings are configured:

```
/ic status --verbose
```

Look for "Vector index status" - if not initialized, run:

```bash
icr doctor --fix
```
