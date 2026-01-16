# ICR Production Roadmap

## Project Positioning

**ICR = Intelligent Context Retrieval**
- Smart context packing for Claude Code
- Semantic search + budget-aware compilation
- RLM-boosted for scattered results

**What RLM Actually Is**: Query expansion with entropy-based gating. NOT recursive LLM calls.

## Current State

| Component | Status | Notes |
|-----------|--------|-------|
| Hybrid Search (semantic + BM25) | ✅ Working | Core value |
| ONNX Embeddings (local) | ✅ Working | No API calls |
| Knapsack Pack Compiler | ✅ Working | Core value |
| Contract Detection | ✅ Working | Interfaces/types prioritized |
| .gitignore/.icrignore | ✅ Working | - |
| RLM Query Expansion | ✅ Wired | Entropy-gated |
| MCP Tools | ✅ Working | 12 tools exposed |
| Skill Integration | ⚠️ Partial | Needs better trigger rules |
| File Watcher | ⚠️ Built | Not auto-started |
| Install Experience | ⚠️ Partial | Requires restart |

## Target User Experience

```
User installs ICR → ICR coexists with native tools →
Claude uses ICR for conceptual questions →
Claude uses native grep/glob for targeted queries →
User sees what ICR found and why
```

**Key principle**: ICR complements native tools, doesn't replace them.

---

## Phase 1: Wire RLM to MCP (Core Value)

### 1.1 Automatic Entropy-Based Gating

**Current:** User must manually call `rlm_plan` then execute sub-queries.

**Target:** Single `memory_pack` call automatically:
1. Does initial retrieval
2. Checks entropy
3. If high → triggers RLM decomposition
4. Executes sub-queries iteratively
5. Returns aggregated context

**Implementation:**
```python
# ic-mcp/src/ic_mcp/tools/memory.py

async def memory_pack(self, input_data, request_id):
    # Initial retrieval
    initial_results = await self._retrieve(input_data.prompt)

    # Check entropy
    entropy = self._calculate_entropy(initial_results)

    if entropy > self.config.rlm.entropy_threshold:
        # Activate RLM
        plan = self.planner.create_plan(input_data.prompt, initial_results)

        for iteration in range(plan.max_iterations):
            for subquery in plan.pending_subqueries():
                sub_results = await self._retrieve(subquery.query)
                plan.add_results(subquery, sub_results)

            if plan.is_confident():
                break

        # Aggregate all results
        final_results = self.aggregator.aggregate(plan)
    else:
        final_results = initial_results

    # Pack within budget
    return self._compile_pack(final_results, input_data.budget_tokens)
```

**Files to modify:**
- `ic-mcp/src/ic_mcp/tools/memory.py` - Add RLM orchestration
- `icd/src/icd/rlm/planner.py` - Expose async interface
- `icd/src/icd/rlm/aggregator.py` - Expose async interface

### 1.2 Sub-Query Type System

Leverage existing `QueryType` enum:
```python
class QueryType(Enum):
    DEFINITION = "definition"      # "What is X?"
    USAGE = "usage"                # "Where is X used?"
    IMPLEMENTATION = "implementation"  # "How does X work?"
    RELATED = "related"            # "What's related to X?"
    CONTRACT = "contract"          # "What interfaces does X have?"
    TEST = "test"                  # "What tests cover X?"
```

**Auto-detect query intent:**
- "how does X work" → IMPLEMENTATION + DEFINITION
- "trace X flow" → USAGE + IMPLEMENTATION
- "plan feature for X" → CONTRACT + IMPLEMENTATION + RELATED + TEST

---

## Phase 2: Skill Integration with Native Tools

### 2.1 Skill Design Philosophy

**Don't replace Claude Code tools. Complement them.**

| Query Type | Best Tool |
|------------|-----------|
| "Find file named X" | Native Glob |
| "Search for string X" | Native Grep |
| "Read file X" | Native Read |
| "How does X work?" | **ICR** |
| "Trace X flow through system" | **ICR + RLM** |
| "Plan new feature Y" | **ICR + RLM** |
| "What would break if I change X?" | **ICR (impact)** |

### 2.2 Smart Skill Trigger

Update `skills/codebase-memory/SKILL.md`:

```markdown
---
name: icr-codebase-memory
description: Use ICR for understanding code by meaning, not keywords.
             Triggers on: "how does", "trace", "plan feature", "what uses",
             "explain the flow", "understand the architecture".
             Does NOT trigger on: "find file", "search for", "read file".
---

# When to Use ICR vs Native Tools

## Use ICR when:
- User asks "how does X work?" (conceptual)
- User asks "trace X flow" (multi-file)
- User asks "plan feature Y" (need context)
- User asks "what would break if I change X" (impact)
- User is NEW to codebase (onboarding)

## Use Native Tools when:
- User knows exact file/function name
- User wants literal string search
- User wants to read specific file
- Simple, targeted queries

## ICR Tool Selection

1. **Simple conceptual question** → `icr__env_search`
2. **Complex flow/architecture question** → `icr__memory_pack` (auto-RLM)
3. **Impact analysis** → `icr__project_impact`
4. **Find symbol by name** → `icr__project_symbol_search`
```

### 2.3 Hook Integration (Using Claude Code Hooks System)

Claude Code provides 10 hook events. The most useful for ICR:

**Available Hooks:**
1. `SessionStart` - Initialize ICR context when session starts
2. `UserPromptSubmit` - Auto-inject ICR context based on user prompts
3. `PreCompact` - Preserve context before compaction (IMPLEMENTED)
4. `Stop` - Persist important context at end of turns
5. `PreToolUse` - Add context before tool execution

**Configuration in `.claude/settings.json`:**
```json
{
  "hooks": {
    "SessionStart": [
      {
        "hooks": [{
          "type": "command",
          "command": ".icr/venv/bin/python .icr/hooks/session_start.py"
        }]
      }
    ],
    "UserPromptSubmit": [
      {
        "hooks": [{
          "type": "command",
          "command": ".icr/venv/bin/python .icr/hooks/prompt_submit.py"
        }]
      }
    ],
    "PreCompact": [
      {
        "hooks": [{
          "type": "command",
          "command": ".icr/venv/bin/python .icr/hooks/precompact.py"
        }]
      }
    ]
  }
}
```

**UserPromptSubmit Hook Implementation:**
```python
# .icr/hooks/prompt_submit.py

import json
import re
import sys

def should_inject_context(prompt: str) -> bool:
    """Detect if prompt would benefit from ICR context."""
    patterns = [
        r"how does .+ work",
        r"trace .+ flow",
        r"plan .+ feature",
        r"explain .+ architecture",
        r"what uses .+",
        r"understand .+",
    ]
    return any(re.search(p, prompt.lower()) for p in patterns)

def main():
    input_data = json.loads(sys.stdin.read())
    prompt = input_data.get("prompt", "")

    if should_inject_context(prompt):
        # Call ICR to get context pack
        # Output additionalContext to stdout as JSON
        result = {"additionalContext": "ICR context here..."}
        print(json.dumps(result))

if __name__ == "__main__":
    main()
```

**SessionStart Hook for Index Freshness:**
```python
# .icr/hooks/session_start.py
# Check index freshness, suggest reindex if stale
```

---

## Phase 3: Installation & Setup

### 3.1 One-Command Install

```bash
curl -fsSL https://raw.githubusercontent.com/idan82labs/icr/main/install.sh | bash
```

This should:
1. ✅ Create `.icr/` with venv
2. ✅ Install packages from GitHub
3. ✅ Create `.mcp.json`
4. ✅ Index codebase
5. ✅ Create default `.icrignore`
6. **NEW:** Copy skill files to project
7. **NEW:** Register MCP server with Claude Code (if possible)

### 3.2 No-Restart Installation

Research Claude Code MCP hot-reload:
- Can we trigger MCP server discovery without restart?
- If not, document clearly: "Restart Claude Code to activate"

### 3.3 Update Detection

Add version check to MCP server startup:
```python
async def check_for_updates():
    # Check GitHub for newer version
    # Notify user if update available
```

---

## Phase 4: Observability & Trust

### 4.1 Show What ICR Found

Current pack header:
```
**Sources Retrieved (5 files):**
  - src/auth/oauth.ts (score: 0.87)
```

Enhanced header:
```
**ICR Analysis**
Mode: RLM (high entropy detected)
Sub-queries executed:
  1. "find authentication entry point" → 3 files
  2. "find token validation" → 2 files
  3. "find session management" → 2 files

**Sources Retrieved (7 files):**
  - src/auth/oauth.ts (score: 0.87, matched: "entry point")
  - src/auth/token.ts (score: 0.82, matched: "token validation")
  ...
```

### 4.2 Confidence Indicator

Show confidence in pack:
```
**Confidence: 0.85** (high - clear matches found)
```

vs

```
**Confidence: 0.42** (low - results scattered, consider refining query)
```

### 4.3 Logging & Diagnostics

```bash
# View what ICR did
cat .icr/mcp.log

# Show index stats
.icr/venv/bin/icd stats
```

---

## Phase 5: Testing & Validation

### 5.1 Best Use Cases for ICR

| Use Case | Without ICR | With ICR |
|----------|-------------|----------|
| **Onboarding to new codebase** | Manual file exploration, grep | "How does auth work?" → instant context |
| **Planning new feature** | Read many files manually | "Plan notification feature" → relevant patterns |
| **Tracing data flow** | grep + manual following | "Trace order flow" → full path |
| **Impact analysis** | Find references manually | "What breaks if I change UserService?" |
| **Understanding patterns** | Search for examples | "How do other endpoints handle errors?" |

### 5.2 Test Scenarios

**Scenario 1: Onboarding**
- Clone unfamiliar repo
- Ask "How does the authentication system work?"
- Compare: grep "auth" vs ICR

**Scenario 2: Feature Planning**
- Pick existing project
- Ask "Plan a notification system feature"
- Compare: manual exploration vs ICR

**Scenario 3: Impact Analysis**
- Pick a core file
- Ask "What would break if I refactor UserService?"
- Compare: find references vs ICR

**Scenario 4: Large Codebase**
- Clone large OSS repo (React, Django, etc.)
- Ask complex architecture question
- Measure: time to understanding

---

## Implementation Priority

| Phase | Effort | Impact | Priority |
|-------|--------|--------|----------|
| 1.1 RLM Auto-Gating | High | High | **P0** |
| 2.2 Smart Skill Trigger | Medium | High | **P0** |
| 4.1 Show What ICR Found | Low | High | **P1** |
| 1.2 Sub-Query Types | Medium | Medium | **P1** |
| 3.1 One-Command Install | Low | Medium | **P2** |
| 2.3 Hook Integration | Medium | Medium | **P2** |
| 4.2 Confidence Indicator | Low | Low | **P3** |

---

---

## Phase 6: Context Compaction Integration (The Killer Feature)

### The Core Insight

**ICR's biggest value isn't search - it's context persistence.**

Claude Code's context window is finite. When it compacts:
- Recent code discussions are summarized/lost
- File contents are dropped
- The "working memory" of the session resets

**ICR can make context feel infinite** by:
1. Persisting important context to disk (indexed)
2. Reconstructing relevant context on demand
3. Surviving compaction with full retrieval capability

### 6.1 Pre-Compaction Hook

When context approaches limit (~80%):

```python
# ic-claude/hooks/pre_compact.py

async def on_pre_compact(session_context):
    """Called before Claude Code compacts context."""

    # Extract key entities from current context
    entities = extract_entities(session_context)
    # Files discussed, functions modified, decisions made

    # Create ICR memory pins for important context
    for entity in entities:
        await icr_pin(
            path=entity.path,
            label=entity.summary,
            ttl=None  # Persist indefinitely
        )

    # Index any new/modified files
    await icr_index_modified()

    return {
        "preserved_entities": len(entities),
        "message": f"ICR preserved {len(entities)} context items"
    }
```

### 6.2 Post-Compaction Context Injection

After compaction, inject ICR retrieval capability:

```python
# ic-claude/hooks/post_compact.py

async def on_post_compact():
    """Called after compaction to restore context access."""

    # Get summary of what's preserved
    stats = await icr_memory_stats()

    # Create context restoration header
    header = f"""
## ICR Context Available

This session has ICR enabled with {stats.total_items} indexed items.
{stats.pinned_count} items were pinned from previous context.

To restore context, I can:
- Search for specific code: "ICR find authentication handling"
- Get pinned context: "ICR show pinned items"
- Retrieve by file: "ICR get src/auth/handler.ts"

Previous session context is preserved and retrievable.
"""

    return {"additionalContext": header}
```

### 6.3 Automatic Context Reconstruction

When user asks about something from pre-compaction:

```python
# Detect "continuation" queries
continuation_patterns = [
    r"what was that .+ we discussed",
    r"the .+ from earlier",
    r"continue with .+",
    r"back to the .+",
    r"that function",
    r"the code we",
]

async def handle_continuation_query(query):
    # Check if this references pre-compaction context
    if is_continuation_query(query):
        # Retrieve from ICR
        pack = await icr_memory_pack(
            query,
            pinned_only=True,  # Prioritize pinned (pre-compaction) items
            budget=4000
        )
        return pack

    return None
```

### 6.4 The "Endless Context" Experience

**Before ICR:**
```
User (turn 1): "Let's work on the auth system"
[... 50 turns of discussion ...]
[COMPACTION]
User (turn 51): "What was that validation function?"
Claude: "I don't have context about previous validation discussions."
```

**With ICR:**
```
User (turn 1): "Let's work on the auth system"
[... 50 turns of discussion ...]
[COMPACTION - ICR preserves key context]
User (turn 51): "What was that validation function?"
Claude: [ICR retrieves pinned context]
         "The validation function we discussed was validateToken()
          in src/auth/validator.ts:45. Here's the code..."
```

### 6.5 Implementation Priority

This is **THE killer feature** that differentiates ICR from simple RAG:

| Feature | Value | Effort | Priority |
|---------|-------|--------|----------|
| Pre-compaction pinning | Very High | Medium | **P0** |
| Post-compaction header | High | Low | **P0** |
| Continuation query detection | High | Medium | **P1** |
| Auto-index on file modify | Medium | Low | **P1** |

### 6.6 User Experience

```
[Session starts]
ICR: Indexed 150 files, ready for retrieval

[50 turns later, context at 85%]
ICR: ⚡ Preserving 12 key context items before compaction

[Compaction occurs]
ICR: Context compacted. 12 items preserved, full retrieval available.

[User continues]
User: "What was that auth handler?"
Claude: [Retrieves from ICR] "Here's the auth handler we discussed..."
```

---

## Success Metrics

1. **Adoption**: Users install and keep using ICR
2. **Accuracy**: ICR finds relevant code users couldn't find with grep
3. **Speed**: Complex questions answered faster than manual exploration
4. **Trust**: Users understand why ICR returned what it did
5. **Context Survival**: Users can continue work after compaction seamlessly

---

## Next Steps

1. [ ] Wire RLM planner to memory_pack (Phase 1.1)
2. [ ] Update skill with smart trigger rules (Phase 2.2)
3. [ ] Add RLM mode indicator to pack output (Phase 4.1)
4. [ ] Test on real projects (Phase 5.2)
5. [ ] Document best practices
