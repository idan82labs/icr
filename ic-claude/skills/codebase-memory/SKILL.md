---
name: icr-codebase-memory
description: Semantic code search and context packing. Use for conceptual questions about unfamiliar codebases.
triggers:
  patterns:
    - "how does .+ work"
    - "explain .+ architecture"
    - "trace .+ flow"
    - "what uses .+"
    - "plan .+ feature"
    - "understand .+"
    - "what would break if"
    - "impact of changing"
  anti_patterns:
    - "find file named"
    - "search for string"
    - "read file"
    - "class \\w+"
    - "function \\w+"
---

# ICR Codebase Memory

Use ICR tools for **semantic** code search - finding code by meaning, not just keywords.

## Automatic Trigger Rules

ICR is **automatically suggested** when the user asks:
- "How does X work?" - Conceptual understanding
- "Trace X flow" - Multi-file analysis
- "Plan feature Y" - Needs broad context
- "What would break if I change X?" - Impact analysis
- "Explain the architecture of X" - Structural questions
- "What uses X?" - Dependency discovery
- "Understand how X works" - Learning unfamiliar code

ICR is **NOT triggered** when:
- User mentions specific file paths - Use Read
- User knows exact symbol name - Use Grep
- User wants file patterns - Use Glob
- Query contains exact error messages - Use Grep

## Decision Tree

```
User Query
    |
    +-- Contains file path? --> Read
    |
    +-- Contains exact symbol (CamelCase/snake_case)? --> Grep
    |
    +-- Contains "find file" or "*.ext"? --> Glob
    |
    +-- Is conceptual (how/why/explain/trace)? --> ICR
    |
    +-- Default: Let Claude decide
```

## When to Use ICR vs Native Tools

### Use ICR when:
- User asks "how does X work?" (conceptual)
- User asks about architecture or patterns
- User is exploring unfamiliar code
- Keywords would return too many/wrong results
- Question spans multiple files
- Understanding intent matters more than exact text

### Use Native Tools when:
- User knows the exact symbol name - use Grep
- User wants file patterns - use Glob
- User wants to read a specific file - use Read
- Simple, targeted queries
- Looking for exact string matches

**ICR complements native tools, doesn't replace them.**

## Available Tools

### `icr__memory_pack` (Primary)
Get compiled context for a query. Uses hybrid search + knapsack packing.

```json
{
  "prompt": "How does the auth system work?",
  "repo_root": ".",
  "budget_tokens": 4000
}
```

Returns: Ranked code chunks with file citations and confidence score.

### `icr__env_search`
Search across the codebase semantically.

```json
{
  "query": "authentication flow",
  "scope": "code",
  "repo_root": "."
}
```

### `icr__project_symbol_search`
Find functions/classes by name (when you know the name).

```json
{
  "query": "handleAuth",
  "repo_root": "."
}
```

### `icr__env_peek`
Read specific lines from a file.

```json
{
  "path": "src/auth.ts",
  "start_line": 10,
  "end_line": 50,
  "repo_root": "."
}
```

### `icr__project_map`
Show project structure.

```json
{
  "repo_root": ".",
  "depth": 2
}
```

## Workflow

1. **Assess the question** - Is this conceptual or targeted?
2. **Choose tool** - Conceptual -> ICR, Known target -> Native
3. **Search** - Use `memory_pack` or `env_search`
4. **Check confidence** - Review the confidence score in results
5. **Cite** - Reference specific files and line numbers

## Example: Conceptual Question

User: "How does the payment flow work?"

```
1. icr__memory_pack(prompt="payment processing flow", budget_tokens=4000)
2. Review returned chunks and confidence score
3. If confidence is low, try more specific sub-queries
4. Explain with citations: "The payment flow starts in `src/payments/processor.ts:45`..."
```

## Example: Known Target

User: "Find the UserService class"

```
1. Use native Grep: pattern="class UserService"
   (Faster and more precise than ICR for exact matches)
```

## Example: Impact Analysis

User: "What would break if I remove the validateUser function?"

```
1. icr__memory_pack(prompt="validateUser function usage and dependencies", budget_tokens=6000)
2. Review the callers and dependent code
3. List affected files and potential breaking changes
```

## Confidence Interpretation

The pack output includes a confidence score:
- **High (0.75+)** - Clear matches found, results are reliable
- **Medium (0.50-0.74)** - Results may need refinement, consider narrower queries
- **Low (<0.50)** - Consider more specific queries or use native tools

## Tips

- Always cite file paths and line numbers
- For complex questions, search multiple times with refined queries
- If results seem scattered (high entropy), ICR will auto-decompose into sub-queries
- Check `icr__memory_stats` to verify index is up to date
- Low confidence? Try breaking down into smaller, focused questions
