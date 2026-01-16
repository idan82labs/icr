---
name: icr-codebase-memory
description: Semantic code search and context packing. Use for conceptual questions about unfamiliar codebases. For known symbols or file patterns, prefer native Grep/Glob.
---

# ICR Codebase Memory

Use ICR tools for **semantic** code search - finding code by meaning, not just keywords.

## When to Use ICR vs Native Tools

### Use ICR when:
- User asks "how does X work?" (conceptual)
- User asks about architecture or patterns
- User is exploring unfamiliar code
- Keywords would return too many/wrong results

### Use Native Tools when:
- User knows the exact symbol name → use Grep
- User wants file patterns → use Glob
- User wants to read a specific file → use Read
- Simple, targeted queries

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

Returns: Ranked code chunks with file citations.

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
2. **Choose tool** - Conceptual → ICR, Known target → Native
3. **Search** - Use `memory_pack` or `env_search`
4. **Cite** - Reference specific files and line numbers

## Example: Conceptual Question

User: "How does the payment flow work?"

```
1. icr__memory_pack(prompt="payment processing flow", budget_tokens=4000)
2. Review returned chunks
3. Explain with citations: "The payment flow starts in `src/payments/processor.ts:45`..."
```

## Example: Known Target

User: "Find the UserService class"

```
1. Use native Grep: pattern="class UserService"
   (Faster and more precise than ICR for exact matches)
```

## Tips

- Always cite file paths and line numbers
- For complex questions, search multiple times with refined queries
- If results seem scattered, try more specific sub-questions
- Check `icr__memory_stats` to verify index is up to date
