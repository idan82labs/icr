---
name: icr-codebase-memory
description: Use ICR tools to search and understand the codebase. Invoke when user asks about code, architecture, how things work, or needs context about the project.
---

# ICR Codebase Memory

When the user asks questions about the codebase, use ICR MCP tools to find relevant code before answering.

## When to Use ICR

Use ICR tools when the user:
- Asks "how does X work?"
- Asks "where is X implemented?"
- Asks "find all usages of X"
- Asks about architecture or design patterns
- Needs context before making changes
- Asks to explain code they haven't shown you

## Available Tools

### `icr__project_symbol_search`
Find functions, classes, methods by name.
```json
{"query": "handleAuth", "repo_root": "."}
```

### `icr__env_peek`
Read specific lines from a file.
```json
{"path": "src/auth.ts", "start_line": 10, "end_line": 50, "repo_root": "."}
```

### `icr__env_search`
Search across the codebase.
```json
{"query": "authentication flow", "scope": "code", "repo_root": "."}
```

### `icr__memory_pack`
Get compiled context for a query (requires indexed repo).
```json
{"prompt": "How does the auth system work?", "repo_root": ".", "budget_tokens": 4000}
```

### `icr__project_map`
Show project structure.
```json
{"repo_root": ".", "depth": 2, "include_files": true}
```

### `icr__project_commands`
Find available build/test commands.
```json
{"repo_root": "."}
```

## Workflow

1. **Understand the question** - What does the user want to know?
2. **Search first** - Use `project_symbol_search` or `env_search` to find relevant code
3. **Read the code** - Use `env_peek` to read the actual implementation
4. **Explain with context** - Reference specific files and line numbers

## Example

User: "How does the login flow work?"

1. Search: `icr__project_symbol_search(query="login")`
2. Found: `LoginController` in `src/auth/login.ts:45`
3. Read: `icr__env_peek(path="src/auth/login.ts", start_line=45, end_line=100)`
4. Explain the code with file references

## Tips

- Always cite file paths and line numbers: `src/auth/login.ts:45`
- If ICR isn't set up, suggest `/icr:setup`
- For complex questions, search multiple times with different queries
- Use `project_map` first if you don't know the project structure
