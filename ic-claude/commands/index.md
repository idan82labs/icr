---
description: Index the current project for smart code retrieval
allowed-tools: Bash, Read
---

# ICR Index

Index the current project so ICR can provide smart code retrieval.

## Steps

1. Check if ICR is set up (`.icr/venv/bin/icd` exists)
   - If not, tell user to run `/icr:setup` first

2. Run the indexer:

```bash
.icr/venv/bin/icd index --repo-root .
```

3. Show progress and summary when complete

## What Gets Indexed

- All source code files (Python, TypeScript, JavaScript, Go, Rust, etc.)
- Contracts and interfaces get special priority
- Documentation files (markdown)
- Configuration files

## What's Excluded

- `.git/`, `node_modules/`, `__pycache__/`, `venv/`
- Binary files
- Files over 500KB

## Success Message

"Indexed X files (Y chunks). ICR is ready! Just ask me questions about your codebase."
