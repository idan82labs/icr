# ICR API Reference

Complete reference documentation for all MCP tools exposed by ICR.

---

## Table of Contents

- [Overview](#overview)
- [Memory Tools](#memory-tools)
- [Environment Tools](#environment-tools)
- [Project Tools](#project-tools)
- [RLM Tools](#rlm-tools)
- [Admin Tools](#admin-tools)
- [Error Handling](#error-handling)
- [Rate Limits and Bounds](#rate-limits-and-bounds)

---

## Overview

### Tool Naming Convention

All ICR tools are exposed through MCP with the `icr` namespace:

```
mcp__icr__<tool_name>
```

Example: `mcp__icr__memory_pack`

### Common Patterns

#### Input Validation

All inputs are validated using Pydantic models with explicit constraints:

```json
{
  "query": {
    "type": "string",
    "minLength": 1,
    "maxLength": 2000
  },
  "limit": {
    "type": "integer",
    "minimum": 1,
    "maximum": 100,
    "default": 20
  }
}
```

#### Output Envelope

Successful responses follow this structure:

```json
{
  "success": true,
  "data": { ... },
  "metadata": {
    "duration_ms": 45,
    "tokens_used": 2345
  }
}
```

Error responses:

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Query cannot be empty",
    "details": { ... }
  }
}
```

---

## Memory Tools

### memory_pack

Compile a context pack for a given prompt.

#### Description

Analyzes the prompt, retrieves relevant code snippets using hybrid search, applies diversity selection, and compiles a token-bounded context pack with citations.

#### Input Schema

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prompt` | string | Yes | - | Query or task prompt (1-10000 chars) |
| `repo_root` | string | Yes | - | Absolute path to repository root |
| `mode` | string | No | "auto" | Resolution mode: "auto", "pack", or "rlm" |
| `budget_tokens` | integer | No | 4000 | Token budget (512-12000) |
| `k` | integer | No | 20 | Number of top sources to consider (5-50) |
| `focus_paths` | array[string] | No | [] | Paths to prioritize (max 100) |
| `pinned_only` | boolean | No | false | Only include pinned items |

#### Example Request

```json
{
  "prompt": "How does the authentication system validate JWT tokens?",
  "repo_root": "/home/user/myproject",
  "mode": "auto",
  "budget_tokens": 4000,
  "k": 20,
  "focus_paths": ["src/auth/"]
}
```

#### Example Response

```json
{
  "success": true,
  "data": {
    "mode_used": "pack",
    "pack_content": "# Context Pack\n\n## Sources\n\n### [1] src/auth/jwt.py:validate_token (lines 45-78)\n\n```python\ndef validate_token(token: str) -> TokenPayload:\n    ...\n```\n\n...",
    "sources": [
      {
        "id": "abc123",
        "path": "src/auth/jwt.py",
        "symbol": "validate_token",
        "start_line": 45,
        "end_line": 78,
        "score": 0.92,
        "token_count": 234
      }
    ],
    "citations": [
      {"ref": 1, "path": "src/auth/jwt.py", "score": 0.92}
    ],
    "entropy": 1.8,
    "gating_reason": "low_entropy"
  },
  "metadata": {
    "duration_ms": 145,
    "tokens_used": 2345,
    "tokens_budget": 4000,
    "sources_considered": 50
  }
}
```

---

### memory_pin

Pin a source to ensure it's always included in context packs.

#### Input Schema

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `source_id` | string | Yes | - | Unique source identifier (1-256 chars) |
| `path` | string | Yes | - | Path to the file or resource |
| `label` | string | No | null | Human-readable label (max 256 chars) |
| `ttl_seconds` | integer | No | null | Time-to-live (60-86400 seconds) |

#### Example Request

```json
{
  "source_id": "core-types",
  "path": "src/types/index.ts",
  "label": "Core type definitions",
  "ttl_seconds": null
}
```

#### Example Response

```json
{
  "success": true,
  "data": {
    "source_id": "core-types",
    "path": "src/types/index.ts",
    "label": "Core type definitions",
    "pinned_at": "2024-01-15T10:30:00Z",
    "expires_at": null
  }
}
```

---

### memory_unpin

Remove a pinned source.

#### Input Schema

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `source_id` | string | Yes | - | Source identifier to unpin |

#### Example Request

```json
{
  "source_id": "core-types"
}
```

#### Example Response

```json
{
  "success": true,
  "data": {
    "unpinned": true,
    "source_id": "core-types"
  }
}
```

---

### memory_list

List memory items with optional filtering.

#### Input Schema

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `filter_type` | string | No | "all" | Filter: "all", "pinned", "recent", "stale" |
| `limit` | integer | No | 50 | Maximum items (1-200) |
| `cursor` | string | No | null | Pagination cursor |

#### Example Request

```json
{
  "filter_type": "pinned",
  "limit": 20
}
```

#### Example Response

```json
{
  "success": true,
  "data": {
    "items": [
      {
        "id": "core-types",
        "path": "src/types/index.ts",
        "label": "Core type definitions",
        "type": "pinned",
        "created_at": "2024-01-15T10:30:00Z"
      }
    ],
    "total_count": 5,
    "has_more": false,
    "next_cursor": null
  }
}
```

---

### memory_get

Retrieve details of a specific memory item.

#### Input Schema

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `source_id` | string | Yes | - | Source identifier |
| `include_content` | boolean | No | true | Include full content |

#### Example Response

```json
{
  "success": true,
  "data": {
    "id": "core-types",
    "path": "src/types/index.ts",
    "label": "Core type definitions",
    "type": "pinned",
    "content": "export interface User {\n  id: string;\n  ...\n}",
    "token_count": 456,
    "created_at": "2024-01-15T10:30:00Z"
  }
}
```

---

### memory_stats

Get memory statistics for the current repository.

#### Input Schema

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `include_breakdown` | boolean | No | false | Include detailed breakdown |

#### Example Response

```json
{
  "success": true,
  "data": {
    "repo_id": "a1b2c3d4",
    "files_indexed": 1234,
    "chunks_stored": 15678,
    "vectors_stored": 15678,
    "contracts_detected": 456,
    "pinned_items": 12,
    "memory_usage": {
      "index_size_bytes": 152043520,
      "vector_size_bytes": 93323264,
      "total_bytes": 245366784
    },
    "breakdown": {
      "by_language": {
        "python": 5234,
        "typescript": 8901,
        "javascript": 1543
      },
      "by_type": {
        "function": 8234,
        "class": 1234,
        "method": 4567,
        "interface": 456
      }
    }
  }
}
```

---

## Environment Tools

### env_search

Search the context environment.

#### Input Schema

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | Yes | - | Search query (1-2000 chars) |
| `scope` | string | Yes | - | Scope: "repo", "transcript", "diffs", "contracts", "all" |
| `path_prefix` | string | No | null | Filter by path prefix |
| `language` | string | No | null | Filter by language (max 50 chars) |
| `time_window_seconds` | integer | No | null | Time window (1-604800) |
| `limit` | integer | No | 20 | Maximum results (1-50) |
| `cursor` | string | No | null | Pagination cursor |
| `explain` | boolean | No | false | Include search strategy explanation |

#### Example Request

```json
{
  "query": "error handling",
  "scope": "repo",
  "language": "python",
  "limit": 20,
  "explain": true
}
```

#### Example Response

```json
{
  "success": true,
  "data": {
    "results": [
      {
        "id": "chunk_abc123",
        "path": "src/utils/errors.py",
        "symbol": "handle_error",
        "content_preview": "def handle_error(e: Exception) -> Response:\n    ...",
        "start_line": 15,
        "end_line": 45,
        "score": 0.89,
        "match_type": "hybrid"
      }
    ],
    "total_count": 45,
    "has_more": true,
    "next_cursor": "eyJvZmZzZXQiOjIwfQ==",
    "explanation": {
      "strategy": "hybrid",
      "semantic_matches": 15,
      "lexical_matches": 30,
      "merged_candidates": 45
    }
  }
}
```

---

### env_peek

View specific lines from a file.

#### Input Schema

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `path` | string | Yes | - | File path |
| `start_line` | integer | Yes | - | Starting line (1-indexed) |
| `end_line` | integer | Yes | - | Ending line (1-indexed, inclusive) |
| `max_lines` | integer | No | 200 | Maximum lines to return (1-400) |

#### Example Request

```json
{
  "path": "src/auth/service.py",
  "start_line": 25,
  "end_line": 60,
  "max_lines": 200
}
```

#### Example Response

```json
{
  "success": true,
  "data": {
    "path": "src/auth/service.py",
    "start_line": 25,
    "end_line": 60,
    "actual_end_line": 60,
    "content": "def authenticate(self, token: str) -> User:\n    ...",
    "truncated": false,
    "total_file_lines": 150
  }
}
```

---

### env_slice

Extract a symbol or range from a file.

#### Input Schema

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `path` | string | Yes | - | File path |
| `symbol` | string | No | null | Symbol name to extract (max 256 chars) |
| `start_line` | integer | No | null | Starting line for range slice |
| `end_line` | integer | No | null | Ending line for range slice |
| `context_lines` | integer | No | 3 | Context lines around slice (0-20) |

#### Example Request

```json
{
  "path": "src/auth/service.py",
  "symbol": "AuthService.authenticate",
  "context_lines": 5
}
```

#### Example Response

```json
{
  "success": true,
  "data": {
    "path": "src/auth/service.py",
    "symbol": "AuthService.authenticate",
    "symbol_type": "method",
    "start_line": 25,
    "end_line": 60,
    "content": "    def authenticate(self, token: str) -> User:\n        ...",
    "context_before": "class AuthService:\n    \"\"\"Authentication service.\"\"\"\n",
    "context_after": "\n    def logout(self, user_id: str) -> None:\n        ..."
  }
}
```

---

### env_aggregate

Perform non-generative aggregation on inputs.

#### Input Schema

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `op` | string | Yes | - | Operation: "extract_regex", "unique", "sort", "group_by", "count", "top_k", "join_on", "diff_sets" |
| `inputs` | array[string] | Yes | - | Input strings (1-200 items) |
| `params` | object | No | {} | Operation-specific parameters |
| `limit` | integer | No | 100 | Maximum results (1-200) |

#### Operations

| Operation | Parameters | Description |
|-----------|------------|-------------|
| `extract_regex` | `pattern: string` | Extract matches for regex pattern |
| `unique` | - | Deduplicate input items |
| `sort` | `key?: string, reverse?: bool` | Sort items |
| `group_by` | `key: string` | Group by extracted key |
| `count` | - | Count occurrences |
| `top_k` | `k: int, key?: string` | Select top k items |
| `join_on` | `key: string` | Join items on common key |
| `diff_sets` | - | Compute set difference |

#### Example Request

```json
{
  "op": "extract_regex",
  "inputs": [
    "def foo(): pass",
    "def bar(): pass",
    "class Baz: pass"
  ],
  "params": {
    "pattern": "def (\\w+)"
  },
  "limit": 100
}
```

#### Example Response

```json
{
  "success": true,
  "data": {
    "operation": "extract_regex",
    "results": ["foo", "bar"],
    "count": 2
  }
}
```

---

## Project Tools

### project_map

Generate a structural map of the repository.

#### Input Schema

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `repo_root` | string | Yes | - | Repository root path |
| `depth` | integer | No | 3 | Directory depth (1-10) |
| `include_patterns` | array[string] | No | [] | Glob patterns to include (max 50) |
| `exclude_patterns` | array[string] | No | [] | Glob patterns to exclude (max 50) |
| `include_stats` | boolean | No | false | Include file statistics |

#### Example Request

```json
{
  "repo_root": "/home/user/myproject",
  "depth": 3,
  "include_stats": true
}
```

#### Example Response

```json
{
  "success": true,
  "data": {
    "tree": {
      "name": "myproject",
      "type": "directory",
      "children": [
        {
          "name": "src",
          "type": "directory",
          "children": [
            {
              "name": "auth",
              "type": "directory",
              "children": [
                {
                  "name": "service.py",
                  "type": "file",
                  "stats": {
                    "size_bytes": 4567,
                    "lines": 150,
                    "language": "python"
                  }
                }
              ]
            }
          ]
        }
      ]
    },
    "summary": {
      "total_files": 234,
      "total_directories": 45,
      "languages": {
        "python": 89,
        "typescript": 123,
        "json": 22
      }
    }
  }
}
```

---

### project_symbol_search

Search for symbols (functions, classes, etc.) in the repository.

#### Input Schema

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | Yes | - | Symbol search query (1-500 chars) |
| `repo_root` | string | Yes | - | Repository root path |
| `symbol_types` | array[string] | No | [] | Filter by type: "function", "class", "method", "variable", "type", "interface" |
| `languages` | array[string] | No | [] | Filter by languages (max 20) |
| `limit` | integer | No | 30 | Maximum results (1-100) |
| `cursor` | string | No | null | Pagination cursor |

#### Example Request

```json
{
  "query": "authenticate",
  "repo_root": "/home/user/myproject",
  "symbol_types": ["function", "method"],
  "languages": ["python"],
  "limit": 20
}
```

#### Example Response

```json
{
  "success": true,
  "data": {
    "symbols": [
      {
        "name": "authenticate",
        "qualified_name": "auth.service.AuthService.authenticate",
        "type": "method",
        "path": "src/auth/service.py",
        "start_line": 25,
        "end_line": 60,
        "signature": "def authenticate(self, token: str) -> User",
        "docstring": "Authenticate a user by token.",
        "score": 0.95
      }
    ],
    "total_count": 5,
    "has_more": false
  }
}
```

---

### project_impact

Analyze the impact of changes to files.

#### Input Schema

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `changed_paths` | array[string] | Yes | - | Changed file paths (1-200 items) |
| `query` | string | No | null | Focus query (max 1000 chars) |
| `max_nodes` | integer | No | 100 | Maximum graph nodes (10-500) |
| `max_edges` | integer | No | 500 | Maximum graph edges (10-2000) |

#### Example Request

```json
{
  "changed_paths": ["src/types/user.ts"],
  "query": "What components use UserType?",
  "max_nodes": 100
}
```

#### Example Response

```json
{
  "success": true,
  "data": {
    "impact_score": 0.78,
    "direct_dependents": [
      {
        "path": "src/services/user_service.ts",
        "references": 12,
        "impact_level": "high"
      },
      {
        "path": "src/api/users.ts",
        "references": 8,
        "impact_level": "high"
      }
    ],
    "transitive_dependents": [
      {
        "path": "src/api/auth.ts",
        "depth": 2,
        "impact_level": "medium"
      }
    ],
    "contract_impact": {
      "modified_contracts": ["UserType", "UserInput"],
      "affected_files": 8,
      "breaking_change_risk": "high"
    },
    "suggested_review": [
      "src/services/user_service.ts",
      "src/api/users.ts",
      "tests/test_user.ts"
    ],
    "graph": {
      "nodes": [...],
      "edges": [...]
    }
  }
}
```

---

### project_commands

Discover build/test/run commands for the project.

#### Input Schema

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `repo_root` | string | Yes | - | Repository root path |
| `command_type` | string | No | "all" | Type: "build", "test", "lint", "format", "run", "all" |

#### Example Response

```json
{
  "success": true,
  "data": {
    "detected_tools": ["npm", "jest", "eslint"],
    "commands": {
      "build": {
        "command": "npm run build",
        "source": "package.json"
      },
      "test": {
        "command": "npm test",
        "source": "package.json"
      },
      "lint": {
        "command": "npm run lint",
        "source": "package.json"
      }
    }
  }
}
```

---

## RLM Tools

### rlm_plan

Generate a retrieval plan for complex queries.

#### Input Schema

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `task` | string | Yes | - | Task description (1-5000 chars) |
| `scope` | string | Yes | - | Scope: "repo", "contracts", "diffs", "all" |
| `budget` | object | No | (see below) | Budget constraints |

**Budget Defaults:**
```json
{
  "max_steps": 10,
  "max_peek_lines": 1000,
  "max_candidates": 100
}
```

#### Example Request

```json
{
  "task": "Find all places where user authentication is bypassed",
  "scope": "repo",
  "budget": {
    "max_steps": 12,
    "max_peek_lines": 1200,
    "max_candidates": 50
  }
}
```

#### Example Response

```json
{
  "success": true,
  "data": {
    "plan": {
      "steps": [
        {
          "step": 1,
          "action": "env_search",
          "params": {
            "query": "authenticate bypass skip",
            "scope": "repo"
          },
          "rationale": "Find authentication-related code with bypass patterns"
        },
        {
          "step": 2,
          "action": "env_search",
          "params": {
            "query": "@skip_auth @no_auth",
            "scope": "repo"
          },
          "rationale": "Search for decorator-based auth skipping"
        },
        {
          "step": 3,
          "action": "env_peek",
          "params": {
            "paths": ["<from_step_1>", "<from_step_2>"]
          },
          "rationale": "Inspect found locations"
        }
      ],
      "estimated_tokens": 2500,
      "estimated_time_ms": 3000
    }
  }
}
```

---

### rlm_map_reduce

Execute a map-reduce operation over multiple sources.

#### Input Schema

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `task` | string | Yes | - | Task description (1-5000 chars) |
| `sources` | array[string] | Yes | - | Source identifiers (1-100 items) |
| `map_prompt` | string | Yes | - | Map phase prompt (1-2000 chars) |
| `reduce_prompt` | string | Yes | - | Reduce phase prompt (1-2000 chars) |
| `max_parallel` | integer | No | 5 | Maximum parallel operations (1-20) |

#### Example Request

```json
{
  "task": "Summarize all error handling patterns",
  "sources": ["src/api/users.py", "src/api/auth.py", "src/api/products.py"],
  "map_prompt": "Extract error handling patterns from this file: {content}",
  "reduce_prompt": "Combine these error handling patterns into a summary: {results}",
  "max_parallel": 5
}
```

#### Example Response

```json
{
  "success": true,
  "data": {
    "map_results": [
      {
        "source": "src/api/users.py",
        "result": "Uses try-except with custom UserError..."
      }
    ],
    "reduce_result": "The codebase uses three main error handling patterns...",
    "sources_processed": 3,
    "sources_failed": 0
  },
  "metadata": {
    "duration_ms": 4500,
    "map_duration_ms": 3000,
    "reduce_duration_ms": 1500
  }
}
```

---

## Admin Tools

### admin_ping

Health check for the ICR service.

#### Input Schema

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `echo` | string | No | null | String to echo back (max 256 chars) |
| `include_diagnostics` | boolean | No | false | Include diagnostics |

#### Example Request

```json
{
  "echo": "hello",
  "include_diagnostics": true
}
```

#### Example Response

```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "echo": "hello",
    "version": "0.1.0",
    "uptime_seconds": 3600,
    "diagnostics": {
      "index_status": "ready",
      "embedding_backend": "local_onnx",
      "last_index_update": "2024-01-15T10:30:00Z",
      "pending_updates": 0
    }
  }
}
```

---

## Error Handling

### Error Codes

| Code | HTTP Equivalent | Description |
|------|-----------------|-------------|
| `VALIDATION_ERROR` | 400 | Invalid input parameters |
| `NOT_FOUND` | 404 | Resource not found |
| `TIMEOUT` | 408 | Operation timed out |
| `RATE_LIMITED` | 429 | Rate limit exceeded |
| `INTERNAL_ERROR` | 500 | Internal server error |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |

### Error Response Format

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Human-readable error message",
    "details": {
      "field": "limit",
      "constraint": "maximum",
      "value": 150,
      "allowed": 100
    }
  }
}
```

### Handling Errors

```python
result = await mcp__icr__memory_pack(...)

if not result["success"]:
    error = result["error"]
    if error["code"] == "TIMEOUT":
        # Retry with smaller budget
        ...
    elif error["code"] == "VALIDATION_ERROR":
        # Fix input parameters
        ...
```

---

## Rate Limits and Bounds

### Per-Operation Bounds

| Tool | Parameter | Limit |
|------|-----------|-------|
| `memory_pack` | `budget_tokens` | 512-12000 |
| `memory_pack` | `k` | 5-50 |
| `env_search` | `limit` | 1-50 |
| `env_peek` | `max_lines` | 1-400 |
| `env_aggregate` | `inputs` | 1-200 |
| `project_impact` | `max_nodes` | 10-500 |
| `rlm_map_reduce` | `sources` | 1-100 |

### Session Bounds (RLM Mode)

| Metric | Limit |
|--------|-------|
| Max steps | 12 |
| Max peek lines (cumulative) | 1200 |
| Max candidates | 50 |

### Time Limits

| Operation | Timeout |
|-----------|---------|
| Pack mode (total) | 8 seconds |
| RLM mode (total) | 20 seconds |
| Single tool call | 5 seconds |
| Hard abort | 30 seconds |

### Performance Targets

| Operation | P50 | P95 |
|-----------|-----|-----|
| ANN lookup (given embedding) | 15ms | 40ms |
| Query embedding (local) | 20ms | 50ms |
| End-to-end semantic | 40ms | 100ms |
| Hybrid search (full) | 70ms | 150ms |
| Pack compilation | 200ms | 500ms |
| RLM plan generation | 300ms | 800ms |
| Map-reduce (total) | 2s | 8s |

---

## Next Steps

- [CONFIGURATION.md](CONFIGURATION.md): Configuration options
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md): Common issues
- [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md): Contributing to ICR
