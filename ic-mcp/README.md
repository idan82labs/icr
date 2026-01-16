# IC-MCP: Model Context Protocol Server for ICR

MCP Server component that exposes safe, bounded tools for Claude Code integration.

## Installation

```bash
pip install ic-mcp
```

## Usage

Run the MCP server:

```bash
ic-mcp
```

Or configure in Claude Code's settings.

## Available Tools

- `env_search` - Unified environment search across code, memory, and context
- `env_peek` - Quick read of specific items with line numbers
- `project_symbol_search` - Search for symbols by name/pattern
- `memory_pack` - Get compiled context pack for a query
- `rlm_map_reduce` - Execute map-reduce queries when context is uncertain
- `focus_set/focus_clear` - Manage focus scope for retrieval
- `memory_pin/memory_unpin` - Pin important chunks
- `memory_record_decision` - Record architectural decisions
- And more...

## License

MIT
