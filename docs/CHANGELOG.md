# ICR Changelog

All notable changes to ICR (Infinite Context Runtime) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Initial project documentation suite

---

## [0.1.0] - 2026-01-16

### Added

#### Core Infrastructure
- **icd (daemon)**: Data plane component for indexing, storage, and retrieval
  - SQLite-based storage with FTS5 full-text search
  - HNSW vector index with float16 storage, float32 compute policy
  - File system watcher with 500ms debouncing
  - Symbol-based chunking using tree-sitter
  - Support for Python, JavaScript, TypeScript, Go, Rust, Java, C/C++

- **ic-mcp (MCP server)**: Tool plane exposing bounded operations
  - Memory tools: `memory_pack`, `memory_pin`, `memory_unpin`, `memory_list`, `memory_get`, `memory_stats`
  - Environment tools: `env_search`, `env_peek`, `env_slice`, `env_aggregate`
  - Project tools: `project_map`, `project_symbol_search`, `project_impact`, `project_commands`
  - RLM tools: `rlm_plan`, `rlm_map_reduce`
  - Admin tools: `admin_ping`

- **ic-claude (plugin)**: Behavior plane for Claude Code integration
  - `UserPromptSubmit` hook for automatic context injection
  - `Stop` hook for ledger extraction
  - `PreCompact` hook for invariant preservation
  - `/ic` command family

#### Retrieval System
- Hybrid retrieval combining semantic (embedding) and lexical (BM25) search
- Configurable scoring weights for embedding, BM25, recency, contracts, focus, and pinned items
- MMR (Maximal Marginal Relevance) diversity selection
- Retrieval entropy computation for mode gating

#### Pack Compilation
- Knapsack-based context pack compilation
- Token budget enforcement (512-12000 tokens)
- Citation generation with source references
- Markdown output formatting

#### RLM-Lite Runtime
- Entropy-based mode gating (pack vs RLM)
- Bounded inspection loops with stop conditions
- Non-generative aggregation operations
- Plan generation for complex queries

#### Embedding System
- Local ONNX backend (default, no network required)
- Support for `all-MiniLM-L6-v2` model (384 dimensions)
- Optional remote backends (OpenAI, Anthropic) with explicit opt-in
- Batch embedding generation

#### Contract Detection
- Heuristic-based detection of interfaces, types, and schemas
- Cross-language pattern matching
- Configurable boost factor for retrieval

#### Configuration
- YAML-based configuration system
- Environment variable overrides
- Per-project configuration support
- Comprehensive ignore patterns for security

#### Telemetry
- Local-only metrics collection
- Per-operation timing
- EWR (Exploration Waste Ratio) proxy counters
- Gating reason codes

### Performance Targets

| Operation | P50 | P95 |
|-----------|-----|-----|
| ANN lookup (given embedding) | 15ms | 40ms |
| Query embedding (local) | 20ms | 50ms |
| End-to-end semantic | 40ms | 100ms |
| Hybrid search (full) | 70ms | 150ms |
| Pack compilation | 200ms | 500ms |

### Scalability Tiers

| Tier | Files | Chunks | RAM | Index Size |
|------|-------|--------|-----|------------|
| Tier 1 (Guaranteed) | 10,000 | 100,000 | <2GB | <1GB |
| Tier 2 (Stretch) | 100,000 | 1,000,000 | <10GB | <5GB |

### Security
- Default local-only mode (no network egress)
- Automatic secret detection and exclusion
- Comprehensive ignore patterns for sensitive files
- Explicit opt-in required for remote embedding backends

### Documentation
- Project README with quick start guide
- Architecture deep dive
- Research foundation (RLM theory)
- User guide with usage examples
- Developer guide for contributors
- Complete API reference
- Configuration reference
- Troubleshooting guide

---

## Version History Format

### Categories

- **Added**: New features
- **Changed**: Changes in existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Security-related changes
- **Performance**: Performance improvements

### Versioning Policy

- **MAJOR** (X.0.0): Breaking changes to API or configuration
- **MINOR** (0.X.0): New features, backward compatible
- **PATCH** (0.0.X): Bug fixes, backward compatible

---

## Roadmap

### Planned for 0.2.0
- [ ] GPU acceleration for local embeddings
- [ ] Incremental HNSW index updates
- [ ] Memory-mapped vector storage for large repos
- [ ] Additional language support (Kotlin, Swift, Scala)
- [ ] Enhanced contract detection with AST analysis

### Planned for 0.3.0
- [ ] Multi-repository support
- [ ] Remote index synchronization
- [ ] Team sharing of pinned invariants
- [ ] Custom embedding model support

### Planned for 1.0.0
- [ ] Stable API guarantee
- [ ] Comprehensive test coverage (>90%)
- [ ] Production hardening
- [ ] Performance benchmarks
- [ ] Security audit

---

## Contributing

See [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) for contribution guidelines.

When submitting PRs, please:
1. Update this CHANGELOG under the `[Unreleased]` section
2. Follow the format established above
3. Link to relevant issues/PRs

---

## Links

- [GitHub Repository](https://github.com/icr/icr)
- [Issue Tracker](https://github.com/icr/icr/issues)
- [Documentation](https://github.com/icr/icr/tree/main/docs)
