# ICD: ICR Daemon

Data plane component for intelligent code retrieval. Handles file watching, indexing, embedding, retrieval, and pack compilation.

## Installation

```bash
pip install icd
```

## Features

- HNSW vector index with float16 storage
- BM25 lexical search with FTS5
- Tree-sitter based code chunking
- Hybrid scoring: semantic + lexical + recency
- MMR diversity selection
- Contract detection and boosting

## License

MIT
