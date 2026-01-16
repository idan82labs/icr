"""
Indexing modules for ICD.

Provides:
- File system watching with debouncing
- Symbol-based code chunking with tree-sitter
- Embedding generation (local ONNX and remote backends)
- Contract detection
- Ignore file parsing (.gitignore, .icrignore)
"""

from icd.indexing.chunker import Chunk, Chunker
from icd.indexing.contract_detector import ContractDetector
from icd.indexing.embedder import EmbeddingBackend, create_embedder
from icd.indexing.ignore_parser import (
    create_default_icrignore,
    load_ignore_patterns,
    parse_ignore_file,
)
from icd.indexing.watcher import FileWatcher

__all__ = [
    "FileWatcher",
    "Chunker",
    "Chunk",
    "EmbeddingBackend",
    "create_embedder",
    "ContractDetector",
    "load_ignore_patterns",
    "parse_ignore_file",
    "create_default_icrignore",
]
