"""
Indexing modules for ICD.

Provides:
- File system watching with debouncing
- Symbol-based code chunking with tree-sitter
- Embedding generation (local ONNX and remote backends)
- Contract detection
"""

from icd.indexing.chunker import Chunk, Chunker
from icd.indexing.contract_detector import ContractDetector
from icd.indexing.embedder import EmbeddingBackend, create_embedder
from icd.indexing.watcher import FileWatcher

__all__ = [
    "FileWatcher",
    "Chunker",
    "Chunk",
    "EmbeddingBackend",
    "create_embedder",
    "ContractDetector",
]
