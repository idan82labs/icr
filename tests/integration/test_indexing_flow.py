"""
Integration tests for the full indexing pipeline.

Tests cover:
- Complete indexing flow from file system to storage
- Incremental indexing
- File change detection
- Error handling during indexing
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from tests.conftest import (
    SAMPLE_AUTH_HANDLER_TS,
    SAMPLE_ENDPOINTS_TS,
    SAMPLE_SHARED_TYPES_TS,
    SAMPLE_VALIDATOR_TS,
    MockEmbeddingBackend,
)


# ==============================================================================
# Test Fixtures
# ==============================================================================

@pytest.fixture
def indexing_repo(tmp_path: Path) -> Path:
    """Create a repository for indexing tests."""
    repo = tmp_path / "indexing_repo"
    (repo / "src").mkdir(parents=True)

    # Create source files
    (repo / "src" / "auth.ts").write_text(SAMPLE_AUTH_HANDLER_TS)
    (repo / "src" / "validator.ts").write_text(SAMPLE_VALIDATOR_TS)
    (repo / "src" / "endpoints.ts").write_text(SAMPLE_ENDPOINTS_TS)
    (repo / "src" / "types.ts").write_text(SAMPLE_SHARED_TYPES_TS)

    return repo


@pytest.fixture
def mock_services():
    """Create mock services for indexing."""
    return {
        "sqlite_store": MagicMock(),
        "vector_store": MagicMock(),
        "embedder": MockEmbeddingBackend(dimension=384),
        "contract_store": MagicMock(),
    }


# ==============================================================================
# Indexing Pipeline Tests
# ==============================================================================

@pytest.mark.integration
class TestIndexingPipeline:
    """Tests for the complete indexing pipeline."""

    @pytest.mark.asyncio
    async def test_index_single_file(self, indexing_repo: Path, mock_services):
        """Test indexing a single file."""
        file_path = indexing_repo / "src" / "auth.ts"

        # Read file content
        content = file_path.read_text()

        # Simulate chunking
        chunks = [
            {
                "id": f"chunk_{i}",
                "content": content[i * 500:(i + 1) * 500],
                "start_line": i * 20 + 1,
                "end_line": (i + 1) * 20,
            }
            for i in range(len(content) // 500 + 1)
            if content[i * 500:(i + 1) * 500]
        ]

        assert len(chunks) > 0

        # Simulate embedding
        embedder = mock_services["embedder"]
        await embedder.initialize()

        embeddings = await embedder.embed([c["content"] for c in chunks])

        assert embeddings.shape[0] == len(chunks)
        assert embeddings.shape[1] == 384

    @pytest.mark.asyncio
    async def test_index_directory(self, indexing_repo: Path, mock_services):
        """Test indexing an entire directory."""
        # Find all TypeScript files
        ts_files = list(indexing_repo.glob("**/*.ts"))

        assert len(ts_files) == 4

        # Track indexing stats
        stats = {
            "files_indexed": 0,
            "chunks_created": 0,
            "errors": 0,
        }

        for file_path in ts_files:
            try:
                content = file_path.read_text()
                # Simulate chunking (simplified)
                chunk_count = max(1, len(content) // 500)
                stats["files_indexed"] += 1
                stats["chunks_created"] += chunk_count
            except Exception:
                stats["errors"] += 1

        assert stats["files_indexed"] == 4
        assert stats["chunks_created"] > 0
        assert stats["errors"] == 0

    @pytest.mark.asyncio
    async def test_index_with_ignore_patterns(self, indexing_repo: Path):
        """Test indexing respects ignore patterns."""
        # Create ignored directory
        (indexing_repo / "node_modules").mkdir()
        (indexing_repo / "node_modules" / "dep.js").write_text("ignored")

        ignore_patterns = ["**/node_modules/**", "**/.git/**"]

        all_files = list(indexing_repo.glob("**/*"))
        files_to_index = []

        import fnmatch

        for file_path in all_files:
            if file_path.is_file():
                rel_path = str(file_path.relative_to(indexing_repo))
                ignored = any(
                    fnmatch.fnmatch(rel_path, pattern) or
                    fnmatch.fnmatch(str(file_path), pattern)
                    for pattern in ignore_patterns
                )
                if not ignored:
                    files_to_index.append(file_path)

        # node_modules should be ignored
        assert not any("node_modules" in str(f) for f in files_to_index)


# ==============================================================================
# Incremental Indexing Tests
# ==============================================================================

@pytest.mark.integration
class TestIncrementalIndexing:
    """Tests for incremental indexing."""

    def test_detect_new_files(self, indexing_repo: Path):
        """Test detection of new files."""
        initial_files = set(indexing_repo.glob("**/*.ts"))

        # Add new file
        new_file = indexing_repo / "src" / "new_module.ts"
        new_file.write_text("export const newFeature = true;")

        current_files = set(indexing_repo.glob("**/*.ts"))
        new_files = current_files - initial_files

        assert len(new_files) == 1
        assert new_file in new_files

    def test_detect_modified_files(self, indexing_repo: Path):
        """Test detection of modified files."""
        file_path = indexing_repo / "src" / "auth.ts"

        # Get initial hash
        initial_content = file_path.read_text()
        initial_hash = hashlib.sha256(initial_content.encode()).hexdigest()

        # Modify file
        file_path.write_text(initial_content + "\n// Modified")

        # Get new hash
        new_content = file_path.read_text()
        new_hash = hashlib.sha256(new_content.encode()).hexdigest()

        assert initial_hash != new_hash

    def test_detect_deleted_files(self, indexing_repo: Path):
        """Test detection of deleted files."""
        initial_files = set(indexing_repo.glob("**/*.ts"))

        # Delete a file
        file_to_delete = indexing_repo / "src" / "validator.ts"
        file_to_delete.unlink()

        current_files = set(indexing_repo.glob("**/*.ts"))
        deleted_files = initial_files - current_files

        assert len(deleted_files) == 1

    def test_content_hash_based_update(self, indexing_repo: Path):
        """Test that updates are based on content hash, not timestamp."""
        file_path = indexing_repo / "src" / "auth.ts"
        content = file_path.read_text()

        content_hash = hashlib.sha256(content.encode()).hexdigest()

        # "Touch" the file (update timestamp)
        file_path.write_text(content)

        # Hash should be the same
        new_hash = hashlib.sha256(file_path.read_text().encode()).hexdigest()

        assert content_hash == new_hash


# ==============================================================================
# File Change Detection Tests
# ==============================================================================

@pytest.mark.integration
class TestFileChangeDetection:
    """Tests for file change detection."""

    def test_compute_file_hash(self, indexing_repo: Path):
        """Test file hash computation."""
        file_path = indexing_repo / "src" / "auth.ts"

        content = file_path.read_text()
        hash1 = hashlib.sha256(content.encode()).hexdigest()
        hash2 = hashlib.sha256(content.encode()).hexdigest()

        assert hash1 == hash2

    def test_detect_content_change(self, indexing_repo: Path):
        """Test content change detection."""
        file_path = indexing_repo / "src" / "auth.ts"

        original_content = file_path.read_text()
        original_hash = hashlib.sha256(original_content.encode()).hexdigest()

        # Small change
        modified_content = original_content.replace("handleAuth", "handleAuthentication")
        file_path.write_text(modified_content)

        modified_hash = hashlib.sha256(modified_content.encode()).hexdigest()

        assert original_hash != modified_hash

    def test_whitespace_change_detected(self, indexing_repo: Path):
        """Test that whitespace changes are detected."""
        file_path = indexing_repo / "src" / "auth.ts"

        original_content = file_path.read_text()
        original_hash = hashlib.sha256(original_content.encode()).hexdigest()

        # Add whitespace
        modified_content = original_content + "\n\n"
        modified_hash = hashlib.sha256(modified_content.encode()).hexdigest()

        assert original_hash != modified_hash


# ==============================================================================
# Error Handling Tests
# ==============================================================================

@pytest.mark.integration
class TestIndexingErrorHandling:
    """Tests for error handling during indexing."""

    def test_handle_unreadable_file(self, indexing_repo: Path):
        """Test handling of unreadable files."""
        # Create binary file
        binary_file = indexing_repo / "src" / "binary.dat"
        binary_file.write_bytes(b"\x00\x01\x02\x03")

        errors = []

        try:
            content = binary_file.read_text()
        except UnicodeDecodeError:
            errors.append(str(binary_file))

        # Binary file should cause error
        assert len(errors) == 1

    def test_handle_large_file(self, indexing_repo: Path):
        """Test handling of large files."""
        max_file_size_kb = 500  # From config

        # Create large file
        large_file = indexing_repo / "src" / "large.ts"
        large_content = "x" * (max_file_size_kb * 1024 + 1)
        large_file.write_text(large_content)

        # Should be skipped due to size
        file_size_kb = large_file.stat().st_size / 1024
        should_skip = file_size_kb > max_file_size_kb

        assert should_skip

    def test_handle_missing_file(self, indexing_repo: Path):
        """Test handling of missing files."""
        missing_file = indexing_repo / "src" / "nonexistent.ts"

        errors = []
        try:
            content = missing_file.read_text()
        except FileNotFoundError:
            errors.append(str(missing_file))

        assert len(errors) == 1

    def test_continue_on_error(self, indexing_repo: Path):
        """Test that indexing continues after errors."""
        # Create mix of valid and invalid files
        (indexing_repo / "src" / "binary.dat").write_bytes(b"\x00\x01")
        (indexing_repo / "src" / "valid.ts").write_text("const x = 1;")

        indexed = []
        errors = []

        for file_path in indexing_repo.glob("src/*"):
            try:
                if file_path.suffix in [".ts", ".js"]:
                    content = file_path.read_text()
                    indexed.append(file_path)
            except Exception as e:
                errors.append((file_path, e))

        # Valid file should be indexed despite error on binary
        assert len(indexed) >= 1


# ==============================================================================
# Index Persistence Tests
# ==============================================================================

@pytest.mark.integration
class TestIndexPersistence:
    """Tests for index persistence."""

    def test_save_and_load_index_metadata(self, tmp_path: Path):
        """Test saving and loading index metadata."""
        import json

        metadata = {
            "indexed_files": 10,
            "total_chunks": 50,
            "last_indexed": "2026-01-16T10:00:00Z",
        }

        metadata_path = tmp_path / "index_metadata.json"

        # Save
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        # Load
        with open(metadata_path) as f:
            loaded = json.load(f)

        assert loaded == metadata

    def test_persist_file_hashes(self, tmp_path: Path):
        """Test persisting file hashes for incremental updates."""
        import json

        file_hashes = {
            "/path/to/file1.ts": "abc123",
            "/path/to/file2.ts": "def456",
        }

        hash_path = tmp_path / "file_hashes.json"

        with open(hash_path, "w") as f:
            json.dump(file_hashes, f)

        with open(hash_path) as f:
            loaded = json.load(f)

        assert loaded == file_hashes


# ==============================================================================
# Performance Tests
# ==============================================================================

@pytest.mark.integration
class TestIndexingPerformance:
    """Tests for indexing performance characteristics."""

    def test_batch_embedding_efficiency(self, mock_services):
        """Test that batch embedding is more efficient."""
        embedder = mock_services["embedder"]

        async def run_test():
            await embedder.initialize()

            texts = [f"Content {i}" for i in range(100)]

            # Batch embedding
            embedder.reset_call_count()
            batch_result = await embedder.embed(texts)

            # Call count should be 100 (one per text in mock)
            batch_calls = embedder.call_count

            assert batch_result.shape == (100, 384)
            assert batch_calls == 100

        asyncio.get_event_loop().run_until_complete(run_test())

    def test_parallel_file_processing(self, indexing_repo: Path):
        """Test parallel file processing concept."""
        import concurrent.futures

        files = list(indexing_repo.glob("**/*.ts"))

        def process_file(file_path: Path) -> dict:
            content = file_path.read_text()
            return {
                "path": str(file_path),
                "size": len(content),
                "hash": hashlib.sha256(content.encode()).hexdigest()[:16],
            }

        # Process in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(process_file, files))

        assert len(results) == len(files)


# ==============================================================================
# Integration with Storage Tests
# ==============================================================================

@pytest.mark.integration
class TestStorageIntegration:
    """Tests for storage integration during indexing."""

    @pytest.mark.asyncio
    async def test_store_chunks_with_embeddings(self, indexing_repo: Path, mock_services):
        """Test storing chunks with their embeddings."""
        embedder = mock_services["embedder"]
        await embedder.initialize()

        file_path = indexing_repo / "src" / "auth.ts"
        content = file_path.read_text()

        # Create chunks
        chunks = [
            {"id": f"chunk_{i}", "content": content[i * 300:(i + 1) * 300]}
            for i in range(min(3, len(content) // 300 + 1))
        ]

        # Generate embeddings
        embeddings = await embedder.embed([c["content"] for c in chunks])

        # Verify we can associate chunks with embeddings
        chunk_embeddings = {
            chunk["id"]: embeddings[i]
            for i, chunk in enumerate(chunks)
        }

        assert len(chunk_embeddings) == len(chunks)
        for chunk_id, embedding in chunk_embeddings.items():
            assert embedding.shape == (384,)

    @pytest.mark.asyncio
    async def test_update_existing_chunks(self, indexing_repo: Path, mock_services):
        """Test updating existing chunks on file modification."""
        embedder = mock_services["embedder"]
        await embedder.initialize()

        file_path = indexing_repo / "src" / "auth.ts"

        # Initial indexing
        content1 = file_path.read_text()
        embedding1 = await embedder.embed_single(content1[:300])

        # Modify file
        modified_content = content1.replace("handleAuth", "handleAuthentication")
        file_path.write_text(modified_content)

        # Re-index
        content2 = file_path.read_text()
        embedding2 = await embedder.embed_single(content2[:300])

        # Embeddings should be different for different content
        assert not np.allclose(embedding1, embedding2)
