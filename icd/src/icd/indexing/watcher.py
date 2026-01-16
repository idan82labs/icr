"""
File system watcher with debouncing.

Monitors the project directory for file changes and triggers
re-indexing with configurable debouncing to avoid excessive updates.
"""

from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import structlog
from watchdog.events import (
    DirCreatedEvent,
    DirDeletedEvent,
    DirModifiedEvent,
    DirMovedEvent,
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
    FileSystemEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer

if TYPE_CHECKING:
    from icd.config import Config
    from icd.indexing.embedder import EmbeddingBackend
    from icd.storage.contract_store import ContractStore
    from icd.storage.sqlite_store import SQLiteStore
    from icd.storage.vector_store import VectorStore

logger = structlog.get_logger(__name__)


class DebouncedHandler(FileSystemEventHandler):
    """
    File system event handler with debouncing.

    Collects events and processes them after a debounce period,
    coalescing multiple events for the same file.
    """

    def __init__(
        self,
        callback: Callable[[set[Path]], None],
        debounce_ms: int = 500,
        ignore_patterns: list[str] | None = None,
        watch_extensions: list[str] | None = None,
    ) -> None:
        """
        Initialize the debounced handler.

        Args:
            callback: Function to call with changed file paths.
            debounce_ms: Debounce delay in milliseconds.
            ignore_patterns: Glob patterns to ignore.
            watch_extensions: File extensions to watch.
        """
        super().__init__()
        self.callback = callback
        self.debounce_seconds = debounce_ms / 1000.0
        self.ignore_patterns = ignore_patterns or []
        self.watch_extensions = set(watch_extensions or [])

        self._pending_changes: set[Path] = set()
        self._deleted_paths: set[Path] = set()
        self._timer: asyncio.TimerHandle | None = None
        self._lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Set the event loop for async operations."""
        self._loop = loop

    def _should_ignore(self, path: str) -> bool:
        """Check if a path should be ignored."""
        for pattern in self.ignore_patterns:
            if fnmatch.fnmatch(path, pattern):
                return True
        return False

    def _should_watch(self, path: str) -> bool:
        """Check if a path should be watched."""
        if not self.watch_extensions:
            return True

        ext = os.path.splitext(path)[1].lower()
        return ext in self.watch_extensions

    def _schedule_callback(self) -> None:
        """Schedule the callback after debounce period."""
        if not self._loop:
            return

        if self._timer:
            self._timer.cancel()

        self._timer = self._loop.call_later(
            self.debounce_seconds,
            self._fire_callback,
        )

    def _fire_callback(self) -> None:
        """Fire the callback with pending changes."""
        if not self._pending_changes and not self._deleted_paths:
            return

        changes = self._pending_changes.copy()
        deleted = self._deleted_paths.copy()
        self._pending_changes.clear()
        self._deleted_paths.clear()

        # Call the callback
        if self._loop:
            self._loop.call_soon_threadsafe(
                lambda: asyncio.create_task(
                    self._async_callback(changes, deleted)
                )
            )

    async def _async_callback(
        self,
        changes: set[Path],
        deleted: set[Path],
    ) -> None:
        """Async wrapper for callback."""
        self.callback(changes, deleted)

    def on_created(self, event: FileCreatedEvent | DirCreatedEvent) -> None:
        """Handle file/directory creation."""
        if isinstance(event, DirCreatedEvent):
            return

        path = event.src_path
        if self._should_ignore(path) or not self._should_watch(path):
            return

        self._pending_changes.add(Path(path))
        self._schedule_callback()

    def on_modified(self, event: FileModifiedEvent | DirModifiedEvent) -> None:
        """Handle file/directory modification."""
        if isinstance(event, DirModifiedEvent):
            return

        path = event.src_path
        if self._should_ignore(path) or not self._should_watch(path):
            return

        self._pending_changes.add(Path(path))
        self._schedule_callback()

    def on_deleted(self, event: FileDeletedEvent | DirDeletedEvent) -> None:
        """Handle file/directory deletion."""
        path = event.src_path
        if self._should_ignore(path):
            return

        self._deleted_paths.add(Path(path))
        self._pending_changes.discard(Path(path))
        self._schedule_callback()

    def on_moved(self, event: FileMovedEvent | DirMovedEvent) -> None:
        """Handle file/directory move."""
        if isinstance(event, DirMovedEvent):
            return

        src_path = event.src_path
        dest_path = event.dest_path

        if not self._should_ignore(src_path):
            self._deleted_paths.add(Path(src_path))

        if not self._should_ignore(dest_path) and self._should_watch(dest_path):
            self._pending_changes.add(Path(dest_path))

        self._schedule_callback()


class FileWatcher:
    """
    File system watcher for code indexing.

    Features:
    - Debounced file change detection
    - Incremental indexing
    - Content-based change detection
    - Parallel file processing
    """

    def __init__(
        self,
        config: "Config",
        sqlite_store: "SQLiteStore",
        vector_store: "VectorStore",
        embedder: "EmbeddingBackend",
        contract_store: "ContractStore",
    ) -> None:
        """
        Initialize the file watcher.

        Args:
            config: ICD configuration.
            sqlite_store: SQLite store for metadata.
            vector_store: Vector store for embeddings.
            embedder: Embedding backend.
            contract_store: Contract store.
        """
        self.config = config
        self.sqlite_store = sqlite_store
        self.vector_store = vector_store
        self.embedder = embedder
        self.contract_store = contract_store

        self._observer: Observer | None = None
        self._handler: DebouncedHandler | None = None
        self._running = False
        self._processing_lock = asyncio.Lock()

        # Lazy imports
        self._chunker: Any = None
        self._contract_detector: Any = None

    def _get_chunker(self) -> Any:
        """Get or create chunker instance."""
        if self._chunker is None:
            from icd.indexing.chunker import Chunker

            self._chunker = Chunker(self.config)
        return self._chunker

    def _get_contract_detector(self) -> Any:
        """Get or create contract detector instance."""
        if self._contract_detector is None:
            from icd.indexing.contract_detector import ContractDetector

            self._contract_detector = ContractDetector(self.config)
        return self._contract_detector

    async def start(self) -> None:
        """Start watching for file changes."""
        if self._running:
            return

        logger.info(
            "Starting file watcher",
            path=str(self.config.project_root),
        )

        self._handler = DebouncedHandler(
            callback=self._handle_changes,
            debounce_ms=self.config.watcher.debounce_ms,
            ignore_patterns=self.config.watcher.ignore_patterns,
            watch_extensions=self.config.watcher.watch_extensions,
        )
        self._handler.set_loop(asyncio.get_event_loop())

        self._observer = Observer()
        self._observer.schedule(
            self._handler,
            str(self.config.project_root),
            recursive=True,
        )
        self._observer.start()
        self._running = True

        logger.info("File watcher started")

    async def stop(self) -> None:
        """Stop watching for file changes."""
        if not self._running:
            return

        logger.info("Stopping file watcher")

        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5.0)
            self._observer = None

        self._handler = None
        self._running = False

        # Save vector index
        await self.vector_store.save()

        logger.info("File watcher stopped")

    def _handle_changes(
        self,
        changed: set[Path],
        deleted: set[Path],
    ) -> None:
        """Handle file changes (called from debounced handler)."""
        asyncio.create_task(self._process_changes(changed, deleted))

    async def _process_changes(
        self,
        changed: set[Path],
        deleted: set[Path],
    ) -> None:
        """Process file changes."""
        async with self._processing_lock:
            # Handle deletions
            for path in deleted:
                await self._remove_file(path)

            # Handle changes/additions
            for path in changed:
                if path.exists() and path.is_file():
                    await self._index_file(path)

            # Save vector index periodically
            await self.vector_store.save()

    async def _remove_file(self, path: Path) -> None:
        """Remove a file from the index."""
        str_path = str(path)
        logger.debug("Removing file from index", path=str_path)

        # Get chunks for this file
        chunks = await self.sqlite_store.get_chunks_by_file(str_path)

        # Remove from vector store
        for chunk in chunks:
            await self.vector_store.delete_vector(chunk.chunk_id)

        # Remove from contract store
        await self.contract_store.delete_contracts_by_file(str_path)

        # Remove from SQLite
        await self.sqlite_store.remove_file(str_path)

        logger.info("Removed file from index", path=str_path)

    async def _index_file(
        self,
        path: Path,
        force: bool = False,
    ) -> dict[str, int]:
        """
        Index a single file.

        Args:
            path: File path.
            force: Force re-indexing even if unchanged.

        Returns:
            Statistics about indexing.
        """
        str_path = str(path)
        stats = {"chunks": 0, "contracts": 0, "errors": 0}

        try:
            # Check file size
            file_size = path.stat().st_size
            max_size = self.config.watcher.max_file_size_kb * 1024
            if file_size > max_size:
                logger.debug(
                    "Skipping large file",
                    path=str_path,
                    size=file_size,
                )
                return stats

            # Read file content
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                logger.debug("Skipping binary file", path=str_path)
                return stats

            # Compute content hash
            content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

            # Check if file has changed
            if not force:
                existing = await self.sqlite_store.get_file_record(str_path)
                if existing and existing.content_hash == content_hash:
                    logger.debug("File unchanged, skipping", path=str_path)
                    return stats

            # Remove existing chunks for this file
            await self._remove_file(path)

            # Chunk the file
            chunker = self._get_chunker()
            chunks = chunker.chunk_file(path, content)

            if not chunks:
                return stats

            # Detect contracts
            contract_detector = self._get_contract_detector()
            for chunk in chunks:
                is_contract = contract_detector.is_contract(chunk)
                chunk.is_contract = is_contract

            # Generate embeddings in batch
            contents = [c.content for c in chunks]
            embeddings = await self.embedder.embed_batch(contents)

            # Store chunks and vectors
            chunk_ids = []
            for chunk, embedding in zip(chunks, embeddings):
                # Store in SQLite
                await self.sqlite_store.store_chunk(
                    chunk_id=chunk.chunk_id,
                    file_path=str_path,
                    content=chunk.content,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    start_byte=chunk.start_byte,
                    end_byte=chunk.end_byte,
                    symbol_name=chunk.symbol_name,
                    symbol_type=chunk.symbol_type,
                    language=chunk.language,
                    token_count=chunk.token_count,
                    is_contract=chunk.is_contract,
                )
                chunk_ids.append(chunk.chunk_id)

                # Store vector
                await self.vector_store.add_vector(chunk.chunk_id, embedding)

                # Store contract if applicable
                if chunk.is_contract:
                    contract = contract_detector.create_contract(chunk)
                    if contract:
                        await self.contract_store.store_contract(contract)
                        stats["contracts"] += 1

                stats["chunks"] += 1

            # Track the file
            await self.sqlite_store.track_file(
                file_path=str_path,
                content_hash=content_hash,
                size_bytes=file_size,
                modified_at=datetime.fromtimestamp(path.stat().st_mtime),
                chunk_count=len(chunks),
                language=chunks[0].language if chunks else None,
            )

            logger.debug(
                "Indexed file",
                path=str_path,
                chunks=stats["chunks"],
                contracts=stats["contracts"],
            )

        except Exception as e:
            logger.error("Error indexing file", path=str_path, error=str(e))
            stats["errors"] += 1

        return stats

    async def index_directory(
        self,
        path: Path | None = None,
        force: bool = False,
    ) -> dict[str, int]:
        """
        Index all files in a directory.

        Args:
            path: Directory path. Uses project_root if not provided.
            force: Force re-indexing of all files.

        Returns:
            Statistics about indexed files.
        """
        target = path or self.config.project_root
        logger.info("Indexing directory", path=str(target), force=force)

        stats = {"files": 0, "chunks": 0, "contracts": 0, "errors": 0}

        # Collect all files to index
        files_to_index: list[Path] = []

        for root, dirs, files in os.walk(target):
            root_path = Path(root)

            # Filter directories
            dirs[:] = [
                d
                for d in dirs
                if not any(
                    fnmatch.fnmatch(str(root_path / d), pattern)
                    for pattern in self.config.watcher.ignore_patterns
                )
            ]

            for filename in files:
                file_path = root_path / filename

                # Check ignore patterns
                str_path = str(file_path)
                if any(
                    fnmatch.fnmatch(str_path, pattern)
                    for pattern in self.config.watcher.ignore_patterns
                ):
                    continue

                # Check extension
                ext = file_path.suffix.lower()
                if ext not in self.config.watcher.watch_extensions:
                    continue

                files_to_index.append(file_path)

        logger.info(f"Found {len(files_to_index)} files to index")

        # Index files
        for file_path in files_to_index:
            file_stats = await self._index_file(file_path, force=force)
            stats["files"] += 1
            stats["chunks"] += file_stats["chunks"]
            stats["contracts"] += file_stats["contracts"]
            stats["errors"] += file_stats["errors"]

            # Progress logging
            if stats["files"] % 100 == 0:
                logger.info(
                    "Indexing progress",
                    files=stats["files"],
                    chunks=stats["chunks"],
                )

        # Save vector index
        await self.vector_store.save()

        logger.info(
            "Directory indexing complete",
            files=stats["files"],
            chunks=stats["chunks"],
            contracts=stats["contracts"],
            errors=stats["errors"],
        )

        return stats

    async def reindex_file(self, path: Path) -> dict[str, int]:
        """
        Force reindex a specific file.

        Args:
            path: File path.

        Returns:
            Indexing statistics.
        """
        return await self._index_file(path, force=True)
