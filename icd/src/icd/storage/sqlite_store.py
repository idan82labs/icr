"""
SQLite storage with FTS5 for metadata and lexical search.

Provides:
- Chunk metadata storage
- Full-text search with BM25 ranking
- File tracking and change detection
- Transaction support
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator

import aiosqlite
import structlog

if TYPE_CHECKING:
    from icd.config import Config

logger = structlog.get_logger(__name__)


@dataclass
class ChunkMetadata:
    """Metadata for a stored chunk."""

    chunk_id: str
    file_path: str
    content_hash: str
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    symbol_name: str | None
    symbol_type: str | None
    language: str
    token_count: int
    created_at: datetime
    updated_at: datetime
    is_contract: bool = False
    is_pinned: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class FileRecord:
    """Record for a tracked file."""

    file_path: str
    content_hash: str
    size_bytes: int
    modified_at: datetime
    indexed_at: datetime
    chunk_count: int
    language: str | None


@dataclass
class BM25Result:
    """Result from BM25 search."""

    chunk_id: str
    score: float
    snippet: str | None = None


class SQLiteStore:
    """
    SQLite storage backend with FTS5 support.

    Features:
    - WAL mode for concurrent reads
    - FTS5 for full-text search with BM25 ranking
    - Content-addressed chunk storage
    - Efficient batch operations
    """

    # SQL Schema
    SCHEMA = """
    -- Chunks table: stores chunk metadata
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id TEXT PRIMARY KEY,
        file_path TEXT NOT NULL,
        content TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        start_line INTEGER NOT NULL,
        end_line INTEGER NOT NULL,
        start_byte INTEGER NOT NULL,
        end_byte INTEGER NOT NULL,
        symbol_name TEXT,
        symbol_type TEXT,
        language TEXT NOT NULL,
        token_count INTEGER NOT NULL,
        is_contract INTEGER DEFAULT 0,
        is_pinned INTEGER DEFAULT 0,
        extra TEXT DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    -- Indexes for common queries
    CREATE INDEX IF NOT EXISTS idx_chunks_file_path ON chunks(file_path);
    CREATE INDEX IF NOT EXISTS idx_chunks_content_hash ON chunks(content_hash);
    CREATE INDEX IF NOT EXISTS idx_chunks_symbol_type ON chunks(symbol_type);
    CREATE INDEX IF NOT EXISTS idx_chunks_is_contract ON chunks(is_contract);
    CREATE INDEX IF NOT EXISTS idx_chunks_is_pinned ON chunks(is_pinned);
    CREATE INDEX IF NOT EXISTS idx_chunks_language ON chunks(language);

    -- Files table: tracks indexed files
    CREATE TABLE IF NOT EXISTS files (
        file_path TEXT PRIMARY KEY,
        content_hash TEXT NOT NULL,
        size_bytes INTEGER NOT NULL,
        modified_at TEXT NOT NULL,
        indexed_at TEXT NOT NULL,
        chunk_count INTEGER DEFAULT 0,
        language TEXT
    );

    -- FTS5 virtual table for full-text search
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
        chunk_id,
        content,
        symbol_name,
        file_path,
        content='chunks',
        content_rowid='rowid',
        tokenize='porter unicode61'
    );

    -- Triggers to keep FTS index in sync
    CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
        INSERT INTO chunks_fts(rowid, chunk_id, content, symbol_name, file_path)
        VALUES (new.rowid, new.chunk_id, new.content, new.symbol_name, new.file_path);
    END;

    CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, chunk_id, content, symbol_name, file_path)
        VALUES('delete', old.rowid, old.chunk_id, old.content, old.symbol_name, old.file_path);
    END;

    CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, chunk_id, content, symbol_name, file_path)
        VALUES('delete', old.rowid, old.chunk_id, old.content, old.symbol_name, old.file_path);
        INSERT INTO chunks_fts(rowid, chunk_id, content, symbol_name, file_path)
        VALUES (new.rowid, new.chunk_id, new.content, new.symbol_name, new.file_path);
    END;
    """

    def __init__(self, config: "Config") -> None:
        """
        Initialize SQLite store.

        Args:
            config: ICD configuration.
        """
        self.config = config
        self.db_path = config.db_path
        self._db: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the database and create schema."""
        logger.info("Initializing SQLite store", db_path=str(self.db_path))

        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._db = await aiosqlite.connect(
            self.db_path,
            isolation_level=None,  # Autocommit mode for explicit transactions
        )

        # Configure SQLite for performance
        if self.config.storage.wal_mode:
            await self._db.execute("PRAGMA journal_mode=WAL")

        cache_size_pages = (self.config.storage.cache_size_mb * 1024 * 1024) // 4096
        await self._db.execute(f"PRAGMA cache_size=-{cache_size_pages}")
        await self._db.execute("PRAGMA synchronous=NORMAL")
        await self._db.execute("PRAGMA temp_store=MEMORY")
        await self._db.execute("PRAGMA mmap_size=268435456")  # 256MB

        # Create schema
        await self._db.executescript(self.SCHEMA)

        logger.info("SQLite store initialized")

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None
            logger.info("SQLite store closed")

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[aiosqlite.Connection]:
        """Context manager for transactions."""
        if not self._db:
            raise RuntimeError("Database not initialized")

        async with self._lock:
            await self._db.execute("BEGIN")
            try:
                yield self._db
                await self._db.execute("COMMIT")
            except Exception:
                await self._db.execute("ROLLBACK")
                raise

    async def store_chunk(
        self,
        chunk_id: str,
        file_path: str,
        content: str,
        start_line: int,
        end_line: int,
        start_byte: int,
        end_byte: int,
        symbol_name: str | None,
        symbol_type: str | None,
        language: str,
        token_count: int,
        is_contract: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> str:
        """
        Store a chunk with metadata.

        Args:
            chunk_id: Unique chunk identifier.
            file_path: Source file path.
            content: Chunk content.
            start_line: Starting line number.
            end_line: Ending line number.
            start_byte: Starting byte offset.
            end_byte: Ending byte offset.
            symbol_name: Name of the symbol (function, class, etc.).
            symbol_type: Type of symbol.
            language: Programming language.
            token_count: Number of tokens in the chunk.
            is_contract: Whether this chunk is a contract.
            extra: Additional metadata.

        Returns:
            The chunk_id.
        """
        if not self._db:
            raise RuntimeError("Database not initialized")

        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        now = datetime.utcnow().isoformat()
        extra_json = json.dumps(extra or {})

        await self._db.execute(
            """
            INSERT OR REPLACE INTO chunks (
                chunk_id, file_path, content, content_hash,
                start_line, end_line, start_byte, end_byte,
                symbol_name, symbol_type, language, token_count,
                is_contract, is_pinned, extra, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)
            """,
            (
                chunk_id,
                file_path,
                content,
                content_hash,
                start_line,
                end_line,
                start_byte,
                end_byte,
                symbol_name,
                symbol_type,
                language,
                token_count,
                1 if is_contract else 0,
                extra_json,
                now,
                now,
            ),
        )

        return chunk_id

    async def store_chunks_batch(
        self,
        chunks: list[dict[str, Any]],
    ) -> list[str]:
        """
        Store multiple chunks in a batch.

        Args:
            chunks: List of chunk dictionaries.

        Returns:
            List of chunk IDs.
        """
        if not self._db:
            raise RuntimeError("Database not initialized")

        now = datetime.utcnow().isoformat()
        chunk_ids = []

        async with self.transaction() as conn:
            for chunk in chunks:
                content_hash = hashlib.sha256(
                    chunk["content"].encode()
                ).hexdigest()[:16]
                extra_json = json.dumps(chunk.get("extra", {}))
                chunk_id = chunk.get("chunk_id") or self._generate_chunk_id(
                    chunk["file_path"], content_hash
                )
                chunk_ids.append(chunk_id)

                await conn.execute(
                    """
                    INSERT OR REPLACE INTO chunks (
                        chunk_id, file_path, content, content_hash,
                        start_line, end_line, start_byte, end_byte,
                        symbol_name, symbol_type, language, token_count,
                        is_contract, is_pinned, extra, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)
                    """,
                    (
                        chunk_id,
                        chunk["file_path"],
                        chunk["content"],
                        content_hash,
                        chunk["start_line"],
                        chunk["end_line"],
                        chunk["start_byte"],
                        chunk["end_byte"],
                        chunk.get("symbol_name"),
                        chunk.get("symbol_type"),
                        chunk["language"],
                        chunk["token_count"],
                        1 if chunk.get("is_contract") else 0,
                        extra_json,
                        now,
                        now,
                    ),
                )

        return chunk_ids

    def _generate_chunk_id(self, file_path: str, content_hash: str) -> str:
        """Generate a stable chunk ID."""
        combined = f"{file_path}:{content_hash}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def _sanitize_fts5_query(self, query: str) -> str:
        """Sanitize query for FTS5 syntax."""
        # Remove characters that cause FTS5 syntax errors
        for char in '?()&|':
            query = query.replace(char, ' ')
        # Escape double quotes
        query = query.replace('"', '""')
        # Remove leading/trailing operators
        query = query.strip('*:')
        # Collapse multiple spaces
        return ' '.join(query.split())

    async def get_chunk(self, chunk_id: str) -> ChunkMetadata | None:
        """
        Get chunk metadata by ID.

        Args:
            chunk_id: Chunk identifier.

        Returns:
            ChunkMetadata or None if not found.
        """
        if not self._db:
            raise RuntimeError("Database not initialized")

        async with self._db.execute(
            "SELECT * FROM chunks WHERE chunk_id = ?",
            (chunk_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return self._row_to_metadata(row, cursor.description)
            return None

    async def get_chunk_content(self, chunk_id: str) -> str | None:
        """
        Get chunk content by ID.

        Args:
            chunk_id: Chunk identifier.

        Returns:
            Content string or None if not found.
        """
        if not self._db:
            raise RuntimeError("Database not initialized")

        async with self._db.execute(
            "SELECT content FROM chunks WHERE chunk_id = ?",
            (chunk_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return row[0]
            return None

    async def get_chunk_with_content(self, chunk_id: str) -> "Chunk | None":
        """
        Get full chunk object (metadata + content) by ID.

        Args:
            chunk_id: Chunk identifier.

        Returns:
            Chunk object or None if not found.
        """
        from icd.retrieval.hybrid import Chunk

        if not self._db:
            raise RuntimeError("Database not initialized")

        async with self._db.execute(
            "SELECT * FROM chunks WHERE chunk_id = ?",
            (chunk_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                metadata = self._row_to_metadata(row, cursor.description)
                # Get content (last column)
                content = row[-1] if row else ""
                return Chunk(
                    chunk_id=metadata.chunk_id,
                    file_path=metadata.file_path,
                    content=content,
                    start_line=metadata.start_line,
                    end_line=metadata.end_line,
                    symbol_name=metadata.symbol_name,
                    symbol_type=metadata.symbol_type,
                    language=metadata.language,
                    token_count=metadata.token_count,
                )
            return None

    async def get_chunks_by_file(self, file_path: str) -> list[ChunkMetadata]:
        """
        Get all chunks for a file.

        Args:
            file_path: File path.

        Returns:
            List of chunk metadata.
        """
        if not self._db:
            raise RuntimeError("Database not initialized")

        async with self._db.execute(
            "SELECT * FROM chunks WHERE file_path = ? ORDER BY start_line",
            (file_path,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                self._row_to_metadata(row, cursor.description) for row in rows
            ]

    async def delete_chunks_by_file(self, file_path: str) -> int:
        """
        Delete all chunks for a file.

        Args:
            file_path: File path.

        Returns:
            Number of deleted chunks.
        """
        if not self._db:
            raise RuntimeError("Database not initialized")

        cursor = await self._db.execute(
            "DELETE FROM chunks WHERE file_path = ?",
            (file_path,),
        )
        return cursor.rowcount

    async def search_bm25(
        self,
        query: str,
        limit: int = 100,
        file_filter: list[str] | None = None,
    ) -> list[BM25Result]:
        """
        Search chunks using BM25 ranking.

        Args:
            query: Search query.
            limit: Maximum results.
            file_filter: Optional list of file paths to filter.

        Returns:
            List of BM25Result with scores.
        """
        if not self._db:
            raise RuntimeError("Database not initialized")

        # Sanitize query for FTS5 syntax
        safe_query = self._sanitize_fts5_query(query)

        if file_filter:
            placeholders = ",".join("?" * len(file_filter))
            sql = f"""
                SELECT
                    chunks_fts.chunk_id,
                    bm25(chunks_fts) as score,
                    snippet(chunks_fts, 1, '<b>', '</b>', '...', 32) as snippet
                FROM chunks_fts
                WHERE chunks_fts MATCH ?
                AND file_path IN ({placeholders})
                ORDER BY bm25(chunks_fts)
                LIMIT ?
            """
            params = [safe_query] + file_filter + [limit]
        else:
            sql = """
                SELECT
                    chunks_fts.chunk_id,
                    bm25(chunks_fts) as score,
                    snippet(chunks_fts, 1, '<b>', '</b>', '...', 32) as snippet
                FROM chunks_fts
                WHERE chunks_fts MATCH ?
                ORDER BY bm25(chunks_fts)
                LIMIT ?
            """
            params = [safe_query, limit]

        try:
            async with self._db.execute(sql, params) as cursor:
                rows = await cursor.fetchall()
                return [
                    BM25Result(
                        chunk_id=row[0],
                        score=-row[1],  # BM25 returns negative scores
                        snippet=row[2],
                    )
                    for row in rows
                ]
        except sqlite3.OperationalError as e:
            # Handle FTS syntax errors gracefully
            logger.warning("BM25 search failed", query=query, error=str(e))
            return []

    async def track_file(
        self,
        file_path: str,
        content_hash: str,
        size_bytes: int,
        modified_at: datetime,
        chunk_count: int,
        language: str | None = None,
    ) -> None:
        """
        Track an indexed file.

        Args:
            file_path: File path.
            content_hash: Hash of file content.
            size_bytes: File size.
            modified_at: File modification time.
            chunk_count: Number of chunks created.
            language: Detected language.
        """
        if not self._db:
            raise RuntimeError("Database not initialized")

        now = datetime.utcnow().isoformat()

        await self._db.execute(
            """
            INSERT OR REPLACE INTO files (
                file_path, content_hash, size_bytes, modified_at,
                indexed_at, chunk_count, language
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                file_path,
                content_hash,
                size_bytes,
                modified_at.isoformat(),
                now,
                chunk_count,
                language,
            ),
        )

    async def get_file_record(self, file_path: str) -> FileRecord | None:
        """
        Get file tracking record.

        Args:
            file_path: File path.

        Returns:
            FileRecord or None.
        """
        if not self._db:
            raise RuntimeError("Database not initialized")

        async with self._db.execute(
            "SELECT * FROM files WHERE file_path = ?",
            (file_path,),
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return FileRecord(
                    file_path=row[0],
                    content_hash=row[1],
                    size_bytes=row[2],
                    modified_at=datetime.fromisoformat(row[3]),
                    indexed_at=datetime.fromisoformat(row[4]),
                    chunk_count=row[5],
                    language=row[6],
                )
            return None

    async def remove_file(self, file_path: str) -> None:
        """
        Remove a file and its chunks from the index.

        Args:
            file_path: File path to remove.
        """
        if not self._db:
            raise RuntimeError("Database not initialized")

        async with self.transaction() as conn:
            await conn.execute(
                "DELETE FROM chunks WHERE file_path = ?",
                (file_path,),
            )
            await conn.execute(
                "DELETE FROM files WHERE file_path = ?",
                (file_path,),
            )

    async def get_all_chunk_ids(self) -> list[str]:
        """Get all chunk IDs in the store."""
        if not self._db:
            raise RuntimeError("Database not initialized")

        async with self._db.execute("SELECT chunk_id FROM chunks") as cursor:
            rows = await cursor.fetchall()
            return [row[0] for row in rows]

    async def get_chunk_ids_by_filter(
        self,
        is_contract: bool | None = None,
        is_pinned: bool | None = None,
        symbol_types: list[str] | None = None,
    ) -> list[str]:
        """
        Get chunk IDs matching filter criteria.

        Args:
            is_contract: Filter by contract status.
            is_pinned: Filter by pinned status.
            symbol_types: Filter by symbol types.

        Returns:
            List of matching chunk IDs.
        """
        if not self._db:
            raise RuntimeError("Database not initialized")

        conditions = []
        params: list[Any] = []

        if is_contract is not None:
            conditions.append("is_contract = ?")
            params.append(1 if is_contract else 0)

        if is_pinned is not None:
            conditions.append("is_pinned = ?")
            params.append(1 if is_pinned else 0)

        if symbol_types:
            placeholders = ",".join("?" * len(symbol_types))
            conditions.append(f"symbol_type IN ({placeholders})")
            params.extend(symbol_types)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        sql = f"SELECT chunk_id FROM chunks WHERE {where_clause}"

        async with self._db.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
            return [row[0] for row in rows]

    async def set_pinned(self, chunk_id: str, pinned: bool) -> bool:
        """
        Set the pinned status of a chunk.

        Args:
            chunk_id: Chunk identifier.
            pinned: Pinned status.

        Returns:
            True if chunk was updated.
        """
        if not self._db:
            raise RuntimeError("Database not initialized")

        now = datetime.utcnow().isoformat()
        cursor = await self._db.execute(
            "UPDATE chunks SET is_pinned = ?, updated_at = ? WHERE chunk_id = ?",
            (1 if pinned else 0, now, chunk_id),
        )
        return cursor.rowcount > 0

    async def get_stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        if not self._db:
            raise RuntimeError("Database not initialized")

        stats: dict[str, Any] = {}

        async with self._db.execute("SELECT COUNT(*) FROM chunks") as cursor:
            row = await cursor.fetchone()
            stats["total_chunks"] = row[0] if row else 0

        async with self._db.execute("SELECT COUNT(*) FROM files") as cursor:
            row = await cursor.fetchone()
            stats["total_files"] = row[0] if row else 0

        async with self._db.execute(
            "SELECT COUNT(*) FROM chunks WHERE is_contract = 1"
        ) as cursor:
            row = await cursor.fetchone()
            stats["contract_chunks"] = row[0] if row else 0

        async with self._db.execute(
            "SELECT COUNT(*) FROM chunks WHERE is_pinned = 1"
        ) as cursor:
            row = await cursor.fetchone()
            stats["pinned_chunks"] = row[0] if row else 0

        async with self._db.execute(
            "SELECT language, COUNT(*) FROM chunks GROUP BY language"
        ) as cursor:
            rows = await cursor.fetchall()
            stats["chunks_by_language"] = {row[0]: row[1] for row in rows}

        return stats

    def _row_to_metadata(
        self,
        row: tuple,
        description: Any,
    ) -> ChunkMetadata:
        """Convert a database row to ChunkMetadata."""
        columns = [d[0] for d in description]
        data = dict(zip(columns, row))

        return ChunkMetadata(
            chunk_id=data["chunk_id"],
            file_path=data["file_path"],
            content_hash=data["content_hash"],
            start_line=data["start_line"],
            end_line=data["end_line"],
            start_byte=data["start_byte"],
            end_byte=data["end_byte"],
            symbol_name=data.get("symbol_name"),
            symbol_type=data.get("symbol_type"),
            language=data["language"],
            token_count=data["token_count"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            is_contract=bool(data.get("is_contract", 0)),
            is_pinned=bool(data.get("is_pinned", 0)),
            extra=json.loads(data.get("extra", "{}")),
        )
