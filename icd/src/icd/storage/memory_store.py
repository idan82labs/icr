"""
Memory store for derived memory (pinned invariants, ledgers, decisions).

Provides persistent storage for:
- Pinned chunks (invariants that should always be included)
- Memory ledger (decisions, context, historical information)
- Session state
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from icd.config import Config
    from icd.storage.sqlite_store import SQLiteStore

logger = structlog.get_logger(__name__)


class LedgerCategory(str, Enum):
    """Categories for ledger entries."""

    DECISION = "decision"
    CONTEXT = "context"
    ARCHITECTURE = "architecture"
    CONSTRAINT = "constraint"
    PATTERN = "pattern"
    BUG_FIX = "bug_fix"
    GENERAL = "general"


@dataclass
class PinnedChunk:
    """A pinned chunk (invariant)."""

    chunk_id: str
    reason: str | None
    pinned_at: datetime
    priority: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LedgerEntry:
    """An entry in the memory ledger."""

    entry_id: str
    content: str
    category: LedgerCategory
    created_at: datetime
    updated_at: datetime
    embedding_id: str | None = None
    relevance_score: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionState:
    """Session state for a retrieval session."""

    session_id: str
    query_history: list[str]
    retrieved_chunks: list[str]
    focus_paths: list[str]
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryStore:
    """
    Store for derived memory including pinned invariants and ledgers.

    Features:
    - Pinned chunk management
    - Memory ledger with categories
    - Session state tracking
    - Priority-based retrieval
    """

    # SQL schema for memory store
    SCHEMA = """
    -- Pinned chunks (invariants)
    CREATE TABLE IF NOT EXISTS pinned_chunks (
        chunk_id TEXT PRIMARY KEY,
        reason TEXT,
        priority INTEGER DEFAULT 0,
        pinned_at TEXT NOT NULL,
        metadata TEXT DEFAULT '{}'
    );

    -- Memory ledger
    CREATE TABLE IF NOT EXISTS ledger_entries (
        entry_id TEXT PRIMARY KEY,
        content TEXT NOT NULL,
        category TEXT NOT NULL,
        embedding_id TEXT,
        relevance_score REAL DEFAULT 1.0,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        metadata TEXT DEFAULT '{}'
    );

    -- Session state
    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        query_history TEXT DEFAULT '[]',
        retrieved_chunks TEXT DEFAULT '[]',
        focus_paths TEXT DEFAULT '[]',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        metadata TEXT DEFAULT '{}'
    );

    -- Indexes
    CREATE INDEX IF NOT EXISTS idx_pinned_priority ON pinned_chunks(priority DESC);
    CREATE INDEX IF NOT EXISTS idx_ledger_category ON ledger_entries(category);
    CREATE INDEX IF NOT EXISTS idx_ledger_created ON ledger_entries(created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at DESC);

    -- FTS for ledger search
    CREATE VIRTUAL TABLE IF NOT EXISTS ledger_fts USING fts5(
        entry_id,
        content,
        category,
        content='ledger_entries',
        content_rowid='rowid',
        tokenize='porter unicode61'
    );

    -- Triggers to keep FTS in sync
    CREATE TRIGGER IF NOT EXISTS ledger_ai AFTER INSERT ON ledger_entries BEGIN
        INSERT INTO ledger_fts(rowid, entry_id, content, category)
        VALUES (new.rowid, new.entry_id, new.content, new.category);
    END;

    CREATE TRIGGER IF NOT EXISTS ledger_ad AFTER DELETE ON ledger_entries BEGIN
        INSERT INTO ledger_fts(ledger_fts, rowid, entry_id, content, category)
        VALUES('delete', old.rowid, old.entry_id, old.content, old.category);
    END;

    CREATE TRIGGER IF NOT EXISTS ledger_au AFTER UPDATE ON ledger_entries BEGIN
        INSERT INTO ledger_fts(ledger_fts, rowid, entry_id, content, category)
        VALUES('delete', old.rowid, old.entry_id, old.content, old.category);
        INSERT INTO ledger_fts(rowid, entry_id, content, category)
        VALUES (new.rowid, new.entry_id, new.content, new.category);
    END;
    """

    def __init__(
        self,
        config: "Config",
        sqlite_store: "SQLiteStore",
    ) -> None:
        """
        Initialize memory store.

        Args:
            config: ICD configuration.
            sqlite_store: SQLite store for persistence.
        """
        self.config = config
        self.sqlite_store = sqlite_store
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the memory store schema."""
        logger.info("Initializing memory store")

        if self.sqlite_store._db:
            await self.sqlite_store._db.executescript(self.SCHEMA)

        logger.info("Memory store initialized")

    # ==================== Pinned Chunks ====================

    async def pin_chunk(
        self,
        chunk_id: str,
        reason: str | None = None,
        priority: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Pin a chunk as an invariant.

        Args:
            chunk_id: Chunk identifier.
            reason: Reason for pinning.
            priority: Priority level (higher = more important).
            metadata: Additional metadata.

        Returns:
            True if successfully pinned.
        """
        db = self.sqlite_store._db
        if not db:
            raise RuntimeError("Database not initialized")

        now = datetime.utcnow().isoformat()
        meta_json = json.dumps(metadata or {})

        async with self._lock:
            await db.execute(
                """
                INSERT OR REPLACE INTO pinned_chunks
                (chunk_id, reason, priority, pinned_at, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (chunk_id, reason, priority, now, meta_json),
            )

            # Also update the chunks table
            await db.execute(
                "UPDATE chunks SET is_pinned = 1 WHERE chunk_id = ?",
                (chunk_id,),
            )

        logger.info("Pinned chunk", chunk_id=chunk_id, reason=reason)
        return True

    async def unpin_chunk(self, chunk_id: str) -> bool:
        """
        Unpin a chunk.

        Args:
            chunk_id: Chunk identifier.

        Returns:
            True if successfully unpinned.
        """
        db = self.sqlite_store._db
        if not db:
            raise RuntimeError("Database not initialized")

        async with self._lock:
            cursor = await db.execute(
                "DELETE FROM pinned_chunks WHERE chunk_id = ?",
                (chunk_id,),
            )

            if cursor.rowcount > 0:
                # Update chunks table
                await db.execute(
                    "UPDATE chunks SET is_pinned = 0 WHERE chunk_id = ?",
                    (chunk_id,),
                )
                logger.info("Unpinned chunk", chunk_id=chunk_id)
                return True

        return False

    async def get_pinned_chunks(
        self,
        min_priority: int | None = None,
    ) -> list[str]:
        """
        Get all pinned chunk IDs.

        Args:
            min_priority: Minimum priority filter.

        Returns:
            List of pinned chunk IDs.
        """
        db = self.sqlite_store._db
        if not db:
            raise RuntimeError("Database not initialized")

        if min_priority is not None:
            sql = """
                SELECT chunk_id FROM pinned_chunks
                WHERE priority >= ?
                ORDER BY priority DESC
            """
            params = (min_priority,)
        else:
            sql = "SELECT chunk_id FROM pinned_chunks ORDER BY priority DESC"
            params = ()

        async with db.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
            return [row[0] for row in rows]

    async def get_pinned_chunk_info(self, chunk_id: str) -> PinnedChunk | None:
        """
        Get pinned chunk information.

        Args:
            chunk_id: Chunk identifier.

        Returns:
            PinnedChunk or None.
        """
        db = self.sqlite_store._db
        if not db:
            raise RuntimeError("Database not initialized")

        async with db.execute(
            "SELECT * FROM pinned_chunks WHERE chunk_id = ?",
            (chunk_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return PinnedChunk(
                    chunk_id=row[0],
                    reason=row[1],
                    priority=row[2],
                    pinned_at=datetime.fromisoformat(row[3]),
                    metadata=json.loads(row[4] or "{}"),
                )
            return None

    async def is_pinned(self, chunk_id: str) -> bool:
        """Check if a chunk is pinned."""
        db = self.sqlite_store._db
        if not db:
            raise RuntimeError("Database not initialized")

        async with db.execute(
            "SELECT 1 FROM pinned_chunks WHERE chunk_id = ?",
            (chunk_id,),
        ) as cursor:
            row = await cursor.fetchone()
            return row is not None

    # ==================== Ledger Entries ====================

    async def add_ledger_entry(
        self,
        content: str,
        category: str | LedgerCategory = LedgerCategory.GENERAL,
        relevance_score: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Add an entry to the memory ledger.

        Args:
            content: Entry content.
            category: Entry category.
            relevance_score: Relevance score (0-1).
            metadata: Additional metadata.

        Returns:
            Entry ID.
        """
        db = self.sqlite_store._db
        if not db:
            raise RuntimeError("Database not initialized")

        if isinstance(category, str):
            category = LedgerCategory(category)

        entry_id = hashlib.sha256(
            f"{content}:{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]

        now = datetime.utcnow().isoformat()
        meta_json = json.dumps(metadata or {})

        await db.execute(
            """
            INSERT INTO ledger_entries
            (entry_id, content, category, relevance_score, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry_id,
                content,
                category.value,
                relevance_score,
                now,
                now,
                meta_json,
            ),
        )

        logger.info("Added ledger entry", entry_id=entry_id, category=category.value)
        return entry_id

    async def get_ledger_entry(self, entry_id: str) -> LedgerEntry | None:
        """
        Get a ledger entry by ID.

        Args:
            entry_id: Entry identifier.

        Returns:
            LedgerEntry or None.
        """
        db = self.sqlite_store._db
        if not db:
            raise RuntimeError("Database not initialized")

        async with db.execute(
            "SELECT * FROM ledger_entries WHERE entry_id = ?",
            (entry_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return LedgerEntry(
                    entry_id=row[0],
                    content=row[1],
                    category=LedgerCategory(row[2]),
                    embedding_id=row[3],
                    relevance_score=row[4],
                    created_at=datetime.fromisoformat(row[5]),
                    updated_at=datetime.fromisoformat(row[6]),
                    metadata=json.loads(row[7] or "{}"),
                )
            return None

    async def search_ledger(
        self,
        query: str,
        category: LedgerCategory | None = None,
        limit: int = 10,
    ) -> list[LedgerEntry]:
        """
        Search the memory ledger.

        Args:
            query: Search query.
            category: Optional category filter.
            limit: Maximum results.

        Returns:
            List of matching entries.
        """
        db = self.sqlite_store._db
        if not db:
            raise RuntimeError("Database not initialized")

        safe_query = query.replace('"', '""')

        if category:
            sql = """
                SELECT entry_id FROM ledger_fts
                WHERE ledger_fts MATCH ? AND category = ?
                ORDER BY bm25(ledger_fts)
                LIMIT ?
            """
            params = (safe_query, category.value, limit)
        else:
            sql = """
                SELECT entry_id FROM ledger_fts
                WHERE ledger_fts MATCH ?
                ORDER BY bm25(ledger_fts)
                LIMIT ?
            """
            params = (safe_query, limit)

        try:
            async with db.execute(sql, params) as cursor:
                rows = await cursor.fetchall()
        except Exception:
            return []

        entries = []
        for row in rows:
            entry = await self.get_ledger_entry(row[0])
            if entry:
                entries.append(entry)

        return entries

    async def get_recent_ledger_entries(
        self,
        limit: int = 10,
        category: LedgerCategory | None = None,
    ) -> list[LedgerEntry]:
        """
        Get recent ledger entries.

        Args:
            limit: Maximum results.
            category: Optional category filter.

        Returns:
            List of entries.
        """
        db = self.sqlite_store._db
        if not db:
            raise RuntimeError("Database not initialized")

        if category:
            sql = """
                SELECT entry_id FROM ledger_entries
                WHERE category = ?
                ORDER BY created_at DESC
                LIMIT ?
            """
            params = (category.value, limit)
        else:
            sql = """
                SELECT entry_id FROM ledger_entries
                ORDER BY created_at DESC
                LIMIT ?
            """
            params = (limit,)

        async with db.execute(sql, params) as cursor:
            rows = await cursor.fetchall()

        entries = []
        for row in rows:
            entry = await self.get_ledger_entry(row[0])
            if entry:
                entries.append(entry)

        return entries

    async def update_ledger_entry(
        self,
        entry_id: str,
        content: str | None = None,
        relevance_score: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Update a ledger entry.

        Args:
            entry_id: Entry identifier.
            content: New content (optional).
            relevance_score: New relevance score (optional).
            metadata: New metadata (optional).

        Returns:
            True if updated.
        """
        db = self.sqlite_store._db
        if not db:
            raise RuntimeError("Database not initialized")

        updates = []
        params: list[Any] = []

        if content is not None:
            updates.append("content = ?")
            params.append(content)

        if relevance_score is not None:
            updates.append("relevance_score = ?")
            params.append(relevance_score)

        if metadata is not None:
            updates.append("metadata = ?")
            params.append(json.dumps(metadata))

        if not updates:
            return False

        updates.append("updated_at = ?")
        params.append(datetime.utcnow().isoformat())
        params.append(entry_id)

        sql = f"UPDATE ledger_entries SET {', '.join(updates)} WHERE entry_id = ?"
        cursor = await db.execute(sql, params)

        return cursor.rowcount > 0

    async def delete_ledger_entry(self, entry_id: str) -> bool:
        """
        Delete a ledger entry.

        Args:
            entry_id: Entry identifier.

        Returns:
            True if deleted.
        """
        db = self.sqlite_store._db
        if not db:
            raise RuntimeError("Database not initialized")

        cursor = await db.execute(
            "DELETE FROM ledger_entries WHERE entry_id = ?",
            (entry_id,),
        )

        return cursor.rowcount > 0

    # ==================== Session State ====================

    async def create_session(
        self,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Create a new session.

        Args:
            session_id: Optional session ID (generated if not provided).
            metadata: Additional metadata.

        Returns:
            Session ID.
        """
        db = self.sqlite_store._db
        if not db:
            raise RuntimeError("Database not initialized")

        if not session_id:
            session_id = hashlib.sha256(
                datetime.utcnow().isoformat().encode()
            ).hexdigest()[:16]

        now = datetime.utcnow().isoformat()
        meta_json = json.dumps(metadata or {})

        await db.execute(
            """
            INSERT INTO sessions
            (session_id, query_history, retrieved_chunks, focus_paths,
             created_at, updated_at, metadata)
            VALUES (?, '[]', '[]', '[]', ?, ?, ?)
            """,
            (session_id, now, now, meta_json),
        )

        return session_id

    async def get_session(self, session_id: str) -> SessionState | None:
        """
        Get session state.

        Args:
            session_id: Session identifier.

        Returns:
            SessionState or None.
        """
        db = self.sqlite_store._db
        if not db:
            raise RuntimeError("Database not initialized")

        async with db.execute(
            "SELECT * FROM sessions WHERE session_id = ?",
            (session_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return SessionState(
                    session_id=row[0],
                    query_history=json.loads(row[1] or "[]"),
                    retrieved_chunks=json.loads(row[2] or "[]"),
                    focus_paths=json.loads(row[3] or "[]"),
                    created_at=datetime.fromisoformat(row[4]),
                    updated_at=datetime.fromisoformat(row[5]),
                    metadata=json.loads(row[6] or "{}"),
                )
            return None

    async def update_session(
        self,
        session_id: str,
        query: str | None = None,
        retrieved_chunks: list[str] | None = None,
        focus_paths: list[str] | None = None,
    ) -> bool:
        """
        Update session state.

        Args:
            session_id: Session identifier.
            query: Query to add to history.
            retrieved_chunks: Chunks to add to retrieved list.
            focus_paths: Paths to set as focus.

        Returns:
            True if updated.
        """
        db = self.sqlite_store._db
        if not db:
            raise RuntimeError("Database not initialized")

        session = await self.get_session(session_id)
        if not session:
            return False

        if query:
            session.query_history.append(query)

        if retrieved_chunks:
            session.retrieved_chunks.extend(retrieved_chunks)
            # Keep only unique chunks
            session.retrieved_chunks = list(dict.fromkeys(session.retrieved_chunks))

        if focus_paths is not None:
            session.focus_paths = focus_paths

        now = datetime.utcnow().isoformat()

        await db.execute(
            """
            UPDATE sessions SET
                query_history = ?,
                retrieved_chunks = ?,
                focus_paths = ?,
                updated_at = ?
            WHERE session_id = ?
            """,
            (
                json.dumps(session.query_history),
                json.dumps(session.retrieved_chunks),
                json.dumps(session.focus_paths),
                now,
                session_id,
            ),
        )

        return True

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.

        Args:
            session_id: Session identifier.

        Returns:
            True if deleted.
        """
        db = self.sqlite_store._db
        if not db:
            raise RuntimeError("Database not initialized")

        cursor = await db.execute(
            "DELETE FROM sessions WHERE session_id = ?",
            (session_id,),
        )

        return cursor.rowcount > 0

    async def get_stats(self) -> dict[str, Any]:
        """Get memory store statistics."""
        db = self.sqlite_store._db
        if not db:
            raise RuntimeError("Database not initialized")

        stats: dict[str, Any] = {}

        async with db.execute("SELECT COUNT(*) FROM pinned_chunks") as cursor:
            row = await cursor.fetchone()
            stats["pinned_chunks"] = row[0] if row else 0

        async with db.execute("SELECT COUNT(*) FROM ledger_entries") as cursor:
            row = await cursor.fetchone()
            stats["ledger_entries"] = row[0] if row else 0

        async with db.execute(
            "SELECT category, COUNT(*) FROM ledger_entries GROUP BY category"
        ) as cursor:
            rows = await cursor.fetchall()
            stats["entries_by_category"] = {row[0]: row[1] for row in rows}

        async with db.execute("SELECT COUNT(*) FROM sessions") as cursor:
            row = await cursor.fetchone()
            stats["sessions"] = row[0] if row else 0

        return stats
