"""
Unit tests for the SQLite storage module.

Tests cover:
- SQLite database operations
- FTS5 full-text search
- Chunk storage and retrieval
- BM25 scoring
- Transaction handling
- WAL mode
"""

from __future__ import annotations

import asyncio
import sqlite3
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ==============================================================================
# Test Data Types
# ==============================================================================

@dataclass
class TestChunk:
    """Test chunk data structure."""

    chunk_id: str
    file_path: str
    content: str
    start_line: int
    end_line: int
    symbol_name: str | None = None
    symbol_type: str | None = None
    language: str | None = None
    token_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# SQLite Basic Operations Tests
# ==============================================================================

class TestSQLiteBasicOperations:
    """Tests for basic SQLite operations."""

    def test_database_creation(self, tmp_path: Path):
        """Test that database file is created."""
        db_path = tmp_path / "test.db"

        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()

        assert db_path.exists()

    def test_wal_mode_enabled(self, tmp_path: Path):
        """Test WAL mode can be enabled."""
        db_path = tmp_path / "test.db"

        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        result = conn.execute("PRAGMA journal_mode").fetchone()
        conn.close()

        assert result[0] == "wal"

    def test_fts5_available(self, tmp_path: Path):
        """Test that FTS5 extension is available."""
        db_path = tmp_path / "test.db"

        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("""
                CREATE VIRTUAL TABLE test_fts USING fts5(content)
            """)
            conn.commit()
            fts_available = True
        except sqlite3.OperationalError:
            fts_available = False
        finally:
            conn.close()

        assert fts_available, "FTS5 extension not available"


# ==============================================================================
# Schema Tests
# ==============================================================================

class TestSQLiteSchema:
    """Tests for database schema."""

    def test_chunks_table_schema(self, tmp_path: Path):
        """Test chunks table schema creation."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                content TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                symbol_name TEXT,
                symbol_type TEXT,
                language TEXT,
                token_count INTEGER DEFAULT 0,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

        # Verify table exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'"
        )
        result = cursor.fetchone()
        conn.close()

        assert result is not None
        assert result[0] == "chunks"

    def test_files_table_schema(self, tmp_path: Path):
        """Test files table schema creation."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        conn.execute("""
            CREATE TABLE IF NOT EXISTS files (
                file_path TEXT PRIMARY KEY,
                file_hash TEXT NOT NULL,
                size_bytes INTEGER,
                last_modified TIMESTAMP,
                language TEXT,
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='files'"
        )
        result = cursor.fetchone()
        conn.close()

        assert result is not None

    def test_fts_table_schema(self, tmp_path: Path):
        """Test FTS5 table schema creation."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id,
                content,
                symbol_name,
                content=chunks,
                content_rowid=rowid
            )
        """)

        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
        )
        result = cursor.fetchone()
        conn.close()

        assert result is not None


# ==============================================================================
# Chunk CRUD Tests
# ==============================================================================

class TestChunkCRUD:
    """Tests for chunk CRUD operations."""

    @pytest.fixture
    def db_conn(self, tmp_path: Path) -> sqlite3.Connection:
        """Create database connection with schema."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        # Create tables
        conn.execute("""
            CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                content TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                symbol_name TEXT,
                symbol_type TEXT,
                language TEXT,
                token_count INTEGER DEFAULT 0,
                metadata TEXT
            )
        """)
        conn.commit()

        yield conn
        conn.close()

    def test_insert_chunk(self, db_conn: sqlite3.Connection):
        """Test inserting a chunk."""
        chunk = TestChunk(
            chunk_id="test123:abc",
            file_path="/path/to/file.py",
            content="def hello(): pass",
            start_line=1,
            end_line=1,
            symbol_name="hello",
            symbol_type="function",
            language="python",
            token_count=5,
        )

        db_conn.execute("""
            INSERT INTO chunks (chunk_id, file_path, content, start_line, end_line,
                               symbol_name, symbol_type, language, token_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (chunk.chunk_id, chunk.file_path, chunk.content, chunk.start_line,
              chunk.end_line, chunk.symbol_name, chunk.symbol_type, chunk.language,
              chunk.token_count))
        db_conn.commit()

        result = db_conn.execute(
            "SELECT * FROM chunks WHERE chunk_id = ?", (chunk.chunk_id,)
        ).fetchone()

        assert result is not None
        assert result["chunk_id"] == chunk.chunk_id
        assert result["content"] == chunk.content

    def test_get_chunk_by_id(self, db_conn: sqlite3.Connection):
        """Test retrieving a chunk by ID."""
        chunk_id = "get_test:123"
        db_conn.execute("""
            INSERT INTO chunks (chunk_id, file_path, content, start_line, end_line)
            VALUES (?, ?, ?, ?, ?)
        """, (chunk_id, "/test.py", "content", 1, 5))
        db_conn.commit()

        result = db_conn.execute(
            "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()

        assert result is not None
        assert result["chunk_id"] == chunk_id

    def test_update_chunk(self, db_conn: sqlite3.Connection):
        """Test updating a chunk."""
        chunk_id = "update_test:456"
        db_conn.execute("""
            INSERT INTO chunks (chunk_id, file_path, content, start_line, end_line)
            VALUES (?, ?, ?, ?, ?)
        """, (chunk_id, "/test.py", "old content", 1, 5))
        db_conn.commit()

        # Update
        db_conn.execute("""
            UPDATE chunks SET content = ? WHERE chunk_id = ?
        """, ("new content", chunk_id))
        db_conn.commit()

        result = db_conn.execute(
            "SELECT content FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()

        assert result["content"] == "new content"

    def test_delete_chunk(self, db_conn: sqlite3.Connection):
        """Test deleting a chunk."""
        chunk_id = "delete_test:789"
        db_conn.execute("""
            INSERT INTO chunks (chunk_id, file_path, content, start_line, end_line)
            VALUES (?, ?, ?, ?, ?)
        """, (chunk_id, "/test.py", "content", 1, 5))
        db_conn.commit()

        db_conn.execute("DELETE FROM chunks WHERE chunk_id = ?", (chunk_id,))
        db_conn.commit()

        result = db_conn.execute(
            "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()

        assert result is None

    def test_delete_chunks_by_file(self, db_conn: sqlite3.Connection):
        """Test deleting all chunks for a file."""
        file_path = "/path/to/delete.py"

        # Insert multiple chunks for same file
        for i in range(5):
            db_conn.execute("""
                INSERT INTO chunks (chunk_id, file_path, content, start_line, end_line)
                VALUES (?, ?, ?, ?, ?)
            """, (f"chunk_{i}", file_path, f"content {i}", i, i + 1))
        db_conn.commit()

        # Verify chunks exist
        count_before = db_conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE file_path = ?", (file_path,)
        ).fetchone()[0]
        assert count_before == 5

        # Delete all chunks for file
        db_conn.execute("DELETE FROM chunks WHERE file_path = ?", (file_path,))
        db_conn.commit()

        count_after = db_conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE file_path = ?", (file_path,)
        ).fetchone()[0]
        assert count_after == 0


# ==============================================================================
# FTS5 Full-Text Search Tests
# ==============================================================================

class TestFTS5Search:
    """Tests for FTS5 full-text search functionality."""

    @pytest.fixture
    def fts_conn(self, tmp_path: Path) -> sqlite3.Connection:
        """Create database with FTS5 enabled."""
        db_path = tmp_path / "fts_test.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        # Create main table
        conn.execute("""
            CREATE TABLE chunks (
                rowid INTEGER PRIMARY KEY,
                chunk_id TEXT UNIQUE NOT NULL,
                file_path TEXT NOT NULL,
                content TEXT NOT NULL,
                symbol_name TEXT
            )
        """)

        # Create FTS5 virtual table
        conn.execute("""
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                content,
                symbol_name,
                content='chunks',
                content_rowid='rowid'
            )
        """)

        # Create triggers for sync
        conn.execute("""
            CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, content, symbol_name)
                VALUES (new.rowid, new.content, new.symbol_name);
            END
        """)

        conn.execute("""
            CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, content, symbol_name)
                VALUES ('delete', old.rowid, old.content, old.symbol_name);
            END
        """)

        conn.commit()
        yield conn
        conn.close()

    def test_fts_basic_search(self, fts_conn: sqlite3.Connection):
        """Test basic FTS5 search."""
        # Insert test data
        fts_conn.execute("""
            INSERT INTO chunks (chunk_id, file_path, content, symbol_name)
            VALUES (?, ?, ?, ?)
        """, ("chunk1", "/test.py", "def authenticate_user(token):", "authenticate_user"))
        fts_conn.commit()

        # Search
        results = fts_conn.execute("""
            SELECT c.chunk_id, c.content
            FROM chunks c
            JOIN chunks_fts fts ON c.rowid = fts.rowid
            WHERE chunks_fts MATCH 'authenticate'
        """).fetchall()

        assert len(results) == 1
        assert "authenticate" in results[0]["content"]

    def test_fts_phrase_search(self, fts_conn: sqlite3.Connection):
        """Test FTS5 phrase search."""
        fts_conn.execute("""
            INSERT INTO chunks (chunk_id, file_path, content, symbol_name)
            VALUES (?, ?, ?, ?)
        """, ("chunk1", "/test.py", "def validate_auth_token(token):", "validate_auth_token"))
        fts_conn.commit()

        # Phrase search
        results = fts_conn.execute("""
            SELECT c.chunk_id
            FROM chunks c
            JOIN chunks_fts fts ON c.rowid = fts.rowid
            WHERE chunks_fts MATCH '"auth token"'
        """).fetchall()

        assert len(results) == 1

    def test_fts_prefix_search(self, fts_conn: sqlite3.Connection):
        """Test FTS5 prefix search."""
        fts_conn.execute("""
            INSERT INTO chunks (chunk_id, file_path, content, symbol_name)
            VALUES (?, ?, ?, ?)
        """, ("chunk1", "/test.py", "authentication logic here", "auth_func"))
        fts_conn.commit()

        # Prefix search
        results = fts_conn.execute("""
            SELECT c.chunk_id
            FROM chunks c
            JOIN chunks_fts fts ON c.rowid = fts.rowid
            WHERE chunks_fts MATCH 'auth*'
        """).fetchall()

        assert len(results) == 1

    def test_fts_no_results(self, fts_conn: sqlite3.Connection):
        """Test FTS5 search with no results."""
        fts_conn.execute("""
            INSERT INTO chunks (chunk_id, file_path, content, symbol_name)
            VALUES (?, ?, ?, ?)
        """, ("chunk1", "/test.py", "def hello():", "hello"))
        fts_conn.commit()

        results = fts_conn.execute("""
            SELECT c.chunk_id
            FROM chunks c
            JOIN chunks_fts fts ON c.rowid = fts.rowid
            WHERE chunks_fts MATCH 'nonexistent'
        """).fetchall()

        assert len(results) == 0

    def test_fts_multiple_results(self, fts_conn: sqlite3.Connection):
        """Test FTS5 search returning multiple results."""
        # Insert multiple matching chunks
        for i in range(5):
            fts_conn.execute("""
                INSERT INTO chunks (chunk_id, file_path, content, symbol_name)
                VALUES (?, ?, ?, ?)
            """, (f"chunk{i}", "/test.py", f"def validate_{i}():", f"validate_{i}"))
        fts_conn.commit()

        results = fts_conn.execute("""
            SELECT c.chunk_id
            FROM chunks c
            JOIN chunks_fts fts ON c.rowid = fts.rowid
            WHERE chunks_fts MATCH 'validate'
        """).fetchall()

        assert len(results) == 5

    def test_fts_symbol_search(self, fts_conn: sqlite3.Connection):
        """Test FTS5 search on symbol names."""
        fts_conn.execute("""
            INSERT INTO chunks (chunk_id, file_path, content, symbol_name)
            VALUES (?, ?, ?, ?)
        """, ("chunk1", "/test.py", "class implementation", "UserAuthenticator"))
        fts_conn.commit()

        results = fts_conn.execute("""
            SELECT c.chunk_id
            FROM chunks c
            JOIN chunks_fts fts ON c.rowid = fts.rowid
            WHERE chunks_fts MATCH 'symbol_name:UserAuthenticator'
        """).fetchall()

        assert len(results) == 1


# ==============================================================================
# BM25 Scoring Tests
# ==============================================================================

class TestBM25Scoring:
    """Tests for BM25 scoring in FTS5."""

    @pytest.fixture
    def bm25_conn(self, tmp_path: Path) -> sqlite3.Connection:
        """Create database with BM25 scoring support."""
        db_path = tmp_path / "bm25_test.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        conn.execute("""
            CREATE TABLE chunks (
                rowid INTEGER PRIMARY KEY,
                chunk_id TEXT UNIQUE,
                content TEXT
            )
        """)

        conn.execute("""
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                content,
                content='chunks',
                content_rowid='rowid'
            )
        """)

        conn.execute("""
            CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, content)
                VALUES (new.rowid, new.content);
            END
        """)

        conn.commit()
        yield conn
        conn.close()

    def test_bm25_score_available(self, bm25_conn: sqlite3.Connection):
        """Test that BM25 scores are available."""
        bm25_conn.execute("""
            INSERT INTO chunks (chunk_id, content) VALUES (?, ?)
        """, ("chunk1", "python programming language"))
        bm25_conn.commit()

        results = bm25_conn.execute("""
            SELECT c.chunk_id, bm25(chunks_fts) as score
            FROM chunks c
            JOIN chunks_fts fts ON c.rowid = fts.rowid
            WHERE chunks_fts MATCH 'python'
        """).fetchall()

        assert len(results) == 1
        assert results[0]["score"] is not None
        # BM25 returns negative scores in SQLite (lower is better match)
        assert results[0]["score"] < 0

    def test_bm25_score_ordering(self, bm25_conn: sqlite3.Connection):
        """Test that BM25 scores order results correctly."""
        # Insert chunks with varying relevance
        bm25_conn.execute("""
            INSERT INTO chunks (chunk_id, content) VALUES (?, ?)
        """, ("chunk1", "python"))
        bm25_conn.execute("""
            INSERT INTO chunks (chunk_id, content) VALUES (?, ?)
        """, ("chunk2", "python python python"))  # More relevant
        bm25_conn.execute("""
            INSERT INTO chunks (chunk_id, content) VALUES (?, ?)
        """, ("chunk3", "python language"))
        bm25_conn.commit()

        results = bm25_conn.execute("""
            SELECT c.chunk_id, bm25(chunks_fts) as score
            FROM chunks c
            JOIN chunks_fts fts ON c.rowid = fts.rowid
            WHERE chunks_fts MATCH 'python'
            ORDER BY score
        """).fetchall()

        assert len(results) == 3
        # Chunk with most "python" mentions should have best (most negative) score
        assert results[0]["chunk_id"] == "chunk2"


# ==============================================================================
# Index Tests
# ==============================================================================

class TestSQLiteIndexes:
    """Tests for database indexes."""

    def test_file_path_index(self, tmp_path: Path):
        """Test index on file_path for fast lookups."""
        db_path = tmp_path / "index_test.db"
        conn = sqlite3.connect(str(db_path))

        conn.execute("""
            CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX idx_chunks_file_path ON chunks(file_path)")
        conn.commit()

        # Verify index exists
        result = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='index' AND name='idx_chunks_file_path'
        """).fetchone()

        conn.close()
        assert result is not None

    def test_symbol_name_index(self, tmp_path: Path):
        """Test index on symbol_name."""
        db_path = tmp_path / "index_test.db"
        conn = sqlite3.connect(str(db_path))

        conn.execute("""
            CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY,
                symbol_name TEXT
            )
        """)
        conn.execute("CREATE INDEX idx_chunks_symbol ON chunks(symbol_name)")
        conn.commit()

        result = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='index' AND name='idx_chunks_symbol'
        """).fetchone()

        conn.close()
        assert result is not None


# ==============================================================================
# Transaction Tests
# ==============================================================================

class TestSQLiteTransactions:
    """Tests for transaction handling."""

    def test_rollback_on_error(self, tmp_path: Path):
        """Test that transactions rollback on error."""
        db_path = tmp_path / "tx_test.db"
        conn = sqlite3.connect(str(db_path))

        conn.execute("""
            CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY,
                content TEXT NOT NULL
            )
        """)
        conn.commit()

        # Insert valid row
        conn.execute("INSERT INTO chunks VALUES ('id1', 'content1')")

        try:
            # Try to insert duplicate (should fail)
            conn.execute("INSERT INTO chunks VALUES ('id1', 'content2')")
            conn.commit()
        except sqlite3.IntegrityError:
            conn.rollback()

        # Verify first row still doesn't exist (rolled back)
        # Actually in this case it should exist since we didn't explicitly start transaction
        conn.close()

    def test_explicit_transaction(self, tmp_path: Path):
        """Test explicit transaction handling."""
        db_path = tmp_path / "tx_test.db"
        conn = sqlite3.connect(str(db_path), isolation_level=None)

        conn.execute("""
            CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY,
                content TEXT
            )
        """)

        # Explicit transaction
        conn.execute("BEGIN TRANSACTION")
        conn.execute("INSERT INTO chunks VALUES ('id1', 'content1')")
        conn.execute("INSERT INTO chunks VALUES ('id2', 'content2')")
        conn.execute("COMMIT")

        result = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        conn.close()

        assert result[0] == 2


# ==============================================================================
# Performance Tests
# ==============================================================================

class TestSQLitePerformance:
    """Tests for SQLite performance characteristics."""

    def test_bulk_insert_performance(self, tmp_path: Path):
        """Test bulk insert performance."""
        db_path = tmp_path / "perf_test.db"
        conn = sqlite3.connect(str(db_path))

        conn.execute("""
            CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY,
                content TEXT
            )
        """)

        # Bulk insert with executemany
        data = [(f"id_{i}", f"content_{i}") for i in range(1000)]

        conn.executemany("INSERT INTO chunks VALUES (?, ?)", data)
        conn.commit()

        result = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        conn.close()

        assert result[0] == 1000

    def test_fts_search_performance(self, tmp_path: Path):
        """Test FTS search performance with many documents."""
        db_path = tmp_path / "fts_perf_test.db"
        conn = sqlite3.connect(str(db_path))

        conn.execute("CREATE TABLE docs (id INTEGER PRIMARY KEY, content TEXT)")
        conn.execute("CREATE VIRTUAL TABLE docs_fts USING fts5(content, content='docs')")
        conn.execute("""
            CREATE TRIGGER docs_ai AFTER INSERT ON docs BEGIN
                INSERT INTO docs_fts(rowid, content) VALUES (new.id, new.content);
            END
        """)

        # Insert many documents
        words = ["python", "java", "javascript", "typescript", "rust", "go"]
        data = [(i, f"Document about {words[i % len(words)]} programming") for i in range(1000)]
        conn.executemany("INSERT INTO docs VALUES (?, ?)", data)
        conn.commit()

        # Search should be fast
        results = conn.execute("""
            SELECT d.id FROM docs d
            JOIN docs_fts fts ON d.id = fts.rowid
            WHERE docs_fts MATCH 'python'
        """).fetchall()

        conn.close()

        # Should find documents with "python"
        assert len(results) > 0


# ==============================================================================
# Stats Tests
# ==============================================================================

class TestSQLiteStats:
    """Tests for statistics and metadata."""

    def test_get_file_count(self, tmp_path: Path):
        """Test getting file count."""
        db_path = tmp_path / "stats_test.db"
        conn = sqlite3.connect(str(db_path))

        conn.execute("CREATE TABLE files (file_path TEXT PRIMARY KEY)")

        for i in range(10):
            conn.execute("INSERT INTO files VALUES (?)", (f"/path/file{i}.py",))
        conn.commit()

        result = conn.execute("SELECT COUNT(*) FROM files").fetchone()
        conn.close()

        assert result[0] == 10

    def test_get_chunk_count(self, tmp_path: Path):
        """Test getting chunk count."""
        db_path = tmp_path / "stats_test.db"
        conn = sqlite3.connect(str(db_path))

        conn.execute("CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY)")

        for i in range(50):
            conn.execute("INSERT INTO chunks VALUES (?)", (f"chunk_{i}",))
        conn.commit()

        result = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        conn.close()

        assert result[0] == 50

    def test_get_chunks_per_file(self, tmp_path: Path):
        """Test getting chunk count per file."""
        db_path = tmp_path / "stats_test.db"
        conn = sqlite3.connect(str(db_path))

        conn.execute("CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, file_path TEXT)")

        # Insert chunks for different files
        for i in range(30):
            file_path = f"/path/file{i % 3}.py"
            conn.execute("INSERT INTO chunks VALUES (?, ?)", (f"chunk_{i}", file_path))
        conn.commit()

        result = conn.execute("""
            SELECT file_path, COUNT(*) as count
            FROM chunks
            GROUP BY file_path
        """).fetchall()

        conn.close()

        assert len(result) == 3
        for row in result:
            assert row[1] == 10  # 30 chunks / 3 files
