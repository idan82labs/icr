"""
Pytest configuration and fixtures for ICD tests.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import AsyncIterator, Iterator

import numpy as np
import pytest

from icd.config import Config


@pytest.fixture(scope="session")
def event_loop() -> Iterator[asyncio.AbstractEventLoop]:
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Iterator[Path]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_config(temp_dir: Path) -> Config:
    """Create a test configuration."""
    return Config(
        project_root=temp_dir,
        data_dir=temp_dir / ".icd",
        log_level="DEBUG",
    )


@pytest.fixture
def sample_code_files(temp_dir: Path) -> dict[str, Path]:
    """Create sample code files for testing."""
    files = {}

    # Python file
    python_content = '''
"""Sample Python module."""

from typing import Protocol


class UserProtocol(Protocol):
    """Protocol for user objects."""

    @property
    def id(self) -> int:
        ...

    @property
    def name(self) -> str:
        ...


class User:
    """User implementation."""

    def __init__(self, id: int, name: str) -> None:
        self._id = id
        self._name = name

    @property
    def id(self) -> int:
        return self._id

    @property
    def name(self) -> str:
        return self._name


def get_user_by_id(user_id: int) -> User | None:
    """Get a user by their ID."""
    # Implementation would go here
    return None


def list_all_users() -> list[User]:
    """List all users in the system."""
    return []
'''
    python_path = temp_dir / "user.py"
    python_path.write_text(python_content)
    files["python"] = python_path

    # TypeScript file
    typescript_content = '''
interface UserInterface {
    id: number;
    name: string;
    email: string;
}

type UserRole = "admin" | "user" | "guest";

class UserService {
    private users: Map<number, UserInterface> = new Map();

    async getUser(id: number): Promise<UserInterface | undefined> {
        return this.users.get(id);
    }

    async createUser(data: Omit<UserInterface, "id">): Promise<UserInterface> {
        const id = Date.now();
        const user = { id, ...data };
        this.users.set(id, user);
        return user;
    }

    async deleteUser(id: number): Promise<boolean> {
        return this.users.delete(id);
    }
}

export { UserInterface, UserRole, UserService };
'''
    typescript_path = temp_dir / "user.ts"
    typescript_path.write_text(typescript_content)
    files["typescript"] = typescript_path

    # Go file
    go_content = '''
package user

import "errors"

// User represents a user in the system.
type User struct {
    ID    int64  `json:"id"`
    Name  string `json:"name"`
    Email string `json:"email"`
}

// UserRepository defines the interface for user storage.
type UserRepository interface {
    GetByID(id int64) (*User, error)
    Create(user *User) error
    Delete(id int64) error
}

// InMemoryRepository implements UserRepository in memory.
type InMemoryRepository struct {
    users map[int64]*User
}

// NewInMemoryRepository creates a new in-memory repository.
func NewInMemoryRepository() *InMemoryRepository {
    return &InMemoryRepository{
        users: make(map[int64]*User),
    }
}

// GetByID retrieves a user by ID.
func (r *InMemoryRepository) GetByID(id int64) (*User, error) {
    user, ok := r.users[id]
    if !ok {
        return nil, errors.New("user not found")
    }
    return user, nil
}

// Create adds a new user.
func (r *InMemoryRepository) Create(user *User) error {
    r.users[user.ID] = user
    return nil
}

// Delete removes a user.
func (r *InMemoryRepository) Delete(id int64) error {
    delete(r.users, id)
    return nil
}
'''
    go_path = temp_dir / "user.go"
    go_path.write_text(go_content)
    files["go"] = go_path

    return files


@pytest.fixture
def sample_embeddings() -> list[np.ndarray]:
    """Generate sample embedding vectors."""
    np.random.seed(42)
    return [
        np.random.randn(384).astype(np.float32)
        for _ in range(10)
    ]


@pytest.fixture
async def sqlite_store(test_config: Config) -> AsyncIterator:
    """Create a SQLite store for testing."""
    from icd.storage.sqlite_store import SQLiteStore

    store = SQLiteStore(test_config)
    await store.initialize()
    yield store
    await store.close()


@pytest.fixture
async def vector_store(test_config: Config) -> AsyncIterator:
    """Create a vector store for testing."""
    from icd.storage.vector_store import VectorStore

    store = VectorStore(test_config)
    await store.initialize()
    yield store
    await store.close()


@pytest.fixture
def chunker(test_config: Config):
    """Create a chunker for testing."""
    from icd.indexing.chunker import Chunker

    return Chunker(test_config)


@pytest.fixture
def mock_embedder():
    """Create a mock embedder for testing."""
    from unittest.mock import AsyncMock, MagicMock

    embedder = MagicMock()
    embedder.dimension = 384

    async def mock_embed(text: str) -> np.ndarray:
        # Generate deterministic embedding based on text hash
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(384).astype(np.float32)

    async def mock_embed_batch(texts: list[str]) -> list[np.ndarray]:
        return [await mock_embed(t) for t in texts]

    embedder.embed = mock_embed
    embedder.embed_batch = mock_embed_batch
    embedder.initialize = AsyncMock()

    return embedder


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    from icd.retrieval.hybrid import Chunk

    return [
        Chunk(
            chunk_id=f"chunk_{i}",
            file_path=f"/path/to/file_{i}.py",
            content=f"def function_{i}(): pass",
            start_line=1,
            end_line=1,
            symbol_name=f"function_{i}",
            symbol_type="function_definition",
            language="python",
            token_count=10,
            is_contract=i % 3 == 0,
            is_pinned=i % 5 == 0,
        )
        for i in range(10)
    ]
