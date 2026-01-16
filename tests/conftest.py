"""
Shared fixtures for ICR test suite.

Provides common test fixtures including:
- Temporary ICR root directories
- Sample repository fixtures
- Pre-indexed test data
- Mock embedding backend
- Configuration overrides
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

# Ensure event loop is available for async fixtures
pytest_plugins = ["pytest_asyncio"]


# ==============================================================================
# Type Definitions
# ==============================================================================

@dataclass
class Chunk:
    """Test chunk representation."""

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

    def __post_init__(self):
        if self.token_count == 0:
            # Rough estimate: 4 chars per token
            self.token_count = len(self.content) // 4


@dataclass
class RetrievalResult:
    """Test retrieval result."""

    chunks: list[Chunk]
    scores: list[float]
    entropy: float
    query: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PackResult:
    """Test pack result."""

    content: str
    token_count: int
    chunk_ids: list[str]
    citations: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# Path Fixtures
# ==============================================================================

@pytest.fixture
def project_root() -> Path:
    """Get the ICR project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def fixtures_dir() -> Path:
    """Get the fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def tmp_icr_root(tmp_path: Path) -> Path:
    """Create a temporary ICR root directory with proper structure."""
    icr_root = tmp_path / ".icr"
    icr_root.mkdir(parents=True)

    # Create subdirectories
    (icr_root / "repos").mkdir()
    (icr_root / "cache").mkdir()
    (icr_root / "logs").mkdir()

    return icr_root


@pytest.fixture
def tmp_repo_dir(tmp_path: Path) -> Path:
    """Create a temporary repository directory."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir(parents=True)
    return repo_dir


# ==============================================================================
# Sample Repository Fixtures
# ==============================================================================

SAMPLE_AUTH_HANDLER_TS = '''import { AuthToken, User } from '../types/shared';
import { validateToken } from './validator';

/**
 * Validates the authentication token and returns user info.
 * @param token - The authentication token to validate
 * @returns User object if valid, null otherwise
 */
export async function handleAuth(token: AuthToken): Promise<User | null> {
  if (!validateToken(token)) {
    return null;
  }

  // Verify token expiration
  if (token.expiresAt < Date.now()) {
    console.warn('Token expired');
    return null;
  }

  // Fetch user from database
  const user = await fetchUserById(token.userId);
  return user;
}

export async function refreshToken(token: AuthToken): Promise<AuthToken | null> {
  const user = await handleAuth(token);
  if (!user) {
    return null;
  }

  return generateNewToken(user);
}

async function fetchUserById(userId: string): Promise<User | null> {
  // Database lookup logic
  return null;
}

function generateNewToken(user: User): AuthToken {
  return {
    userId: user.id,
    token: crypto.randomUUID(),
    expiresAt: Date.now() + 3600000,
  };
}
'''

SAMPLE_VALIDATOR_TS = '''import { AuthToken } from '../types/shared';

/**
 * Validates the structure and signature of an auth token.
 */
export function validateToken(token: AuthToken): boolean {
  if (!token || !token.token || !token.userId) {
    return false;
  }

  // Verify token format
  if (typeof token.token !== 'string' || token.token.length < 32) {
    return false;
  }

  // Verify signature (simplified)
  return verifySignature(token);
}

function verifySignature(token: AuthToken): boolean {
  // Cryptographic verification would happen here
  return true;
}
'''

SAMPLE_ENDPOINTS_TS = '''import { handleAuth, refreshToken } from '../auth/handler';
import { AuthToken, User, ApiResponse } from '../types/shared';

/**
 * POST /api/auth
 * Authenticates a user with the provided token.
 */
export async function authenticateEndpoint(
  request: { body: { token: string } }
): Promise<ApiResponse<User>> {
  const token: AuthToken = {
    token: request.body.token,
    userId: extractUserIdFromToken(request.body.token),
    expiresAt: extractExpiryFromToken(request.body.token),
  };

  const user = await handleAuth(token);

  if (!user) {
    return {
      success: false,
      error: 'Authentication failed',
      data: null,
    };
  }

  return {
    success: true,
    data: user,
  };
}

/**
 * POST /api/auth/refresh
 * Refreshes an expired or expiring token.
 */
export async function refreshEndpoint(
  request: { body: { token: string } }
): Promise<ApiResponse<AuthToken>> {
  const token = parseToken(request.body.token);
  const newToken = await refreshToken(token);

  if (!newToken) {
    return {
      success: false,
      error: 'Token refresh failed',
      data: null,
    };
  }

  return {
    success: true,
    data: newToken,
  };
}

function extractUserIdFromToken(token: string): string {
  // JWT parsing logic
  return 'user-id';
}

function extractExpiryFromToken(token: string): number {
  // JWT parsing logic
  return Date.now() + 3600000;
}

function parseToken(tokenString: string): AuthToken {
  return {
    token: tokenString,
    userId: extractUserIdFromToken(tokenString),
    expiresAt: extractExpiryFromToken(tokenString),
  };
}
'''

SAMPLE_SHARED_TYPES_TS = '''/**
 * Shared type definitions for the authentication system.
 */

export interface AuthToken {
  token: string;
  userId: string;
  expiresAt: number;
}

export interface User {
  id: string;
  email: string;
  name: string;
  roles: string[];
  createdAt: number;
  lastLogin: number;
}

export interface ApiResponse<T> {
  success: boolean;
  data: T | null;
  error?: string;
  metadata?: Record<string, unknown>;
}

export type AuthRole = 'admin' | 'user' | 'guest';

export interface Permission {
  resource: string;
  action: 'read' | 'write' | 'delete' | 'admin';
}
'''

SAMPLE_API_YAML = '''openapi: 3.0.0
info:
  title: Auth API
  version: 1.0.0
  description: Authentication API for the application

paths:
  /api/auth:
    post:
      operationId: authenticate
      summary: Authenticate user
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/AuthRequest'
      responses:
        '200':
          description: Successful authentication
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/AuthResponse'
        '401':
          description: Authentication failed

  /api/auth/refresh:
    post:
      operationId: refreshToken
      summary: Refresh authentication token
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/RefreshRequest'
      responses:
        '200':
          description: Token refreshed
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/TokenResponse'

components:
  schemas:
    AuthRequest:
      type: object
      required:
        - token
      properties:
        token:
          type: string
          description: Authentication token

    AuthResponse:
      type: object
      properties:
        success:
          type: boolean
        data:
          $ref: '#/components/schemas/User'
        error:
          type: string

    RefreshRequest:
      type: object
      required:
        - token
      properties:
        token:
          type: string

    TokenResponse:
      type: object
      properties:
        token:
          type: string
        expiresAt:
          type: integer

    User:
      type: object
      properties:
        id:
          type: string
        email:
          type: string
        name:
          type: string
        roles:
          type: array
          items:
            type: string
'''

SAMPLE_PACKAGE_JSON = '''{
  "name": "sample-repo",
  "version": "1.0.0",
  "description": "Sample repository for ICR testing",
  "main": "dist/index.js",
  "scripts": {
    "build": "tsc",
    "test": "jest",
    "lint": "eslint src/",
    "start": "node dist/index.js"
  },
  "dependencies": {
    "express": "^4.18.0"
  },
  "devDependencies": {
    "@types/node": "^20.0.0",
    "typescript": "^5.0.0",
    "jest": "^29.0.0"
  }
}
'''


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    """Create a sample repository with realistic TypeScript code."""
    repo_dir = tmp_path / "sample_repo"

    # Create directory structure
    (repo_dir / "src" / "auth").mkdir(parents=True)
    (repo_dir / "src" / "api").mkdir(parents=True)
    (repo_dir / "src" / "types").mkdir(parents=True)
    (repo_dir / "contracts").mkdir(parents=True)

    # Write source files
    (repo_dir / "src" / "auth" / "handler.ts").write_text(SAMPLE_AUTH_HANDLER_TS)
    (repo_dir / "src" / "auth" / "validator.ts").write_text(SAMPLE_VALIDATOR_TS)
    (repo_dir / "src" / "api" / "endpoints.ts").write_text(SAMPLE_ENDPOINTS_TS)
    (repo_dir / "src" / "types" / "shared.ts").write_text(SAMPLE_SHARED_TYPES_TS)
    (repo_dir / "contracts" / "api.yaml").write_text(SAMPLE_API_YAML)
    (repo_dir / "package.json").write_text(SAMPLE_PACKAGE_JSON)

    return repo_dir


@pytest.fixture
def sample_transcript(tmp_path: Path) -> Path:
    """Create a sample conversation transcript."""
    transcript_path = tmp_path / "transcript.jsonl"

    entries = [
        {
            "timestamp": "2026-01-16T10:00:00Z",
            "role": "user",
            "content": "Where is the auth token validated?",
        },
        {
            "timestamp": "2026-01-16T10:00:05Z",
            "role": "assistant",
            "content": "The auth token is validated in `src/auth/validator.ts` in the `validateToken` function.",
            "tool_calls": [
                {"name": "project_symbol_search", "args": {"query": "validateToken"}}
            ],
        },
        {
            "timestamp": "2026-01-16T10:01:00Z",
            "role": "user",
            "content": "How does the refresh endpoint work?",
        },
        {
            "timestamp": "2026-01-16T10:01:10Z",
            "role": "assistant",
            "content": "The refresh endpoint is in `src/api/endpoints.ts`. It calls `refreshToken` from the auth handler.",
            "ledger": {
                "decisions": ["Use handleAuth before refreshing token"],
                "invariants": ["Token must not be expired for refresh"],
            },
        },
    ]

    with open(transcript_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    return transcript_path


# ==============================================================================
# Mock Embedding Backend
# ==============================================================================

class MockEmbeddingBackend:
    """
    Mock embedding backend for fast tests.

    Generates deterministic embeddings based on content hash.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._initialized = False
        self._call_count = 0

    async def initialize(self) -> None:
        """Initialize the mock backend."""
        self._initialized = True

    async def embed(self, texts: list[str]) -> np.ndarray:
        """Generate mock embeddings for texts."""
        self._call_count += len(texts)
        embeddings = []

        for text in texts:
            # Generate deterministic embedding from content hash
            hash_bytes = hashlib.sha256(text.encode()).digest()
            # Use hash to seed random generator for reproducibility
            rng = np.random.default_rng(int.from_bytes(hash_bytes[:8], "little"))
            embedding = rng.standard_normal(self.dimension).astype(np.float32)
            # Normalize to unit vector
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)

        return np.array(embeddings, dtype=np.float32)

    async def embed_single(self, text: str) -> np.ndarray:
        """Generate mock embedding for a single text."""
        result = await self.embed([text])
        return result[0]

    @property
    def call_count(self) -> int:
        """Number of texts embedded."""
        return self._call_count

    def reset_call_count(self) -> None:
        """Reset the call counter."""
        self._call_count = 0


@pytest.fixture
def mock_embedder() -> MockEmbeddingBackend:
    """Get a mock embedding backend."""
    return MockEmbeddingBackend(dimension=384)


@pytest.fixture
async def initialized_mock_embedder(mock_embedder: MockEmbeddingBackend) -> MockEmbeddingBackend:
    """Get an initialized mock embedding backend."""
    await mock_embedder.initialize()
    return mock_embedder


# ==============================================================================
# Configuration Fixtures
# ==============================================================================

@pytest.fixture
def base_config_dict(tmp_path: Path) -> dict[str, Any]:
    """Get a base configuration dictionary for testing."""
    return {
        "project_root": str(tmp_path / "project"),
        "data_dir": str(tmp_path / ".icd"),
        "log_level": "DEBUG",
        "storage": {
            "db_path": str(tmp_path / ".icd" / "index.db"),
            "wal_mode": True,
            "cache_size_mb": 16,
            "max_vectors_per_repo": 10000,
            "vector_dtype": "float16",
            "hnsw_m": 16,
            "hnsw_ef_construction": 100,
            "hnsw_ef_search": 50,
        },
        "embedding": {
            "backend": "local_onnx",
            "model_name": "all-MiniLM-L6-v2",
            "dimension": 384,
            "batch_size": 32,
            "max_tokens": 512,
            "normalize": True,
        },
        "chunking": {
            "min_tokens": 200,
            "target_tokens": 500,
            "max_tokens": 1200,
            "overlap_tokens": 50,
            "preserve_symbols": True,
        },
        "retrieval": {
            "weight_embedding": 0.4,
            "weight_bm25": 0.3,
            "weight_recency": 0.1,
            "weight_contract": 0.1,
            "weight_focus": 0.05,
            "weight_pinned": 0.05,
            "recency_tau_days": 30.0,
            "mmr_lambda": 0.7,
            "initial_candidates": 100,
            "final_results": 20,
            "entropy_temperature": 1.0,
        },
        "pack": {
            "default_budget_tokens": 8000,
            "max_budget_tokens": 32000,
            "include_metadata": True,
            "include_citations": True,
        },
        "watcher": {
            "debounce_ms": 100,
            "max_file_size_kb": 500,
        },
        "rlm": {
            "enabled": True,
            "entropy_threshold": 2.5,
            "max_iterations": 5,
            "budget_per_iteration": 2000,
        },
        "contract": {
            "enabled": True,
            "boost_factor": 1.5,
        },
        "network": {
            "enabled": False,
        },
        "telemetry": {
            "enabled": False,
        },
    }


@pytest.fixture
def test_config(base_config_dict: dict[str, Any], tmp_path: Path):
    """Create a test configuration object."""
    # Ensure directories exist
    project_dir = Path(base_config_dict["project_root"])
    data_dir = Path(base_config_dict["data_dir"])
    project_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Import here to avoid import errors in test collection
    try:
        from icd.config import Config
        return Config(**base_config_dict)
    except ImportError:
        # Return dict if Config class not available
        return base_config_dict


# ==============================================================================
# Chunk Fixtures
# ==============================================================================

def make_chunk_id(content: str, file_path: str, symbol_path: str = "") -> str:
    """Generate a stable chunk ID."""
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
    struct_path = f"{file_path}:{symbol_path}"
    struct_hash = hashlib.md5(struct_path.encode()).hexdigest()[:8]
    return f"{content_hash}:{struct_hash}"


@pytest.fixture
def sample_chunks(sample_repo: Path) -> list[Chunk]:
    """Generate sample chunks from the sample repository."""
    chunks = []

    # Auth handler chunks
    handler_content = SAMPLE_AUTH_HANDLER_TS
    handler_path = str(sample_repo / "src" / "auth" / "handler.ts")

    chunks.append(Chunk(
        chunk_id=make_chunk_id(handler_content[:500], handler_path, "handleAuth"),
        file_path=handler_path,
        content=handler_content[:500],
        start_line=1,
        end_line=25,
        symbol_name="handleAuth",
        symbol_type="function",
        language="typescript",
    ))

    chunks.append(Chunk(
        chunk_id=make_chunk_id(handler_content[500:900], handler_path, "refreshToken"),
        file_path=handler_path,
        content=handler_content[500:900],
        start_line=26,
        end_line=35,
        symbol_name="refreshToken",
        symbol_type="function",
        language="typescript",
    ))

    # Validator chunks
    validator_content = SAMPLE_VALIDATOR_TS
    validator_path = str(sample_repo / "src" / "auth" / "validator.ts")

    chunks.append(Chunk(
        chunk_id=make_chunk_id(validator_content, validator_path, "validateToken"),
        file_path=validator_path,
        content=validator_content,
        start_line=1,
        end_line=20,
        symbol_name="validateToken",
        symbol_type="function",
        language="typescript",
    ))

    # Endpoints chunks
    endpoints_content = SAMPLE_ENDPOINTS_TS
    endpoints_path = str(sample_repo / "src" / "api" / "endpoints.ts")

    chunks.append(Chunk(
        chunk_id=make_chunk_id(endpoints_content[:600], endpoints_path, "authenticateEndpoint"),
        file_path=endpoints_path,
        content=endpoints_content[:600],
        start_line=1,
        end_line=30,
        symbol_name="authenticateEndpoint",
        symbol_type="function",
        language="typescript",
    ))

    chunks.append(Chunk(
        chunk_id=make_chunk_id(endpoints_content[600:1100], endpoints_path, "refreshEndpoint"),
        file_path=endpoints_path,
        content=endpoints_content[600:1100],
        start_line=31,
        end_line=55,
        symbol_name="refreshEndpoint",
        symbol_type="function",
        language="typescript",
    ))

    # Types chunks (contract)
    types_content = SAMPLE_SHARED_TYPES_TS
    types_path = str(sample_repo / "src" / "types" / "shared.ts")

    chunks.append(Chunk(
        chunk_id=make_chunk_id(types_content, types_path, "types"),
        file_path=types_path,
        content=types_content,
        start_line=1,
        end_line=35,
        symbol_name="AuthToken",
        symbol_type="interface",
        language="typescript",
        metadata={"is_contract": True},
    ))

    # API contract
    api_content = SAMPLE_API_YAML
    api_path = str(sample_repo / "contracts" / "api.yaml")

    chunks.append(Chunk(
        chunk_id=make_chunk_id(api_content, api_path, "api_contract"),
        file_path=api_path,
        content=api_content,
        start_line=1,
        end_line=100,
        symbol_name="Auth API",
        symbol_type="contract",
        language="yaml",
        metadata={"is_contract": True},
    ))

    return chunks


@pytest.fixture
def sample_embeddings(sample_chunks: list[Chunk], mock_embedder: MockEmbeddingBackend) -> dict[str, np.ndarray]:
    """Generate embeddings for sample chunks."""
    embeddings = {}

    async def generate():
        for chunk in sample_chunks:
            embedding = await mock_embedder.embed_single(chunk.content)
            embeddings[chunk.chunk_id] = embedding

    asyncio.get_event_loop().run_until_complete(generate())
    return embeddings


# ==============================================================================
# Pre-indexed Data Fixtures
# ==============================================================================

@pytest.fixture
def pre_indexed_data(
    sample_chunks: list[Chunk],
    sample_embeddings: dict[str, np.ndarray],
    sample_repo: Path,
) -> dict[str, Any]:
    """
    Provide pre-indexed test data.

    Returns a dictionary with all indexed data ready for use.
    """
    return {
        "repo_path": sample_repo,
        "chunks": sample_chunks,
        "embeddings": sample_embeddings,
        "files": [
            str(sample_repo / "src" / "auth" / "handler.ts"),
            str(sample_repo / "src" / "auth" / "validator.ts"),
            str(sample_repo / "src" / "api" / "endpoints.ts"),
            str(sample_repo / "src" / "types" / "shared.ts"),
            str(sample_repo / "contracts" / "api.yaml"),
            str(sample_repo / "package.json"),
        ],
        "contracts": [
            str(sample_repo / "src" / "types" / "shared.ts"),
            str(sample_repo / "contracts" / "api.yaml"),
        ],
        "symbols": {
            "handleAuth": {"file": "src/auth/handler.ts", "type": "function", "line": 9},
            "refreshToken": {"file": "src/auth/handler.ts", "type": "function", "line": 26},
            "validateToken": {"file": "src/auth/validator.ts", "type": "function", "line": 6},
            "authenticateEndpoint": {"file": "src/api/endpoints.ts", "type": "function", "line": 8},
            "refreshEndpoint": {"file": "src/api/endpoints.ts", "type": "function", "line": 35},
            "AuthToken": {"file": "src/types/shared.ts", "type": "interface", "line": 5},
            "User": {"file": "src/types/shared.ts", "type": "interface", "line": 11},
            "ApiResponse": {"file": "src/types/shared.ts", "type": "interface", "line": 20},
        },
    }


# ==============================================================================
# Mock Store Fixtures
# ==============================================================================

@pytest.fixture
def mock_sqlite_store():
    """Create a mock SQLite store."""
    store = MagicMock()
    store.initialize = AsyncMock()
    store.close = AsyncMock()
    store.get_stats = AsyncMock(return_value={"files": 6, "chunks": 7})
    store.insert_chunk = AsyncMock()
    store.get_chunk = AsyncMock()
    store.search_fts = AsyncMock(return_value=[])
    store.get_chunks_by_file = AsyncMock(return_value=[])
    store.delete_chunks_by_file = AsyncMock()
    return store


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    store = MagicMock()
    store.initialize = AsyncMock()
    store.close = AsyncMock()
    store.get_stats = AsyncMock(return_value={"vectors": 7, "dimension": 384})
    store.add_vectors = AsyncMock()
    store.search = AsyncMock(return_value=([], []))
    store.delete_vectors = AsyncMock()
    return store


@pytest.fixture
def mock_contract_store():
    """Create a mock contract store."""
    store = MagicMock()
    store.initialize = AsyncMock()
    store.close = AsyncMock()
    store.get_stats = AsyncMock(return_value={"contracts": 2})
    store.is_contract = MagicMock(return_value=False)
    store.get_contracts = AsyncMock(return_value=[])
    return store


@pytest.fixture
def mock_memory_store():
    """Create a mock memory store."""
    store = MagicMock()
    store.initialize = AsyncMock()
    store.close = AsyncMock()
    store.pin_chunk = AsyncMock(return_value=True)
    store.unpin_chunk = AsyncMock(return_value=True)
    store.get_pinned_chunks = AsyncMock(return_value=[])
    store.add_ledger_entry = AsyncMock(return_value="entry-1")
    store.get_ledger_entries = AsyncMock(return_value=[])
    return store


# ==============================================================================
# Async Test Helpers
# ==============================================================================

@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ==============================================================================
# Parametrized Test Data
# ==============================================================================

# Sample queries for retrieval tests
SAMPLE_QUERIES = [
    ("Where is auth token validated?", ["validateToken", "handleAuth"]),
    ("How does refresh endpoint work?", ["refreshEndpoint", "refreshToken"]),
    ("What are the shared types?", ["AuthToken", "User", "ApiResponse"]),
    ("How does authentication flow work?", ["handleAuth", "authenticateEndpoint"]),
]

# Sample scores for MMR tests
MMR_TEST_SCORES = [
    # (scores, expected_diversity_order)
    ([0.9, 0.85, 0.8, 0.75, 0.7], [0, 4, 2]),  # High similarity, need diversity
    ([0.9, 0.5, 0.4, 0.3, 0.2], [0, 1, 2]),   # Already diverse
    ([0.5, 0.5, 0.5, 0.5, 0.5], [0, 1, 2]),   # Equal scores
]

# Sample entropy test cases
ENTROPY_TEST_CASES = [
    # (scores, expected_entropy_range)
    ([0.9, 0.05, 0.05], (0.0, 1.0)),     # Low entropy (concentrated)
    ([0.33, 0.33, 0.34], (1.5, 2.0)),    # High entropy (uniform)
    ([0.7, 0.2, 0.1], (0.8, 1.5)),       # Medium entropy
]


@pytest.fixture(params=SAMPLE_QUERIES)
def sample_query_with_expected(request):
    """Parametrized fixture for sample queries."""
    return request.param


@pytest.fixture(params=MMR_TEST_SCORES)
def mmr_test_case(request):
    """Parametrized fixture for MMR test cases."""
    return request.param


@pytest.fixture(params=ENTROPY_TEST_CASES)
def entropy_test_case(request):
    """Parametrized fixture for entropy test cases."""
    return request.param


# ==============================================================================
# Cleanup Fixtures
# ==============================================================================

@pytest.fixture(autouse=True)
def cleanup_env():
    """Clean up environment variables after each test."""
    original_env = os.environ.copy()
    yield
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


# ==============================================================================
# Marks and Markers
# ==============================================================================

def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "acceptance: marks tests as acceptance tests")
    config.addinivalue_line("markers", "requires_network: marks tests that require network")
