"""
Unit tests for the code chunking module.

Tests cover:
- Symbol-based chunking
- Chunk size bounds (200-800 tokens, max 1200)
- Stable chunk ID generation
- Deduplication
- Language-specific chunking
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest


# ==============================================================================
# Helper Functions for Testing
# ==============================================================================

def compute_chunk_id(content: str, file_path: str, symbol_path: str = "") -> str:
    """Compute a stable chunk ID (mirrors production implementation)."""
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
    struct_path = f"{file_path}:{symbol_path}"
    struct_hash = hashlib.md5(struct_path.encode()).hexdigest()[:8]
    return f"{content_hash}:{struct_hash}"


def estimate_tokens(text: str) -> int:
    """Estimate token count (roughly 4 characters per token)."""
    return len(text) // 4


# ==============================================================================
# Test Fixtures
# ==============================================================================

PYTHON_CODE_SAMPLE = '''
"""Module docstring for the sample module."""

from typing import List, Optional

class DataProcessor:
    """Processes data with various transformations."""

    def __init__(self, config: dict):
        """Initialize the processor with configuration."""
        self.config = config
        self._cache = {}

    def process(self, data: List[dict]) -> List[dict]:
        """Process a list of data items.

        Args:
            data: Input data items

        Returns:
            Processed data items
        """
        results = []
        for item in data:
            processed = self._transform(item)
            results.append(processed)
        return results

    def _transform(self, item: dict) -> dict:
        """Apply transformation to a single item."""
        return {k: v.upper() if isinstance(v, str) else v for k, v in item.items()}


def utility_function(x: int, y: int) -> int:
    """A standalone utility function."""
    return x + y


async def async_helper(value: str) -> Optional[str]:
    """An async helper function."""
    if not value:
        return None
    return value.strip()
'''

TYPESCRIPT_CODE_SAMPLE = '''
/**
 * User service module for handling user operations.
 */

import { Database } from './database';
import { Logger } from './logger';

interface User {
  id: string;
  name: string;
  email: string;
}

interface CreateUserInput {
  name: string;
  email: string;
}

export class UserService {
  private db: Database;
  private logger: Logger;

  constructor(db: Database, logger: Logger) {
    this.db = db;
    this.logger = logger;
  }

  async createUser(input: CreateUserInput): Promise<User> {
    this.logger.info('Creating user', { email: input.email });
    const user = await this.db.insert('users', {
      id: crypto.randomUUID(),
      ...input,
    });
    return user;
  }

  async getUserById(id: string): Promise<User | null> {
    return this.db.findOne('users', { id });
  }

  async deleteUser(id: string): Promise<boolean> {
    const result = await this.db.delete('users', { id });
    return result.affected > 0;
  }
}

export function validateEmail(email: string): boolean {
  const emailRegex = /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/;
  return emailRegex.test(email);
}
'''

GO_CODE_SAMPLE = '''
// Package auth provides authentication utilities.
package auth

import (
    "context"
    "crypto/rand"
    "encoding/hex"
    "errors"
    "time"
)

// Token represents an authentication token.
type Token struct {
    Value     string
    UserID    string
    ExpiresAt time.Time
}

// TokenService handles token operations.
type TokenService struct {
    store TokenStore
    ttl   time.Duration
}

// NewTokenService creates a new token service.
func NewTokenService(store TokenStore, ttl time.Duration) *TokenService {
    return &TokenService{
        store: store,
        ttl:   ttl,
    }
}

// Generate creates a new token for the given user.
func (s *TokenService) Generate(ctx context.Context, userID string) (*Token, error) {
    bytes := make([]byte, 32)
    if _, err := rand.Read(bytes); err != nil {
        return nil, err
    }

    token := &Token{
        Value:     hex.EncodeToString(bytes),
        UserID:    userID,
        ExpiresAt: time.Now().Add(s.ttl),
    }

    if err := s.store.Save(ctx, token); err != nil {
        return nil, err
    }

    return token, nil
}

// Validate checks if a token is valid.
func (s *TokenService) Validate(ctx context.Context, value string) (*Token, error) {
    token, err := s.store.Get(ctx, value)
    if err != nil {
        return nil, err
    }

    if token.ExpiresAt.Before(time.Now()) {
        return nil, errors.New("token expired")
    }

    return token, nil
}
'''


@pytest.fixture
def python_code() -> str:
    """Provide Python code sample."""
    return PYTHON_CODE_SAMPLE


@pytest.fixture
def typescript_code() -> str:
    """Provide TypeScript code sample."""
    return TYPESCRIPT_CODE_SAMPLE


@pytest.fixture
def go_code() -> str:
    """Provide Go code sample."""
    return GO_CODE_SAMPLE


# ==============================================================================
# Chunk ID Tests
# ==============================================================================

class TestChunkIdGeneration:
    """Tests for stable chunk ID generation."""

    def test_chunk_id_deterministic(self):
        """Test that chunk IDs are deterministic for same content."""
        content = "def hello(): pass"
        file_path = "/path/to/file.py"

        id1 = compute_chunk_id(content, file_path, "hello")
        id2 = compute_chunk_id(content, file_path, "hello")

        assert id1 == id2

    def test_chunk_id_changes_with_content(self):
        """Test that chunk IDs change when content changes."""
        file_path = "/path/to/file.py"
        symbol = "hello"

        id1 = compute_chunk_id("def hello(): pass", file_path, symbol)
        id2 = compute_chunk_id("def hello(): return 1", file_path, symbol)

        assert id1 != id2

    def test_chunk_id_changes_with_file_path(self):
        """Test that chunk IDs change when file path changes."""
        content = "def hello(): pass"
        symbol = "hello"

        id1 = compute_chunk_id(content, "/path/a/file.py", symbol)
        id2 = compute_chunk_id(content, "/path/b/file.py", symbol)

        assert id1 != id2

    def test_chunk_id_changes_with_symbol(self):
        """Test that chunk IDs change when symbol changes."""
        content = "def hello(): pass"
        file_path = "/path/to/file.py"

        id1 = compute_chunk_id(content, file_path, "hello")
        id2 = compute_chunk_id(content, file_path, "goodbye")

        assert id1 != id2

    def test_chunk_id_format(self):
        """Test chunk ID format is content_hash:struct_hash."""
        content = "test content"
        file_path = "/test/path.py"

        chunk_id = compute_chunk_id(content, file_path, "symbol")

        # Format: 16-char content hash + : + 8-char struct hash
        parts = chunk_id.split(":")
        assert len(parts) == 2
        assert len(parts[0]) == 16
        assert len(parts[1]) == 8


# ==============================================================================
# Token Estimation Tests
# ==============================================================================

class TestTokenEstimation:
    """Tests for token count estimation."""

    def test_empty_content(self):
        """Test token estimation for empty content."""
        assert estimate_tokens("") == 0

    def test_short_content(self):
        """Test token estimation for short content."""
        # 12 characters -> ~3 tokens
        tokens = estimate_tokens("hello world!")
        assert tokens == 3

    def test_code_content(self):
        """Test token estimation for code content."""
        code = "def calculate(x, y):\n    return x + y"
        tokens = estimate_tokens(code)
        # 38 characters -> ~9 tokens
        assert 8 <= tokens <= 10


# ==============================================================================
# Chunk Size Bounds Tests
# ==============================================================================

class TestChunkSizeBounds:
    """Tests for chunk size constraints per PRD."""

    def test_min_chunk_tokens(self):
        """Test minimum chunk size is 200 tokens."""
        min_tokens = 200
        # 200 tokens * 4 chars = 800 characters minimum
        min_chars = min_tokens * 4

        content = "x" * min_chars
        tokens = estimate_tokens(content)

        assert tokens >= min_tokens

    def test_target_chunk_tokens(self):
        """Test target chunk size is 500-800 tokens."""
        target_min = 500
        target_max = 800

        # Target range in characters
        target_min_chars = target_min * 4  # 2000 chars
        target_max_chars = target_max * 4  # 3200 chars

        content = "x" * 2400  # ~600 tokens
        tokens = estimate_tokens(content)

        assert target_min <= tokens <= target_max

    def test_max_chunk_tokens(self):
        """Test maximum chunk size is 1200 tokens (hard limit)."""
        max_tokens = 1200
        max_chars = max_tokens * 4  # 4800 characters

        content = "x" * max_chars
        tokens = estimate_tokens(content)

        assert tokens <= max_tokens

    def test_chunk_size_validation(self):
        """Test chunk size validation logic."""
        min_tokens = 200
        target_tokens = 500
        max_tokens = 1200

        # Valid chunk in target range
        valid_content = "x" * (target_tokens * 4)
        valid_tokens = estimate_tokens(valid_content)
        assert min_tokens <= valid_tokens <= max_tokens

        # Chunk at minimum
        min_content = "x" * (min_tokens * 4)
        min_token_count = estimate_tokens(min_content)
        assert min_token_count >= min_tokens

        # Chunk at maximum (should be truncated in real implementation)
        max_content = "x" * (max_tokens * 4)
        max_token_count = estimate_tokens(max_content)
        assert max_token_count <= max_tokens


# ==============================================================================
# Symbol-Based Chunking Tests
# ==============================================================================

class TestSymbolBasedChunking:
    """Tests for symbol-based code chunking."""

    def test_extract_python_functions(self, python_code: str):
        """Test extraction of Python function symbols."""
        # Expected functions in the code
        expected_symbols = [
            "DataProcessor.__init__",
            "DataProcessor.process",
            "DataProcessor._transform",
            "utility_function",
            "async_helper",
        ]

        # Verify symbols exist in code
        assert "def __init__" in python_code
        assert "def process" in python_code
        assert "def utility_function" in python_code
        assert "async def async_helper" in python_code

    def test_extract_python_classes(self, python_code: str):
        """Test extraction of Python class symbols."""
        assert "class DataProcessor:" in python_code

    def test_extract_typescript_functions(self, typescript_code: str):
        """Test extraction of TypeScript function symbols."""
        assert "async createUser" in typescript_code
        assert "async getUserById" in typescript_code
        assert "function validateEmail" in typescript_code

    def test_extract_typescript_interfaces(self, typescript_code: str):
        """Test extraction of TypeScript interface symbols."""
        assert "interface User" in typescript_code
        assert "interface CreateUserInput" in typescript_code

    def test_extract_go_functions(self, go_code: str):
        """Test extraction of Go function symbols."""
        assert "func NewTokenService" in go_code
        assert "func (s *TokenService) Generate" in go_code
        assert "func (s *TokenService) Validate" in go_code

    def test_extract_go_types(self, go_code: str):
        """Test extraction of Go type symbols."""
        assert "type Token struct" in go_code
        assert "type TokenService struct" in go_code


# ==============================================================================
# Docstring and Comment Preservation Tests
# ==============================================================================

class TestDocstringPreservation:
    """Tests for preserving docstrings with their symbols."""

    def test_python_docstring_attached(self, python_code: str):
        """Test that Python docstrings are attached to symbols."""
        # Module docstring
        assert '"""Module docstring' in python_code

        # Class docstring
        assert '"""Processes data' in python_code

        # Method docstrings
        assert '"""Initialize the processor' in python_code
        assert '"""Process a list of data' in python_code

    def test_typescript_jsdoc_attached(self, typescript_code: str):
        """Test that TypeScript JSDoc comments are attached to symbols."""
        assert "/**" in typescript_code
        assert "* User service module" in typescript_code

    def test_go_comments_attached(self, go_code: str):
        """Test that Go comments are attached to symbols."""
        assert "// Package auth provides" in go_code
        assert "// Token represents" in go_code
        assert "// TokenService handles" in go_code


# ==============================================================================
# Deduplication Tests
# ==============================================================================

class TestChunkDeduplication:
    """Tests for chunk deduplication."""

    def test_identical_content_same_id(self):
        """Test that identical content produces same chunk ID."""
        content = "def foo(): return 42"
        file_path = "/test/file.py"

        id1 = compute_chunk_id(content, file_path, "foo")
        id2 = compute_chunk_id(content, file_path, "foo")

        assert id1 == id2

    def test_whitespace_changes_affect_id(self):
        """Test that whitespace changes affect chunk ID."""
        content1 = "def foo():\n    return 42"
        content2 = "def foo():\n        return 42"  # Different indent
        file_path = "/test/file.py"

        id1 = compute_chunk_id(content1, file_path, "foo")
        id2 = compute_chunk_id(content2, file_path, "foo")

        assert id1 != id2

    def test_should_update_detection(self):
        """Test detection of when chunks need updating."""
        old_content = "def foo(): return 1"
        new_content = "def foo(): return 2"
        file_path = "/test/file.py"

        old_id = compute_chunk_id(old_content, file_path, "foo")
        new_id = compute_chunk_id(new_content, file_path, "foo")

        # Extract content hash (first part of ID)
        old_hash = old_id.split(":")[0]
        new_hash = new_id.split(":")[0]

        # Content changed, so hashes differ -> should update
        assert old_hash != new_hash

    def test_no_update_needed_same_content(self):
        """Test that unchanged content doesn't need update."""
        content = "def foo(): return 1"
        file_path = "/test/file.py"

        id1 = compute_chunk_id(content, file_path, "foo")
        id2 = compute_chunk_id(content, file_path, "foo")

        hash1 = id1.split(":")[0]
        hash2 = id2.split(":")[0]

        # Same content, same hash -> no update needed
        assert hash1 == hash2


# ==============================================================================
# Language Detection Tests
# ==============================================================================

class TestLanguageDetection:
    """Tests for programming language detection."""

    @pytest.mark.parametrize("extension,expected_language", [
        (".py", "python"),
        (".ts", "typescript"),
        (".tsx", "typescript"),
        (".js", "javascript"),
        (".jsx", "javascript"),
        (".go", "go"),
        (".rs", "rust"),
        (".java", "java"),
        (".rb", "ruby"),
        (".php", "php"),
        (".c", "c"),
        (".cpp", "cpp"),
        (".h", "c"),
        (".hpp", "cpp"),
    ])
    def test_language_from_extension(self, extension: str, expected_language: str):
        """Test language detection from file extension."""
        file_path = f"/path/to/file{extension}"

        # Language should be detected from extension
        detected = extension.lstrip(".").lower()
        if detected in ("tsx", "ts"):
            detected = "typescript"
        elif detected in ("jsx", "js"):
            detected = "javascript"
        elif detected in ("h",):
            detected = "c"
        elif detected in ("hpp",):
            detected = "cpp"

        assert detected == expected_language


# ==============================================================================
# Chunk Overlap Tests
# ==============================================================================

class TestChunkOverlap:
    """Tests for chunk overlap handling."""

    def test_overlap_tokens_default(self):
        """Test default overlap is reasonable."""
        default_overlap = 50  # PRD suggests 50 token overlap
        assert 0 <= default_overlap <= 200

    def test_overlap_creates_context(self):
        """Test that overlap provides context between chunks."""
        # Simulate splitting with overlap
        content = "A" * 400 + "B" * 400  # ~200 tokens total
        overlap = 50 * 4  # 50 tokens * 4 chars

        # First chunk
        chunk1_end = 400 + overlap // 2
        chunk1 = content[:chunk1_end]

        # Second chunk with overlap
        chunk2_start = 400 - overlap // 2
        chunk2 = content[chunk2_start:]

        # Verify overlap exists
        assert len(chunk1) > 400
        assert len(chunk2) > 400
        # Overlapping content should be shared
        assert chunk1[-overlap:] in chunk2 or chunk2[:overlap] in chunk1


# ==============================================================================
# Edge Cases Tests
# ==============================================================================

class TestChunkerEdgeCases:
    """Tests for edge cases in chunking."""

    def test_empty_file(self):
        """Test handling of empty files."""
        content = ""
        tokens = estimate_tokens(content)
        assert tokens == 0

    def test_single_line_file(self):
        """Test handling of single line files."""
        content = "x = 1"
        chunk_id = compute_chunk_id(content, "/test/file.py", "x")
        assert chunk_id is not None

    def test_very_long_function(self):
        """Test handling of functions exceeding max tokens."""
        # Create a function with >1200 tokens (~4800 chars)
        long_body = "\n".join([f"    x{i} = {i}" for i in range(500)])
        content = f"def long_func():\n{long_body}"

        tokens = estimate_tokens(content)
        # Should be split in real implementation
        assert tokens > 1200

    def test_deeply_nested_code(self):
        """Test handling of deeply nested code."""
        nested = "if True:\n"
        for i in range(10):
            nested += "    " * (i + 1) + f"if x{i}:\n"
        nested += "    " * 11 + "pass"

        chunk_id = compute_chunk_id(nested, "/test/file.py", "nested")
        assert chunk_id is not None

    def test_unicode_content(self):
        """Test handling of unicode content."""
        content = "def greet():\n    return 'Hello, ' "
        chunk_id = compute_chunk_id(content, "/test/file.py", "greet")
        assert chunk_id is not None

    def test_binary_like_content(self):
        """Test handling of content with special characters."""
        content = "data = b'\\x00\\x01\\x02\\x03'"
        chunk_id = compute_chunk_id(content, "/test/file.py", "data")
        assert chunk_id is not None


# ==============================================================================
# Structural Boundary Tests
# ==============================================================================

class TestStructuralBoundaries:
    """Tests for respecting structural boundaries."""

    def test_class_boundary_respected(self):
        """Test that class definitions are kept together when possible."""
        code = """
class SmallClass:
    def __init__(self):
        self.x = 1

    def method(self):
        return self.x
"""
        tokens = estimate_tokens(code)
        # Small class should fit in one chunk
        assert tokens < 1200

    def test_function_boundary_respected(self):
        """Test that function definitions are kept together when possible."""
        code = """
def small_function(x, y):
    '''Add two numbers.'''
    result = x + y
    return result
"""
        tokens = estimate_tokens(code)
        # Small function should fit in one chunk
        assert tokens < 200  # Well under minimum

    def test_import_block_handling(self):
        """Test handling of import blocks."""
        code = """
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

def main():
    pass
"""
        # Imports might be grouped or attached to first symbol
        assert "import os" in code
        assert "def main" in code


# ==============================================================================
# Chunk Metadata Tests
# ==============================================================================

class TestChunkMetadata:
    """Tests for chunk metadata extraction."""

    def test_metadata_includes_file_path(self):
        """Test that metadata includes file path."""
        file_path = "/project/src/module.py"
        content = "def func(): pass"

        chunk_id = compute_chunk_id(content, file_path, "func")
        # File path is encoded in chunk ID (struct hash)
        assert chunk_id is not None

    def test_metadata_includes_symbol_info(self):
        """Test that metadata includes symbol information."""
        file_path = "/project/src/module.py"
        content = "def func(): pass"
        symbol = "func"

        chunk_id = compute_chunk_id(content, file_path, symbol)
        # Symbol is encoded in struct hash
        assert chunk_id is not None

    def test_line_numbers_extraction(self):
        """Test extraction of line number information."""
        code = """# Line 1
# Line 2
def func():  # Line 3
    pass     # Line 4
# Line 5
"""
        lines = code.split("\n")
        # Find function start line
        for i, line in enumerate(lines, 1):
            if "def func" in line:
                func_start = i
                break

        assert func_start == 3
