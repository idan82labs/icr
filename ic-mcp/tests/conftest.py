"""
Pytest configuration and fixtures for IC-MCP tests.
"""

import os
import tempfile
from pathlib import Path
from typing import Generator
from uuid import uuid4

import pytest

from ic_mcp.server import ICRMCPServer, create_server
from ic_mcp.tools.admin import set_server_start_time
from ic_mcp.tools.memory import MemoryStore, get_memory_store


@pytest.fixture
def request_id():
    """Generate a unique request ID for tests."""
    return uuid4()


@pytest.fixture
def temp_repo() -> Generator[Path, None, None]:
    """Create a temporary repository structure for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = Path(tmpdir)

        # Create a basic project structure
        (repo_root / "src").mkdir()
        (repo_root / "src" / "main.py").write_text(
            '''"""Main module."""

def main():
    """Entry point."""
    print("Hello, World!")


def helper_function(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y


class Calculator:
    """A simple calculator."""

    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    def subtract(self, a: int, b: int) -> int:
        """Subtract b from a."""
        return a - b


if __name__ == "__main__":
    main()
'''
        )

        (repo_root / "src" / "utils.py").write_text(
            '''"""Utility functions."""

import re
from typing import List


def parse_input(text: str) -> List[str]:
    """Parse input text into tokens."""
    return re.findall(r"\\w+", text)


def format_output(items: List[str]) -> str:
    """Format items as comma-separated string."""
    return ", ".join(items)


CONSTANT_VALUE = 42
'''
        )

        (repo_root / "tests").mkdir()
        (repo_root / "tests" / "test_main.py").write_text(
            '''"""Tests for main module."""

import pytest
from src.main import main, helper_function, Calculator


def test_helper_function():
    """Test helper_function."""
    assert helper_function(2, 3) == 5


def test_calculator_add():
    """Test Calculator.add."""
    calc = Calculator()
    assert calc.add(2, 3) == 5


def test_calculator_subtract():
    """Test Calculator.subtract."""
    calc = Calculator()
    assert calc.subtract(5, 3) == 2
'''
        )

        # Create config files
        (repo_root / "pyproject.toml").write_text(
            '''[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "test-project"
version = "0.1.0"

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
'''
        )

        (repo_root / "package.json").write_text(
            '''{
  "name": "test-project",
  "version": "0.1.0",
  "scripts": {
    "test": "jest",
    "build": "tsc",
    "lint": "eslint .",
    "format": "prettier --write .",
    "start": "node dist/index.js"
  }
}
'''
        )

        # Create a TypeScript file
        (repo_root / "src" / "index.ts").write_text(
            '''/**
 * Main entry point.
 */

export interface User {
  id: string;
  name: string;
  email: string;
}

export function createUser(name: string, email: string): User {
  return {
    id: crypto.randomUUID(),
    name,
    email,
  };
}

export class UserService {
  private users: Map<string, User> = new Map();

  addUser(user: User): void {
    this.users.set(user.id, user);
  }

  getUser(id: string): User | undefined {
    return this.users.get(id);
  }

  listUsers(): User[] {
    return Array.from(this.users.values());
  }
}
'''
        )

        # Create .git directory to simulate a git repo
        (repo_root / ".git").mkdir()
        (repo_root / ".git" / "config").write_text("[core]\n\trepositoryformatversion = 0\n")

        yield repo_root


@pytest.fixture
def server(temp_repo: Path) -> ICRMCPServer:
    """Create a server instance with the temp repo."""
    set_server_start_time()
    return create_server(str(temp_repo))


@pytest.fixture
def memory_store() -> MemoryStore:
    """Get a fresh memory store for testing."""
    # Reset the global store
    import ic_mcp.tools.memory as memory_module

    memory_module._memory_store = None
    return get_memory_store()


@pytest.fixture
def sample_files(temp_repo: Path) -> dict[str, Path]:
    """Return paths to sample files in the temp repo."""
    return {
        "main_py": temp_repo / "src" / "main.py",
        "utils_py": temp_repo / "src" / "utils.py",
        "test_main_py": temp_repo / "tests" / "test_main.py",
        "index_ts": temp_repo / "src" / "index.ts",
        "pyproject": temp_repo / "pyproject.toml",
        "package_json": temp_repo / "package.json",
    }
