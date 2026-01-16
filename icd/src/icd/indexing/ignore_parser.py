"""
Ignore file parser for .gitignore and .icrignore.

Parses and combines ignore patterns from multiple sources:
1. Built-in defaults (node_modules, .git, __pycache__, etc.)
2. .gitignore patterns (if present)
3. .icrignore patterns (if present)
"""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)


def parse_ignore_file(path: Path) -> list[str]:
    """
    Parse an ignore file (.gitignore or .icrignore format).

    Args:
        path: Path to the ignore file.

    Returns:
        List of glob patterns to ignore.
    """
    if not path.exists():
        return []

    patterns: list[str] = []

    try:
        content = path.read_text(encoding="utf-8")
        for line in content.splitlines():
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Handle negation (we don't support it yet, just skip)
            if line.startswith("!"):
                logger.debug("Negation patterns not supported", pattern=line)
                continue

            # Convert gitignore pattern to glob pattern
            pattern = gitignore_to_glob(line)
            patterns.append(pattern)

    except Exception as e:
        logger.warning("Failed to parse ignore file", path=str(path), error=str(e))

    return patterns


def gitignore_to_glob(pattern: str) -> str:
    """
    Convert a gitignore pattern to a glob pattern.

    Gitignore patterns:
    - foo/ matches directories named foo
    - /foo matches foo only in root
    - *.py matches .py files anywhere
    - foo matches files and directories named foo anywhere

    Args:
        pattern: Gitignore pattern.

    Returns:
        Glob pattern suitable for fnmatch.
    """
    # Remove trailing slash (directory indicator)
    if pattern.endswith("/"):
        pattern = pattern[:-1]

    # Handle patterns starting with /
    if pattern.startswith("/"):
        # Anchored to root
        pattern = pattern[1:]
    else:
        # Match anywhere in tree
        if not pattern.startswith("**/"):
            pattern = "**/" + pattern

    # Ensure pattern matches both files and directories
    if not pattern.endswith("/**"):
        # Add /** to also match contents
        patterns_extended = pattern + "/**"
    else:
        patterns_extended = pattern

    return pattern


def load_ignore_patterns(
    repo_root: Path,
    include_gitignore: bool = True,
    include_icrignore: bool = True,
) -> list[str]:
    """
    Load all ignore patterns for a repository.

    Combines patterns from:
    1. .gitignore (if include_gitignore=True)
    2. .icrignore (if include_icrignore=True)

    Args:
        repo_root: Repository root directory.
        include_gitignore: Whether to include .gitignore patterns.
        include_icrignore: Whether to include .icrignore patterns.

    Returns:
        Combined list of glob patterns.
    """
    patterns: list[str] = []

    if include_gitignore:
        gitignore_path = repo_root / ".gitignore"
        gitignore_patterns = parse_ignore_file(gitignore_path)
        if gitignore_patterns:
            logger.info(
                "Loaded .gitignore patterns",
                count=len(gitignore_patterns),
            )
            patterns.extend(gitignore_patterns)

    if include_icrignore:
        icrignore_path = repo_root / ".icrignore"
        icrignore_patterns = parse_ignore_file(icrignore_path)
        if icrignore_patterns:
            logger.info(
                "Loaded .icrignore patterns",
                count=len(icrignore_patterns),
            )
            patterns.extend(icrignore_patterns)

    return patterns


def should_ignore(
    path: str | Path,
    patterns: list[str],
    repo_root: Path | None = None,
) -> bool:
    """
    Check if a path should be ignored based on patterns.

    Args:
        path: Path to check (absolute or relative).
        patterns: List of glob patterns.
        repo_root: Repository root for relative path calculation.

    Returns:
        True if the path should be ignored.
    """
    path = Path(path)

    # Make path relative if repo_root provided
    if repo_root:
        try:
            path = path.relative_to(repo_root)
        except ValueError:
            pass

    str_path = str(path)

    for pattern in patterns:
        if fnmatch.fnmatch(str_path, pattern):
            return True
        # Also check each path component
        for i, part in enumerate(path.parts):
            partial = str(Path(*path.parts[: i + 1]))
            if fnmatch.fnmatch(partial, pattern):
                return True

    return False


def create_default_icrignore() -> str:
    """
    Create default .icrignore content.

    Returns:
        Default .icrignore file content.
    """
    return """# ICR Ignore File
# Patterns here will be excluded from indexing
# Uses gitignore syntax

# Secrets and credentials
.env
.env.*
*.pem
*.key
credentials.json
secrets.yaml
secrets.yml

# Large generated files
*.min.js
*.min.css
*.bundle.js
*.chunk.js

# Lock files (usually not useful for context)
package-lock.json
yarn.lock
pnpm-lock.yaml
Gemfile.lock
poetry.lock
Pipfile.lock
composer.lock
Cargo.lock

# Build artifacts
*.map
*.d.ts.map

# Test fixtures and snapshots (often noisy)
__snapshots__/
*.snap

# ICR's own data
.icr/
"""
