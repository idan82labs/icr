"""
Incremental indexing utilities.

Provides fast staleness detection and incremental reindexing
without requiring a running daemon.
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from icd.indexing.ignore_parser import load_ignore_patterns

if TYPE_CHECKING:
    from icd.config import Config

logger = structlog.get_logger(__name__)


@dataclass
class StalenessReport:
    """Report on index staleness."""

    is_stale: bool
    total_files: int
    stale_files: int
    new_files: int
    deleted_files: int
    stale_paths: list[str]
    new_paths: list[str]
    deleted_paths: list[str]
    index_age_seconds: float

    @property
    def needs_reindex(self) -> bool:
        """Check if reindexing is needed."""
        return self.stale_files > 0 or self.new_files > 0 or self.deleted_files > 0


def find_index_db(project_root: Path) -> Path | None:
    """Find the index database file."""
    candidates = [
        project_root / ".icd" / "index.db",
        project_root / ".icr" / "index.db",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def get_watch_extensions() -> set[str]:
    """Get default file extensions to watch."""
    return {
        ".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs",
        ".java", ".kt", ".scala", ".rb", ".php", ".cs",
        ".c", ".cpp", ".h", ".hpp", ".swift", ".m",
        ".json", ".yaml", ".yml", ".toml", ".md",
    }


def check_staleness(
    project_root: Path,
    max_stale_files: int = 100,
) -> StalenessReport:
    """
    Quick check for stale files without full indexing.

    Compares file modification times against index timestamps.

    Args:
        project_root: Project root directory.
        max_stale_files: Maximum stale files to report (for performance).

    Returns:
        StalenessReport with details about stale files.
    """
    db_path = find_index_db(project_root)

    if db_path is None:
        return StalenessReport(
            is_stale=True,
            total_files=0,
            stale_files=0,
            new_files=0,
            deleted_files=0,
            stale_paths=[],
            new_paths=[],
            deleted_paths=[],
            index_age_seconds=float("inf"),
        )

    try:
        # Get index age
        index_mtime = db_path.stat().st_mtime
        index_age = (datetime.now().timestamp() - index_mtime)

        # Connect to database
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Get indexed files with their timestamps
        try:
            cursor.execute("""
                SELECT file_path, modified_at FROM files
            """)
            indexed_files = {row[0]: row[1] for row in cursor.fetchall()}
        except sqlite3.OperationalError:
            # Table might not exist
            indexed_files = {}

        conn.close()

        # Load ignore patterns
        ignore_patterns = load_ignore_patterns(
            project_root,
            include_gitignore=True,
            include_icrignore=True,
        )

        # Add default ignores
        default_ignores = [
            "**/node_modules/**",
            "**/.git/**",
            "**/venv/**",
            "**/.venv/**",
            "**/__pycache__/**",
            "**/dist/**",
            "**/build/**",
            "**/.icr/**",
            "**/.icd/**",
        ]
        ignore_patterns.extend(default_ignores)

        watch_extensions = get_watch_extensions()

        # Scan filesystem
        current_files: dict[str, float] = {}

        for root, dirs, files in os.walk(project_root):
            # Filter directories
            dirs[:] = [
                d for d in dirs
                if not any(
                    fnmatch.fnmatch(os.path.join(root, d), p)
                    for p in ignore_patterns
                )
                and not d.startswith(".")
            ]

            for filename in files:
                file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(file_path, project_root)

                # Check ignore patterns
                if any(fnmatch.fnmatch(file_path, p) for p in ignore_patterns):
                    continue

                # Check extension
                ext = os.path.splitext(filename)[1].lower()
                if ext not in watch_extensions:
                    continue

                try:
                    mtime = os.path.getmtime(file_path)
                    current_files[rel_path] = mtime
                except OSError:
                    continue

        # Find stale, new, and deleted files
        stale_paths = []
        new_paths = []
        deleted_paths = []

        # Check for stale and new files
        for path, mtime in current_files.items():
            if path in indexed_files:
                indexed_mtime = indexed_files[path]
                # Parse indexed mtime if it's a string
                if isinstance(indexed_mtime, str):
                    try:
                        indexed_mtime = datetime.fromisoformat(indexed_mtime).timestamp()
                    except ValueError:
                        indexed_mtime = 0

                if mtime > indexed_mtime:
                    stale_paths.append(path)
            else:
                new_paths.append(path)

            # Limit for performance
            if len(stale_paths) + len(new_paths) >= max_stale_files:
                break

        # Check for deleted files
        for path in indexed_files:
            if path not in current_files:
                deleted_paths.append(path)
                if len(deleted_paths) >= max_stale_files:
                    break

        is_stale = len(stale_paths) > 0 or len(new_paths) > 0 or len(deleted_paths) > 0

        return StalenessReport(
            is_stale=is_stale,
            total_files=len(current_files),
            stale_files=len(stale_paths),
            new_files=len(new_paths),
            deleted_files=len(deleted_paths),
            stale_paths=stale_paths[:max_stale_files],
            new_paths=new_paths[:max_stale_files],
            deleted_paths=deleted_paths[:max_stale_files],
            index_age_seconds=index_age,
        )

    except Exception as e:
        logger.warning(f"Error checking staleness: {e}")
        return StalenessReport(
            is_stale=True,
            total_files=0,
            stale_files=0,
            new_files=0,
            deleted_files=0,
            stale_paths=[],
            new_paths=[],
            deleted_paths=[],
            index_age_seconds=float("inf"),
        )


async def incremental_reindex(
    project_root: Path,
    max_files: int = 50,
) -> dict[str, int]:
    """
    Perform incremental reindexing of stale files.

    This is a lightweight reindex that only updates changed files.
    For large changes, a full reindex is recommended.

    Args:
        project_root: Project root directory.
        max_files: Maximum files to reindex in one pass.

    Returns:
        Statistics about reindexed files.
    """
    from icd.config import load_config
    from icd.main import ICDService

    stats = {"files": 0, "chunks": 0, "errors": 0, "skipped": 0}

    # Check staleness first
    report = check_staleness(project_root, max_stale_files=max_files)

    if not report.needs_reindex:
        logger.info("Index is up to date, no reindexing needed")
        return stats

    logger.info(
        "Starting incremental reindex",
        stale=report.stale_files,
        new=report.new_files,
        deleted=report.deleted_files,
    )

    try:
        # Load config and create service
        config = load_config(project_root=project_root)
        service = ICDService(config)

        async with service.session():
            # Reindex stale files
            files_to_reindex = report.stale_paths + report.new_paths

            for rel_path in files_to_reindex[:max_files]:
                file_path = project_root / rel_path
                if file_path.exists():
                    try:
                        result = await service.reindex_file(file_path)
                        stats["files"] += 1
                        stats["chunks"] += result.get("chunks", 0)
                        stats["errors"] += result.get("errors", 0)
                    except Exception as e:
                        logger.warning(f"Error reindexing {rel_path}: {e}")
                        stats["errors"] += 1

            # Handle deleted files (if supported by service)
            # For now, deleted files are handled on next full reindex
            stats["skipped"] = len(report.deleted_paths)

            if report.stale_files + report.new_files > max_files:
                stats["skipped"] += (
                    report.stale_files + report.new_files - max_files
                )

        logger.info("Incremental reindex complete", stats=stats)

    except Exception as e:
        logger.error(f"Incremental reindex failed: {e}")
        stats["errors"] += 1

    return stats


def run_incremental_reindex(project_root: Path, max_files: int = 50) -> dict[str, int]:
    """
    Synchronous wrapper for incremental_reindex.

    Useful for calling from hooks or scripts.
    """
    return asyncio.run(incremental_reindex(project_root, max_files))
