#!/usr/bin/env python3
"""
SessionStart Hook Handler for ICR

This hook is invoked when a new Claude Code session starts.
It checks index freshness, triggers incremental reindex if needed,
and provides context about ICR availability.

Input (via stdin):
{
  "session_id": "uuid",
  "cwd": "/current/working/directory"
}

Output (via stdout):
{
  "additionalContext": "## ICR Available\n...",
  "warnings": []
}
"""

import json
import logging
import os
import sqlite3
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

# Configure logging to stderr (stdout is reserved for hook output)
logging.basicConfig(
    level=logging.DEBUG if os.environ.get("ICR_DEBUG") else logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("icr.hook.sessionstart")


@dataclass
class HookInput:
    """Input data from Claude Code SessionStart hook."""

    session_id: str
    cwd: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HookInput":
        """Create HookInput from dictionary."""
        return cls(
            session_id=data.get("session_id", ""),
            cwd=data.get("cwd"),
        )


@dataclass
class HookOutput:
    """Output data for Claude Code SessionStart hook."""

    additional_context: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON output."""
        return {
            "additionalContext": self.additional_context,
            "warnings": self.warnings,
        }


@dataclass
class IndexFreshness:
    """Index freshness status."""

    status: str  # "fresh", "stale", "missing", "reindexed"
    message: str
    age_days: int = 0
    file_count: int = 0
    chunk_count: int = 0
    stale_files: int = 0
    reindexed_files: int = 0


def find_index_db(project_root: Path) -> Optional[Path]:
    """Find the ICR index database file."""
    # Check common locations
    candidates = [
        project_root / ".icd" / "index.db",
        project_root / ".icr" / "index.db",
        project_root / ".icr" / "icd.db",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def check_file_staleness(project_root: Path) -> tuple[int, int, int]:
    """
    Quick check for stale files using icd.indexing.incremental if available.

    Returns (stale_count, new_count, deleted_count).
    """
    try:
        # Try to import the incremental module
        sys.path.insert(0, str(project_root / ".icr" / "venv" / "lib" / "python3.11" / "site-packages"))
        sys.path.insert(0, str(project_root / ".icr" / "venv" / "lib" / "python3.10" / "site-packages"))
        sys.path.insert(0, str(project_root / ".icr" / "venv" / "lib" / "python3.12" / "site-packages"))

        from icd.indexing.incremental import check_staleness

        report = check_staleness(project_root, max_stale_files=100)
        return report.stale_files, report.new_files, report.deleted_files

    except ImportError:
        logger.debug("icd.indexing.incremental not available")
        return 0, 0, 0
    except Exception as e:
        logger.debug(f"Error checking staleness: {e}")
        return 0, 0, 0


def trigger_incremental_reindex(project_root: Path, max_files: int = 20) -> int:
    """
    Trigger incremental reindex for stale files.

    Returns number of files reindexed.
    """
    try:
        # Try using the Python API directly
        sys.path.insert(0, str(project_root / ".icr" / "venv" / "lib" / "python3.11" / "site-packages"))
        sys.path.insert(0, str(project_root / ".icr" / "venv" / "lib" / "python3.10" / "site-packages"))
        sys.path.insert(0, str(project_root / ".icr" / "venv" / "lib" / "python3.12" / "site-packages"))

        from icd.indexing.incremental import run_incremental_reindex

        stats = run_incremental_reindex(project_root, max_files=max_files)
        return stats.get("files", 0)

    except ImportError:
        logger.debug("icd.indexing.incremental not available, trying CLI")

        # Fallback to CLI
        try:
            icd_path = project_root / ".icr" / "venv" / "bin" / "icd"
            if not icd_path.exists():
                return 0

            result = subprocess.run(
                [str(icd_path), "-p", str(project_root), "index"],
                capture_output=True,
                text=True,
                timeout=60,  # 1 minute timeout
            )

            if result.returncode == 0:
                # Parse output for file count
                for line in result.stdout.split("\n"):
                    if "Indexed" in line and "files" in line:
                        try:
                            return int(line.split()[1])
                        except (IndexError, ValueError):
                            pass
                return 1  # At least indicate something was done

        except subprocess.TimeoutExpired:
            logger.warning("Reindex timed out")
        except Exception as e:
            logger.warning(f"CLI reindex failed: {e}")

        return 0

    except Exception as e:
        logger.warning(f"Incremental reindex failed: {e}")
        return 0


def check_index_freshness(project_root: Path, auto_reindex: bool = True) -> IndexFreshness:
    """Check if index is fresh or stale, optionally trigger reindex."""
    db_path = find_index_db(project_root)

    if db_path is None:
        return IndexFreshness(
            status="missing",
            message="No ICR index found. Run `icr index` or reinstall ICR to create.",
        )

    try:
        # Check modification time
        mtime = datetime.fromtimestamp(db_path.stat().st_mtime)
        age = datetime.now() - mtime
        age_days = age.days

        # Get index stats
        file_count, chunk_count = get_index_stats(db_path)

        # Check for stale files (modified since last index)
        stale_count, new_count, deleted_count = check_file_staleness(project_root)
        total_stale = stale_count + new_count + deleted_count

        # If files are stale and auto_reindex is enabled, reindex them
        reindexed_files = 0
        if total_stale > 0 and auto_reindex:
            logger.info(f"Found {total_stale} stale files, triggering incremental reindex")
            reindexed_files = trigger_incremental_reindex(project_root, max_files=20)

            if reindexed_files > 0:
                # Refresh stats after reindex
                file_count, chunk_count = get_index_stats(db_path)

                return IndexFreshness(
                    status="reindexed",
                    message=f"Auto-reindexed {reindexed_files} changed files.",
                    age_days=age_days,
                    file_count=file_count,
                    chunk_count=chunk_count,
                    stale_files=total_stale,
                    reindexed_files=reindexed_files,
                )

        # Check if index is old (>7 days)
        if age > timedelta(days=7):
            return IndexFreshness(
                status="stale",
                message=f"Index is {age_days} days old. Consider running `icr index` to refresh.",
                age_days=age_days,
                file_count=file_count,
                chunk_count=chunk_count,
                stale_files=total_stale,
            )

        # Check if there are stale files that couldn't be reindexed
        if total_stale > 0:
            return IndexFreshness(
                status="stale",
                message=f"{total_stale} files changed since last index. Run `icr index` to update.",
                age_days=age_days,
                file_count=file_count,
                chunk_count=chunk_count,
                stale_files=total_stale,
            )

        return IndexFreshness(
            status="fresh",
            message="",
            age_days=age_days,
            file_count=file_count,
            chunk_count=chunk_count,
        )

    except Exception as e:
        logger.warning(f"Error checking index freshness: {e}")
        return IndexFreshness(
            status="missing",
            message=f"Error accessing index: {e}",
        )


def get_index_stats(db_path: Path) -> tuple[int, int]:
    """Get basic index statistics from SQLite database."""
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Try to get file count
        file_count = 0
        try:
            cursor.execute("SELECT COUNT(*) FROM files")
            file_count = cursor.fetchone()[0]
        except sqlite3.OperationalError:
            # Table might not exist or have different name
            try:
                cursor.execute("SELECT COUNT(DISTINCT file_path) FROM chunks")
                file_count = cursor.fetchone()[0]
            except sqlite3.OperationalError:
                pass

        # Try to get chunk count
        chunk_count = 0
        try:
            cursor.execute("SELECT COUNT(*) FROM chunks")
            chunk_count = cursor.fetchone()[0]
        except sqlite3.OperationalError:
            pass

        conn.close()
        return file_count, chunk_count

    except Exception as e:
        logger.debug(f"Failed to get index stats: {e}")
        return 0, 0


def build_context(freshness: IndexFreshness) -> str:
    """Build additional context message based on index freshness."""
    if freshness.status not in ("fresh", "reindexed"):
        return ""

    parts = ["## ICR Available"]

    if freshness.status == "reindexed":
        parts.append(f"ICR auto-reindexed {freshness.reindexed_files} changed files on session start.")
    elif freshness.age_days == 0:
        parts.append(f"ICR semantic search is ready. Index is up to date.")
    else:
        parts.append(f"ICR semantic search is ready. Index is {freshness.age_days} day(s) old.")

    if freshness.file_count > 0 or freshness.chunk_count > 0:
        parts.append(f"Indexed: {freshness.file_count} files, {freshness.chunk_count} chunks.")

    parts.append("")
    parts.append("Use ICR for conceptual questions like:")
    parts.append("- \"How does X work?\"")
    parts.append("- \"Trace the Y flow\"")
    parts.append("- \"What uses Z?\"")
    parts.append("")
    parts.append("For known symbols or file patterns, prefer native Grep/Glob.")

    return "\n".join(parts)


def handle_hook(input_data: dict[str, Any]) -> dict[str, Any]:
    """
    Main hook handler function.

    Args:
        input_data: Dictionary containing hook input

    Returns:
        Dictionary containing hook output
    """
    warnings: list[str] = []

    # Parse input
    try:
        hook_input = HookInput.from_dict(input_data)
    except Exception as e:
        logger.error(f"Failed to parse input: {e}")
        return HookOutput(warnings=[f"Failed to parse input: {e}"]).to_dict()

    # Check if ICR is disabled
    if os.environ.get("ICR_DISABLE_HOOKS"):
        logger.info("ICR hooks disabled via environment variable")
        return HookOutput(warnings=["ICR hooks disabled"]).to_dict()

    # Determine project root
    project_root = Path(hook_input.cwd) if hook_input.cwd else Path.cwd()

    # Check index freshness
    freshness = check_index_freshness(project_root)

    # Build output based on freshness status
    if freshness.status == "missing":
        warnings.append(freshness.message)
        additional_context = ""
    elif freshness.status == "stale":
        warnings.append(freshness.message)
        # Still provide context for stale index
        additional_context = build_context(
            IndexFreshness(
                status="fresh",  # Treat as fresh for context building
                message="",
                age_days=freshness.age_days,
                file_count=freshness.file_count,
                chunk_count=freshness.chunk_count,
            )
        )
    else:
        additional_context = build_context(freshness)

    output = HookOutput(
        additional_context=additional_context,
        warnings=warnings,
    )

    return output.to_dict()


def main() -> None:
    """Main entry point for the hook."""
    try:
        # Read input from stdin
        input_text = sys.stdin.read()

        if not input_text.strip():
            # No input provided, output empty response
            print(json.dumps(HookOutput().to_dict()))
            return

        # Parse JSON input
        try:
            input_data = json.loads(input_text)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON input: {e}")
            print(json.dumps(HookOutput(
                warnings=[f"Invalid JSON input: {e}"]
            ).to_dict()))
            return

        # Handle the hook
        output = handle_hook(input_data)

        # Output JSON to stdout
        print(json.dumps(output))

    except Exception as e:
        logger.exception("Unexpected error in hook handler")
        # Always output valid JSON even on error
        print(json.dumps({
            "additionalContext": "",
            "warnings": [f"Hook error: {e}"],
        }))
        sys.exit(1)


if __name__ == "__main__":
    main()
