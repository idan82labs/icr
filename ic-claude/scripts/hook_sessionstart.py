#!/usr/bin/env python3
"""
SessionStart Hook Handler for ICR

This hook is invoked when a new Claude Code session starts.
It checks index freshness and provides context about ICR availability.

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

    status: str  # "fresh", "stale", "missing"
    message: str
    age_days: int = 0
    file_count: int = 0
    chunk_count: int = 0


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


def check_index_freshness(project_root: Path) -> IndexFreshness:
    """Check if index is fresh or stale."""
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

        if age > timedelta(days=7):
            return IndexFreshness(
                status="stale",
                message=f"Index is {age_days} days old. Consider running `icr index` to refresh.",
                age_days=age_days,
                file_count=file_count,
                chunk_count=chunk_count,
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
    if freshness.status != "fresh":
        return ""

    parts = ["## ICR Available"]

    if freshness.age_days == 0:
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
