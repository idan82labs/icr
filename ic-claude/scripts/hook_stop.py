#!/usr/bin/env python3
"""
Stop Hook Handler for ICR

This hook is invoked after Claude completes a response.
It extracts structured ledger information from the response,
updates priors based on the interaction, and computes metrics.

Input (via stdin):
{
  "session_id": "uuid",
  "response": "claude response text",
  "transcript_path": "/path/to/transcript.jsonl"
}

Output (via stdout):
{
  "success": true,
  "ledger_entries": [...],
  "warnings": []
}

IMPORTANT: Ledger extraction is DETERMINISTIC.
Only structured ledger blocks are extracted - no free-text inference.

Expected ledger format in response:
```
Ledger:
- Decisions: [list]
- Todos: [list]
- Open Questions: [list]
- Files touched: [list]
```
"""

import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Configure logging to stderr
logging.basicConfig(
    level=logging.DEBUG if os.environ.get("ICR_DEBUG") else logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("icr.hook.stop")


@dataclass
class HookInput:
    """Input data from Claude Code Stop hook."""

    session_id: str
    response: str
    transcript_path: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HookInput":
        """Create HookInput from dictionary."""
        return cls(
            session_id=data.get("session_id", ""),
            response=data.get("response", ""),
            transcript_path=data.get("transcript_path"),
        )


@dataclass
class LedgerEntry:
    """A single ledger entry extracted from the response."""

    entry_type: str  # decision, todo, question, file
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = "response"  # response, inferred, manual
    confidence: float = 1.0  # 1.0 for structured extraction

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.entry_type,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "confidence": self.confidence,
        }


@dataclass
class HookOutput:
    """Output data for Claude Code Stop hook."""

    success: bool = True
    ledger_entries: list[LedgerEntry] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON output."""
        return {
            "success": self.success,
            "ledger_entries": [e.to_dict() for e in self.ledger_entries],
            "metrics": self.metrics,
            "warnings": self.warnings,
        }


class LedgerExtractor:
    """
    Extract structured ledger entries from Claude responses.

    This extractor is DETERMINISTIC - it only extracts explicitly
    structured ledger blocks, never infers from free text.
    """

    # Pattern for the main ledger block
    LEDGER_BLOCK_PATTERN = re.compile(
        r"(?:^|\n)Ledger:\s*\n((?:[-*]\s+.+\n?)+)",
        re.MULTILINE | re.IGNORECASE,
    )

    # Patterns for individual ledger sections
    SECTION_PATTERNS = {
        "decision": re.compile(
            r"[-*]\s*Decisions?:\s*\[?([^\]]*)\]?",
            re.IGNORECASE,
        ),
        "todo": re.compile(
            r"[-*]\s*Todos?:\s*\[?([^\]]*)\]?",
            re.IGNORECASE,
        ),
        "question": re.compile(
            r"[-*]\s*(?:Open\s+)?Questions?:\s*\[?([^\]]*)\]?",
            re.IGNORECASE,
        ),
        "file": re.compile(
            r"[-*]\s*Files?\s+touched:\s*\[?([^\]]*)\]?",
            re.IGNORECASE,
        ),
    }

    # Pattern for list items within a section
    LIST_ITEM_PATTERN = re.compile(r"['\"]([^'\"]+)['\"]|(\S[^,\]]+)")

    def extract(self, response: str) -> list[LedgerEntry]:
        """
        Extract ledger entries from response.

        Returns only entries from structured ledger blocks.
        Never infers from free text.
        """
        entries: list[LedgerEntry] = []

        # Find all ledger blocks
        ledger_blocks = self.LEDGER_BLOCK_PATTERN.findall(response)

        for block in ledger_blocks:
            entries.extend(self._extract_from_block(block))

        # Also check for alternative formats
        entries.extend(self._extract_alternative_formats(response))

        return entries

    def _extract_from_block(self, block: str) -> list[LedgerEntry]:
        """Extract entries from a single ledger block."""
        entries: list[LedgerEntry] = []

        for entry_type, pattern in self.SECTION_PATTERNS.items():
            match = pattern.search(block)
            if match:
                content = match.group(1).strip()
                if content and content.lower() not in ("none", "n/a", "[]", ""):
                    # Parse list items
                    items = self._parse_list_items(content)
                    for item in items:
                        if item.strip():
                            entries.append(LedgerEntry(
                                entry_type=entry_type,
                                content=item.strip(),
                                source="response",
                                confidence=1.0,
                            ))

        return entries

    def _extract_alternative_formats(self, response: str) -> list[LedgerEntry]:
        """
        Extract from alternative structured formats.

        Supports:
        - Markdown task lists: - [ ] or - [x]
        - DECISION: prefix
        - TODO: prefix
        - QUESTION: prefix
        """
        entries: list[LedgerEntry] = []

        # Markdown task lists as todos
        task_pattern = re.compile(r"^[-*]\s*\[([ x])\]\s*(.+)$", re.MULTILINE)
        for match in task_pattern.finditer(response):
            status = match.group(1)
            content = match.group(2).strip()
            if status == " ":  # Unchecked = todo
                entries.append(LedgerEntry(
                    entry_type="todo",
                    content=content,
                    source="response",
                    confidence=0.9,  # Slightly lower confidence for inferred format
                ))

        # Prefixed entries
        prefix_patterns = {
            "decision": re.compile(r"^DECISION:\s*(.+)$", re.MULTILINE),
            "todo": re.compile(r"^TODO:\s*(.+)$", re.MULTILINE),
            "question": re.compile(r"^QUESTION:\s*(.+)$", re.MULTILINE),
        }

        for entry_type, pattern in prefix_patterns.items():
            for match in pattern.finditer(response):
                content = match.group(1).strip()
                if content:
                    entries.append(LedgerEntry(
                        entry_type=entry_type,
                        content=content,
                        source="response",
                        confidence=0.95,
                    ))

        return entries

    def _parse_list_items(self, content: str) -> list[str]:
        """Parse comma-separated or quoted list items."""
        items: list[str] = []

        # Try to parse as comma-separated list
        if "," in content:
            for part in content.split(","):
                part = part.strip().strip("'\"[]")
                if part:
                    items.append(part)
        else:
            # Single item or space-separated
            content = content.strip().strip("'\"[]")
            if content:
                items.append(content)

        return items


class TranscriptReader:
    """Read and parse Claude Code transcript files."""

    def __init__(self, transcript_path: str | None = None):
        """Initialize transcript reader."""
        self.transcript_path = transcript_path
        self._fallback_paths = [
            Path.home() / ".claude" / "sessions",
            Path.home() / ".config" / "claude" / "sessions",
            Path("/tmp") / "claude-sessions",
        ]

    def find_transcript(self, session_id: str | None = None) -> Path | None:
        """
        Find the transcript file.

        Falls back to searching known directories if path is invalid.
        """
        # Try provided path first
        if self.transcript_path:
            path = Path(self.transcript_path)
            if path.exists() and path.is_file():
                return path
            logger.warning(f"Provided transcript path invalid: {self.transcript_path}")

        # Search fallback directories
        for fallback_dir in self._fallback_paths:
            if not fallback_dir.exists():
                continue

            # Look for .jsonl files
            jsonl_files = list(fallback_dir.glob("*.jsonl"))

            if not jsonl_files:
                continue

            # If we have a session_id, try to match
            if session_id:
                for f in jsonl_files:
                    if session_id in f.name:
                        logger.info(f"Found matching transcript: {f}")
                        return f

            # Otherwise, return most recently modified
            jsonl_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            logger.info(f"Using most recent transcript: {jsonl_files[0]}")
            return jsonl_files[0]

        logger.warning("No transcript file found")
        return None

    def read_transcript(self) -> list[dict[str, Any]]:
        """Read and parse the transcript file."""
        path = self.find_transcript()
        if not path:
            return []

        entries: list[dict[str, Any]] = []
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.warning(f"Failed to read transcript: {e}")

        return entries


class ICRClient:
    """Client for interacting with ICR core services."""

    def __init__(self, config_path: str | None = None):
        """Initialize ICR client."""
        self.config_path = config_path or os.environ.get(
            "ICR_CONFIG_PATH",
            os.path.expanduser("~/.icr/config.yaml")
        )
        self.db_path = os.environ.get(
            "ICR_DB_PATH",
            os.path.expanduser("~/.icr/icr.db")
        )
        self._initialized = False

    def _ensure_initialized(self) -> bool:
        """Ensure ICR is initialized."""
        if self._initialized:
            return True

        if not Path(self.config_path).exists():
            logger.warning(f"ICR config not found at {self.config_path}")
            return False

        self._initialized = True
        return True

    def store_ledger_entries(
        self,
        session_id: str,
        entries: list[LedgerEntry],
    ) -> None:
        """Store ledger entries in the database."""
        if not self._ensure_initialized():
            return

        try:
            from icr.core.ledger import LedgerStore
            store = LedgerStore(db_path=self.db_path)
            for entry in entries:
                store.add(
                    session_id=session_id,
                    entry_type=entry.entry_type,
                    content=entry.content,
                    source=entry.source,
                    confidence=entry.confidence,
                    timestamp=entry.timestamp,
                )
        except ImportError:
            logger.debug("icr-core not available, skipping ledger storage")
        except Exception as e:
            logger.warning(f"Failed to store ledger entries: {e}")

    def update_priors(
        self,
        session_id: str,
        response: str,
        entries: list[LedgerEntry],
    ) -> None:
        """
        Update priors based on the response and extracted entries.

        This updates relevance scores and adds new priors from
        files touched and decisions made.
        """
        if not self._ensure_initialized():
            return

        try:
            from icr.core.priors import PriorManager
            manager = PriorManager(db_path=self.db_path)

            # Boost priors for files mentioned in ledger
            file_entries = [e for e in entries if e.entry_type == "file"]
            for entry in file_entries:
                manager.boost_relevance(entry.content, boost=0.2)

            # Add decisions as high-value priors
            decision_entries = [e for e in entries if e.entry_type == "decision"]
            for entry in decision_entries:
                manager.add_prior(
                    content=entry.content,
                    source="decision",
                    session_id=session_id,
                    initial_relevance=0.8,
                )

        except ImportError:
            logger.debug("icr-core not available, skipping prior update")
        except Exception as e:
            logger.warning(f"Failed to update priors: {e}")

    def record_response_metrics(
        self,
        session_id: str,
        response: str,
        entries: list[LedgerEntry],
    ) -> dict[str, Any]:
        """Record metrics about the response."""
        metrics = {
            "response_length": len(response),
            "ledger_entries": len(entries),
            "entries_by_type": {},
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Count entries by type
        for entry in entries:
            if entry.entry_type not in metrics["entries_by_type"]:
                metrics["entries_by_type"][entry.entry_type] = 0
            metrics["entries_by_type"][entry.entry_type] += 1

        # Try to record to database
        if self._ensure_initialized():
            try:
                from icr.core.metrics import MetricsRecorder
                recorder = MetricsRecorder(db_path=self.db_path)
                recorder.record_response(
                    session_id=session_id,
                    response_length=metrics["response_length"],
                    ledger_entries=metrics["ledger_entries"],
                    timestamp=datetime.utcnow(),
                )
            except ImportError:
                logger.debug("icr-core not available, skipping metrics recording")
            except Exception as e:
                logger.warning(f"Failed to record metrics: {e}")

        return metrics


def handle_hook(input_data: dict[str, Any]) -> dict[str, Any]:
    """
    Main hook handler function.

    Args:
        input_data: Dictionary containing hook input

    Returns:
        Dictionary containing hook output
    """
    warnings: list[str] = []
    output = HookOutput()

    # Parse input
    try:
        hook_input = HookInput.from_dict(input_data)
    except Exception as e:
        logger.error(f"Failed to parse input: {e}")
        return HookOutput(
            success=False,
            warnings=[f"Failed to parse input: {e}"],
        ).to_dict()

    # Check if ICR is disabled
    if os.environ.get("ICR_DISABLE_HOOKS"):
        logger.info("ICR hooks disabled via environment variable")
        return HookOutput(warnings=["ICR hooks disabled"]).to_dict()

    # Skip if response is empty
    if not hook_input.response.strip():
        return HookOutput().to_dict()

    # Extract ledger entries
    extractor = LedgerExtractor()
    try:
        entries = extractor.extract(hook_input.response)
        output.ledger_entries = entries

        if entries:
            logger.info(f"Extracted {len(entries)} ledger entries")
        else:
            logger.debug("No ledger entries found in response")

    except Exception as e:
        logger.error(f"Ledger extraction failed: {e}")
        warnings.append(f"Ledger extraction failed: {e}")

    # Initialize client and store/update
    client = ICRClient()

    # Store ledger entries
    if output.ledger_entries:
        try:
            client.store_ledger_entries(
                session_id=hook_input.session_id,
                entries=output.ledger_entries,
            )
        except Exception as e:
            logger.warning(f"Failed to store ledger: {e}")
            warnings.append(f"Failed to store ledger: {e}")

    # Update priors
    try:
        client.update_priors(
            session_id=hook_input.session_id,
            response=hook_input.response,
            entries=output.ledger_entries,
        )
    except Exception as e:
        logger.warning(f"Failed to update priors: {e}")
        warnings.append(f"Failed to update priors: {e}")

    # Record metrics
    try:
        output.metrics = client.record_response_metrics(
            session_id=hook_input.session_id,
            response=hook_input.response,
            entries=output.ledger_entries,
        )
    except Exception as e:
        logger.warning(f"Failed to record metrics: {e}")
        warnings.append(f"Failed to record metrics: {e}")

    output.warnings = warnings
    return output.to_dict()


def main() -> None:
    """Main entry point for the hook."""
    try:
        # Read input from stdin
        input_text = sys.stdin.read()

        if not input_text.strip():
            print(json.dumps(HookOutput().to_dict()))
            return

        # Parse JSON input
        try:
            input_data = json.loads(input_text)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON input: {e}")
            print(json.dumps(HookOutput(
                success=False,
                warnings=[f"Invalid JSON input: {e}"],
            ).to_dict()))
            return

        # Handle the hook
        output = handle_hook(input_data)

        # Output JSON to stdout
        print(json.dumps(output))

    except Exception as e:
        logger.exception("Unexpected error in hook handler")
        print(json.dumps({
            "success": False,
            "ledger_entries": [],
            "metrics": {},
            "warnings": [f"Hook error: {e}"],
        }))
        sys.exit(1)


if __name__ == "__main__":
    main()
