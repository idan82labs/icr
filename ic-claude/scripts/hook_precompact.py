#!/usr/bin/env python3
"""
PreCompact Hook Handler for ICR

This hook is invoked before Claude Code compacts the conversation context.
It persists invariants and creates a compact-safe snapshot to ensure
critical context survives compaction.

Input (via stdin):
{
  "session_id": "uuid",
  "transcript_path": "/path/to/transcript.jsonl",
  "compaction_reason": "token_limit|manual|auto",
  "current_tokens": 150000,
  "target_tokens": 100000
}

Output (via stdout):
{
  "persistedContext": "## ICR Preserved Context\n...",
  "warnings": []
}

The persistedContext will be included in the compacted context,
ensuring invariants and critical state survive.
"""

import json
import logging
import os
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
logger = logging.getLogger("icr.hook.precompact")


@dataclass
class HookInput:
    """Input data from Claude Code PreCompact hook."""

    session_id: str
    transcript_path: str | None = None
    compaction_reason: str = "unknown"
    current_tokens: int = 0
    target_tokens: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HookInput":
        """Create HookInput from dictionary."""
        return cls(
            session_id=data.get("session_id", ""),
            transcript_path=data.get("transcript_path"),
            compaction_reason=data.get("compaction_reason", "unknown"),
            current_tokens=data.get("current_tokens", 0),
            target_tokens=data.get("target_tokens", 0),
        )


@dataclass
class HookOutput:
    """Output data for Claude Code PreCompact hook."""

    persisted_context: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON output."""
        return {
            "persistedContext": self.persisted_context,
            "warnings": self.warnings,
        }


@dataclass
class CompactSnapshot:
    """Snapshot of critical context to preserve across compaction."""

    invariants: list[dict[str, Any]] = field(default_factory=list)
    critical_decisions: list[dict[str, Any]] = field(default_factory=list)
    active_todos: list[dict[str, Any]] = field(default_factory=list)
    open_questions: list[dict[str, Any]] = field(default_factory=list)
    key_files: list[str] = field(default_factory=list)
    session_summary: str = ""
    compaction_count: int = 0
    original_start: str = ""

    def to_markdown(self) -> str:
        """Render snapshot as markdown for persistence."""
        sections = []

        # Header with compaction info
        sections.append("## ICR Preserved Context")
        sections.append(f"*Compaction #{self.compaction_count + 1} at {datetime.utcnow().isoformat()}Z*")
        if self.original_start:
            sections.append(f"*Session started: {self.original_start}*")
        sections.append("")

        # Pinned Invariants (ALWAYS preserved)
        if self.invariants:
            sections.append("### Pinned Invariants (MUST preserve)")
            for inv in self.invariants:
                inv_id = inv.get("id", "unknown")
                priority = inv.get("priority", 5)
                content = inv.get("content", "")
                created = inv.get("created", "")
                sections.append(f"- **[{inv_id}]** (P{priority}) {content}")
                if created:
                    sections.append(f"  - *Pinned: {created}*")
            sections.append("")

        # Critical Decisions
        if self.critical_decisions:
            sections.append("### Key Decisions Made")
            for decision in self.critical_decisions:
                content = decision.get("content", "")
                timestamp = decision.get("timestamp", "")
                sections.append(f"- {content}")
                if timestamp:
                    sections.append(f"  - *Decided: {timestamp}*")
            sections.append("")

        # Active Todos
        if self.active_todos:
            sections.append("### Active TODOs")
            for todo in self.active_todos:
                content = todo.get("content", "")
                sections.append(f"- [ ] {content}")
            sections.append("")

        # Open Questions
        if self.open_questions:
            sections.append("### Open Questions")
            for question in self.open_questions:
                content = question.get("content", "")
                sections.append(f"- {content}")
            sections.append("")

        # Key Files
        if self.key_files:
            sections.append("### Key Files in Context")
            for f in self.key_files[:15]:  # Limit to top 15
                sections.append(f"- `{f}`")
            sections.append("")

        # Session Summary
        if self.session_summary:
            sections.append("### Session Summary")
            sections.append(self.session_summary)
            sections.append("")

        # Footer
        sections.append("---")
        sections.append("*This context was preserved by ICR during compaction.*")
        sections.append("*Use `/ic status` to see full ICR state.*")

        return "\n".join(sections)

    def estimate_tokens(self) -> int:
        """Estimate token count of the snapshot."""
        content = self.to_markdown()
        return len(content) // 4  # Rough estimate


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
        self._config: dict[str, Any] | None = None
        self._initialized = False

    def _ensure_initialized(self) -> bool:
        """Ensure ICR is initialized."""
        if self._initialized:
            return True

        if not Path(self.config_path).exists():
            logger.warning(f"ICR config not found at {self.config_path}")
            return False

        try:
            import yaml
            with open(self.config_path) as f:
                self._config = yaml.safe_load(f)
            self._initialized = True
            return True
        except ImportError:
            self._config = {}
            self._initialized = True
            return True
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
            return False

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        if not self._ensure_initialized():
            return default
        return self._config.get(key, default) if self._config else default

    def get_invariants(self) -> list[dict[str, Any]]:
        """Get all pinned invariants."""
        if not self._ensure_initialized():
            return []

        try:
            from icr.core.invariants import InvariantStore
            store = InvariantStore(db_path=self.db_path)
            return store.get_all_active()
        except ImportError:
            logger.debug("icr-core not available, skipping invariants")
            return []
        except Exception as e:
            logger.warning(f"Failed to get invariants: {e}")
            return []

    def get_critical_decisions(
        self,
        session_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get critical decisions from the session."""
        if not self._ensure_initialized():
            return []

        try:
            from icr.core.ledger import LedgerStore
            store = LedgerStore(db_path=self.db_path)
            entries = store.get_by_type(
                session_id=session_id,
                entry_type="decision",
                limit=limit,
            )
            return entries
        except ImportError:
            logger.debug("icr-core not available, skipping decisions")
            return []
        except Exception as e:
            logger.warning(f"Failed to get decisions: {e}")
            return []

    def get_active_todos(
        self,
        session_id: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Get active (uncompleted) todos from the session."""
        if not self._ensure_initialized():
            return []

        try:
            from icr.core.ledger import LedgerStore
            store = LedgerStore(db_path=self.db_path)
            entries = store.get_by_type(
                session_id=session_id,
                entry_type="todo",
                limit=limit,
            )
            # Filter to active only (would need completion tracking)
            return entries
        except ImportError:
            logger.debug("icr-core not available, skipping todos")
            return []
        except Exception as e:
            logger.warning(f"Failed to get todos: {e}")
            return []

    def get_open_questions(
        self,
        session_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get open questions from the session."""
        if not self._ensure_initialized():
            return []

        try:
            from icr.core.ledger import LedgerStore
            store = LedgerStore(db_path=self.db_path)
            entries = store.get_by_type(
                session_id=session_id,
                entry_type="question",
                limit=limit,
            )
            return entries
        except ImportError:
            logger.debug("icr-core not available, skipping questions")
            return []
        except Exception as e:
            logger.warning(f"Failed to get questions: {e}")
            return []

    def get_key_files(
        self,
        session_id: str,
        limit: int = 15,
    ) -> list[str]:
        """Get key files from the session."""
        if not self._ensure_initialized():
            return []

        try:
            from icr.core.ledger import LedgerStore
            store = LedgerStore(db_path=self.db_path)
            entries = store.get_by_type(
                session_id=session_id,
                entry_type="file",
                limit=limit,
            )
            return [e.get("content", "") for e in entries if e.get("content")]
        except ImportError:
            logger.debug("icr-core not available, skipping files")
            return []
        except Exception as e:
            logger.warning(f"Failed to get files: {e}")
            return []

    def get_session_info(self, session_id: str) -> dict[str, Any]:
        """Get session information."""
        if not self._ensure_initialized():
            return {}

        try:
            from icr.core.sessions import SessionStore
            store = SessionStore(db_path=self.db_path)
            return store.get(session_id) or {}
        except ImportError:
            logger.debug("icr-core not available, skipping session info")
            return {}
        except Exception as e:
            logger.warning(f"Failed to get session info: {e}")
            return {}

    def generate_session_summary(self, session_id: str) -> str:
        """Generate a summary of the session."""
        if not self._ensure_initialized():
            return ""

        try:
            from icr.core.summarization import SessionSummarizer
            summarizer = SessionSummarizer(db_path=self.db_path)
            return summarizer.summarize(session_id)
        except ImportError:
            # Provide basic summary from ledger if core not available
            decisions = self.get_critical_decisions(session_id, limit=5)
            if decisions:
                summary_parts = ["Key activities this session:"]
                for d in decisions[:5]:
                    summary_parts.append(f"- {d.get('content', '')}")
                return "\n".join(summary_parts)
            return ""
        except Exception as e:
            logger.warning(f"Failed to generate summary: {e}")
            return ""

    def record_compaction(
        self,
        session_id: str,
        reason: str,
        tokens_before: int,
        tokens_after: int,
    ) -> None:
        """Record compaction event."""
        if not self._ensure_initialized():
            return

        try:
            from icr.core.metrics import MetricsRecorder
            recorder = MetricsRecorder(db_path=self.db_path)
            recorder.record_compaction(
                session_id=session_id,
                reason=reason,
                tokens_before=tokens_before,
                tokens_after=tokens_after,
                timestamp=datetime.utcnow(),
            )
        except ImportError:
            logger.debug("icr-core not available, skipping compaction recording")
        except Exception as e:
            logger.warning(f"Failed to record compaction: {e}")

    def increment_compaction_count(self, session_id: str) -> int:
        """Increment and return the compaction count for this session."""
        if not self._ensure_initialized():
            return 0

        try:
            from icr.core.sessions import SessionStore
            store = SessionStore(db_path=self.db_path)
            return store.increment_compaction_count(session_id)
        except ImportError:
            return 0
        except Exception as e:
            logger.warning(f"Failed to increment compaction count: {e}")
            return 0


def build_compact_snapshot(
    client: ICRClient,
    hook_input: HookInput,
) -> CompactSnapshot:
    """
    Build a snapshot of critical context to preserve.

    This assembles all invariants, decisions, todos, and other
    critical state that must survive compaction.
    """
    snapshot = CompactSnapshot()

    # Always include all invariants
    snapshot.invariants = client.get_invariants()

    # Get critical decisions
    snapshot.critical_decisions = client.get_critical_decisions(
        session_id=hook_input.session_id,
        limit=10,
    )

    # Get active todos
    snapshot.active_todos = client.get_active_todos(
        session_id=hook_input.session_id,
        limit=20,
    )

    # Get open questions
    snapshot.open_questions = client.get_open_questions(
        session_id=hook_input.session_id,
        limit=10,
    )

    # Get key files
    snapshot.key_files = client.get_key_files(
        session_id=hook_input.session_id,
        limit=15,
    )

    # Get session info
    session_info = client.get_session_info(hook_input.session_id)
    snapshot.compaction_count = session_info.get("compaction_count", 0)
    snapshot.original_start = session_info.get("started", "")

    # Generate session summary
    snapshot.session_summary = client.generate_session_summary(
        session_id=hook_input.session_id
    )

    return snapshot


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

    # Initialize client
    client = ICRClient()

    # Check if invariant persistence is enabled
    if not client.get_config("persist_invariants", True):
        logger.info("Invariant persistence disabled in config")
        return HookOutput().to_dict()

    # Build snapshot
    try:
        snapshot = build_compact_snapshot(client, hook_input)
    except Exception as e:
        logger.error(f"Failed to build snapshot: {e}")
        warnings.append(f"Snapshot generation failed: {e}")
        return HookOutput(warnings=warnings).to_dict()

    # Check if we have anything to persist
    if not (snapshot.invariants or snapshot.critical_decisions or
            snapshot.active_todos or snapshot.open_questions):
        logger.info("No critical context to preserve")
        return HookOutput().to_dict()

    # Estimate tokens and warn if large
    estimated_tokens = snapshot.estimate_tokens()
    if estimated_tokens > 2000:
        warnings.append(f"Preserved context is {estimated_tokens} tokens")

    # Record compaction event
    try:
        client.record_compaction(
            session_id=hook_input.session_id,
            reason=hook_input.compaction_reason,
            tokens_before=hook_input.current_tokens,
            tokens_after=hook_input.target_tokens,
        )
        client.increment_compaction_count(hook_input.session_id)
    except Exception as e:
        logger.warning(f"Failed to record compaction: {e}")
        # Non-fatal, continue

    # Generate output
    output = HookOutput(
        persisted_context=snapshot.to_markdown(),
        warnings=warnings,
    )

    logger.info(
        f"Preserving context: {len(snapshot.invariants)} invariants, "
        f"{len(snapshot.critical_decisions)} decisions, "
        f"{len(snapshot.active_todos)} todos"
    )

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
                warnings=[f"Invalid JSON input: {e}"]
            ).to_dict()))
            return

        # Handle the hook
        output = handle_hook(input_data)

        # Output JSON to stdout
        print(json.dumps(output))

    except Exception as e:
        logger.exception("Unexpected error in hook handler")
        print(json.dumps({
            "persistedContext": "",
            "warnings": [f"Hook error: {e}"],
        }))
        sys.exit(1)


if __name__ == "__main__":
    main()
