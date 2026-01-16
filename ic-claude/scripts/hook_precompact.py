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
    """
    Client for interacting with ICR/ICD services.

    This client extracts context from the transcript file directly since
    the conversation transcript contains all the information we need to
    preserve across compaction.
    """

    def __init__(self, config_path: str | None = None):
        """Initialize ICR client."""
        # Look for .icr config in current directory or home
        self.config_path = config_path or os.environ.get(
            "ICR_CONFIG_PATH",
            ".icr/config.yaml"
        )
        self._config: dict[str, Any] | None = None
        self._initialized = False
        self._compaction_count_file = Path(".icr/compaction_count")

    def _ensure_initialized(self) -> bool:
        """Ensure ICR is initialized."""
        if self._initialized:
            return True

        # Try to load config
        config_paths = [
            Path(self.config_path),
            Path(".icr/config.yaml"),
            Path.home() / ".icr" / "config.yaml",
        ]

        for path in config_paths:
            if path.exists():
                try:
                    import yaml
                    with open(path) as f:
                        self._config = yaml.safe_load(f) or {}
                    self._initialized = True
                    return True
                except Exception as e:
                    logger.debug(f"Failed to load config from {path}: {e}")
                    continue

        # No config found, use defaults
        self._config = {}
        self._initialized = True
        return True

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        if not self._ensure_initialized():
            return default
        return self._config.get(key, default) if self._config else default

    def get_invariants(self) -> list[dict[str, Any]]:
        """
        Get pinned invariants from .icr/invariants.json if it exists.

        Users can pin important context by adding to this file.
        """
        invariants_path = Path(".icr/invariants.json")
        if not invariants_path.exists():
            return []

        try:
            import json
            with open(invariants_path) as f:
                data = json.load(f)
            return data.get("invariants", [])
        except Exception as e:
            logger.debug(f"Failed to load invariants: {e}")
            return []

    def extract_from_transcript(
        self,
        transcript_path: str | None,
    ) -> dict[str, Any]:
        """
        Extract key information from the conversation transcript.

        This parses the JSONL transcript to find:
        - Key files mentioned/modified
        - Decisions made (from assistant messages)
        - Active todos (from tool calls)
        - Open questions
        """
        result = {
            "key_files": [],
            "decisions": [],
            "todos": [],
            "questions": [],
            "summary": "",
        }

        if not transcript_path or not Path(transcript_path).exists():
            return result

        try:
            import json
            import re

            files_mentioned: set[str] = set()
            decisions: list[str] = []
            todos: list[str] = []

            with open(transcript_path) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Extract file paths from tool calls
                    if entry.get("type") == "tool_use":
                        tool_input = entry.get("input", {})
                        if isinstance(tool_input, dict):
                            for key in ["file_path", "path", "file"]:
                                if key in tool_input:
                                    files_mentioned.add(tool_input[key])

                    # Extract from assistant messages
                    if entry.get("type") == "text" and entry.get("role") == "assistant":
                        text = entry.get("text", "")

                        # Look for file paths in the text
                        path_pattern = r'[`"]?([a-zA-Z0-9_/.-]+\.[a-zA-Z0-9]+)[`"]?'
                        for match in re.findall(path_pattern, text):
                            if "/" in match and not match.startswith("http"):
                                files_mentioned.add(match)

                        # Extract decisions (lines starting with decision-like language)
                        for line_text in text.split("\n"):
                            line_lower = line_text.lower().strip()
                            if any(line_lower.startswith(d) for d in [
                                "decided", "decision:", "we'll", "i'll", "let's",
                                "the approach", "using", "implemented"
                            ]):
                                if len(line_text) < 200:
                                    decisions.append(line_text.strip())

                    # Extract todos from TodoWrite tool calls
                    if entry.get("type") == "tool_use" and entry.get("name") == "TodoWrite":
                        tool_input = entry.get("input", {})
                        for todo in tool_input.get("todos", []):
                            if todo.get("status") in ["pending", "in_progress"]:
                                todos.append(todo.get("content", ""))

            # Limit and deduplicate
            result["key_files"] = list(files_mentioned)[:15]
            result["decisions"] = list(dict.fromkeys(decisions))[:10]  # Dedupe, keep order
            result["todos"] = todos[:20]

            # Generate summary from decisions
            if decisions:
                result["summary"] = "Key activities: " + "; ".join(decisions[:5])

        except Exception as e:
            logger.warning(f"Failed to extract from transcript: {e}")

        return result

    def get_critical_decisions(
        self,
        session_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get critical decisions - delegated to transcript extraction."""
        # This is now handled by extract_from_transcript
        return []

    def get_active_todos(
        self,
        session_id: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Get active todos - delegated to transcript extraction."""
        return []

    def get_open_questions(
        self,
        session_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get open questions - delegated to transcript extraction."""
        return []

    def get_key_files(
        self,
        session_id: str,
        limit: int = 15,
    ) -> list[str]:
        """Get key files - delegated to transcript extraction."""
        return []

    def get_session_info(self, session_id: str) -> dict[str, Any]:
        """Get session information from compaction count file."""
        count = 0
        if self._compaction_count_file.exists():
            try:
                count = int(self._compaction_count_file.read_text().strip())
            except Exception:
                pass

        return {
            "compaction_count": count,
            "started": "",
        }

    def generate_session_summary(self, session_id: str) -> str:
        """Generate a summary - handled by extract_from_transcript."""
        return ""

    def record_compaction(
        self,
        session_id: str,
        reason: str,
        tokens_before: int,
        tokens_after: int,
    ) -> None:
        """Record compaction event to a log file."""
        log_path = Path(".icr/compaction.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(log_path, "a") as f:
                f.write(
                    f"{datetime.utcnow().isoformat()} | {reason} | "
                    f"{tokens_before} -> {tokens_after}\n"
                )
        except Exception as e:
            logger.debug(f"Failed to record compaction: {e}")

    def increment_compaction_count(self, session_id: str) -> int:
        """Increment and return the compaction count."""
        self._compaction_count_file.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        if self._compaction_count_file.exists():
            try:
                count = int(self._compaction_count_file.read_text().strip())
            except Exception:
                pass

        count += 1
        try:
            self._compaction_count_file.write_text(str(count))
        except Exception as e:
            logger.debug(f"Failed to write compaction count: {e}")

        return count


def build_compact_snapshot(
    client: ICRClient,
    hook_input: HookInput,
) -> CompactSnapshot:
    """
    Build a snapshot of critical context to preserve.

    This assembles all invariants, decisions, todos, and other
    critical state that must survive compaction by extracting
    from the conversation transcript.
    """
    snapshot = CompactSnapshot()

    # Always include all invariants from .icr/invariants.json
    snapshot.invariants = client.get_invariants()

    # Extract context from transcript
    transcript_data = client.extract_from_transcript(hook_input.transcript_path)

    # Convert extracted decisions to the expected format
    snapshot.critical_decisions = [
        {"content": d, "timestamp": ""} for d in transcript_data.get("decisions", [])
    ]

    # Convert extracted todos
    snapshot.active_todos = [
        {"content": t} for t in transcript_data.get("todos", [])
    ]

    # Open questions (not extracted yet, could be enhanced)
    snapshot.open_questions = []

    # Key files from transcript
    snapshot.key_files = transcript_data.get("key_files", [])

    # Get session info (compaction count)
    session_info = client.get_session_info(hook_input.session_id)
    snapshot.compaction_count = session_info.get("compaction_count", 0)
    snapshot.original_start = session_info.get("started", "")

    # Session summary from transcript
    snapshot.session_summary = transcript_data.get("summary", "")

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
