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
from typing import Any, Optional

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
    transcript_path: Optional[str] = None
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
    key_symbols: list[str] = field(default_factory=list)
    session_summary: str = ""
    compaction_count: int = 0
    original_start: str = ""
    target_tokens: int = 0

    def to_markdown(self, budget_tokens: int = 0) -> str:
        """
        Render snapshot as markdown for persistence.

        Args:
            budget_tokens: If > 0, try to fit within this token budget
        """
        sections = []

        # Header with compaction info
        sections.append("## ICR Preserved Context")
        sections.append(f"*Compaction #{self.compaction_count + 1} at {datetime.utcnow().isoformat()}Z*")
        if self.original_start:
            sections.append(f"*Session started: {self.original_start}*")
        sections.append("")

        # Pinned Invariants (ALWAYS preserved - highest priority)
        if self.invariants:
            sections.append("### Pinned Invariants (MUST preserve)")
            for inv in self.invariants:
                inv_id = inv.get("id", "unknown")
                priority = inv.get("priority", 5)
                content = inv.get("content", "")
                sections.append(f"- **[{inv_id}]** (P{priority}) {content}")
            sections.append("")

        # Session Summary (high priority - gives Claude quick context)
        if self.session_summary:
            sections.append("### Session Summary")
            sections.append(self.session_summary)
            sections.append("")

        # Active Todos (important for continuity)
        if self.active_todos:
            sections.append("### Active TODOs")
            for todo in self.active_todos[:10]:  # Limit if many
                content = todo.get("content", "")
                status = todo.get("status", "pending")
                marker = "[x]" if status == "completed" else "[ ]"
                sections.append(f"- {marker} {content}")
            if len(self.active_todos) > 10:
                sections.append(f"  - *...and {len(self.active_todos) - 10} more*")
            sections.append("")

        # Critical Decisions (important for understanding approach)
        if self.critical_decisions:
            sections.append("### Key Decisions Made")
            for decision in self.critical_decisions[:7]:  # Limit if many
                content = decision.get("content", "")
                sections.append(f"- {content}")
            if len(self.critical_decisions) > 7:
                sections.append(f"  - *...and {len(self.critical_decisions) - 7} more*")
            sections.append("")

        # Open Questions (if space permits)
        if self.open_questions:
            sections.append("### Open Questions")
            for question in self.open_questions[:5]:
                content = question.get("content", "")
                sections.append(f"- {content}")
            sections.append("")

        # Key Files (condensed)
        if self.key_files:
            sections.append("### Key Files")
            # Group by directory for compactness
            files_by_dir: dict[str, list[str]] = {}
            for f in self.key_files[:15]:
                parts = f.rsplit("/", 1)
                if len(parts) == 2:
                    dir_path, filename = parts
                    dir_path = dir_path or "."
                else:
                    dir_path, filename = ".", f
                if dir_path not in files_by_dir:
                    files_by_dir[dir_path] = []
                files_by_dir[dir_path].append(filename)

            for dir_path, files in list(files_by_dir.items())[:5]:
                if len(files) == 1:
                    sections.append(f"- `{dir_path}/{files[0]}`")
                else:
                    sections.append(f"- `{dir_path}/` ({', '.join(files[:4])})")
            sections.append("")

        # Key Symbols (if space permits and we have them)
        if self.key_symbols:
            sections.append("### Key Symbols")
            sections.append(f"- {', '.join(self.key_symbols[:15])}")
            sections.append("")

        # Footer
        sections.append("---")
        sections.append("*Context preserved by ICR. Use semantic search to retrieve more.*")

        return "\n".join(sections)

    def estimate_tokens(self) -> int:
        """Estimate token count of the snapshot."""
        content = self.to_markdown()
        return len(content) // 4  # Rough estimate

    def trim_to_budget(self, budget_tokens: int) -> "CompactSnapshot":
        """
        Create a trimmed copy that fits within the token budget.

        Priority order (preserved first):
        1. Invariants (always)
        2. Session summary
        3. Active todos
        4. Decisions
        5. Questions
        6. Files
        7. Symbols
        """
        if self.estimate_tokens() <= budget_tokens:
            return self

        # Create a copy and progressively trim
        trimmed = CompactSnapshot(
            invariants=self.invariants,  # Always keep
            session_summary=self.session_summary,
            active_todos=self.active_todos,
            critical_decisions=self.critical_decisions,
            open_questions=self.open_questions,
            key_files=self.key_files,
            key_symbols=self.key_symbols,
            compaction_count=self.compaction_count,
            original_start=self.original_start,
            target_tokens=budget_tokens,
        )

        # Progressively trim lower-priority items
        if trimmed.estimate_tokens() > budget_tokens:
            trimmed.key_symbols = []
        if trimmed.estimate_tokens() > budget_tokens:
            trimmed.key_files = trimmed.key_files[:5]
        if trimmed.estimate_tokens() > budget_tokens:
            trimmed.open_questions = trimmed.open_questions[:3]
        if trimmed.estimate_tokens() > budget_tokens:
            trimmed.critical_decisions = trimmed.critical_decisions[:5]
        if trimmed.estimate_tokens() > budget_tokens:
            trimmed.active_todos = trimmed.active_todos[:5]
        if trimmed.estimate_tokens() > budget_tokens:
            # Truncate summary as last resort
            if len(trimmed.session_summary) > 200:
                trimmed.session_summary = trimmed.session_summary[:200] + "..."

        return trimmed


class ICRClient:
    """
    Client for interacting with ICR/ICD services.

    This client extracts context from the transcript file directly since
    the conversation transcript contains all the information we need to
    preserve across compaction.
    """

    def __init__(self, config_path: Optional[str] = None):
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
        transcript_path: Optional[str],
    ) -> dict[str, Any]:
        """
        Extract key information from the conversation transcript.

        This parses the JSONL transcript to find:
        - Key files mentioned/modified (with edit actions prioritized)
        - Decisions made (from assistant messages)
        - Active todos (from tool calls)
        - Open questions (detected from conversation)
        - Code symbols mentioned (classes, functions, etc.)
        """
        result = {
            "key_files": [],
            "decisions": [],
            "todos": [],
            "questions": [],
            "symbols": [],
            "summary": "",
            "context_items": [],
        }

        if not transcript_path or not Path(transcript_path).exists():
            return result

        try:
            import json
            import re

            files_mentioned: dict[str, int] = {}  # file -> action weight
            decisions: list[dict[str, str]] = []
            todos: list[str] = []
            questions: list[str] = []
            symbols: set[str] = set()
            activities: list[str] = []

            with open(transcript_path) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Extract file paths from tool calls with action weighting
                    if entry.get("type") == "tool_use":
                        tool_name = entry.get("name", "")
                        tool_input = entry.get("input", {})
                        if isinstance(tool_input, dict):
                            # Weight files by importance of action
                            weight = 1
                            if tool_name in ["Edit", "Write", "NotebookEdit"]:
                                weight = 3  # Edits are most important
                            elif tool_name in ["Read"]:
                                weight = 2  # Reads are important
                            elif tool_name in ["Glob", "Grep"]:
                                weight = 1  # Searches less so

                            for key in ["file_path", "path", "file", "notebook_path"]:
                                if key in tool_input:
                                    file_path = tool_input[key]
                                    files_mentioned[file_path] = max(
                                        files_mentioned.get(file_path, 0), weight
                                    )

                            # Extract symbols from Edit tool (old_string/new_string)
                            if tool_name == "Edit":
                                for key in ["old_string", "new_string"]:
                                    if key in tool_input:
                                        code = tool_input[key]
                                        symbols.update(self._extract_symbols(code))

                    # Extract from assistant messages
                    if entry.get("type") == "text" and entry.get("role") == "assistant":
                        text = entry.get("text", "")

                        # Look for file paths in the text
                        path_pattern = r'[`"]?([a-zA-Z0-9_/.-]+\.[a-zA-Z0-9]+)[`"]?'
                        for match in re.findall(path_pattern, text):
                            if "/" in match and not match.startswith("http"):
                                files_mentioned[match] = max(files_mentioned.get(match, 0), 1)

                        # Extract decisions with better patterns
                        for line_text in text.split("\n"):
                            line_lower = line_text.lower().strip()
                            line_clean = line_text.strip()

                            # Decision patterns
                            decision_starters = [
                                "decided", "decision:", "we'll", "i'll",
                                "let's", "the approach", "using", "implemented",
                                "chose", "selected", "opted for", "going with",
                                "the plan is", "strategy:", "will use"
                            ]
                            if any(line_lower.startswith(d) for d in decision_starters):
                                if 10 < len(line_clean) < 200:
                                    decisions.append({
                                        "content": line_clean,
                                        "category": "approach"
                                    })

                            # Activity patterns for summary
                            activity_starters = [
                                "added", "created", "fixed", "updated",
                                "refactored", "implemented", "removed", "changed"
                            ]
                            if any(line_lower.startswith(a) for a in activity_starters):
                                if 10 < len(line_clean) < 100:
                                    activities.append(line_clean)

                        # Extract open questions
                        question_patterns = [
                            r"(?:should we|do we need to|what about|how should|consider whether)[^?]*\?",
                            r"(?:todo|fixme|xxx|hack):\s*(.+)",
                            r"(?:open question|unresolved|tbd):\s*(.+)",
                        ]
                        for pattern in question_patterns:
                            for match in re.findall(pattern, text.lower()):
                                if isinstance(match, str) and 10 < len(match) < 150:
                                    questions.append(match.strip())

                        # Extract symbols from code blocks
                        code_block_pattern = r'```(?:\w+)?\n(.*?)```'
                        for code_match in re.findall(code_block_pattern, text, re.DOTALL):
                            symbols.update(self._extract_symbols(code_match))

                    # Extract from user messages for questions
                    if entry.get("type") == "text" and entry.get("role") == "user":
                        text = entry.get("text", "")
                        # User questions that might still be open
                        if text.strip().endswith("?") and len(text) < 150:
                            questions.append(text.strip())

                    # Extract todos from TodoWrite tool calls
                    if entry.get("type") == "tool_use" and entry.get("name") == "TodoWrite":
                        tool_input = entry.get("input", {})
                        for todo in tool_input.get("todos", []):
                            if todo.get("status") in ["pending", "in_progress"]:
                                todos.append(todo.get("content", ""))

            # Sort files by importance (weight) and take top 15
            sorted_files = sorted(files_mentioned.items(), key=lambda x: -x[1])
            result["key_files"] = [f[0] for f in sorted_files[:15]]

            # Deduplicate decisions, keep most recent
            seen_decisions = set()
            unique_decisions = []
            for d in reversed(decisions):
                content_key = d["content"].lower()[:50]
                if content_key not in seen_decisions:
                    seen_decisions.add(content_key)
                    unique_decisions.append(d)
            result["decisions"] = list(reversed(unique_decisions))[:10]

            # Deduplicate and limit todos
            result["todos"] = list(dict.fromkeys(todos))[:20]

            # Deduplicate and limit questions
            result["questions"] = list(dict.fromkeys(questions))[:10]

            # Limit symbols
            result["symbols"] = list(symbols)[:20]

            # Generate improved summary
            result["summary"] = self._generate_session_summary(
                activities, decisions, sorted_files, todos
            )

        except Exception as e:
            logger.warning(f"Failed to extract from transcript: {e}")

        return result

    def _extract_symbols(self, code: str) -> set[str]:
        """Extract code symbols (classes, functions, variables) from code."""
        import re
        symbols = set()

        # Python/JS class definitions
        for match in re.findall(r'\bclass\s+(\w+)', code):
            symbols.add(match)

        # Python/JS function definitions
        for match in re.findall(r'\bdef\s+(\w+)', code):
            symbols.add(match)
        for match in re.findall(r'\bfunction\s+(\w+)', code):
            symbols.add(match)
        for match in re.findall(r'\basync\s+(?:def|function)\s+(\w+)', code):
            symbols.add(match)

        # TypeScript/Go type definitions
        for match in re.findall(r'\b(?:type|interface)\s+(\w+)', code):
            symbols.add(match)

        # Struct definitions (Go/Rust)
        for match in re.findall(r'\bstruct\s+(\w+)', code):
            symbols.add(match)

        # CamelCase identifiers (likely classes/types)
        for match in re.findall(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b', code):
            if len(match) > 3:
                symbols.add(match)

        return symbols

    def _generate_session_summary(
        self,
        activities: list[str],
        decisions: list[dict[str, str]],
        files: list[tuple[str, int]],
        todos: list[str],
    ) -> str:
        """Generate a coherent session summary."""
        parts = []

        # Summarize activities
        if activities:
            unique_activities = list(dict.fromkeys(activities))[:5]
            parts.append("**Activities:** " + "; ".join(unique_activities))

        # Summarize decisions
        if decisions:
            decision_texts = [d["content"][:80] for d in decisions[:3]]
            parts.append("**Key decisions:** " + "; ".join(decision_texts))

        # Summarize file focus
        if files:
            edited_files = [f[0] for f in files if f[1] >= 3][:5]
            if edited_files:
                parts.append("**Files modified:** " + ", ".join(
                    [f.split("/")[-1] for f in edited_files]
                ))

        # Note pending work
        if todos:
            parts.append(f"**Pending tasks:** {len(todos)} items remaining")

        if parts:
            return "\n".join(parts)
        return "Session in progress."

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
    decisions = transcript_data.get("decisions", [])
    if isinstance(decisions, list) and decisions:
        if isinstance(decisions[0], dict):
            # New format with content and category
            snapshot.critical_decisions = decisions
        else:
            # Old format - just strings
            snapshot.critical_decisions = [
                {"content": d, "timestamp": ""} for d in decisions
            ]

    # Convert extracted todos
    snapshot.active_todos = [
        {"content": t, "status": "pending"} for t in transcript_data.get("todos", [])
    ]

    # Open questions - now extracted from transcript
    snapshot.open_questions = [
        {"content": q} for q in transcript_data.get("questions", [])
    ]

    # Key files from transcript (now sorted by importance)
    snapshot.key_files = transcript_data.get("key_files", [])

    # Key symbols (classes, functions, etc.)
    snapshot.key_symbols = transcript_data.get("symbols", [])

    # Get session info (compaction count)
    session_info = client.get_session_info(hook_input.session_id)
    snapshot.compaction_count = session_info.get("compaction_count", 0)
    snapshot.original_start = session_info.get("started", "")

    # Session summary from transcript (now more detailed)
    snapshot.session_summary = transcript_data.get("summary", "")

    # Store target tokens for budget management
    snapshot.target_tokens = hook_input.target_tokens

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

    # Check if we have anything meaningful to persist
    has_content = (
        snapshot.invariants or
        snapshot.critical_decisions or
        snapshot.active_todos or
        snapshot.open_questions or
        snapshot.key_files or
        snapshot.session_summary
    )
    if not has_content:
        logger.info("No critical context to preserve")
        return HookOutput().to_dict()

    # Estimate tokens and apply budget management
    estimated_tokens = snapshot.estimate_tokens()

    # Budget for preserved context: aim for 10% of target tokens, max 4000
    context_budget = min(hook_input.target_tokens // 10, 4000) if hook_input.target_tokens else 2000

    if estimated_tokens > context_budget:
        logger.info(
            f"Trimming snapshot from {estimated_tokens} to {context_budget} tokens"
        )
        snapshot = snapshot.trim_to_budget(context_budget)
        estimated_tokens = snapshot.estimate_tokens()
        warnings.append(f"Context trimmed to fit budget ({estimated_tokens} tokens)")

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
