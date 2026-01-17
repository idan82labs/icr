#!/usr/bin/env python3
"""
UserPromptSubmit Hook Handler for ICR

This hook is invoked before each user prompt is sent to Claude.
It injects the ICR context pack (priors, invariants, environment summary)
into the additionalContext field.

Input (via stdin):
{
  "session_id": "uuid",
  "prompt": "user prompt text",
  "transcript_path": "/path/to/transcript.jsonl",
  "cwd": "/current/working/directory"
}

Output (via stdout):
{
  "additionalContext": "## ICR Context Pack\n...",
  "warnings": []
}
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

# Configure logging to stderr (stdout is reserved for hook output)
logging.basicConfig(
    level=logging.DEBUG if os.environ.get("ICR_DEBUG") else logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("icr.hook.userpromptsubmit")


@dataclass
class HookInput:
    """Input data from Claude Code UserPromptSubmit hook."""

    session_id: str
    prompt: str
    transcript_path: str | None = None
    cwd: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HookInput":
        """Create HookInput from dictionary."""
        return cls(
            session_id=data.get("session_id", ""),
            prompt=data.get("prompt", ""),
            transcript_path=data.get("transcript_path"),
            cwd=data.get("cwd"),
        )


@dataclass
class HookOutput:
    """Output data for Claude Code UserPromptSubmit hook."""

    additional_context: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON output."""
        return {
            "additionalContext": self.additional_context,
            "warnings": self.warnings,
        }


@dataclass
class ContextPack:
    """ICR Context Pack containing priors, invariants, and environment info."""

    priors: list[dict[str, Any]] = field(default_factory=list)
    invariants: list[dict[str, Any]] = field(default_factory=list)
    environment_summary: str = ""
    active_files: list[dict[str, Any]] = field(default_factory=list)
    ledger_summary: str = ""
    token_budget: int = 4000
    tokens_used: int = 0

    def to_markdown(self) -> str:
        """Render context pack as markdown for injection."""
        sections = []

        # Header
        sections.append("## ICR Context Pack")
        sections.append(f"*Generated: {datetime.utcnow().isoformat()}Z*")
        sections.append("")

        # Pinned Invariants (always first, highest priority)
        if self.invariants:
            sections.append("### Pinned Invariants")
            for inv in self.invariants:
                priority = inv.get("priority", 5)
                content = inv.get("content", "")
                inv_id = inv.get("id", "unknown")
                sections.append(f"- [{inv_id}] (P{priority}) {content}")
            sections.append("")

        # Active Priors
        if self.priors:
            sections.append("### Active Priors")
            for prior in self.priors:
                source = prior.get("source", "unknown")
                relevance = prior.get("relevance", 0.0)
                content = prior.get("content", "")
                # Truncate long content
                if len(content) > 500:
                    content = content[:497] + "..."
                sections.append(f"**{source}** (relevance: {relevance:.2f})")
                sections.append(f"```\n{content}\n```")
                sections.append("")

        # Environment Summary
        if self.environment_summary:
            sections.append("### Environment")
            sections.append(self.environment_summary)
            sections.append("")

        # Active Files
        if self.active_files:
            sections.append("### Active Files")
            for f in self.active_files[:10]:  # Limit to top 10
                path = f.get("path", "unknown")
                score = f.get("relevance", 0.0)
                sections.append(f"- `{path}` (relevance: {score:.2f})")
            sections.append("")

        # Recent Ledger
        if self.ledger_summary:
            sections.append("### Recent Decisions")
            sections.append(self.ledger_summary)
            sections.append("")

        # Footer with token info
        sections.append(f"*Context tokens: {self.tokens_used}/{self.token_budget}*")

        return "\n".join(sections)


# Continuation query detection patterns
CONTINUATION_PATTERNS = [
    r"that (?:function|class|file|code|method|module) (?:we|from|I|you|earlier)",
    r"the (?:function|class|file|code|method|module) from (?:before|earlier)",
    r"what was (?:that|the) .+",
    r"continue with .+",
    r"back to (?:the|that) .+",
    r"as we discussed",
    r"remember (?:when|that|the)",
    r"like (?:we|I) (?:mentioned|said|discussed)",
    r"the (?:thing|stuff|code) (?:we|I) (?:were|was) (?:working on|looking at)",
    r"earlier (?:we|I|you)",
    r"(?:that|the) (?:same|previous) .+",
]


def is_continuation_query(prompt: str) -> bool:
    """Detect if query references pre-compaction context."""
    prompt_lower = prompt.lower()
    return any(re.search(p, prompt_lower) for p in CONTINUATION_PATTERNS)


def get_pinned_context(project_root: Path) -> str:
    """Retrieve pinned context from last compaction."""
    pins_file = project_root / ".icr" / "pins.json"
    if not pins_file.exists():
        return ""

    try:
        pins = json.loads(pins_file.read_text())
        if not pins:
            return ""

        # Format pinned items as context
        parts = ["## Preserved Context (from before compaction)"]
        parts.append("")

        # Group pins by type
        files = [p for p in pins if p.get("type") == "file"]
        decisions = [p for p in pins if p.get("type") == "decision"]
        todos = [p for p in pins if p.get("type") == "todo"]

        if files:
            parts.append("### Key Files")
            for pin in files[:5]:
                label = pin.get("label", "unknown")
                score = pin.get("score", 0)
                parts.append(f"- `{label}` (relevance: {score:.2f})")
            parts.append("")

        if decisions:
            parts.append("### Recent Decisions")
            for pin in decisions[:3]:
                content = pin.get("content", "")[:200]
                parts.append(f"- {content}")
            parts.append("")

        if todos:
            parts.append("### Active TODOs")
            for pin in todos[:5]:
                content = pin.get("content", "")
                parts.append(f"- [ ] {content}")
            parts.append("")

        parts.append("---")
        return "\n".join(parts)

    except Exception as e:
        logger.warning(f"Failed to load pinned context: {e}")
        return ""


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
        """Ensure ICR is initialized, return False if not available."""
        if self._initialized:
            return True

        # Check if config exists
        if not Path(self.config_path).exists():
            logger.warning(f"ICR config not found at {self.config_path}")
            return False

        # Try to load config
        try:
            import yaml
            with open(self.config_path) as f:
                self._config = yaml.safe_load(f)
            self._initialized = True
            return True
        except ImportError:
            logger.warning("PyYAML not installed, using defaults")
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

    def get_priors(
        self,
        query: str,
        cwd: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Retrieve relevant priors for the given query.

        This queries the ICR vector store for semantically similar context.
        """
        if not self._ensure_initialized():
            return []

        try:
            # Try to use icr-core if available
            from icr.core.retrieval import PriorRetriever
            retriever = PriorRetriever(db_path=self.db_path)
            return retriever.search(query, limit=limit, cwd=cwd)
        except ImportError:
            logger.debug("icr-core not available, skipping prior retrieval")
            return []
        except Exception as e:
            logger.warning(f"Prior retrieval failed: {e}")
            return []

    def get_invariants(self) -> list[dict[str, Any]]:
        """Retrieve all pinned invariants."""
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
            logger.warning(f"Invariant retrieval failed: {e}")
            return []

    def get_environment_summary(self, cwd: str | None = None) -> str:
        """Get a summary of the current environment."""
        if not self._ensure_initialized():
            return ""

        try:
            from icr.core.environment import EnvironmentAnalyzer
            analyzer = EnvironmentAnalyzer(db_path=self.db_path)
            return analyzer.summarize(cwd=cwd)
        except ImportError:
            # Provide basic environment info as fallback
            parts = []
            if cwd:
                parts.append(f"Working directory: {cwd}")
                # Check for common project indicators
                cwd_path = Path(cwd)
                if (cwd_path / "package.json").exists():
                    parts.append("Project type: Node.js")
                elif (cwd_path / "pyproject.toml").exists():
                    parts.append("Project type: Python")
                elif (cwd_path / "Cargo.toml").exists():
                    parts.append("Project type: Rust")
                elif (cwd_path / "go.mod").exists():
                    parts.append("Project type: Go")
            return "\n".join(parts) if parts else ""
        except Exception as e:
            logger.warning(f"Environment summary failed: {e}")
            return ""

    def get_active_files(
        self,
        cwd: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get recently active files with relevance scores."""
        if not self._ensure_initialized():
            return []

        try:
            from icr.core.files import FileTracker
            tracker = FileTracker(db_path=self.db_path)
            return tracker.get_recent(cwd=cwd, limit=limit)
        except ImportError:
            logger.debug("icr-core not available, skipping file tracking")
            return []
        except Exception as e:
            logger.warning(f"Active files retrieval failed: {e}")
            return []

    def get_ledger_summary(self, session_id: str, limit: int = 5) -> str:
        """Get summary of recent ledger entries."""
        if not self._ensure_initialized():
            return ""

        try:
            from icr.core.ledger import LedgerStore
            store = LedgerStore(db_path=self.db_path)
            entries = store.get_recent(session_id=session_id, limit=limit)

            if not entries:
                return ""

            lines = []
            for entry in entries:
                entry_type = entry.get("type", "unknown")
                content = entry.get("content", "")
                lines.append(f"- [{entry_type}] {content}")

            return "\n".join(lines)
        except ImportError:
            logger.debug("icr-core not available, skipping ledger")
            return ""
        except Exception as e:
            logger.warning(f"Ledger summary failed: {e}")
            return ""

    def record_prompt(self, session_id: str, prompt: str, cwd: str | None = None) -> None:
        """Record the prompt for metrics and learning."""
        if not self._ensure_initialized():
            return

        try:
            from icr.core.metrics import MetricsRecorder
            recorder = MetricsRecorder(db_path=self.db_path)
            recorder.record_prompt(
                session_id=session_id,
                prompt=prompt,
                cwd=cwd,
                timestamp=datetime.utcnow(),
            )
        except ImportError:
            logger.debug("icr-core not available, skipping metrics")
        except Exception as e:
            logger.warning(f"Metrics recording failed: {e}")


def build_context_pack(
    client: ICRClient,
    hook_input: HookInput,
) -> ContextPack:
    """
    Build a context pack for the given prompt.

    This is the core logic for assembling relevant context.
    """
    pack = ContextPack()

    # Get token budget from config
    pack.token_budget = client.get_config("max_context_tokens", 4000)

    # Always include invariants first (they're pinned for a reason)
    pack.invariants = client.get_invariants()

    # Get relevant priors based on the prompt
    pack.priors = client.get_priors(
        query=hook_input.prompt,
        cwd=hook_input.cwd,
        limit=5,
    )

    # Get environment summary
    pack.environment_summary = client.get_environment_summary(cwd=hook_input.cwd)

    # Get active files
    pack.active_files = client.get_active_files(cwd=hook_input.cwd, limit=10)

    # Get ledger summary
    pack.ledger_summary = client.get_ledger_summary(
        session_id=hook_input.session_id,
        limit=5,
    )

    # Estimate tokens used (rough estimate: 4 chars per token)
    content = pack.to_markdown()
    pack.tokens_used = len(content) // 4

    return pack


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

    # Find config path based on cwd (check .icd/ and .icr/)
    config_path = None
    if hook_input.cwd:
        project_root = Path(hook_input.cwd)
        for config_dir in [".icd", ".icr"]:
            candidate = project_root / config_dir / "config.yaml"
            if candidate.exists():
                config_path = str(candidate)
                break

    # Initialize client with project config
    client = ICRClient(config_path=config_path)

    # Check if auto-injection is enabled
    if not client.get_config("auto_inject", True):
        logger.info("Auto-injection disabled in config")
        return HookOutput().to_dict()

    # Build context pack
    try:
        pack = build_context_pack(client, hook_input)
    except Exception as e:
        logger.error(f"Failed to build context pack: {e}")
        warnings.append(f"Context pack generation failed: {e}")
        return HookOutput(warnings=warnings).to_dict()

    # Check for continuation query and inject pinned context
    additional_context = pack.to_markdown()
    if is_continuation_query(hook_input.prompt):
        project_root = Path(hook_input.cwd) if hook_input.cwd else Path.cwd()
        pinned_context = get_pinned_context(project_root)
        if pinned_context:
            # Prepend pinned context to the pack
            additional_context = pinned_context + "\n\n" + additional_context
            logger.info("Injected pinned context for continuation query")

    # Record prompt for metrics
    try:
        client.record_prompt(
            session_id=hook_input.session_id,
            prompt=hook_input.prompt,
            cwd=hook_input.cwd,
        )
    except Exception as e:
        logger.warning(f"Failed to record prompt: {e}")
        # Non-fatal, continue

    # Generate output
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
