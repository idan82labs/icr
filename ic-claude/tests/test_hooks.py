"""
Tests for ICR Claude Code Hook Handlers

These tests verify the hook handlers work correctly:
- UserPromptSubmit: Context pack generation
- Stop: Ledger extraction
- PreCompact: Invariant persistence
"""

import json
import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

# Import hook modules
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.hook_userpromptsubmit import (
    HookInput as UserPromptInput,
    HookOutput as UserPromptOutput,
    ContextPack,
    handle_hook as handle_user_prompt,
    build_context_pack,
)

from scripts.hook_stop import (
    HookInput as StopInput,
    HookOutput as StopOutput,
    LedgerEntry,
    LedgerExtractor,
    handle_hook as handle_stop,
)

from scripts.hook_precompact import (
    HookInput as PreCompactInput,
    HookOutput as PreCompactOutput,
    CompactSnapshot,
    handle_hook as handle_precompact,
)


# =============================================================================
# UserPromptSubmit Hook Tests
# =============================================================================

class TestUserPromptSubmitHook:
    """Tests for UserPromptSubmit hook handler."""

    def test_input_parsing(self):
        """Test parsing of hook input."""
        data = {
            "session_id": "test-session-123",
            "prompt": "Help me with this code",
            "transcript_path": "/tmp/transcript.jsonl",
            "cwd": "/home/user/project",
        }

        hook_input = UserPromptInput.from_dict(data)

        assert hook_input.session_id == "test-session-123"
        assert hook_input.prompt == "Help me with this code"
        assert hook_input.transcript_path == "/tmp/transcript.jsonl"
        assert hook_input.cwd == "/home/user/project"

    def test_input_missing_optional_fields(self):
        """Test parsing with missing optional fields."""
        data = {
            "session_id": "test-session",
            "prompt": "test prompt",
        }

        hook_input = UserPromptInput.from_dict(data)

        assert hook_input.session_id == "test-session"
        assert hook_input.prompt == "test prompt"
        assert hook_input.transcript_path is None
        assert hook_input.cwd is None

    def test_output_serialization(self):
        """Test output serialization to dict."""
        output = UserPromptOutput(
            additional_context="## ICR Context Pack\nTest content",
            warnings=["Warning 1", "Warning 2"],
        )

        data = output.to_dict()

        assert data["additionalContext"] == "## ICR Context Pack\nTest content"
        assert data["warnings"] == ["Warning 1", "Warning 2"]

    def test_context_pack_markdown_generation(self):
        """Test context pack markdown generation."""
        pack = ContextPack(
            invariants=[
                {"id": "inv_123", "priority": 8, "content": "Always use TypeScript"},
            ],
            priors=[
                {"source": "auth.py", "relevance": 0.9, "content": "def login():..."},
            ],
            environment_summary="Working directory: /project",
            active_files=[
                {"path": "src/main.py", "relevance": 0.8},
            ],
            ledger_summary="- [decision] Use async handlers",
            token_budget=4000,
            tokens_used=500,
        )

        markdown = pack.to_markdown()

        assert "## ICR Context Pack" in markdown
        assert "### Pinned Invariants" in markdown
        assert "inv_123" in markdown
        assert "Always use TypeScript" in markdown
        assert "### Active Priors" in markdown
        assert "auth.py" in markdown
        assert "### Environment" in markdown
        assert "### Active Files" in markdown
        assert "src/main.py" in markdown
        assert "### Recent Decisions" in markdown
        assert "500/4000" in markdown

    def test_empty_context_pack(self):
        """Test empty context pack generation."""
        pack = ContextPack()
        markdown = pack.to_markdown()

        assert "## ICR Context Pack" in markdown
        # Should not have section headers for empty sections
        assert "### Pinned Invariants" not in markdown
        assert "### Active Priors" not in markdown

    def test_handle_hook_disabled(self):
        """Test hook when ICR is disabled."""
        with patch.dict("os.environ", {"ICR_DISABLE_HOOKS": "1"}):
            result = handle_user_prompt({
                "session_id": "test",
                "prompt": "test",
            })

            assert "warnings" in result
            assert any("disabled" in w.lower() for w in result["warnings"])

    def test_handle_hook_empty_input(self):
        """Test hook with empty input."""
        result = handle_user_prompt({})

        assert "additionalContext" in result
        assert "warnings" in result


# =============================================================================
# Stop Hook Tests (Ledger Extraction)
# =============================================================================

class TestLedgerExtractor:
    """Tests for deterministic ledger extraction."""

    def test_extract_structured_ledger(self):
        """Test extraction of properly structured ledger."""
        response = """
Here's what I did:

Ledger:
- Decisions: ['Use async/await', 'Add error handling']
- Todos: ['Write tests', 'Update docs']
- Open Questions: ['Should we use Redis?']
- Files touched: ['src/main.py', 'src/utils.py']

The implementation is complete.
"""
        extractor = LedgerExtractor()
        entries = extractor.extract(response)

        # Should extract all entries
        decisions = [e for e in entries if e.entry_type == "decision"]
        todos = [e for e in entries if e.entry_type == "todo"]
        questions = [e for e in entries if e.entry_type == "question"]
        files = [e for e in entries if e.entry_type == "file"]

        assert len(decisions) == 2
        assert len(todos) == 2
        assert len(questions) == 1
        assert len(files) == 2

        # Check content
        assert any("async/await" in d.content for d in decisions)
        assert any("Write tests" in t.content for t in todos)
        assert any("Redis" in q.content for q in questions)
        assert any("main.py" in f.content for f in files)

    def test_extract_empty_ledger(self):
        """Test extraction from response without ledger."""
        response = """
I've made some changes to the code.
Let me know if you have questions.
"""
        extractor = LedgerExtractor()
        entries = extractor.extract(response)

        # Should return empty list - no inference
        assert len(entries) == 0

    def test_extract_partial_ledger(self):
        """Test extraction of partial ledger."""
        response = """
Ledger:
- Decisions: ['Implement caching']
- Todos: None
- Open Questions: []
- Files touched: ['cache.py']
"""
        extractor = LedgerExtractor()
        entries = extractor.extract(response)

        decisions = [e for e in entries if e.entry_type == "decision"]
        todos = [e for e in entries if e.entry_type == "todo"]
        files = [e for e in entries if e.entry_type == "file"]

        assert len(decisions) == 1
        assert len(todos) == 0  # None should not create entries
        assert len(files) == 1

    def test_extract_alternative_format_prefix(self):
        """Test extraction of prefix-style entries."""
        response = """
DECISION: Use PostgreSQL instead of MySQL
TODO: Set up database migrations
QUESTION: What about connection pooling?
"""
        extractor = LedgerExtractor()
        entries = extractor.extract(response)

        assert len(entries) == 3
        assert any(e.entry_type == "decision" for e in entries)
        assert any(e.entry_type == "todo" for e in entries)
        assert any(e.entry_type == "question" for e in entries)

    def test_extract_markdown_tasks(self):
        """Test extraction of markdown task lists."""
        response = """
## Tasks
- [ ] Implement authentication
- [x] Set up project structure
- [ ] Write documentation
"""
        extractor = LedgerExtractor()
        entries = extractor.extract(response)

        # Only unchecked tasks should be todos
        todos = [e for e in entries if e.entry_type == "todo"]
        assert len(todos) == 2
        assert any("authentication" in t.content for t in todos)
        assert any("documentation" in t.content for t in todos)
        # Checked task should not be included
        assert not any("project structure" in t.content for t in todos)

    def test_deterministic_extraction(self):
        """Test that extraction is deterministic (no randomness)."""
        response = """
Ledger:
- Decisions: ['A', 'B', 'C']
- Todos: ['X', 'Y']
"""
        extractor = LedgerExtractor()

        # Extract multiple times
        results = [extractor.extract(response) for _ in range(5)]

        # All results should be identical
        for result in results[1:]:
            assert len(result) == len(results[0])
            for i, entry in enumerate(result):
                assert entry.entry_type == results[0][i].entry_type
                assert entry.content == results[0][i].content

    def test_no_free_text_inference(self):
        """Test that extractor doesn't infer from free text."""
        response = """
I decided to use React for the frontend.
I think we should probably add some tests later.
One question I have is about the deployment process.
I modified the main.py file.
"""
        extractor = LedgerExtractor()
        entries = extractor.extract(response)

        # Should NOT extract anything - these are not structured
        assert len(entries) == 0


class TestStopHook:
    """Tests for Stop hook handler."""

    def test_input_parsing(self):
        """Test parsing of stop hook input."""
        data = {
            "session_id": "test-session",
            "response": "Here's the implementation...",
            "transcript_path": "/tmp/transcript.jsonl",
        }

        hook_input = StopInput.from_dict(data)

        assert hook_input.session_id == "test-session"
        assert hook_input.response == "Here's the implementation..."
        assert hook_input.transcript_path == "/tmp/transcript.jsonl"

    def test_output_serialization(self):
        """Test output serialization."""
        output = StopOutput(
            success=True,
            ledger_entries=[
                LedgerEntry(entry_type="decision", content="Use async"),
            ],
            metrics={"response_length": 100},
            warnings=[],
        )

        data = output.to_dict()

        assert data["success"] is True
        assert len(data["ledger_entries"]) == 1
        assert data["ledger_entries"][0]["type"] == "decision"
        assert data["metrics"]["response_length"] == 100

    def test_handle_empty_response(self):
        """Test hook with empty response."""
        result = handle_stop({
            "session_id": "test",
            "response": "",
        })

        assert result["success"] is True
        assert len(result["ledger_entries"]) == 0

    def test_handle_hook_disabled(self):
        """Test hook when ICR is disabled."""
        with patch.dict("os.environ", {"ICR_DISABLE_HOOKS": "1"}):
            result = handle_stop({
                "session_id": "test",
                "response": "Ledger:\n- Decisions: ['test']",
            })

            assert any("disabled" in w.lower() for w in result["warnings"])


# =============================================================================
# PreCompact Hook Tests
# =============================================================================

class TestPreCompactHook:
    """Tests for PreCompact hook handler."""

    def test_input_parsing(self):
        """Test parsing of precompact hook input."""
        data = {
            "session_id": "test-session",
            "transcript_path": "/tmp/transcript.jsonl",
            "compaction_reason": "token_limit",
            "current_tokens": 150000,
            "target_tokens": 100000,
        }

        hook_input = PreCompactInput.from_dict(data)

        assert hook_input.session_id == "test-session"
        assert hook_input.compaction_reason == "token_limit"
        assert hook_input.current_tokens == 150000
        assert hook_input.target_tokens == 100000

    def test_snapshot_markdown_generation(self):
        """Test compact snapshot markdown generation."""
        snapshot = CompactSnapshot(
            invariants=[
                {
                    "id": "inv_1",
                    "priority": 9,
                    "content": "Always validate input",
                    "created": "2024-01-01T00:00:00Z",
                }
            ],
            critical_decisions=[
                {"content": "Use microservices", "timestamp": "2024-01-02"},
            ],
            active_todos=[
                {"content": "Add logging"},
            ],
            open_questions=[
                {"content": "Scale strategy?"},
            ],
            key_files=["main.py", "config.py"],
            session_summary="Built authentication system",
            compaction_count=2,
            original_start="2024-01-01T00:00:00Z",
        )

        markdown = snapshot.to_markdown()

        assert "## ICR Preserved Context" in markdown
        assert "Compaction #3" in markdown  # count + 1
        assert "### Pinned Invariants" in markdown
        assert "inv_1" in markdown
        assert "Always validate input" in markdown
        assert "### Key Decisions Made" in markdown
        assert "microservices" in markdown
        assert "### Active TODOs" in markdown
        assert "Add logging" in markdown
        assert "### Open Questions" in markdown
        assert "Scale strategy" in markdown
        assert "### Key Files" in markdown
        assert "main.py" in markdown
        assert "### Session Summary" in markdown
        assert "authentication" in markdown

    def test_empty_snapshot(self):
        """Test empty snapshot generation."""
        snapshot = CompactSnapshot()
        markdown = snapshot.to_markdown()

        assert "## ICR Preserved Context" in markdown
        # Should not have content sections
        assert "### Pinned Invariants" not in markdown
        assert "### Key Decisions Made" not in markdown

    def test_output_serialization(self):
        """Test output serialization."""
        output = PreCompactOutput(
            persisted_context="## ICR Preserved Context\n...",
            warnings=["Large context"],
        )

        data = output.to_dict()

        assert data["persistedContext"] == "## ICR Preserved Context\n..."
        assert data["warnings"] == ["Large context"]

    def test_handle_hook_disabled(self):
        """Test hook when ICR is disabled."""
        with patch.dict("os.environ", {"ICR_DISABLE_HOOKS": "1"}):
            result = handle_precompact({
                "session_id": "test",
                "compaction_reason": "manual",
            })

            assert any("disabled" in w.lower() for w in result["warnings"])


# =============================================================================
# Integration Tests
# =============================================================================

class TestHookIntegration:
    """Integration tests for hook handlers."""

    def test_full_hook_cycle(self):
        """Test a full cycle: prompt -> response -> compact."""
        # 1. UserPromptSubmit
        user_result = handle_user_prompt({
            "session_id": "integration-test",
            "prompt": "Help me refactor this code",
            "cwd": "/project",
        })

        assert "additionalContext" in user_result
        assert isinstance(user_result["warnings"], list)

        # 2. Stop (with ledger in response)
        stop_result = handle_stop({
            "session_id": "integration-test",
            "response": """
I've refactored the code.

Ledger:
- Decisions: ['Extract to separate module']
- Todos: ['Add unit tests']
- Open Questions: []
- Files touched: ['refactored.py']

Let me know if you need anything else.
""",
        })

        assert stop_result["success"] is True
        entries = stop_result["ledger_entries"]
        assert len(entries) >= 2  # At least decision and file

        # 3. PreCompact
        compact_result = handle_precompact({
            "session_id": "integration-test",
            "compaction_reason": "token_limit",
            "current_tokens": 150000,
            "target_tokens": 100000,
        })

        assert "persistedContext" in compact_result
        assert isinstance(compact_result["warnings"], list)

    def test_hooks_handle_malformed_input(self):
        """Test that hooks handle malformed input gracefully."""
        # None values
        result1 = handle_user_prompt({"session_id": None, "prompt": None})
        assert "warnings" in result1 or "additionalContext" in result1

        # Wrong types
        result2 = handle_stop({"session_id": 123, "response": ["not", "string"]})
        assert "warnings" in result2 or "success" in result2

        # Completely wrong structure
        result3 = handle_precompact("not a dict")
        assert isinstance(result3, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
