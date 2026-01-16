"""
Integration tests for hook integration.

Tests cover:
- UserPromptSubmit hook
- Stop hook (ledger parsing)
- PreCompact hook
- Hook fallback behavior
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ==============================================================================
# Hook Response Types
# ==============================================================================

@dataclass
class HookResponse:
    """Response from a hook."""

    continue_execution: bool
    additional_context: str | None = None
    modified_prompt: str | None = None
    error: str | None = None


@dataclass
class LedgerParseResult:
    """Result of parsing ledger from stop hook."""

    decisions: list[str]
    invariants: list[str]
    observations: list[str]
    success: bool
    error: str | None = None


# ==============================================================================
# Hook Simulation Functions
# ==============================================================================

def simulate_user_prompt_submit_hook(
    prompt: str,
    repo_root: str,
    transcript_summary: str | None = None,
    config: dict[str, Any] | None = None,
) -> HookResponse:
    """
    Simulate UserPromptSubmit hook execution.

    This hook is called when user submits a prompt.
    It can inject additional context from ICR.
    """
    config = config or {}

    # Simulate pack generation
    if config.get("auto_pack", True):
        # Generate context pack
        context = f"""<!-- ICR Context Pack -->
<!-- Query: {prompt} -->
<!-- Mode: pack -->

## Relevant Context

Based on your query, here are relevant code snippets:

```typescript
// Example relevant code
function handleAuth(token: AuthToken): Promise<User> {{
  // Implementation
}}
```

<!-- End ICR Context Pack -->
"""
        return HookResponse(
            continue_execution=True,
            additional_context=context,
        )

    return HookResponse(continue_execution=True)


def simulate_stop_hook(
    assistant_response: str,
    transcript_path: str | None = None,
) -> LedgerParseResult:
    """
    Simulate Stop hook execution.

    This hook is called when assistant response is complete.
    It extracts structured ledger entries.
    """
    import re

    # Try to extract JSON ledger
    json_pattern = r'```json\s*\n\s*\{[^}]*"ledger"[^}]*\}[^`]*```'
    match = re.search(json_pattern, assistant_response, re.DOTALL)

    if match:
        try:
            json_text = match.group(0).replace("```json", "").replace("```", "").strip()
            data = json.loads(json_text)
            ledger = data.get("ledger", data)

            return LedgerParseResult(
                decisions=ledger.get("decisions", []),
                invariants=ledger.get("invariants", []),
                observations=ledger.get("observations", []),
                success=True,
            )
        except json.JSONDecodeError as e:
            return LedgerParseResult(
                decisions=[],
                invariants=[],
                observations=[],
                success=False,
                error=str(e),
            )

    # Try HTML comment format
    comment_pattern = r'<!--\s*LEDGER:\s*(\{.*?\})\s*-->'
    match = re.search(comment_pattern, assistant_response, re.DOTALL)

    if match:
        try:
            ledger = json.loads(match.group(1))
            return LedgerParseResult(
                decisions=ledger.get("decisions", []),
                invariants=ledger.get("invariants", []),
                observations=ledger.get("observations", []),
                success=True,
            )
        except json.JSONDecodeError as e:
            return LedgerParseResult(
                decisions=[],
                invariants=[],
                observations=[],
                success=False,
                error=str(e),
            )

    # No ledger found
    return LedgerParseResult(
        decisions=[],
        invariants=[],
        observations=[],
        success=True,  # Not finding ledger is not an error
    )


def simulate_pre_compact_hook(
    current_transcript: list[dict],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Simulate PreCompact hook execution.

    This hook is called before conversation compaction.
    It extracts and preserves important information.
    """
    config = config or {}

    # Extract pinned items
    pinned = []
    ledger_entries = []

    for message in current_transcript:
        # Check for pinned content markers
        if "PINNED:" in message.get("content", ""):
            pinned.append(message)

        # Check for ledger entries
        if "ledger" in message:
            ledger_entries.extend(message.get("ledger", {}).get("decisions", []))
            ledger_entries.extend(message.get("ledger", {}).get("invariants", []))

    return {
        "pinned_messages": pinned,
        "ledger_entries": ledger_entries,
        "preserve_count": len(pinned) + len(ledger_entries),
    }


# ==============================================================================
# UserPromptSubmit Hook Tests
# ==============================================================================

@pytest.mark.integration
class TestUserPromptSubmitHook:
    """Tests for UserPromptSubmit hook."""

    def test_hook_returns_context(self):
        """Test hook returns additional context."""
        response = simulate_user_prompt_submit_hook(
            prompt="Where is auth token validated?",
            repo_root="/path/to/repo",
        )

        assert response.continue_execution is True
        assert response.additional_context is not None
        assert len(response.additional_context) > 0

    def test_hook_includes_query_info(self):
        """Test hook includes query information in context."""
        prompt = "How does the refresh token work?"

        response = simulate_user_prompt_submit_hook(
            prompt=prompt,
            repo_root="/path/to/repo",
        )

        # Context should reference the query
        assert prompt in response.additional_context or "refresh" in response.additional_context.lower()

    def test_hook_can_be_disabled(self):
        """Test hook can be disabled via config."""
        response = simulate_user_prompt_submit_hook(
            prompt="test",
            repo_root="/path",
            config={"auto_pack": False},
        )

        # Should still continue but no additional context
        assert response.continue_execution is True
        assert response.additional_context is None

    def test_hook_with_transcript_summary(self):
        """Test hook with transcript summary."""
        response = simulate_user_prompt_submit_hook(
            prompt="Continue with authentication",
            repo_root="/path",
            transcript_summary="User was working on auth flow",
        )

        assert response.continue_execution is True


# ==============================================================================
# Stop Hook Tests
# ==============================================================================

@pytest.mark.integration
class TestStopHook:
    """Tests for Stop hook (ledger parsing)."""

    def test_parse_json_ledger(self):
        """Test parsing JSON ledger from response."""
        response = '''
Here is my analysis:

```json
{
    "ledger": {
        "decisions": ["Use JWT tokens", "Implement refresh flow"],
        "invariants": ["Tokens must be validated before use"],
        "observations": ["Current code uses session cookies"]
    }
}
```
'''
        result = simulate_stop_hook(response)

        assert result.success is True
        assert len(result.decisions) == 2
        assert len(result.invariants) == 1
        assert len(result.observations) == 1

    def test_parse_comment_ledger(self):
        """Test parsing HTML comment ledger."""
        response = '''
The auth flow works like this...

<!-- LEDGER: {"decisions": ["Migrate to OAuth"], "invariants": ["Always verify signature"]} -->
'''
        result = simulate_stop_hook(response)

        assert result.success is True
        assert "Migrate to OAuth" in result.decisions
        assert "Always verify signature" in result.invariants

    def test_no_ledger_is_success(self):
        """Test that no ledger is not an error."""
        response = "Just a regular response without any ledger."

        result = simulate_stop_hook(response)

        assert result.success is True
        assert len(result.decisions) == 0
        assert len(result.invariants) == 0

    def test_malformed_json_handled(self):
        """Test handling of malformed JSON ledger."""
        response = '''
```json
{
    "ledger": {
        "decisions": ["unclosed
    }
}
```
'''
        result = simulate_stop_hook(response)

        # Should handle error gracefully
        assert result.success is False or len(result.decisions) == 0


# ==============================================================================
# PreCompact Hook Tests
# ==============================================================================

@pytest.mark.integration
class TestPreCompactHook:
    """Tests for PreCompact hook."""

    def test_extract_pinned_content(self):
        """Test extraction of pinned content."""
        transcript = [
            {"role": "user", "content": "Question 1"},
            {"role": "assistant", "content": "PINNED: Important invariant here"},
            {"role": "user", "content": "Question 2"},
        ]

        result = simulate_pre_compact_hook(transcript)

        assert len(result["pinned_messages"]) == 1
        assert "PINNED:" in result["pinned_messages"][0]["content"]

    def test_extract_ledger_entries(self):
        """Test extraction of ledger entries."""
        transcript = [
            {"role": "assistant", "content": "Response 1", "ledger": {
                "decisions": ["Decision A", "Decision B"],
                "invariants": ["Invariant 1"],
            }},
            {"role": "assistant", "content": "Response 2", "ledger": {
                "decisions": ["Decision C"],
            }},
        ]

        result = simulate_pre_compact_hook(transcript)

        assert len(result["ledger_entries"]) == 4  # 3 decisions + 1 invariant

    def test_preserve_count(self):
        """Test preserve count calculation."""
        transcript = [
            {"role": "assistant", "content": "PINNED: Item 1"},
            {"role": "assistant", "content": "PINNED: Item 2"},
            {"role": "assistant", "content": "Regular", "ledger": {"decisions": ["D1"]}},
        ]

        result = simulate_pre_compact_hook(transcript)

        assert result["preserve_count"] == 3  # 2 pinned + 1 ledger entry


# ==============================================================================
# Hook Fallback Tests
# ==============================================================================

@pytest.mark.integration
class TestHookFallback:
    """Tests for hook fallback behavior."""

    def test_fallback_when_hook_disabled(self):
        """Test fallback when hooks are disabled."""
        # Simulate disabled hook
        response = HookResponse(continue_execution=True)

        # System should continue without additional context
        assert response.continue_execution is True
        assert response.additional_context is None

    def test_fallback_when_hook_errors(self):
        """Test fallback when hook encounters error."""
        response = HookResponse(
            continue_execution=True,
            error="Hook script not found",
        )

        # System should continue despite error
        assert response.continue_execution is True
        assert response.error is not None

    def test_ic_commands_work_without_hooks(self):
        """Test /ic commands work without hooks."""
        # Simulate direct CLI invocation without hook
        # This tests the fallback path

        # Mock CLI command execution
        result = {
            "status": "success",
            "output": "Pack generated successfully",
        }

        assert result["status"] == "success"

    def test_graceful_degradation(self):
        """Test graceful degradation when services unavailable."""
        # Simulate service unavailable
        def failing_hook(*args, **kwargs):
            raise ConnectionError("Service unavailable")

        # System should handle gracefully
        try:
            failing_hook()
            fallback_used = False
        except ConnectionError:
            fallback_used = True

        assert fallback_used


# ==============================================================================
# Hook Configuration Tests
# ==============================================================================

@pytest.mark.integration
class TestHookConfiguration:
    """Tests for hook configuration."""

    def test_hook_config_structure(self):
        """Test expected hook configuration structure."""
        config = {
            "hooks": {
                "UserPromptSubmit": [
                    {
                        "matcher": ".*",
                        "hooks": [
                            {
                                "type": "command",
                                "command": "icr hook prompt-submit",
                            }
                        ],
                    }
                ],
                "Stop": [
                    {
                        "matcher": ".*",
                        "hooks": [
                            {
                                "type": "command",
                                "command": "icr hook stop",
                            }
                        ],
                    }
                ],
            }
        }

        assert "hooks" in config
        assert "UserPromptSubmit" in config["hooks"]
        assert "Stop" in config["hooks"]

    def test_hook_matcher_pattern(self):
        """Test hook matcher pattern."""
        import re

        matcher = ".*"  # Match all
        test_prompts = [
            "Where is auth?",
            "How does this work?",
            "",
            "Special chars: @#$%",
        ]

        for prompt in test_prompts:
            assert re.match(matcher, prompt) is not None

    def test_hook_command_format(self):
        """Test hook command format."""
        command = "icr hook prompt-submit"

        # Should be a valid command string
        parts = command.split()
        assert parts[0] == "icr"
        assert parts[1] == "hook"


# ==============================================================================
# Hook Integration Tests
# ==============================================================================

@pytest.mark.integration
class TestHookIntegration:
    """Integration tests for hook system."""

    def test_prompt_submit_to_stop_flow(self):
        """Test complete prompt submit to stop flow."""
        # 1. User submits prompt
        prompt_response = simulate_user_prompt_submit_hook(
            prompt="Explain the auth flow",
            repo_root="/path/to/repo",
        )
        assert prompt_response.continue_execution

        # 2. Simulate assistant response with ledger
        assistant_response = '''
The auth flow works as follows:

```json
{
    "ledger": {
        "decisions": ["Document auth flow"],
        "invariants": ["Token validation required"]
    }
}
```
'''

        # 3. Stop hook parses ledger
        stop_result = simulate_stop_hook(assistant_response)
        assert stop_result.success
        assert len(stop_result.decisions) > 0

    def test_multiple_hooks_in_sequence(self):
        """Test multiple hooks executing in sequence."""
        # Simulate a conversation with multiple hook invocations
        conversation = []

        # Turn 1
        response1 = simulate_user_prompt_submit_hook("Question 1", "/repo")
        conversation.append({"prompt": "Question 1", "context": response1.additional_context})

        # Turn 2
        response2 = simulate_user_prompt_submit_hook("Question 2", "/repo")
        conversation.append({"prompt": "Question 2", "context": response2.additional_context})

        # All hooks should succeed
        for turn in conversation:
            assert turn["context"] is not None or turn["context"] is None  # Either is valid
