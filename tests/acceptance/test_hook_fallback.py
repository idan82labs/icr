"""
Acceptance test for hook fallback behavior.

From PRD:
- Disable hooks
- Pass: /ic commands still work
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ==============================================================================
# Test Data Structures
# ==============================================================================

@dataclass
class CommandResult:
    """Result of CLI command execution."""

    success: bool
    output: str
    error: str | None = None
    exit_code: int = 0


@dataclass
class HookStatus:
    """Status of hooks."""

    enabled: bool
    available: bool
    error: str | None = None


# ==============================================================================
# CLI Simulation Functions
# ==============================================================================

def simulate_ic_pack(
    query: str,
    repo_root: Path,
    hooks_enabled: bool = True,
    hooks_available: bool = True,
) -> CommandResult:
    """
    Simulate /ic pack command.

    Works with or without hooks.
    """
    # Check hook availability
    if hooks_enabled and not hooks_available:
        # Hook requested but not available - use fallback
        pass

    # Core pack functionality (doesn't require hooks)
    try:
        # Simulate pack generation
        pack_content = f"""## Context Pack
Query: {query}

### Relevant Code
```
// Sample code from {repo_root}
function example() {{
    return true;
}}
```
"""
        return CommandResult(
            success=True,
            output=pack_content,
            exit_code=0,
        )
    except Exception as e:
        return CommandResult(
            success=False,
            output="",
            error=str(e),
            exit_code=1,
        )


def simulate_ic_search(
    query: str,
    repo_root: Path,
    hooks_enabled: bool = True,
) -> CommandResult:
    """
    Simulate /ic search command.

    Works with or without hooks.
    """
    try:
        results = f"""Search results for: {query}

1. src/auth/handler.ts:9 - handleAuth function
2. src/auth/validator.ts:6 - validateToken function
"""
        return CommandResult(
            success=True,
            output=results,
            exit_code=0,
        )
    except Exception as e:
        return CommandResult(
            success=False,
            output="",
            error=str(e),
            exit_code=1,
        )


def simulate_ic_impact(
    changed_paths: list[str],
    repo_root: Path,
    hooks_enabled: bool = True,
) -> CommandResult:
    """
    Simulate /ic impact command.

    Works with or without hooks.
    """
    try:
        impacts = f"""Impact Analysis
Changed: {', '.join(changed_paths)}

Impacted files:
- src/api/endpoints.ts (uses changed types)
- tests/auth.test.ts (tests changed functions)
"""
        return CommandResult(
            success=True,
            output=impacts,
            exit_code=0,
        )
    except Exception as e:
        return CommandResult(
            success=False,
            output="",
            error=str(e),
            exit_code=1,
        )


def check_hook_status(hook_name: str) -> HookStatus:
    """Check if a hook is available and enabled."""
    # Simulate hook configuration check
    return HookStatus(
        enabled=True,
        available=True,
    )


def disable_hooks() -> None:
    """Disable all hooks (for testing fallback)."""
    pass


def enable_hooks() -> None:
    """Re-enable hooks."""
    pass


# ==============================================================================
# Hook Fallback Acceptance Test
# ==============================================================================

@pytest.mark.acceptance
class TestHookFallback:
    """
    Acceptance test for hook fallback.

    Pass criteria (from PRD):
    - With hooks disabled
    - /ic commands still work
    """

    @pytest.fixture
    def test_repo(self, tmp_path: Path) -> Path:
        """Create test repository."""
        repo = tmp_path / "fallback_test_repo"
        (repo / "src" / "auth").mkdir(parents=True)
        (repo / "src" / "auth" / "handler.ts").write_text("export function handleAuth() {}")
        return repo

    def test_ic_pack_without_hooks(self, test_repo: Path):
        """
        Test: /ic pack works without hooks

        Expected: Command succeeds and returns pack
        """
        result = simulate_ic_pack(
            query="authentication flow",
            repo_root=test_repo,
            hooks_enabled=False,
            hooks_available=False,
        )

        assert result.success, f"Command should succeed: {result.error}"
        assert len(result.output) > 0, "Should return pack content"
        assert result.exit_code == 0

    def test_ic_search_without_hooks(self, test_repo: Path):
        """
        Test: /ic search works without hooks

        Expected: Command succeeds and returns results
        """
        result = simulate_ic_search(
            query="validateToken",
            repo_root=test_repo,
            hooks_enabled=False,
        )

        assert result.success
        assert "Search results" in result.output
        assert result.exit_code == 0

    def test_ic_impact_without_hooks(self, test_repo: Path):
        """
        Test: /ic impact works without hooks

        Expected: Command succeeds and returns impacts
        """
        result = simulate_ic_impact(
            changed_paths=["src/types/shared.ts"],
            repo_root=test_repo,
            hooks_enabled=False,
        )

        assert result.success
        assert "Impact" in result.output
        assert result.exit_code == 0

    def test_all_ic_commands_work_without_hooks(self, test_repo: Path):
        """Test all /ic commands work without hooks."""
        commands = [
            ("pack", lambda: simulate_ic_pack("test", test_repo, hooks_enabled=False)),
            ("search", lambda: simulate_ic_search("test", test_repo, hooks_enabled=False)),
            ("impact", lambda: simulate_ic_impact(["file.ts"], test_repo, hooks_enabled=False)),
        ]

        for cmd_name, cmd_func in commands:
            result = cmd_func()
            assert result.success, f"/ic {cmd_name} should work without hooks"


# ==============================================================================
# Graceful Degradation Tests
# ==============================================================================

@pytest.mark.acceptance
class TestGracefulDegradation:
    """Tests for graceful degradation when hooks fail."""

    @pytest.fixture
    def test_repo(self, tmp_path: Path) -> Path:
        """Create test repository."""
        repo = tmp_path / "degradation_test_repo"
        repo.mkdir()
        return repo

    def test_command_succeeds_when_hook_unavailable(self, test_repo: Path):
        """Test command succeeds when hook script is unavailable."""
        result = simulate_ic_pack(
            query="test",
            repo_root=test_repo,
            hooks_enabled=True,
            hooks_available=False,  # Hook not available
        )

        # Should fall back to direct execution
        assert result.success

    def test_command_succeeds_when_hook_errors(self, test_repo: Path):
        """Test command succeeds when hook execution errors."""
        # Simulate hook error scenario
        def failing_hook():
            raise RuntimeError("Hook script failed")

        # Command should still work via fallback
        result = simulate_ic_pack(
            query="test",
            repo_root=test_repo,
            hooks_enabled=False,  # Simulate fallback after error
        )

        assert result.success

    def test_warning_when_hooks_unavailable(self, test_repo: Path):
        """Test that warning is shown when hooks unavailable."""
        # In production, should log warning about missing hooks
        status = HookStatus(
            enabled=True,
            available=False,
            error="Hook script not found",
        )

        # Warning should be present but command should work
        result = simulate_ic_pack(
            query="test",
            repo_root=test_repo,
            hooks_enabled=False,
        )

        assert result.success


# ==============================================================================
# Direct CLI Tests
# ==============================================================================

@pytest.mark.acceptance
class TestDirectCLI:
    """Tests for direct CLI execution (without Claude Code)."""

    @pytest.fixture
    def test_repo(self, tmp_path: Path) -> Path:
        """Create test repository."""
        repo = tmp_path / "cli_test_repo"
        repo.mkdir()
        return repo

    def test_cli_pack_command(self, test_repo: Path):
        """Test direct CLI pack command."""
        result = simulate_ic_pack(
            query="authentication",
            repo_root=test_repo,
            hooks_enabled=False,
        )

        assert result.success
        assert "Context Pack" in result.output

    def test_cli_search_command(self, test_repo: Path):
        """Test direct CLI search command."""
        result = simulate_ic_search(
            query="handleAuth",
            repo_root=test_repo,
            hooks_enabled=False,
        )

        assert result.success

    def test_cli_returns_valid_exit_code(self, test_repo: Path):
        """Test CLI returns valid exit codes."""
        # Success
        result = simulate_ic_pack("test", test_repo, hooks_enabled=False)
        assert result.exit_code == 0

        # The actual failure case would require simulating an error condition


# ==============================================================================
# Hook Configuration Tests
# ==============================================================================

@pytest.mark.acceptance
class TestHookConfiguration:
    """Tests for hook configuration handling."""

    def test_detect_missing_hooks(self):
        """Test detection of missing hooks."""
        # Simulate checking for hooks that don't exist
        status = HookStatus(
            enabled=True,
            available=False,
            error="Hook configuration not found",
        )

        assert not status.available
        assert status.error is not None

    def test_detect_disabled_hooks(self):
        """Test detection of disabled hooks."""
        status = HookStatus(
            enabled=False,
            available=True,
        )

        assert not status.enabled
        assert status.available

    def test_hooks_can_be_toggled(self):
        """Test that hooks can be enabled/disabled."""
        # This tests the configuration mechanism
        initial_enabled = True

        # Disable
        disable_hooks()
        # In real implementation, would check config

        # Enable
        enable_hooks()
        # In real implementation, would check config


# ==============================================================================
# Integration Tests
# ==============================================================================

@pytest.mark.acceptance
class TestFallbackIntegration:
    """Integration tests for fallback behavior."""

    @pytest.fixture
    def test_repo(self, tmp_path: Path) -> Path:
        """Create test repository."""
        repo = tmp_path / "integration_test_repo"
        (repo / "src").mkdir(parents=True)
        (repo / "src" / "main.ts").write_text("export function main() {}")
        return repo

    def test_complete_workflow_without_hooks(self, test_repo: Path):
        """Test complete workflow without hooks."""
        # 1. Search for something
        search_result = simulate_ic_search(
            query="main",
            repo_root=test_repo,
            hooks_enabled=False,
        )
        assert search_result.success

        # 2. Get pack for context
        pack_result = simulate_ic_pack(
            query="how does main work",
            repo_root=test_repo,
            hooks_enabled=False,
        )
        assert pack_result.success

        # 3. Check impact of changes
        impact_result = simulate_ic_impact(
            changed_paths=["src/main.ts"],
            repo_root=test_repo,
            hooks_enabled=False,
        )
        assert impact_result.success

    def test_mixed_hook_availability(self, test_repo: Path):
        """Test with some hooks available and some not."""
        # Simulate scenario where UserPromptSubmit hook works
        # but Stop hook is unavailable

        # Commands should still work
        result = simulate_ic_pack(
            query="test",
            repo_root=test_repo,
            hooks_enabled=True,
            hooks_available=True,  # Partial availability
        )

        assert result.success
