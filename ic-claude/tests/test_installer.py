"""
Tests for ICR Claude Code Plugin Installer

Tests cover:
- Installation of hooks
- Installation of MCP server configuration
- Uninstallation
- Health checks (doctor)
"""

import json
import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from installer.install import (
    InstallConfig,
    InstallResult,
    install,
    install_user_hooks,
    install_mcp_server,
    get_hook_definitions,
    get_mcp_config,
    load_json_file,
    save_json_file,
    backup_file,
)

from installer.uninstall import (
    UninstallConfig,
    UninstallResult,
    uninstall,
    remove_icr_hooks,
    remove_mcp_server,
)

from installer.doctor import (
    HealthChecker,
    HealthReport,
    CheckResult,
)


# =============================================================================
# Install Tests
# =============================================================================

class TestInstallConfig:
    """Tests for installation configuration."""

    def test_default_paths(self):
        """Test default configuration paths."""
        config = InstallConfig()

        assert config.user_settings_path == Path.home() / ".claude" / "settings.json"
        assert config.user_claude_json == Path.home() / ".claude.json"
        assert config.icr_config_path == Path.home() / ".icr" / "config.yaml"

    def test_custom_project_path(self):
        """Test setting custom project path."""
        config = InstallConfig(project_path=Path("/custom/project"))

        assert config.project_path == Path("/custom/project")


class TestHookDefinitions:
    """Tests for hook definition generation."""

    def test_hook_definitions_structure(self):
        """Test that hook definitions have correct structure."""
        hooks = get_hook_definitions()

        assert "UserPromptSubmit" in hooks
        assert "Stop" in hooks
        assert "PreCompact" in hooks

        for hook_name, handlers in hooks.items():
            assert isinstance(handlers, list)
            assert len(handlers) > 0

            for handler in handlers:
                assert "type" in handler
                assert handler["type"] == "command"
                assert "command" in handler
                assert "timeout_ms" in handler

    def test_hook_commands_point_to_scripts(self):
        """Test that hook commands reference actual script files."""
        hooks = get_hook_definitions()

        for hook_name, handlers in hooks.items():
            for handler in handlers:
                command = handler["command"]
                # Should contain path to a .py file
                assert ".py" in command
                # Should use python3
                assert "python3" in command


class TestMCPConfig:
    """Tests for MCP server configuration."""

    def test_mcp_config_structure(self):
        """Test MCP config has correct structure."""
        config = get_mcp_config()

        assert "icr" in config
        assert "command" in config["icr"]
        assert "args" in config["icr"]
        assert config["icr"]["command"] == "icr"
        assert "mcp-serve" in config["icr"]["args"]


class TestJSONFileOperations:
    """Tests for JSON file operations."""

    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist."""
        result = load_json_file(Path("/nonexistent/path.json"))
        assert result == {}

    def test_load_valid_json(self):
        """Test loading valid JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"key": "value"}, f)
            temp_path = Path(f.name)

        try:
            result = load_json_file(temp_path)
            assert result == {"key": "value"}
        finally:
            temp_path.unlink()

    def test_load_invalid_json(self):
        """Test loading invalid JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json {{{")
            temp_path = Path(f.name)

        try:
            result = load_json_file(temp_path)
            assert result == {}
        finally:
            temp_path.unlink()

    def test_save_json_file(self):
        """Test saving JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            data = {"key": "value", "number": 42}

            result = save_json_file(path, data)

            assert result is True
            assert path.exists()

            with open(path) as f:
                loaded = json.load(f)

            assert loaded == data

    def test_save_creates_parent_dirs(self):
        """Test that save creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "deep" / "nested" / "test.json"

            result = save_json_file(path, {"test": True})

            assert result is True
            assert path.exists()


class TestBackup:
    """Tests for file backup."""

    def test_backup_existing_file(self):
        """Test backing up an existing file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"original": true}')
            original_path = Path(f.name)

        try:
            backup_path = backup_file(original_path)

            assert backup_path is not None
            assert backup_path.exists()
            assert backup_path.suffix == ".json.icr-backup"

            with open(backup_path) as f:
                content = f.read()
            assert '{"original": true}' in content
        finally:
            original_path.unlink()
            if backup_path and backup_path.exists():
                backup_path.unlink()

    def test_backup_nonexistent_file(self):
        """Test backing up a file that doesn't exist."""
        result = backup_file(Path("/nonexistent/file.json"))
        assert result is None


class TestInstallUserHooks:
    """Tests for user-level hook installation."""

    def test_install_hooks_new_file(self):
        """Test installing hooks to new settings file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = InstallConfig(
                user_settings_path=Path(tmpdir) / "settings.json",
            )
            result = InstallResult()

            install_user_hooks(config, result)

            assert result.hooks_installed is True
            assert config.user_settings_path.exists()

            with open(config.user_settings_path) as f:
                settings = json.load(f)

            assert "hooks" in settings
            assert "UserPromptSubmit" in settings["hooks"]
            assert "Stop" in settings["hooks"]
            assert "PreCompact" in settings["hooks"]

    def test_install_hooks_existing_file(self):
        """Test installing hooks to existing settings file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "settings.json"

            # Create existing settings
            existing = {
                "other_setting": "value",
                "hooks": {
                    "SomeOtherHook": [{"type": "command", "command": "other"}],
                },
            }
            with open(settings_path, "w") as f:
                json.dump(existing, f)

            config = InstallConfig(
                user_settings_path=settings_path,
                force=True,
            )
            result = InstallResult()

            install_user_hooks(config, result)

            assert result.hooks_installed is True

            with open(settings_path) as f:
                settings = json.load(f)

            # Should preserve existing settings
            assert settings["other_setting"] == "value"
            assert "SomeOtherHook" in settings["hooks"]
            # Should add ICR hooks
            assert "UserPromptSubmit" in settings["hooks"]


class TestInstallMCPServer:
    """Tests for MCP server configuration installation."""

    def test_install_mcp_new_file(self):
        """Test installing MCP config to new file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = InstallConfig(
                user_claude_json=Path(tmpdir) / ".claude.json",
            )
            result = InstallResult()

            install_mcp_server(config, result)

            assert result.mcp_installed is True
            assert config.user_claude_json.exists()

            with open(config.user_claude_json) as f:
                claude_config = json.load(f)

            assert "mcpServers" in claude_config
            assert "icr" in claude_config["mcpServers"]

    def test_install_mcp_existing_servers(self):
        """Test installing MCP config with existing servers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            claude_json = Path(tmpdir) / ".claude.json"

            existing = {
                "mcpServers": {
                    "other-server": {"command": "other"},
                },
            }
            with open(claude_json, "w") as f:
                json.dump(existing, f)

            config = InstallConfig(
                user_claude_json=claude_json,
                force=True,
            )
            result = InstallResult()

            install_mcp_server(config, result)

            assert result.mcp_installed is True

            with open(claude_json) as f:
                claude_config = json.load(f)

            # Should preserve existing servers
            assert "other-server" in claude_config["mcpServers"]
            # Should add ICR server
            assert "icr" in claude_config["mcpServers"]


class TestFullInstall:
    """Tests for complete installation."""

    def test_full_install(self):
        """Test complete installation process."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = InstallConfig(
                user_settings_path=Path(tmpdir) / ".claude" / "settings.json",
                user_claude_json=Path(tmpdir) / ".claude.json",
                icr_config_path=Path(tmpdir) / ".icr" / "config.yaml",
                plugin_dir=Path(tmpdir) / ".claude" / "plugins" / "icr",
                install_hooks=True,
                install_mcp=True,
                install_commands=False,  # Skip to avoid file copying issues
            )

            result = install(config)

            assert result.success is True
            assert result.hooks_installed is True
            assert result.mcp_installed is True


# =============================================================================
# Uninstall Tests
# =============================================================================

class TestUninstall:
    """Tests for uninstallation."""

    def test_remove_icr_hooks(self):
        """Test removing ICR hooks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "settings.json"

            # Create settings with ICR hooks
            settings = {
                "hooks": {
                    "UserPromptSubmit": [
                        {"type": "command", "command": "other-tool"},
                        {"type": "command", "command": "python3 hook_userpromptsubmit.py"},
                    ],
                    "Stop": [
                        {"type": "command", "command": "icr-stop-hook"},
                    ],
                },
            }
            with open(settings_path, "w") as f:
                json.dump(settings, f)

            config = UninstallConfig(user_settings_path=settings_path)
            result = UninstallResult()

            remove_icr_hooks(config, result)

            assert result.hooks_removed is True

            with open(settings_path) as f:
                new_settings = json.load(f)

            # Should have removed ICR hooks
            # Other hooks should remain
            assert len(new_settings["hooks"]["UserPromptSubmit"]) == 1
            assert "other-tool" in new_settings["hooks"]["UserPromptSubmit"][0]["command"]

    def test_remove_mcp_server(self):
        """Test removing MCP server."""
        with tempfile.TemporaryDirectory() as tmpdir:
            claude_json = Path(tmpdir) / ".claude.json"

            existing = {
                "mcpServers": {
                    "icr": {"command": "icr", "args": ["mcp-serve"]},
                    "other": {"command": "other"},
                },
            }
            with open(claude_json, "w") as f:
                json.dump(existing, f)

            config = UninstallConfig(user_claude_json=claude_json)
            result = UninstallResult()

            remove_mcp_server(config, result)

            assert result.mcp_removed is True

            with open(claude_json) as f:
                new_config = json.load(f)

            assert "icr" not in new_config["mcpServers"]
            assert "other" in new_config["mcpServers"]

    def test_uninstall_dry_run(self):
        """Test dry run doesn't make changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "settings.json"

            original = {
                "hooks": {
                    "Stop": [{"type": "command", "command": "icr-hook"}],
                },
            }
            with open(settings_path, "w") as f:
                json.dump(original, f)

            config = UninstallConfig(
                user_settings_path=settings_path,
                dry_run=True,
            )

            result = uninstall(config)

            # Should not have removed anything
            with open(settings_path) as f:
                current = json.load(f)

            assert current == original


# =============================================================================
# Doctor Tests
# =============================================================================

class TestHealthChecker:
    """Tests for health checker."""

    def test_check_result_structure(self):
        """Test CheckResult structure."""
        result = CheckResult(
            name="Test Check",
            passed=True,
            message="All good",
            fixable=False,
        )

        assert result.name == "Test Check"
        assert result.passed is True
        assert result.message == "All good"
        assert result.fixable is False

    def test_health_report_aggregation(self):
        """Test HealthReport aggregates results."""
        report = HealthReport()

        report.add(CheckResult("Check 1", True, "OK"))
        report.add(CheckResult("Check 2", False, "Failed", fixable=True))
        report.add(CheckResult("Check 3", True, "OK"))

        assert report.passed == 2
        assert report.failed == 1
        assert report.healthy is False
        assert len(report.checks) == 3

    def test_check_python_version(self):
        """Test Python version check."""
        checker = HealthChecker()
        result = checker.check_python_version()

        # We're running tests, so Python should be valid
        assert result.passed is True
        assert "Python" in result.message

    def test_check_config_missing(self):
        """Test config check when file is missing."""
        checker = HealthChecker(
            config_path=Path("/nonexistent/config.yaml")
        )
        result = checker.check_config()

        assert result.passed is False
        assert result.fixable is True
        assert "not found" in result.message.lower()

    def test_check_database_missing(self):
        """Test database check when file is missing."""
        checker = HealthChecker(
            db_path=Path("/nonexistent/icr.db")
        )
        result = checker.check_database()

        assert result.passed is False
        assert result.fixable is True

    def test_check_hooks_config_missing(self):
        """Test hooks check when settings missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checker = HealthChecker()
            checker.user_settings_path = Path(tmpdir) / "nonexistent.json"

            result = checker.check_hooks_config()

            assert result.passed is False
            assert result.fixable is True

    def test_check_hooks_config_present(self):
        """Test hooks check when properly configured."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "settings.json"

            settings = {
                "hooks": {
                    "UserPromptSubmit": [
                        {"type": "command", "command": "icr-hook"}
                    ],
                },
            }
            with open(settings_path, "w") as f:
                json.dump(settings, f)

            checker = HealthChecker()
            checker.user_settings_path = settings_path

            result = checker.check_hooks_config()

            assert result.passed is True

    def test_full_health_check(self):
        """Test full health check returns report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checker = HealthChecker(
                config_path=Path(tmpdir) / "config.yaml",
                db_path=Path(tmpdir) / "icr.db",
            )

            report = checker.check_all()

            assert isinstance(report, HealthReport)
            assert len(report.checks) > 0
            # Most checks should fail since nothing exists
            assert report.failed > 0


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_install_with_corrupted_existing_file(self):
        """Test install handles corrupted existing files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "settings.json"

            # Write corrupted JSON
            with open(settings_path, "w") as f:
                f.write("{corrupted json")

            config = InstallConfig(
                user_settings_path=settings_path,
                force=True,
            )
            result = InstallResult()

            # Should handle gracefully
            install_user_hooks(config, result)

            # Should have created valid file
            with open(settings_path) as f:
                settings = json.load(f)

            assert "hooks" in settings

    def test_uninstall_with_no_icr_hooks(self):
        """Test uninstall when no ICR hooks present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "settings.json"

            settings = {
                "hooks": {
                    "SomeHook": [{"type": "command", "command": "other"}],
                },
            }
            with open(settings_path, "w") as f:
                json.dump(settings, f)

            config = UninstallConfig(user_settings_path=settings_path)
            result = UninstallResult()

            remove_icr_hooks(config, result)

            # Should not error, just not remove anything
            assert result.hooks_removed is False

    def test_concurrent_access_safety(self):
        """Test that operations are safe for concurrent access."""
        # This is a basic test - real concurrent testing would need threads
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "settings.json"

            # Simulate concurrent read/write
            for i in range(10):
                data = {"iteration": i}
                save_json_file(settings_path, data)
                loaded = load_json_file(settings_path)
                assert loaded["iteration"] == i


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
