#!/usr/bin/env python3
"""
ICR Claude Code Plugin Installer

This script installs ICR hooks and configuration for Claude Code.

Installation Strategy (reliability-focused):
1. User-level hooks (critical) - ~/.claude/settings.json
2. Project-level hooks (fallback) - .claude/settings.json
3. Plugin-level hooks (optional) - ~/.claude/plugins/icr/

Explicit /ic commands MUST work even if hooks fail.
"""

import argparse
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("icr.installer")


@dataclass
class InstallConfig:
    """Configuration for installation."""

    # Paths
    user_settings_path: Path = field(
        default_factory=lambda: Path.home() / ".claude" / "settings.json"
    )
    user_claude_json: Path = field(
        default_factory=lambda: Path.home() / ".claude.json"
    )
    icr_config_path: Path = field(
        default_factory=lambda: Path.home() / ".icr" / "config.yaml"
    )
    icr_db_path: Path = field(
        default_factory=lambda: Path.home() / ".icr" / "icr.db"
    )
    plugin_dir: Path = field(
        default_factory=lambda: Path.home() / ".claude" / "plugins" / "icr"
    )

    # Options
    install_hooks: bool = True
    install_mcp: bool = True
    install_commands: bool = True
    force: bool = False
    project_path: Path | None = None


@dataclass
class InstallResult:
    """Result of installation."""

    success: bool = True
    hooks_installed: bool = False
    mcp_installed: bool = False
    commands_installed: bool = False
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def get_hook_definitions() -> dict[str, list[dict[str, Any]]]:
    """Get the hook definitions for ICR."""
    # Get the path to the scripts directory
    scripts_dir = Path(__file__).parent.parent / "scripts"

    return {
        "UserPromptSubmit": [
            {
                "type": "command",
                "command": f"python3 {scripts_dir / 'hook_userpromptsubmit.py'}",
                "timeout_ms": 5000,
            }
        ],
        "Stop": [
            {
                "type": "command",
                "command": f"python3 {scripts_dir / 'hook_stop.py'}",
                "timeout_ms": 10000,
            }
        ],
        "PreCompact": [
            {
                "type": "command",
                "command": f"python3 {scripts_dir / 'hook_precompact.py'}",
                "timeout_ms": 15000,
            }
        ],
    }


def get_mcp_config() -> dict[str, Any]:
    """Get the MCP server configuration."""
    return {
        "icr": {
            "command": "icr",
            "args": ["mcp-serve"],
        }
    }


def load_json_file(path: Path) -> dict[str, Any]:
    """Load a JSON file, returning empty dict if not exists."""
    if not path.exists():
        return {}

    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.warning(f"Invalid JSON in {path}: {e}")
        return {}
    except Exception as e:
        logger.warning(f"Failed to read {path}: {e}")
        return {}


def save_json_file(path: Path, data: dict[str, Any]) -> bool:
    """Save data to a JSON file."""
    try:
        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write with pretty formatting
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        return True
    except Exception as e:
        logger.error(f"Failed to write {path}: {e}")
        return False


def backup_file(path: Path) -> Path | None:
    """Create a backup of a file."""
    if not path.exists():
        return None

    backup_path = path.with_suffix(path.suffix + ".icr-backup")
    try:
        shutil.copy2(path, backup_path)
        logger.info(f"Created backup: {backup_path}")
        return backup_path
    except Exception as e:
        logger.warning(f"Failed to create backup: {e}")
        return None


def install_user_hooks(config: InstallConfig, result: InstallResult) -> None:
    """Install hooks at user level (~/.claude/settings.json)."""
    logger.info("Installing user-level hooks...")

    settings_path = config.user_settings_path

    # Backup existing file
    if settings_path.exists() and not config.force:
        backup_file(settings_path)

    # Load existing settings
    settings = load_json_file(settings_path)

    # Get hook definitions
    hook_defs = get_hook_definitions()

    # Merge hooks
    if "hooks" not in settings:
        settings["hooks"] = {}

    for hook_name, hook_handlers in hook_defs.items():
        if hook_name not in settings["hooks"]:
            settings["hooks"][hook_name] = []

        # Check if ICR hooks already exist
        existing_icr_hooks = [
            h for h in settings["hooks"][hook_name]
            if "icr" in h.get("command", "").lower() or
               "hook_" in h.get("command", "").lower()
        ]

        if existing_icr_hooks and not config.force:
            logger.info(f"  {hook_name}: ICR hooks already present, skipping")
            continue

        # Remove existing ICR hooks if forcing
        if config.force:
            settings["hooks"][hook_name] = [
                h for h in settings["hooks"][hook_name]
                if "icr" not in h.get("command", "").lower() and
                   "hook_" not in h.get("command", "").lower()
            ]

        # Add new hooks
        settings["hooks"][hook_name].extend(hook_handlers)
        logger.info(f"  {hook_name}: Added {len(hook_handlers)} handler(s)")

    # Save settings
    if save_json_file(settings_path, settings):
        result.hooks_installed = True
        logger.info(f"User hooks installed to {settings_path}")
    else:
        result.errors.append(f"Failed to save hooks to {settings_path}")


def install_project_hooks(config: InstallConfig, result: InstallResult) -> None:
    """Install hooks at project level (.claude/settings.json)."""
    if not config.project_path:
        return

    logger.info("Installing project-level hooks...")

    settings_path = config.project_path / ".claude" / "settings.json"

    # Backup existing
    if settings_path.exists() and not config.force:
        backup_file(settings_path)

    # Load existing
    settings = load_json_file(settings_path)

    # Get hook definitions
    hook_defs = get_hook_definitions()

    # Merge hooks
    if "hooks" not in settings:
        settings["hooks"] = {}

    for hook_name, hook_handlers in hook_defs.items():
        if hook_name not in settings["hooks"]:
            settings["hooks"][hook_name] = []

        # Add hooks (simpler for project level, just add)
        settings["hooks"][hook_name].extend(hook_handlers)

    # Save
    if save_json_file(settings_path, settings):
        logger.info(f"Project hooks installed to {settings_path}")
    else:
        result.warnings.append(f"Failed to install project hooks to {settings_path}")


def install_mcp_server(config: InstallConfig, result: InstallResult) -> None:
    """Install MCP server configuration."""
    logger.info("Installing MCP server configuration...")

    claude_json = config.user_claude_json

    # Backup existing
    if claude_json.exists() and not config.force:
        backup_file(claude_json)

    # Load existing
    claude_config = load_json_file(claude_json)

    # Get MCP config
    mcp_config = get_mcp_config()

    # Merge MCP servers
    if "mcpServers" not in claude_config:
        claude_config["mcpServers"] = {}

    # Check if ICR MCP already exists
    if "icr" in claude_config["mcpServers"] and not config.force:
        logger.info("  ICR MCP server already configured, skipping")
        result.mcp_installed = True
        return

    # Add ICR MCP server
    claude_config["mcpServers"].update(mcp_config)

    # Save
    if save_json_file(claude_json, claude_config):
        result.mcp_installed = True
        logger.info(f"MCP server configured in {claude_json}")
    else:
        result.errors.append(f"Failed to configure MCP server in {claude_json}")


def install_plugin_files(config: InstallConfig, result: InstallResult) -> None:
    """Install plugin files to plugin directory."""
    logger.info("Installing plugin files...")

    plugin_dir = config.plugin_dir
    source_dir = Path(__file__).parent.parent

    try:
        # Create plugin directory
        plugin_dir.mkdir(parents=True, exist_ok=True)

        # Copy plugin.json
        shutil.copy2(source_dir / "plugin.json", plugin_dir / "plugin.json")

        # Copy commands
        commands_dir = plugin_dir / "commands"
        commands_dir.mkdir(exist_ok=True)
        shutil.copy2(
            source_dir / "commands" / "ic.md",
            commands_dir / "ic.md"
        )

        # Copy hooks.json template
        hooks_dir = plugin_dir / "hooks"
        hooks_dir.mkdir(exist_ok=True)
        shutil.copy2(
            source_dir / "hooks" / "hooks.json",
            hooks_dir / "hooks.json"
        )

        result.commands_installed = True
        logger.info(f"Plugin files installed to {plugin_dir}")

    except Exception as e:
        result.warnings.append(f"Failed to install plugin files: {e}")


def create_default_config(config: InstallConfig, result: InstallResult) -> None:
    """Create default ICR configuration if not exists."""
    if config.icr_config_path.exists():
        logger.info("ICR config already exists, skipping")
        return

    logger.info("Creating default ICR configuration...")

    default_config = """# ICR Configuration
# See https://github.com/icr/icr for documentation

# Context injection settings
auto_inject: true
max_context_tokens: 4000

# Ledger settings
ledger_extraction: true

# Compaction settings
persist_invariants: true

# Embedding settings
embedding_backend: "local"  # Options: local, openai, anthropic

# Logging
log_level: "INFO"
"""

    try:
        config.icr_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config.icr_config_path, "w") as f:
            f.write(default_config)
        logger.info(f"Created default config at {config.icr_config_path}")
    except Exception as e:
        result.warnings.append(f"Failed to create default config: {e}")


def install(config: InstallConfig) -> InstallResult:
    """
    Perform full installation.

    Args:
        config: Installation configuration

    Returns:
        InstallResult with success status and details
    """
    result = InstallResult()

    logger.info("Starting ICR Claude Code Plugin installation...")
    logger.info(f"  User settings: {config.user_settings_path}")
    logger.info(f"  MCP config: {config.user_claude_json}")
    logger.info(f"  ICR config: {config.icr_config_path}")

    # Install hooks (user-level is critical)
    if config.install_hooks:
        try:
            install_user_hooks(config, result)
        except Exception as e:
            result.errors.append(f"User hook installation failed: {e}")
            logger.error(f"User hook installation failed: {e}")

        # Project hooks are optional
        if config.project_path:
            try:
                install_project_hooks(config, result)
            except Exception as e:
                result.warnings.append(f"Project hook installation failed: {e}")

    # Install MCP server
    if config.install_mcp:
        try:
            install_mcp_server(config, result)
        except Exception as e:
            result.warnings.append(f"MCP installation failed: {e}")

    # Install plugin files
    if config.install_commands:
        try:
            install_plugin_files(config, result)
        except Exception as e:
            result.warnings.append(f"Plugin file installation failed: {e}")

    # Create default config
    try:
        create_default_config(config, result)
    except Exception as e:
        result.warnings.append(f"Config creation failed: {e}")

    # Determine overall success
    result.success = len(result.errors) == 0

    # Summary
    if result.success:
        logger.info("Installation completed successfully!")
        logger.info(f"  Hooks installed: {result.hooks_installed}")
        logger.info(f"  MCP configured: {result.mcp_installed}")
        logger.info(f"  Commands installed: {result.commands_installed}")

        if result.warnings:
            logger.warning("Warnings during installation:")
            for warning in result.warnings:
                logger.warning(f"  - {warning}")
    else:
        logger.error("Installation failed!")
        for error in result.errors:
            logger.error(f"  - {error}")

    return result


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Install ICR Claude Code Plugin",
    )
    parser.add_argument(
        "--no-hooks",
        action="store_true",
        help="Skip hook installation",
    )
    parser.add_argument(
        "--no-mcp",
        action="store_true",
        help="Skip MCP server configuration",
    )
    parser.add_argument(
        "--no-commands",
        action="store_true",
        help="Skip command installation",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reinstall, overwriting existing configuration",
    )
    parser.add_argument(
        "--project",
        type=Path,
        help="Also install project-level hooks at this path",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = InstallConfig(
        install_hooks=not args.no_hooks,
        install_mcp=not args.no_mcp,
        install_commands=not args.no_commands,
        force=args.force,
        project_path=args.project,
    )

    result = install(config)

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
