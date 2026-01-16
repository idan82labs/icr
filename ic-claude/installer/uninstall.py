#!/usr/bin/env python3
"""
ICR Claude Code Plugin Uninstaller

This script removes ICR hooks and configuration from Claude Code.

It safely removes:
1. User-level hooks from ~/.claude/settings.json
2. MCP server config from ~/.claude.json
3. Plugin files from ~/.claude/plugins/icr/

Optionally:
4. ICR configuration from ~/.icr/
5. ICR database from ~/.icr/
"""

import argparse
import json
import logging
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("icr.uninstaller")


@dataclass
class UninstallConfig:
    """Configuration for uninstallation."""

    # Paths
    user_settings_path: Path = field(
        default_factory=lambda: Path.home() / ".claude" / "settings.json"
    )
    user_claude_json: Path = field(
        default_factory=lambda: Path.home() / ".claude.json"
    )
    icr_config_dir: Path = field(
        default_factory=lambda: Path.home() / ".icr"
    )
    plugin_dir: Path = field(
        default_factory=lambda: Path.home() / ".claude" / "plugins" / "icr"
    )

    # Options
    remove_hooks: bool = True
    remove_mcp: bool = True
    remove_plugin: bool = True
    remove_config: bool = False  # Off by default - destructive
    remove_data: bool = False    # Off by default - destructive
    dry_run: bool = False


@dataclass
class UninstallResult:
    """Result of uninstallation."""

    success: bool = True
    hooks_removed: bool = False
    mcp_removed: bool = False
    plugin_removed: bool = False
    config_removed: bool = False
    data_removed: bool = False
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def load_json_file(path: Path) -> dict[str, Any]:
    """Load a JSON file, returning empty dict if not exists."""
    if not path.exists():
        return {}

    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to read {path}: {e}")
        return {}


def save_json_file(path: Path, data: dict[str, Any]) -> bool:
    """Save data to a JSON file."""
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Failed to write {path}: {e}")
        return False


def remove_icr_hooks(config: UninstallConfig, result: UninstallResult) -> None:
    """Remove ICR hooks from user settings."""
    logger.info("Removing ICR hooks from user settings...")

    settings_path = config.user_settings_path

    if not settings_path.exists():
        logger.info("  No settings file found, skipping")
        return

    settings = load_json_file(settings_path)

    if "hooks" not in settings:
        logger.info("  No hooks configured, skipping")
        return

    hooks_modified = False
    for hook_name in list(settings["hooks"].keys()):
        original_count = len(settings["hooks"][hook_name])

        # Filter out ICR hooks
        settings["hooks"][hook_name] = [
            h for h in settings["hooks"][hook_name]
            if not _is_icr_hook(h)
        ]

        new_count = len(settings["hooks"][hook_name])
        if new_count < original_count:
            removed = original_count - new_count
            logger.info(f"  {hook_name}: Removed {removed} ICR handler(s)")
            hooks_modified = True

        # Remove empty hook arrays
        if not settings["hooks"][hook_name]:
            del settings["hooks"][hook_name]

    # Remove empty hooks object
    if not settings["hooks"]:
        del settings["hooks"]

    if hooks_modified:
        if config.dry_run:
            logger.info("  [DRY RUN] Would save modified settings")
        else:
            if save_json_file(settings_path, settings):
                result.hooks_removed = True
                logger.info(f"  Hooks removed from {settings_path}")
            else:
                result.errors.append(f"Failed to save {settings_path}")
    else:
        logger.info("  No ICR hooks found")


def _is_icr_hook(hook: dict[str, Any]) -> bool:
    """Check if a hook is an ICR hook."""
    command = hook.get("command", "").lower()
    return (
        "icr" in command or
        "hook_userpromptsubmit" in command or
        "hook_stop" in command or
        "hook_precompact" in command
    )


def remove_mcp_server(config: UninstallConfig, result: UninstallResult) -> None:
    """Remove ICR MCP server configuration."""
    logger.info("Removing MCP server configuration...")

    claude_json = config.user_claude_json

    if not claude_json.exists():
        logger.info("  No claude.json found, skipping")
        return

    claude_config = load_json_file(claude_json)

    if "mcpServers" not in claude_config:
        logger.info("  No MCP servers configured, skipping")
        return

    if "icr" not in claude_config["mcpServers"]:
        logger.info("  ICR MCP server not found, skipping")
        return

    if config.dry_run:
        logger.info("  [DRY RUN] Would remove ICR MCP server")
    else:
        del claude_config["mcpServers"]["icr"]

        # Remove empty mcpServers
        if not claude_config["mcpServers"]:
            del claude_config["mcpServers"]

        if save_json_file(claude_json, claude_config):
            result.mcp_removed = True
            logger.info(f"  MCP server removed from {claude_json}")
        else:
            result.errors.append(f"Failed to save {claude_json}")


def remove_plugin_files(config: UninstallConfig, result: UninstallResult) -> None:
    """Remove ICR plugin files."""
    logger.info("Removing plugin files...")

    plugin_dir = config.plugin_dir

    if not plugin_dir.exists():
        logger.info("  Plugin directory not found, skipping")
        return

    if config.dry_run:
        logger.info(f"  [DRY RUN] Would remove {plugin_dir}")
    else:
        try:
            shutil.rmtree(plugin_dir)
            result.plugin_removed = True
            logger.info(f"  Removed {plugin_dir}")
        except Exception as e:
            result.errors.append(f"Failed to remove plugin directory: {e}")


def remove_icr_config(config: UninstallConfig, result: UninstallResult) -> None:
    """Remove ICR configuration files."""
    logger.info("Removing ICR configuration...")

    config_dir = config.icr_config_dir
    config_file = config_dir / "config.yaml"

    if not config_file.exists():
        logger.info("  Config file not found, skipping")
        return

    if config.dry_run:
        logger.info(f"  [DRY RUN] Would remove {config_file}")
    else:
        try:
            config_file.unlink()
            result.config_removed = True
            logger.info(f"  Removed {config_file}")
        except Exception as e:
            result.errors.append(f"Failed to remove config: {e}")


def remove_icr_data(config: UninstallConfig, result: UninstallResult) -> None:
    """Remove ICR database and data files."""
    logger.info("Removing ICR data...")

    config_dir = config.icr_config_dir

    if not config_dir.exists():
        logger.info("  ICR directory not found, skipping")
        return

    if config.dry_run:
        logger.info(f"  [DRY RUN] Would remove entire {config_dir}")
    else:
        try:
            shutil.rmtree(config_dir)
            result.data_removed = True
            logger.info(f"  Removed {config_dir}")
        except Exception as e:
            result.errors.append(f"Failed to remove data directory: {e}")


def uninstall(config: UninstallConfig) -> UninstallResult:
    """
    Perform uninstallation.

    Args:
        config: Uninstallation configuration

    Returns:
        UninstallResult with status and details
    """
    result = UninstallResult()

    if config.dry_run:
        logger.info("DRY RUN - No changes will be made")

    logger.info("Starting ICR Claude Code Plugin uninstallation...")

    # Remove hooks
    if config.remove_hooks:
        try:
            remove_icr_hooks(config, result)
        except Exception as e:
            result.errors.append(f"Hook removal failed: {e}")

    # Remove MCP server
    if config.remove_mcp:
        try:
            remove_mcp_server(config, result)
        except Exception as e:
            result.errors.append(f"MCP removal failed: {e}")

    # Remove plugin files
    if config.remove_plugin:
        try:
            remove_plugin_files(config, result)
        except Exception as e:
            result.errors.append(f"Plugin removal failed: {e}")

    # Remove config (if requested)
    if config.remove_config:
        try:
            remove_icr_config(config, result)
        except Exception as e:
            result.errors.append(f"Config removal failed: {e}")

    # Remove data (if requested)
    if config.remove_data:
        try:
            remove_icr_data(config, result)
        except Exception as e:
            result.errors.append(f"Data removal failed: {e}")

    # Determine success
    result.success = len(result.errors) == 0

    # Summary
    if result.success:
        if config.dry_run:
            logger.info("Dry run completed - no changes made")
        else:
            logger.info("Uninstallation completed successfully!")

        if result.warnings:
            for warning in result.warnings:
                logger.warning(f"  - {warning}")
    else:
        logger.error("Uninstallation encountered errors:")
        for error in result.errors:
            logger.error(f"  - {error}")

    return result


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Uninstall ICR Claude Code Plugin",
    )
    parser.add_argument(
        "--keep-hooks",
        action="store_true",
        help="Keep hooks installed",
    )
    parser.add_argument(
        "--keep-mcp",
        action="store_true",
        help="Keep MCP server configuration",
    )
    parser.add_argument(
        "--keep-plugin",
        action="store_true",
        help="Keep plugin files",
    )
    parser.add_argument(
        "--remove-config",
        action="store_true",
        help="Also remove ICR configuration (~/.icr/config.yaml)",
    )
    parser.add_argument(
        "--remove-data",
        action="store_true",
        help="Also remove all ICR data (~/.icr/)",
    )
    parser.add_argument(
        "--purge",
        action="store_true",
        help="Remove everything including config and data",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without making changes",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Handle --purge
    if args.purge:
        args.remove_config = True
        args.remove_data = True

    config = UninstallConfig(
        remove_hooks=not args.keep_hooks,
        remove_mcp=not args.keep_mcp,
        remove_plugin=not args.keep_plugin,
        remove_config=args.remove_config,
        remove_data=args.remove_data,
        dry_run=args.dry_run,
    )

    # Confirmation for destructive operations
    if (args.remove_data or args.purge) and not args.dry_run:
        print("WARNING: This will permanently delete ICR data!")
        response = input("Are you sure? Type 'yes' to confirm: ")
        if response.lower() != "yes":
            print("Cancelled.")
            return 0

    result = uninstall(config)

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
