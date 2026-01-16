#!/usr/bin/env python3
"""
ICR CLI Commands for Claude Code

This module provides CLI fallback commands that work even when hooks fail.
These commands can be invoked via /ic <command> in Claude Code.

Commands:
- pack: Generate context pack manually
- search: Search environment
- impact: Show impact analysis
- pin: Pin an invariant
- unpin: Remove pinned invariant
- status: Show ICR status
- ledger: View ledger entries
- compact: Manual compaction
- config: View/modify configuration
- sync: Environment synchronization
- clear: Clear ICR state
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure we can import from parent
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class CLIContext:
    """Context for CLI command execution."""

    config_path: str
    db_path: str
    cwd: str
    verbose: bool = False

    @classmethod
    def from_env(cls) -> "CLIContext":
        """Create context from environment."""
        return cls(
            config_path=os.environ.get(
                "ICR_CONFIG_PATH",
                os.path.expanduser("~/.icr/config.yaml")
            ),
            db_path=os.environ.get(
                "ICR_DB_PATH",
                os.path.expanduser("~/.icr/icr.db")
            ),
            cwd=os.getcwd(),
            verbose=os.environ.get("ICR_DEBUG", "").lower() in ("1", "true"),
        )


class ICRClient:
    """Lightweight ICR client for CLI commands."""

    def __init__(self, ctx: CLIContext):
        """Initialize client with context."""
        self.ctx = ctx
        self._config: dict[str, Any] | None = None

    def _load_config(self) -> dict[str, Any]:
        """Load configuration."""
        if self._config is not None:
            return self._config

        if not Path(self.ctx.config_path).exists():
            self._config = {}
            return self._config

        try:
            import yaml
            with open(self.ctx.config_path) as f:
                self._config = yaml.safe_load(f) or {}
        except ImportError:
            self._config = {}
        except Exception:
            self._config = {}

        return self._config

    def is_initialized(self) -> bool:
        """Check if ICR is properly initialized."""
        return (
            Path(self.ctx.config_path).exists() and
            Path(self.ctx.db_path).exists()
        )


# =============================================================================
# Command: pack
# =============================================================================

def cmd_pack(args: argparse.Namespace, ctx: CLIContext) -> int:
    """Generate and display the current context pack."""
    client = ICRClient(ctx)

    if not client.is_initialized():
        print("ICR not initialized. Run 'icr init' first.")
        return 1

    # Import the hook module for pack generation
    try:
        from scripts.hook_userpromptsubmit import (
            ICRClient as HookClient,
            HookInput,
            build_context_pack,
        )

        hook_client = HookClient()
        hook_input = HookInput(
            session_id="cli-" + datetime.utcnow().strftime("%Y%m%d%H%M%S"),
            prompt=args.query if hasattr(args, "query") and args.query else "",
            cwd=ctx.cwd,
        )

        pack = build_context_pack(hook_client, hook_input)

        if args.verbose:
            print(f"Token budget: {pack.token_budget}")
            print(f"Tokens used: {pack.tokens_used}")
            print(f"Invariants: {len(pack.invariants)}")
            print(f"Priors: {len(pack.priors)}")
            print(f"Active files: {len(pack.active_files)}")
            print()

        print(pack.to_markdown())
        return 0

    except ImportError as e:
        print(f"Failed to import pack generator: {e}")
        return 1
    except Exception as e:
        print(f"Pack generation failed: {e}")
        return 1


# =============================================================================
# Command: search
# =============================================================================

def cmd_search(args: argparse.Namespace, ctx: CLIContext) -> int:
    """Search the ICR environment."""
    client = ICRClient(ctx)

    if not client.is_initialized():
        print("ICR not initialized. Run 'icr init' first.")
        return 1

    query = args.query
    limit = args.limit or 10
    result_type = args.type

    try:
        from icr.core.retrieval import PriorRetriever

        retriever = PriorRetriever(db_path=ctx.db_path)
        results = retriever.search(
            query=query,
            limit=limit,
            cwd=ctx.cwd,
            result_type=result_type,
        )

        if not results:
            print(f"No results found for: {query}")
            return 0

        print(f"Search results for: {query}")
        print("=" * 50)

        for i, result in enumerate(results, 1):
            source = result.get("source", "unknown")
            relevance = result.get("relevance", 0.0)
            content = result.get("content", "")
            preview = content[:200] + "..." if len(content) > 200 else content

            print(f"\n{i}. {source} (relevance: {relevance:.2f})")
            print(f"   {preview}")

        return 0

    except ImportError:
        print("Search requires icr-core. Install with: pip install icr-core")
        return 1
    except Exception as e:
        print(f"Search failed: {e}")
        return 1


# =============================================================================
# Command: impact
# =============================================================================

def cmd_impact(args: argparse.Namespace, ctx: CLIContext) -> int:
    """Show impact analysis for files."""
    client = ICRClient(ctx)

    if not client.is_initialized():
        print("ICR not initialized. Run 'icr init' first.")
        return 1

    paths = args.paths
    depth = args.depth or 2

    try:
        from icr.core.impact import ImpactAnalyzer

        analyzer = ImpactAnalyzer(db_path=ctx.db_path)

        for path in paths:
            print(f"\nImpact analysis for: {path}")
            print("=" * 50)

            analysis = analyzer.analyze(path, depth=depth)

            if analysis.get("direct_dependents"):
                print("\nDirect dependents:")
                for dep in analysis["direct_dependents"]:
                    print(f"  - {dep}")

            if analysis.get("transitive_dependents"):
                print(f"\nTransitive dependents (depth {depth}):")
                for dep in analysis["transitive_dependents"]:
                    print(f"  - {dep}")

            if analysis.get("affected_tests"):
                print("\nAffected tests:")
                for test in analysis["affected_tests"]:
                    print(f"  - {test}")

            risk = analysis.get("risk_level", "unknown")
            print(f"\nRisk level: {risk}")

        return 0

    except ImportError:
        print("Impact analysis requires icr-core. Install with: pip install icr-core")
        return 1
    except Exception as e:
        print(f"Impact analysis failed: {e}")
        return 1


# =============================================================================
# Command: pin
# =============================================================================

def cmd_pin(args: argparse.Namespace, ctx: CLIContext) -> int:
    """Pin an invariant."""
    client = ICRClient(ctx)

    if not client.is_initialized():
        print("ICR not initialized. Run 'icr init' first.")
        return 1

    content = args.invariant
    priority = args.priority or 5
    expires = args.expires

    try:
        from icr.core.invariants import InvariantStore

        store = InvariantStore(db_path=ctx.db_path)
        inv_id = store.add(
            content=content,
            priority=priority,
            expires=expires,
        )

        print(f"Invariant pinned successfully!")
        print(f"  ID: {inv_id}")
        print(f"  Priority: {priority}")
        print(f"  Content: {content}")
        if expires:
            print(f"  Expires: {expires}")

        # Show count
        all_invariants = store.get_all_active()
        print(f"\nTotal pinned invariants: {len(all_invariants)}")

        return 0

    except ImportError:
        print("Pin requires icr-core. Install with: pip install icr-core")
        return 1
    except Exception as e:
        print(f"Pin failed: {e}")
        return 1


# =============================================================================
# Command: unpin
# =============================================================================

def cmd_unpin(args: argparse.Namespace, ctx: CLIContext) -> int:
    """Remove a pinned invariant."""
    client = ICRClient(ctx)

    if not client.is_initialized():
        print("ICR not initialized. Run 'icr init' first.")
        return 1

    inv_id = args.id

    try:
        from icr.core.invariants import InvariantStore

        store = InvariantStore(db_path=ctx.db_path)
        success = store.remove(inv_id)

        if success:
            print(f"Invariant {inv_id} removed successfully.")
            all_invariants = store.get_all_active()
            print(f"Remaining pinned invariants: {len(all_invariants)}")
            return 0
        else:
            print(f"Invariant {inv_id} not found.")
            return 1

    except ImportError:
        print("Unpin requires icr-core. Install with: pip install icr-core")
        return 1
    except Exception as e:
        print(f"Unpin failed: {e}")
        return 1


# =============================================================================
# Command: status
# =============================================================================

def cmd_status(args: argparse.Namespace, ctx: CLIContext) -> int:
    """Show ICR status."""
    print("ICR Status")
    print("=" * 50)

    # Version
    try:
        from icr import __version__
        print(f"Version: {__version__}")
    except ImportError:
        print("Version: unknown")

    # Configuration
    config_exists = Path(ctx.config_path).exists()
    print(f"\nConfiguration:")
    print(f"  Path: {ctx.config_path}")
    print(f"  Exists: {'Yes' if config_exists else 'No'}")

    # Database
    db_exists = Path(ctx.db_path).exists()
    print(f"\nDatabase:")
    print(f"  Path: {ctx.db_path}")
    print(f"  Exists: {'Yes' if db_exists else 'No'}")

    if db_exists:
        try:
            db_size = Path(ctx.db_path).stat().st_size
            print(f"  Size: {db_size / 1024 / 1024:.2f} MB")
        except Exception:
            pass

    # Try to get detailed stats if core is available
    if args.verbose:
        try:
            from icr.core.stats import get_stats
            stats = get_stats(ctx.db_path)

            print(f"\nDatabase Statistics:")
            print(f"  Files indexed: {stats.get('files', 0)}")
            print(f"  Chunks: {stats.get('chunks', 0)}")
            print(f"  Priors: {stats.get('priors', 0)}")
            print(f"  Invariants: {stats.get('invariants', 0)}")
            print(f"  Ledger entries: {stats.get('ledger_entries', 0)}")

        except ImportError:
            print("\n(Detailed stats require icr-core)")
        except Exception as e:
            print(f"\n(Failed to get stats: {e})")

    # Hook status
    print(f"\nHook Status:")
    hooks_configured = _check_hooks_configured()
    print(f"  Configured: {'Yes' if hooks_configured else 'No'}")

    if args.verbose and hooks_configured:
        _show_hook_details()

    # MCP status
    print(f"\nMCP Server:")
    mcp_configured = _check_mcp_configured()
    print(f"  Configured: {'Yes' if mcp_configured else 'No'}")

    return 0


def _check_hooks_configured() -> bool:
    """Check if ICR hooks are configured."""
    settings_path = Path.home() / ".claude" / "settings.json"
    if not settings_path.exists():
        return False

    try:
        with open(settings_path) as f:
            settings = json.load(f)
        hooks = settings.get("hooks", {})
        return "UserPromptSubmit" in hooks or "Stop" in hooks
    except Exception:
        return False


def _check_mcp_configured() -> bool:
    """Check if MCP server is configured."""
    claude_json = Path.home() / ".claude.json"
    if not claude_json.exists():
        return False

    try:
        with open(claude_json) as f:
            config = json.load(f)
        servers = config.get("mcpServers", {})
        return "icr" in servers
    except Exception:
        return False


def _show_hook_details() -> None:
    """Show detailed hook configuration."""
    settings_path = Path.home() / ".claude" / "settings.json"
    try:
        with open(settings_path) as f:
            settings = json.load(f)
        hooks = settings.get("hooks", {})

        for hook_name in ["UserPromptSubmit", "Stop", "PreCompact"]:
            hook_config = hooks.get(hook_name, [])
            if hook_config:
                print(f"    {hook_name}: {len(hook_config)} handler(s)")
    except Exception:
        pass


# =============================================================================
# Command: ledger
# =============================================================================

def cmd_ledger(args: argparse.Namespace, ctx: CLIContext) -> int:
    """View ledger entries."""
    client = ICRClient(ctx)

    if not client.is_initialized():
        print("ICR not initialized. Run 'icr init' first.")
        return 1

    limit = args.last or 10
    entry_type = args.type

    try:
        from icr.core.ledger import LedgerStore

        store = LedgerStore(db_path=ctx.db_path)

        if entry_type:
            entries = store.get_by_type(
                session_id=None,  # All sessions
                entry_type=entry_type,
                limit=limit,
            )
        else:
            entries = store.get_recent(session_id=None, limit=limit)

        if not entries:
            print("No ledger entries found.")
            return 0

        print(f"Ledger Entries (last {limit})")
        print("=" * 50)

        for entry in entries:
            timestamp = entry.get("timestamp", "")
            etype = entry.get("type", "unknown")
            content = entry.get("content", "")
            session = entry.get("session_id", "")[:8]

            print(f"\n[{timestamp}] ({etype}) [{session}...]")
            print(f"  {content}")

        return 0

    except ImportError:
        print("Ledger requires icr-core. Install with: pip install icr-core")
        return 1
    except Exception as e:
        print(f"Failed to read ledger: {e}")
        return 1


# =============================================================================
# Command: compact
# =============================================================================

def cmd_compact(args: argparse.Namespace, ctx: CLIContext) -> int:
    """Manually trigger compaction."""
    client = ICRClient(ctx)

    if not client.is_initialized():
        print("ICR not initialized. Run 'icr init' first.")
        return 1

    dry_run = args.dry_run
    preserve = args.preserve_invariants

    try:
        from icr.core.compaction import Compactor

        compactor = Compactor(db_path=ctx.db_path)

        if dry_run:
            print("Dry run - showing what would be compacted:")
            print("=" * 50)

            preview = compactor.preview()
            print(f"\nCurrent state:")
            print(f"  Total priors: {preview.get('total_priors', 0)}")
            print(f"  Total tokens: {preview.get('total_tokens', 0)}")

            print(f"\nAfter compaction:")
            print(f"  Retained priors: {preview.get('retained_priors', 0)}")
            print(f"  Retained tokens: {preview.get('retained_tokens', 0)}")

            if preview.get("discarded"):
                print(f"\nWould discard:")
                for item in preview["discarded"][:10]:
                    print(f"  - {item.get('source', 'unknown')}: {item.get('reason', '')}")

            return 0

        result = compactor.compact(preserve_invariants=preserve)

        print("Compaction complete!")
        print(f"  Tokens before: {result.get('tokens_before', 0)}")
        print(f"  Tokens after: {result.get('tokens_after', 0)}")
        print(f"  Priors removed: {result.get('removed', 0)}")

        return 0

    except ImportError:
        print("Compact requires icr-core. Install with: pip install icr-core")
        return 1
    except Exception as e:
        print(f"Compaction failed: {e}")
        return 1


# =============================================================================
# Command: config
# =============================================================================

def cmd_config(args: argparse.Namespace, ctx: CLIContext) -> int:
    """View or modify configuration."""
    client = ICRClient(ctx)
    config = client._load_config()

    key = args.key if hasattr(args, "key") else None
    value = args.value if hasattr(args, "value") else None

    if not key:
        # Show all config
        print("ICR Configuration")
        print("=" * 50)
        print(f"Config file: {ctx.config_path}")
        print()

        if not config:
            print("No configuration found.")
            return 0

        for k, v in sorted(config.items()):
            print(f"  {k}: {v}")
        return 0

    if not value:
        # Show single key
        if key in config:
            print(f"{key}: {config[key]}")
        else:
            print(f"Key '{key}' not found.")
        return 0

    # Set value
    try:
        import yaml

        # Parse value
        try:
            parsed_value = yaml.safe_load(value)
        except Exception:
            parsed_value = value

        config[key] = parsed_value

        # Ensure directory exists
        Path(ctx.config_path).parent.mkdir(parents=True, exist_ok=True)

        with open(ctx.config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        print(f"Set {key} = {parsed_value}")
        return 0

    except ImportError:
        print("Config modification requires PyYAML. Install with: pip install pyyaml")
        return 1
    except Exception as e:
        print(f"Failed to set config: {e}")
        return 1


# =============================================================================
# Command: sync
# =============================================================================

def cmd_sync(args: argparse.Namespace, ctx: CLIContext) -> int:
    """Synchronize environment."""
    client = ICRClient(ctx)

    if not client.is_initialized():
        print("ICR not initialized. Run 'icr init' first.")
        return 1

    full = args.full
    path = args.path or ctx.cwd

    try:
        from icr.core.indexing import Indexer

        indexer = Indexer(db_path=ctx.db_path)

        print(f"Synchronizing: {path}")
        if full:
            print("Mode: Full re-index")
        else:
            print("Mode: Incremental")

        result = indexer.sync(path=path, full=full)

        print("\nSync complete!")
        print(f"  Files processed: {result.get('files_processed', 0)}")
        print(f"  Chunks created: {result.get('chunks_created', 0)}")
        print(f"  Time: {result.get('duration_ms', 0)}ms")

        return 0

    except ImportError:
        print("Sync requires icr-core. Install with: pip install icr-core")
        return 1
    except Exception as e:
        print(f"Sync failed: {e}")
        return 1


# =============================================================================
# Command: clear
# =============================================================================

def cmd_clear(args: argparse.Namespace, ctx: CLIContext) -> int:
    """Clear ICR state."""
    client = ICRClient(ctx)

    if not client.is_initialized():
        print("ICR not initialized.")
        return 1

    target = args.target

    if target not in ("priors", "ledger", "invariants", "all"):
        print(f"Invalid target: {target}")
        print("Valid targets: priors, ledger, invariants, all")
        return 1

    # Confirm
    print(f"WARNING: This will clear {target} from ICR.")
    response = input("Are you sure? (yes/no): ")
    if response.lower() != "yes":
        print("Cancelled.")
        return 0

    try:
        from icr.core.storage import Storage

        storage = Storage(db_path=ctx.db_path)

        if target == "all":
            storage.clear_all()
            print("All ICR state cleared.")
        elif target == "priors":
            storage.clear_priors()
            print("Priors cleared.")
        elif target == "ledger":
            storage.clear_ledger()
            print("Ledger cleared.")
        elif target == "invariants":
            storage.clear_invariants()
            print("Invariants cleared.")

        return 0

    except ImportError:
        print("Clear requires icr-core. Install with: pip install icr-core")
        return 1
    except Exception as e:
        print(f"Clear failed: {e}")
        return 1


# =============================================================================
# Main CLI
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="ic",
        description="ICR CLI - Infinite Context Runtime commands for Claude Code",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # pack
    pack_parser = subparsers.add_parser("pack", help="Generate context pack")
    pack_parser.add_argument("-v", "--verbose", action="store_true")
    pack_parser.add_argument("--tokens", type=int, help="Max tokens")
    pack_parser.add_argument("query", nargs="?", help="Optional query for relevance")
    pack_parser.set_defaults(func=cmd_pack)

    # search
    search_parser = subparsers.add_parser("search", help="Search environment")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--limit", type=int, default=10)
    search_parser.add_argument("--type", choices=["file", "chunk", "memory", "decision"])
    search_parser.set_defaults(func=cmd_search)

    # impact
    impact_parser = subparsers.add_parser("impact", help="Impact analysis")
    impact_parser.add_argument("paths", nargs="+", help="File paths to analyze")
    impact_parser.add_argument("--depth", type=int, default=2)
    impact_parser.set_defaults(func=cmd_impact)

    # pin
    pin_parser = subparsers.add_parser("pin", help="Pin an invariant")
    pin_parser.add_argument("invariant", help="Invariant content")
    pin_parser.add_argument("--priority", type=int, default=5)
    pin_parser.add_argument("--expires", help="Expiration (e.g., '7d', '1h')")
    pin_parser.set_defaults(func=cmd_pin)

    # unpin
    unpin_parser = subparsers.add_parser("unpin", help="Remove pinned invariant")
    unpin_parser.add_argument("id", help="Invariant ID")
    unpin_parser.set_defaults(func=cmd_unpin)

    # status
    status_parser = subparsers.add_parser("status", help="Show ICR status")
    status_parser.add_argument("-v", "--verbose", action="store_true")
    status_parser.set_defaults(func=cmd_status)

    # ledger
    ledger_parser = subparsers.add_parser("ledger", help="View ledger")
    ledger_parser.add_argument("--last", type=int, default=10)
    ledger_parser.add_argument("--type", choices=["decision", "todo", "question", "file"])
    ledger_parser.set_defaults(func=cmd_ledger)

    # compact
    compact_parser = subparsers.add_parser("compact", help="Manual compaction")
    compact_parser.add_argument("--dry-run", action="store_true")
    compact_parser.add_argument("--preserve-invariants", action="store_true")
    compact_parser.set_defaults(func=cmd_compact)

    # config
    config_parser = subparsers.add_parser("config", help="View/modify config")
    config_parser.add_argument("key", nargs="?", help="Config key")
    config_parser.add_argument("value", nargs="?", help="New value")
    config_parser.set_defaults(func=cmd_config)

    # sync
    sync_parser = subparsers.add_parser("sync", help="Sync environment")
    sync_parser.add_argument("--full", action="store_true")
    sync_parser.add_argument("--path", help="Path to sync")
    sync_parser.set_defaults(func=cmd_sync)

    # clear
    clear_parser = subparsers.add_parser("clear", help="Clear ICR state")
    clear_parser.add_argument("target", choices=["priors", "ledger", "invariants", "all"])
    clear_parser.set_defaults(func=cmd_clear)

    return parser


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    ctx = CLIContext.from_env()
    if args.verbose:
        ctx.verbose = True

    if hasattr(args, "func"):
        return args.func(args, ctx)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
