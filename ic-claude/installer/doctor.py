#!/usr/bin/env python3
"""
ICR Health Check (Doctor)

This script verifies that ICR is properly configured and functioning.

Checks performed:
1. Configuration exists at ~/.icr/config.yaml
2. SQLite database is accessible
3. Vector index is initialized
4. Embedding backend is working
5. Claude Code hooks are configured
6. MCP server configuration is present
7. Python dependencies are available

Can optionally attempt to fix common issues with --fix.
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger("icr.doctor")


@dataclass
class CheckResult:
    """Result of a single health check."""

    name: str
    passed: bool
    message: str
    fixable: bool = False
    fix_command: str | None = None


@dataclass
class HealthReport:
    """Complete health report."""

    checks: list[CheckResult] = field(default_factory=list)
    passed: int = 0
    failed: int = 0
    warnings: int = 0

    def add(self, result: CheckResult) -> None:
        """Add a check result."""
        self.checks.append(result)
        if result.passed:
            self.passed += 1
        else:
            self.failed += 1

    @property
    def healthy(self) -> bool:
        """Return True if all critical checks passed."""
        return self.failed == 0


class HealthChecker:
    """Performs ICR health checks."""

    def __init__(
        self,
        config_path: Path | None = None,
        db_path: Path | None = None,
        verbose: bool = False,
    ):
        """Initialize health checker."""
        self.config_path = config_path or Path(
            os.environ.get("ICR_CONFIG_PATH", Path.home() / ".icr" / "config.yaml")
        )
        self.db_path = db_path or Path(
            os.environ.get("ICR_DB_PATH", Path.home() / ".icr" / "icr.db")
        )
        self.user_settings_path = Path.home() / ".claude" / "settings.json"
        self.user_claude_json = Path.home() / ".claude.json"
        self.verbose = verbose

    def check_all(self) -> HealthReport:
        """Run all health checks."""
        report = HealthReport()

        # Core checks
        report.add(self.check_config())
        report.add(self.check_database())
        report.add(self.check_database_schema())
        report.add(self.check_vector_index())
        report.add(self.check_embedding_backend())

        # Integration checks
        report.add(self.check_hooks_config())
        report.add(self.check_mcp_config())

        # Dependency checks
        report.add(self.check_python_version())
        report.add(self.check_dependencies())

        return report

    def check_config(self) -> CheckResult:
        """Check if configuration file exists and is valid."""
        if not self.config_path.exists():
            return CheckResult(
                name="Configuration",
                passed=False,
                message=f"Config not found at {self.config_path}",
                fixable=True,
                fix_command="icr init",
            )

        try:
            import yaml
            with open(self.config_path) as f:
                config = yaml.safe_load(f)

            if not config:
                return CheckResult(
                    name="Configuration",
                    passed=False,
                    message="Config file is empty",
                    fixable=True,
                    fix_command="icr init --force",
                )

            return CheckResult(
                name="Configuration",
                passed=True,
                message=f"Valid config at {self.config_path}",
            )

        except ImportError:
            return CheckResult(
                name="Configuration",
                passed=False,
                message="PyYAML not installed",
                fixable=True,
                fix_command="pip install pyyaml",
            )
        except Exception as e:
            return CheckResult(
                name="Configuration",
                passed=False,
                message=f"Config error: {e}",
                fixable=True,
                fix_command="icr init --force",
            )

    def check_database(self) -> CheckResult:
        """Check if database exists and is accessible."""
        if not self.db_path.exists():
            return CheckResult(
                name="Database",
                passed=False,
                message=f"Database not found at {self.db_path}",
                fixable=True,
                fix_command="icr init",
            )

        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            conn.close()

            # Get size
            size_mb = self.db_path.stat().st_size / 1024 / 1024

            return CheckResult(
                name="Database",
                passed=True,
                message=f"Accessible ({size_mb:.2f} MB)",
            )

        except sqlite3.Error as e:
            return CheckResult(
                name="Database",
                passed=False,
                message=f"Database error: {e}",
                fixable=False,
            )

    def check_database_schema(self) -> CheckResult:
        """Check if database has required tables."""
        if not self.db_path.exists():
            return CheckResult(
                name="Database Schema",
                passed=False,
                message="Database does not exist",
                fixable=True,
                fix_command="icr init",
            )

        required_tables = [
            "chunks",
            "priors",
            "invariants",
            "ledger",
            "sessions",
            "files",
        ]

        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Get existing tables
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            existing_tables = {row[0] for row in cursor.fetchall()}
            conn.close()

            missing = [t for t in required_tables if t not in existing_tables]

            if missing:
                return CheckResult(
                    name="Database Schema",
                    passed=False,
                    message=f"Missing tables: {', '.join(missing)}",
                    fixable=True,
                    fix_command="icr db migrate",
                )

            return CheckResult(
                name="Database Schema",
                passed=True,
                message=f"All {len(required_tables)} required tables present",
            )

        except sqlite3.Error as e:
            return CheckResult(
                name="Database Schema",
                passed=False,
                message=f"Schema check failed: {e}",
                fixable=False,
            )

    def check_vector_index(self) -> CheckResult:
        """Check if vector index is initialized."""
        if not self.db_path.exists():
            return CheckResult(
                name="Vector Index",
                passed=False,
                message="Database does not exist",
                fixable=True,
                fix_command="icr init",
            )

        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Check for vector data
            cursor.execute(
                "SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL"
            )
            count = cursor.fetchone()[0]
            conn.close()

            if count == 0:
                return CheckResult(
                    name="Vector Index",
                    passed=False,
                    message="No embeddings in database",
                    fixable=True,
                    fix_command="icr sync --full",
                )

            return CheckResult(
                name="Vector Index",
                passed=True,
                message=f"{count} embeddings indexed",
            )

        except sqlite3.OperationalError as e:
            if "no such table" in str(e):
                return CheckResult(
                    name="Vector Index",
                    passed=False,
                    message="Chunks table missing",
                    fixable=True,
                    fix_command="icr db migrate",
                )
            return CheckResult(
                name="Vector Index",
                passed=False,
                message=f"Vector check failed: {e}",
                fixable=False,
            )

    def check_embedding_backend(self) -> CheckResult:
        """Check if embedding backend is working."""
        try:
            from icr.core.embeddings import EmbeddingBackend

            backend = EmbeddingBackend.from_config(self.config_path)

            # Test embedding
            test_text = "Hello, world!"
            embedding = backend.embed(test_text)

            if embedding is None or len(embedding) == 0:
                return CheckResult(
                    name="Embedding Backend",
                    passed=False,
                    message="Backend returned empty embedding",
                    fixable=False,
                )

            return CheckResult(
                name="Embedding Backend",
                passed=True,
                message=f"Working ({backend.name}, dim={len(embedding)})",
            )

        except ImportError:
            return CheckResult(
                name="Embedding Backend",
                passed=False,
                message="icr-core not installed",
                fixable=True,
                fix_command="pip install icr-core",
            )
        except Exception as e:
            return CheckResult(
                name="Embedding Backend",
                passed=False,
                message=f"Backend error: {e}",
                fixable=False,
            )

    def check_hooks_config(self) -> CheckResult:
        """Check if Claude Code hooks are configured."""
        if not self.user_settings_path.exists():
            return CheckResult(
                name="Claude Hooks",
                passed=False,
                message="~/.claude/settings.json not found",
                fixable=True,
                fix_command="icr install --hooks",
            )

        try:
            with open(self.user_settings_path) as f:
                settings = json.load(f)

            hooks = settings.get("hooks", {})

            if not hooks:
                return CheckResult(
                    name="Claude Hooks",
                    passed=False,
                    message="No hooks configured",
                    fixable=True,
                    fix_command="icr install --hooks",
                )

            # Check for ICR hooks
            icr_hooks = []
            for hook_name, hook_list in hooks.items():
                for hook in hook_list:
                    cmd = hook.get("command", "")
                    if "icr" in cmd.lower() or "hook_" in cmd.lower():
                        icr_hooks.append(hook_name)

            if not icr_hooks:
                return CheckResult(
                    name="Claude Hooks",
                    passed=False,
                    message="No ICR hooks found",
                    fixable=True,
                    fix_command="icr install --hooks",
                )

            return CheckResult(
                name="Claude Hooks",
                passed=True,
                message=f"Configured: {', '.join(set(icr_hooks))}",
            )

        except json.JSONDecodeError as e:
            return CheckResult(
                name="Claude Hooks",
                passed=False,
                message=f"Invalid settings.json: {e}",
                fixable=False,
            )

    def check_mcp_config(self) -> CheckResult:
        """Check if MCP server is configured."""
        if not self.user_claude_json.exists():
            return CheckResult(
                name="MCP Server",
                passed=False,
                message="~/.claude.json not found",
                fixable=True,
                fix_command="icr install --mcp",
            )

        try:
            with open(self.user_claude_json) as f:
                config = json.load(f)

            servers = config.get("mcpServers", {})

            if "icr" not in servers:
                return CheckResult(
                    name="MCP Server",
                    passed=False,
                    message="ICR MCP server not configured",
                    fixable=True,
                    fix_command="icr install --mcp",
                )

            icr_config = servers["icr"]
            cmd = icr_config.get("command", "")

            return CheckResult(
                name="MCP Server",
                passed=True,
                message=f"Configured (command: {cmd})",
            )

        except json.JSONDecodeError as e:
            return CheckResult(
                name="MCP Server",
                passed=False,
                message=f"Invalid claude.json: {e}",
                fixable=False,
            )

    def check_python_version(self) -> CheckResult:
        """Check Python version."""
        version = sys.version_info

        if version < (3, 10):
            return CheckResult(
                name="Python Version",
                passed=False,
                message=f"Python {version.major}.{version.minor} (requires >= 3.10)",
                fixable=False,
            )

        return CheckResult(
            name="Python Version",
            passed=True,
            message=f"Python {version.major}.{version.minor}.{version.micro}",
        )

    def check_dependencies(self) -> CheckResult:
        """Check required Python dependencies."""
        required = ["yaml", "sqlite3"]
        optional = ["numpy", "sentence_transformers"]

        missing_required = []
        missing_optional = []

        for pkg in required:
            try:
                __import__(pkg)
            except ImportError:
                # yaml is pyyaml
                if pkg == "yaml":
                    missing_required.append("pyyaml")
                else:
                    missing_required.append(pkg)

        for pkg in optional:
            try:
                __import__(pkg)
            except ImportError:
                missing_optional.append(pkg)

        if missing_required:
            return CheckResult(
                name="Dependencies",
                passed=False,
                message=f"Missing: {', '.join(missing_required)}",
                fixable=True,
                fix_command=f"pip install {' '.join(missing_required)}",
            )

        msg = "All required dependencies installed"
        if missing_optional:
            msg += f" (optional missing: {', '.join(missing_optional)})"

        return CheckResult(
            name="Dependencies",
            passed=True,
            message=msg,
        )


def print_report(report: HealthReport) -> None:
    """Print the health report."""
    print("\nICR Health Check")
    print("=" * 60)

    for check in report.checks:
        status = "[PASS]" if check.passed else "[FAIL]"
        color_start = "\033[92m" if check.passed else "\033[91m"
        color_end = "\033[0m"

        print(f"{color_start}{status}{color_end} {check.name}")
        print(f"       {check.message}")

        if not check.passed and check.fixable and check.fix_command:
            print(f"       Fix: {check.fix_command}")

    print()
    print("-" * 60)
    print(f"Total: {report.passed} passed, {report.failed} failed")

    if report.healthy:
        print("\033[92mICR is healthy!\033[0m")
    else:
        print("\033[91mICR has issues that need attention.\033[0m")


def attempt_fixes(report: HealthReport) -> int:
    """Attempt to fix issues."""
    import subprocess

    fixed = 0
    for check in report.checks:
        if not check.passed and check.fixable and check.fix_command:
            print(f"\nAttempting to fix: {check.name}")
            print(f"  Running: {check.fix_command}")

            try:
                result = subprocess.run(
                    check.fix_command,
                    shell=True,
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    print("  [FIXED]")
                    fixed += 1
                else:
                    print(f"  [FAILED] {result.stderr}")
            except Exception as e:
                print(f"  [ERROR] {e}")

    return fixed


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ICR Health Check",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Attempt to fix detected issues",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    checker = HealthChecker(verbose=args.verbose)
    report = checker.check_all()

    if args.json:
        output = {
            "healthy": report.healthy,
            "passed": report.passed,
            "failed": report.failed,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "message": c.message,
                    "fixable": c.fixable,
                    "fix_command": c.fix_command,
                }
                for c in report.checks
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        print_report(report)

    if args.fix and not report.healthy:
        print("\nAttempting automatic fixes...")
        fixed = attempt_fixes(report)
        print(f"\nFixed {fixed} issue(s)")

        # Re-run checks
        print("\nRe-checking...")
        report = checker.check_all()
        print_report(report)

    return 0 if report.healthy else 1


if __name__ == "__main__":
    sys.exit(main())
