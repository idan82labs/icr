"""
ICR Claude Code Hook Scripts

This module contains the hook handlers for Claude Code integration:
- UserPromptSubmit: Inject context pack into prompts
- Stop: Extract ledger from responses
- PreCompact: Preserve invariants during compaction
"""

__version__ = "1.0.0"
__all__ = [
    "hook_userpromptsubmit",
    "hook_stop",
    "hook_precompact",
    "cli",
]
