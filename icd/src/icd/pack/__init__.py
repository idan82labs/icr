"""
Pack compilation modules for ICD.

Provides:
- Knapsack-based pack compilation
- Markdown formatting with citations
- Mode gating (pack vs RLM)
"""

from icd.pack.compiler import PackCompiler
from icd.pack.formatter import PackFormatter
from icd.pack.gating import ModeGate

__all__ = [
    "PackCompiler",
    "PackFormatter",
    "ModeGate",
]
