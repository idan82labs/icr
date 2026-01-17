"""
Pack compilation modules for ICD.

Provides:
- Knapsack-based pack compilation
- Markdown formatting with citations
- Mode gating (pack vs RLM)

Novel contribution:
- Dependency-Aware Packing (DAC-Pack) - Precedence-constrained knapsack
"""

from icd.pack.compiler import PackCompiler
from icd.pack.formatter import PackFormatter
from icd.pack.gating import ModeGate

# Novel: Dependency-Aware Packing
from icd.pack.dependency_packer import (
    DependencyAwarePacker,
    DependencyAnalyzer,
    DependencyBundle,
    DACPackResult,
    create_dependency_packer,
)

__all__ = [
    # Core packing
    "PackCompiler",
    "PackFormatter",
    "ModeGate",
    # Novel: DAC-Pack
    "DependencyAwarePacker",
    "DependencyAnalyzer",
    "DependencyBundle",
    "DACPackResult",
    "create_dependency_packer",
]
