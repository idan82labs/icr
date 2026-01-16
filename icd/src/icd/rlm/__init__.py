"""
RLM (Retrieval-augmented Language Model) modules for ICD.

Provides:
- Plan generation for iterative retrieval
- Non-generative aggregation operations
- Budget tracking and stop conditions
"""

from icd.rlm.aggregator import Aggregator
from icd.rlm.budget import BudgetTracker
from icd.rlm.planner import RLMPlanner

__all__ = [
    "RLMPlanner",
    "Aggregator",
    "BudgetTracker",
]
