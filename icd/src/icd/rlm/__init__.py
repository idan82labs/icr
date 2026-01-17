"""
RLM (Retrieval-augmented Language Model) modules for ICD.

Provides:
- Plan generation for iterative retrieval
- Non-generative aggregation operations
- Budget tracking and stop conditions
- True RLM with context externalization (research-grade)
"""

from icd.rlm.aggregator import Aggregator
from icd.rlm.budget import BudgetTracker
from icd.rlm.planner import RLMPlanner
from icd.rlm.true_rlm import RLMExecutionResult, RLMProgram, TrueRLMOrchestrator, run_true_rlm

__all__ = [
    "RLMPlanner",
    "Aggregator",
    "BudgetTracker",
    "TrueRLMOrchestrator",
    "RLMProgram",
    "RLMExecutionResult",
    "run_true_rlm",
]
