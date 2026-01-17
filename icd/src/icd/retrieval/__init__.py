"""
Retrieval modules for ICD.

Provides:
- Hybrid retrieval (semantic + BM25)
- MMR diversity selection
- Retrieval entropy computation
- Cross-encoder reranking
"""

from icd.retrieval.entropy import EntropyCalculator
from icd.retrieval.hybrid import HybridRetriever
from icd.retrieval.mmr import MMRSelector
from icd.retrieval.reranker import CrossEncoderReranker

__all__ = [
    "HybridRetriever",
    "MMRSelector",
    "EntropyCalculator",
    "CrossEncoderReranker",
]
