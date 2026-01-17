"""
Retrieval modules for ICD.

Provides:
- Hybrid retrieval (semantic + BM25)
- MMR diversity selection
- Retrieval entropy computation
- Cross-encoder reranking
- CRAG (Corrective RAG) for quality-aware retrieval

Novel contributions:
- Query Intent Router (QIR) - Intent-aware retrieval adjustment
- Multi-Hop Graph Retrieval (MHGR) - Query-guided graph traversal
- Adaptive Entropy Calibration (AEC) - Per-project threshold tuning
- Enhanced Retriever - Integration of all novel components
"""

from icd.retrieval.crag import CRAGRetriever, create_crag_retriever
from icd.retrieval.entropy import EntropyCalculator
from icd.retrieval.hybrid import HybridRetriever
from icd.retrieval.mmr import MMRSelector
from icd.retrieval.reranker import CrossEncoderReranker

# Novel components
from icd.retrieval.query_router import QueryRouter, QueryIntentClassifier, QueryIntent
from icd.retrieval.multihop import MultiHopRetriever, create_multihop_retriever
from icd.retrieval.entropy_calibrator import (
    EntropyCalibrator,
    AdaptiveEntropyGate,
    calibrate_entropy_threshold,
)
from icd.retrieval.enhanced import EnhancedRetriever, create_enhanced_retriever

__all__ = [
    # Core retrieval
    "HybridRetriever",
    "MMRSelector",
    "EntropyCalculator",
    "CrossEncoderReranker",
    "CRAGRetriever",
    "create_crag_retriever",
    # Novel: Query Intent Router
    "QueryRouter",
    "QueryIntentClassifier",
    "QueryIntent",
    # Novel: Multi-Hop Graph Retrieval
    "MultiHopRetriever",
    "create_multihop_retriever",
    # Novel: Adaptive Entropy Calibration
    "EntropyCalibrator",
    "AdaptiveEntropyGate",
    "calibrate_entropy_threshold",
    # Novel: Enhanced Retriever (integration)
    "EnhancedRetriever",
    "create_enhanced_retriever",
]
