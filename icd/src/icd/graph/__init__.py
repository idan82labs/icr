"""
Code graph module for ICR.

Provides AST-derived code graphs for multi-hop retrieval:
- Import/export relationships
- Function call tracking
- Class inheritance
- Dependency-aware context expansion
"""

from icd.graph.builder import CodeGraphBuilder, EdgeType, NodeType
from icd.graph.traversal import GraphRetriever

__all__ = [
    "CodeGraphBuilder",
    "GraphRetriever",
    "NodeType",
    "EdgeType",
]
