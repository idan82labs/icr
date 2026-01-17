"""
Query Intent Router (QIR).

Classifies query intent and routes to specialized retrieval strategies.
Different query types benefit from different retrieval configurations:

- "What is X?" → Boost definitions, contracts, interfaces
- "How does X work?" → Boost implementations, function bodies
- "Where is X used?" → Enable graph expansion, find call sites
- "Why does X do Y?" → Look for comments, docstrings, design docs

This is a novel contribution to code retrieval: intent-aware routing
that adjusts retrieval weights dynamically based on query semantics.

Reference: Inspired by query intent classification in web search,
adapted for code retrieval with code-specific intent taxonomy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from icd.config import Config

logger = structlog.get_logger(__name__)


class QueryIntent(str, Enum):
    """Taxonomy of code query intents."""

    # Definitional: "What is X?", "Define X", "X class/interface"
    DEFINITION = "definition"

    # Implementation: "How does X work?", "X algorithm", "X logic"
    IMPLEMENTATION = "implementation"

    # Usage: "Where is X used?", "X callers", "X references"
    USAGE = "usage"

    # Explanation: "Why does X?", "Purpose of X", "Design of X"
    EXPLANATION = "explanation"

    # Debugging: "X error", "X bug", "X fails"
    DEBUGGING = "debugging"

    # Comparison: "Difference between X and Y", "X vs Y"
    COMPARISON = "comparison"

    # Example: "Example of X", "How to use X", "X tutorial"
    EXAMPLE = "example"

    # General: Doesn't fit other categories
    GENERAL = "general"


@dataclass
class IntentClassification:
    """Result of intent classification."""

    primary_intent: QueryIntent
    confidence: float  # 0.0 - 1.0
    secondary_intent: QueryIntent | None = None
    extracted_entities: list[str] = field(default_factory=list)
    signals: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalStrategy:
    """Retrieval configuration for a specific intent."""

    # Weight adjustments (multipliers applied to base weights)
    weight_embedding_mult: float = 1.0
    weight_bm25_mult: float = 1.0
    weight_contract_mult: float = 1.0
    weight_recency_mult: float = 1.0

    # Feature toggles
    enable_graph_expansion: bool = False
    graph_edge_priority: list[str] = field(default_factory=list)

    # CRAG adjustments
    crag_correct_threshold_adj: float = 0.0  # Added to base threshold
    crag_incorrect_threshold_adj: float = 0.0

    # Symbol type preferences (boost these types)
    preferred_symbol_types: list[str] = field(default_factory=list)

    # Additional context
    include_docstrings: bool = True
    include_tests: bool = False


# Intent-specific retrieval strategies
# Symbol types match tree-sitter output: class_definition, function_definition,
# decorated_definition, module_header, text_block
INTENT_STRATEGIES: dict[QueryIntent, RetrievalStrategy] = {
    QueryIntent.DEFINITION: RetrievalStrategy(
        weight_embedding_mult=0.6,  # Much less semantic - prioritize exact match
        weight_bm25_mult=2.0,  # Strongly boost keyword match for class names
        weight_contract_mult=3.0,  # Heavily prefer contracts/interfaces
        weight_recency_mult=0.3,  # Definitions are stable, don't need recency
        enable_graph_expansion=False,
        preferred_symbol_types=[
            "class_definition", "function_definition", "decorated_definition"
        ],
        include_docstrings=True,
        include_tests=False,
    ),

    QueryIntent.IMPLEMENTATION: RetrievalStrategy(
        weight_embedding_mult=1.5,  # Boost semantic for "how" questions
        weight_bm25_mult=1.2,  # Still need keyword match
        weight_contract_mult=0.5,  # Less interested in interfaces
        weight_recency_mult=0.8,
        enable_graph_expansion=True,
        graph_edge_priority=["CALLS", "CONTAINS", "IMPORTS"],
        preferred_symbol_types=[
            "function_definition", "decorated_definition", "class_definition"
        ],
        include_docstrings=True,
        include_tests=False,
    ),

    QueryIntent.USAGE: RetrievalStrategy(
        weight_embedding_mult=0.7,  # Less semantic, more structural
        weight_bm25_mult=1.5,  # Boost keyword for function names
        weight_contract_mult=0.3,  # Not interested in interfaces for usage
        weight_recency_mult=1.5,  # Recent usage very relevant
        enable_graph_expansion=True,  # Critical for usage queries
        graph_edge_priority=["CALLS", "REFERENCES", "IMPORTS"],
        crag_correct_threshold_adj=-0.1,  # Be more lenient
        preferred_symbol_types=[
            "function_definition", "decorated_definition", "class_definition"
        ],
        include_docstrings=False,
        include_tests=True,  # Tests show usage
    ),

    QueryIntent.EXPLANATION: RetrievalStrategy(
        weight_embedding_mult=1.3,  # Semantic understanding crucial
        weight_bm25_mult=0.8,
        weight_contract_mult=1.2,  # Contracts explain intent
        weight_recency_mult=0.8,
        enable_graph_expansion=False,
        preferred_symbol_types=[
            "class_definition", "module_header", "decorated_definition"
        ],
        include_docstrings=True,  # Critical for explanation
        include_tests=False,
    ),

    QueryIntent.DEBUGGING: RetrievalStrategy(
        weight_embedding_mult=1.0,
        weight_bm25_mult=1.2,  # Error messages are keyword-heavy
        weight_contract_mult=0.8,
        weight_recency_mult=1.5,  # Recent code more likely buggy
        enable_graph_expansion=True,
        graph_edge_priority=["CALLS", "CONTAINS"],
        crag_incorrect_threshold_adj=0.1,  # Be more aggressive
        preferred_symbol_types=[
            "function_definition", "decorated_definition"
        ],
        include_docstrings=True,
        include_tests=True,  # Tests reveal bugs
    ),

    QueryIntent.COMPARISON: RetrievalStrategy(
        weight_embedding_mult=1.1,
        weight_bm25_mult=1.0,
        weight_contract_mult=1.3,  # Interfaces show contracts
        weight_recency_mult=0.7,
        enable_graph_expansion=False,
        preferred_symbol_types=[
            "class_definition", "decorated_definition", "function_definition"
        ],
        include_docstrings=True,
        include_tests=False,
    ),

    QueryIntent.EXAMPLE: RetrievalStrategy(
        weight_embedding_mult=1.0,
        weight_bm25_mult=1.1,
        weight_contract_mult=0.6,
        weight_recency_mult=1.0,
        enable_graph_expansion=True,
        graph_edge_priority=["CALLS", "IMPORTS"],
        preferred_symbol_types=[
            "function_definition", "decorated_definition"
        ],
        include_docstrings=True,
        include_tests=True,  # Tests are examples
    ),

    QueryIntent.GENERAL: RetrievalStrategy(
        # Use base weights
        weight_embedding_mult=1.0,
        weight_bm25_mult=1.0,
        weight_contract_mult=1.0,
        weight_recency_mult=1.0,
        enable_graph_expansion=False,
        include_docstrings=True,
        include_tests=False,
    ),
}


class QueryIntentClassifier:
    """
    Classifies query intent using rule-based heuristics.

    Fast, no LLM required, interpretable.
    Could be enhanced with learned classifier in future.
    """

    # Intent patterns: (regex, intent, confidence_boost)
    INTENT_PATTERNS: list[tuple[str, QueryIntent, float]] = [
        # Definition patterns
        (r"^what\s+is\s+", QueryIntent.DEFINITION, 0.3),
        (r"^define\s+", QueryIntent.DEFINITION, 0.4),
        (r"\bclass\s+\w+\b", QueryIntent.DEFINITION, 0.2),
        (r"\binterface\s+\w+\b", QueryIntent.DEFINITION, 0.3),
        (r"\btype\s+(of\s+)?\w+\b", QueryIntent.DEFINITION, 0.2),
        (r"\bdefinition\b", QueryIntent.DEFINITION, 0.3),

        # Implementation patterns
        (r"^how\s+(does|do|is|are|can)\s+", QueryIntent.IMPLEMENTATION, 0.3),
        (r"\bimplement(s|ed|ation)?\b", QueryIntent.IMPLEMENTATION, 0.3),
        (r"\balgorithm\b", QueryIntent.IMPLEMENTATION, 0.3),
        (r"\blogic\b", QueryIntent.IMPLEMENTATION, 0.2),
        (r"\bwork(s|ing)?\b", QueryIntent.IMPLEMENTATION, 0.1),
        (r"\bcode\s+for\b", QueryIntent.IMPLEMENTATION, 0.2),

        # Usage patterns
        (r"^where\s+(is|are|do|does)\s+", QueryIntent.USAGE, 0.4),
        (r"\bused\b", QueryIntent.USAGE, 0.2),
        (r"\bcall(s|ed|ing)?\b", QueryIntent.USAGE, 0.2),
        (r"\breference(s|d)?\b", QueryIntent.USAGE, 0.2),
        (r"\bcaller(s)?\b", QueryIntent.USAGE, 0.3),
        (r"\bwho\s+uses\b", QueryIntent.USAGE, 0.3),

        # Explanation patterns
        (r"^why\s+(does|do|is|are)\s+", QueryIntent.EXPLANATION, 0.4),
        (r"\bpurpose\b", QueryIntent.EXPLANATION, 0.3),
        (r"\breason\b", QueryIntent.EXPLANATION, 0.2),
        (r"\bdesign\b", QueryIntent.EXPLANATION, 0.2),
        (r"\bexplain\b", QueryIntent.EXPLANATION, 0.3),

        # Debugging patterns
        (r"\berror\b", QueryIntent.DEBUGGING, 0.3),
        (r"\bbug\b", QueryIntent.DEBUGGING, 0.3),
        (r"\bfail(s|ed|ing)?\b", QueryIntent.DEBUGGING, 0.2),
        (r"\bfix\b", QueryIntent.DEBUGGING, 0.2),
        (r"\bcrash(es|ed|ing)?\b", QueryIntent.DEBUGGING, 0.3),
        (r"\bexception\b", QueryIntent.DEBUGGING, 0.2),
        (r"\bnot\s+working\b", QueryIntent.DEBUGGING, 0.3),

        # Comparison patterns
        (r"\bdifference\s+between\b", QueryIntent.COMPARISON, 0.4),
        (r"\bvs\.?\b", QueryIntent.COMPARISON, 0.3),
        (r"\bcompare\b", QueryIntent.COMPARISON, 0.3),
        (r"\bversus\b", QueryIntent.COMPARISON, 0.3),
        (r"\bor\b.*\bwhich\b", QueryIntent.COMPARISON, 0.2),

        # Example patterns
        (r"\bexample\b", QueryIntent.EXAMPLE, 0.4),
        (r"\bhow\s+to\s+use\b", QueryIntent.EXAMPLE, 0.3),
        (r"\btutorial\b", QueryIntent.EXAMPLE, 0.3),
        (r"\bsample\b", QueryIntent.EXAMPLE, 0.2),
        (r"\bdemo\b", QueryIntent.EXAMPLE, 0.2),
    ]

    # Entity extraction patterns
    ENTITY_PATTERNS = [
        r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b",  # CamelCase
        r"\b([a-z]+(?:_[a-z]+)+)\b",  # snake_case
        r"`([^`]+)`",  # Backtick-quoted
        r'"([^"]+)"',  # Double-quoted
        r"'([^']+)'",  # Single-quoted
    ]

    def __init__(self, config: "Config | None" = None) -> None:
        self.config = config

        # Compile patterns for efficiency
        self._compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), intent, boost)
            for pattern, intent, boost in self.INTENT_PATTERNS
        ]

        self._entity_patterns = [
            re.compile(pattern) for pattern in self.ENTITY_PATTERNS
        ]

    def classify(self, query: str) -> IntentClassification:
        """
        Classify query intent.

        Returns primary intent with confidence, optional secondary,
        and extracted entities.
        """
        query_lower = query.lower().strip()

        # Accumulate scores for each intent
        intent_scores: dict[QueryIntent, float] = {
            intent: 0.0 for intent in QueryIntent
        }
        signals: dict[str, Any] = {"matched_patterns": []}

        # Match patterns
        for regex, intent, boost in self._compiled_patterns:
            if regex.search(query_lower):
                intent_scores[intent] += boost
                signals["matched_patterns"].append({
                    "pattern": regex.pattern,
                    "intent": intent.value,
                    "boost": boost,
                })

        # Extract entities
        entities = self._extract_entities(query)

        # Find top two intents
        sorted_intents = sorted(
            intent_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        primary_intent = sorted_intents[0][0]
        primary_score = sorted_intents[0][1]

        # If no patterns matched, use GENERAL
        if primary_score == 0:
            primary_intent = QueryIntent.GENERAL
            confidence = 0.5  # Low confidence default
        else:
            # Normalize confidence
            total_score = sum(s for _, s in sorted_intents if s > 0)
            confidence = min(0.95, primary_score / max(total_score, 0.1) + 0.3)

        # Secondary intent if score is significant
        secondary_intent = None
        if len(sorted_intents) > 1 and sorted_intents[1][1] > 0.1:
            secondary_intent = sorted_intents[1][0]

        signals["all_scores"] = {k.value: v for k, v in intent_scores.items() if v > 0}

        logger.debug(
            "Query intent classified",
            query=query[:50],
            intent=primary_intent.value,
            confidence=confidence,
            entities=entities[:5],
        )

        return IntentClassification(
            primary_intent=primary_intent,
            confidence=confidence,
            secondary_intent=secondary_intent,
            extracted_entities=entities,
            signals=signals,
        )

    def _extract_entities(self, query: str) -> list[str]:
        """Extract potential code entities from query."""
        entities = []
        seen = set()

        for pattern in self._entity_patterns:
            for match in pattern.finditer(query):
                entity = match.group(1) if match.lastindex else match.group(0)
                if entity and entity.lower() not in seen:
                    seen.add(entity.lower())
                    entities.append(entity)

        return entities


class QueryRouter:
    """
    Routes queries to appropriate retrieval strategies based on intent.

    Main entry point for intent-aware retrieval.
    """

    def __init__(self, config: "Config") -> None:
        self.config = config
        self.classifier = QueryIntentClassifier(config)
        self.strategies = INTENT_STRATEGIES

    def route(self, query: str) -> tuple[IntentClassification, RetrievalStrategy]:
        """
        Route a query to its appropriate strategy.

        Returns:
            Tuple of (classification, strategy)
        """
        classification = self.classifier.classify(query)
        strategy = self.strategies.get(
            classification.primary_intent,
            self.strategies[QueryIntent.GENERAL]
        )

        return classification, strategy

    def get_adjusted_weights(
        self,
        query: str,
        base_weights: dict[str, float],
    ) -> dict[str, float]:
        """
        Get adjusted retrieval weights for a query.

        Applies intent-specific multipliers to base weights.
        """
        classification, strategy = self.route(query)

        adjusted = {
            "weight_embedding": base_weights.get("weight_embedding", 0.4) * strategy.weight_embedding_mult,
            "weight_bm25": base_weights.get("weight_bm25", 0.3) * strategy.weight_bm25_mult,
            "weight_contract": base_weights.get("weight_contract", 0.1) * strategy.weight_contract_mult,
            "weight_recency": base_weights.get("weight_recency", 0.1) * strategy.weight_recency_mult,
        }

        # Normalize to sum to ~1.0 (approximately)
        total = sum(adjusted.values())
        if total > 0:
            # Keep relative proportions but don't strictly normalize
            # (some queries should weight certain signals more heavily)
            pass

        return adjusted

    def should_enable_graph(self, query: str) -> bool:
        """Check if graph expansion should be enabled for this query."""
        classification, strategy = self.route(query)
        return strategy.enable_graph_expansion

    def get_graph_edge_priority(self, query: str) -> list[str]:
        """Get prioritized edge types for graph traversal."""
        classification, strategy = self.route(query)
        return strategy.graph_edge_priority

    def get_preferred_symbol_types(self, query: str) -> list[str]:
        """Get preferred symbol types for this query."""
        classification, strategy = self.route(query)
        return strategy.preferred_symbol_types


def create_query_router(config: "Config") -> QueryRouter:
    """Create a query router instance."""
    return QueryRouter(config)
