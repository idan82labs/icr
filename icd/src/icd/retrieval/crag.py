"""
Corrective Retrieval Augmented Generation (CRAG).

Implements retrieval quality evaluation and automatic correction:
1. Evaluate each retrieved chunk for relevance
2. Classify overall quality: Correct / Incorrect / Ambiguous
3. Apply correction strategy based on classification

Research shows +10-20% accuracy improvement on difficult queries (ICLR 2025).

Reference: "Corrective Retrieval Augmented Generation" (Yan et al., 2024)
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog

if TYPE_CHECKING:
    from icd.config import Config
    from icd.retrieval.hybrid import Chunk, RetrievalResult, ScoredChunk

logger = structlog.get_logger(__name__)


@dataclass
class CRAGEvaluation:
    """Result of CRAG quality evaluation."""

    quality: str  # "correct", "incorrect", "ambiguous"
    confidence: float  # 0-1
    chunk_scores: list[float]  # Per-chunk relevance scores
    reason: str


class RelevanceEvaluator:
    """
    Evaluates relevance of retrieved chunks to query.

    Uses lightweight heuristics to avoid LLM calls during retrieval.
    For true CRAG, you'd use a trained classifier or LLM.
    """

    def __init__(self, config: "Config") -> None:
        self.config = config

    def evaluate_chunk(self, query: str, chunk: "Chunk") -> float:
        """
        Evaluate relevance of a single chunk to the query.

        Returns a score in [0, 1].
        """
        query_lower = query.lower()
        content_lower = chunk.content.lower()

        score = 0.0
        factors = []

        # Factor 1: Query term overlap
        query_terms = set(re.findall(r'\b\w+\b', query_lower))
        content_terms = set(re.findall(r'\b\w+\b', content_lower))

        if query_terms:
            term_overlap = len(query_terms & content_terms) / len(query_terms)
            score += 0.3 * term_overlap
            factors.append(f"term_overlap={term_overlap:.2f}")

        # Factor 2: Symbol name match
        if chunk.symbol_name:
            symbol_lower = chunk.symbol_name.lower()
            # Check if any query term appears in symbol name
            for term in query_terms:
                if len(term) > 2 and term in symbol_lower:
                    score += 0.2
                    factors.append(f"symbol_match={term}")
                    break

        # Factor 3: Question-answer alignment
        # If query is a "how" question, prefer implementation code
        if "how" in query_lower or "implement" in query_lower:
            if chunk.symbol_type in ["function_definition", "method_definition"]:
                score += 0.15
                factors.append("question_type_match")

        # If query is a "what" question, prefer contracts/types
        if "what" in query_lower or "type" in query_lower:
            if chunk.is_contract or chunk.symbol_type in [
                "interface_declaration", "type_alias_declaration", "class_definition"
            ]:
                score += 0.15
                factors.append("contract_match")

        # Factor 4: File path relevance
        file_terms = set(re.findall(r'\w+', chunk.file_path.lower()))
        path_overlap = len(query_terms & file_terms) / max(len(query_terms), 1)
        score += 0.15 * path_overlap
        if path_overlap > 0:
            factors.append(f"path_overlap={path_overlap:.2f}")

        # Factor 5: Content length reasonableness
        # Very short chunks (< 50 chars) are often not useful
        # Very long chunks (> 2000 chars) may be unfocused
        content_len = len(chunk.content)
        if 50 < content_len < 2000:
            score += 0.1
            factors.append("length_ok")

        # Factor 6: Docstring/comment presence (good for understanding)
        if '"""' in chunk.content or "'''" in chunk.content or "/**" in chunk.content:
            score += 0.1
            factors.append("has_docstring")

        return min(score, 1.0)


class QueryReformulator:
    """
    Reformulates queries when initial retrieval fails.

    Uses heuristic transformations (no LLM required).
    For true CRAG, you'd use LLM-based reformulation.
    """

    # Words that are too generic to use alone in reformulations
    GENERIC_WORDS = {
        'class', 'function', 'method', 'implementation', 'definition',
        'code', 'file', 'module', 'the', 'how', 'what', 'where', 'when',
        'does', 'is', 'are', 'work', 'works', 'type', 'interface',
    }

    def reformulate(self, query: str, failed_chunks: list["Chunk"]) -> list[str]:
        """
        Generate reformulated queries.

        Returns multiple alternative queries to try.
        """
        reformulations = []

        # Extract compound terms (CamelCase, snake_case) - keep them whole
        camel_terms = re.findall(r'[A-Z][a-z]+(?:[A-Z][a-z]+)+', query)
        snake_terms = re.findall(r'[a-z]+(?:_[a-z]+)+', query)

        # Filter out generic words
        terms = [t for t in (camel_terms + snake_terms) if t.lower() not in self.GENERIC_WORDS]

        # Strategy 1: Use compound terms directly with implementation suffix
        for term in terms[:2]:
            reformulations.append(f"{term} implementation")
            reformulations.append(term)

        # Strategy 2: Convert questions to keyword search
        if query.lower().startswith(("how", "what", "where", "when")):
            keywords = re.sub(r'^(how|what|where|when)\s+(does|do|is|are)\s+', '', query.lower())
            keywords = keywords.strip('?')
            if keywords and keywords.lower() not in self.GENERIC_WORDS:
                reformulations.append(keywords)

        # Strategy 3: Focus on file types mentioned
        if "test" in query.lower() and terms:
            reformulations.append("test " + " ".join(terms[:2]))

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for q in reformulations:
            q_clean = q.strip()
            if q_clean and q_clean.lower() not in seen:
                seen.add(q_clean.lower())
                unique.append(q_clean)

        return unique[:3]  # Return at most 3 reformulations


class CRAGRetriever:
    """
    Corrective RAG retriever wrapper.

    Wraps an existing retriever and adds quality evaluation + correction.
    """

    def __init__(
        self,
        config: "Config",
        base_retriever: Any,  # HybridRetriever
        correct_threshold: float = 0.5,    # Lowered from 0.6
        incorrect_threshold: float = 0.15,  # Lowered from 0.3
    ) -> None:
        self.config = config
        self.base_retriever = base_retriever
        self.evaluator = RelevanceEvaluator(config)
        self.reformulator = QueryReformulator()

        # Thresholds (configurable)
        self.correct_threshold = correct_threshold
        self.incorrect_threshold = incorrect_threshold

    async def retrieve_with_correction(
        self,
        query: str,
        limit: int | None = None,
        **kwargs,
    ) -> "RetrievalResult":
        """
        Retrieve with automatic quality correction.

        1. Initial retrieval
        2. Evaluate quality
        3. If incorrect: reformulate and retry
        4. If ambiguous: combine original + reformulated
        5. Return best results
        """
        # Step 1: Initial retrieval
        initial_result = await self.base_retriever.retrieve(
            query=query,
            limit=limit,
            **kwargs,
        )

        # Step 2: Evaluate quality
        evaluation = self._evaluate_quality(query, initial_result)

        logger.debug(
            "CRAG evaluation",
            quality=evaluation.quality,
            confidence=evaluation.confidence,
            reason=evaluation.reason,
        )

        # Step 3: Apply correction based on quality
        if evaluation.quality == "correct":
            # Results are good, use as-is
            initial_result.metadata["crag_quality"] = "correct"
            initial_result.metadata["crag_confidence"] = evaluation.confidence
            return initial_result

        elif evaluation.quality == "incorrect":
            # Results are poor, try reformulation
            reformulated_result = await self._retrieve_with_reformulation(
                query=query,
                limit=limit,
                **kwargs,
            )

            if reformulated_result is not None:
                reformulated_result.metadata["crag_quality"] = "corrected"
                reformulated_result.metadata["crag_original_query"] = query
                return reformulated_result

            # Reformulation didn't help, return original
            initial_result.metadata["crag_quality"] = "incorrect_uncorrected"
            return initial_result

        else:  # ambiguous
            # Combine original + reformulated
            reformulated_result = await self._retrieve_with_reformulation(
                query=query,
                limit=limit,
                **kwargs,
            )

            if reformulated_result is not None:
                combined = self._merge_results(
                    initial_result,
                    reformulated_result,
                    limit=limit or 20,
                )
                combined.metadata["crag_quality"] = "combined"
                return combined

            initial_result.metadata["crag_quality"] = "ambiguous"
            return initial_result

    def _evaluate_quality(
        self,
        query: str,
        result: "RetrievalResult",
    ) -> CRAGEvaluation:
        """Evaluate quality of retrieval results."""
        if not result.chunks:
            return CRAGEvaluation(
                quality="incorrect",
                confidence=0.0,
                chunk_scores=[],
                reason="no_results",
            )

        # Score each chunk
        chunk_scores = []
        for chunk in result.chunks[:10]:  # Evaluate top 10
            score = self.evaluator.evaluate_chunk(query, chunk)
            chunk_scores.append(score)

        avg_score = np.mean(chunk_scores)
        max_score = np.max(chunk_scores)

        # Classify
        if avg_score >= self.correct_threshold:
            return CRAGEvaluation(
                quality="correct",
                confidence=avg_score,
                chunk_scores=chunk_scores,
                reason=f"avg_score={avg_score:.2f} >= {self.correct_threshold}",
            )
        elif avg_score <= self.incorrect_threshold:
            return CRAGEvaluation(
                quality="incorrect",
                confidence=1.0 - avg_score,
                chunk_scores=chunk_scores,
                reason=f"avg_score={avg_score:.2f} <= {self.incorrect_threshold}",
            )
        else:
            return CRAGEvaluation(
                quality="ambiguous",
                confidence=0.5,
                chunk_scores=chunk_scores,
                reason=f"avg_score={avg_score:.2f} between thresholds",
            )

    async def _retrieve_with_reformulation(
        self,
        query: str,
        limit: int | None,
        **kwargs,
    ) -> "RetrievalResult | None":
        """Try reformulated queries and return best result."""
        reformulations = self.reformulator.reformulate(query, [])

        if not reformulations:
            return None

        best_result = None
        best_score = 0.0

        for reformulated_query in reformulations:
            logger.debug("Trying reformulated query", query=reformulated_query)

            result = await self.base_retriever.retrieve(
                query=reformulated_query,
                limit=limit,
                **kwargs,
            )

            # Evaluate this result
            evaluation = self._evaluate_quality(reformulated_query, result)

            if evaluation.confidence > best_score:
                best_score = evaluation.confidence
                best_result = result
                best_result.metadata["crag_reformulated_query"] = reformulated_query

        return best_result if best_score > self.incorrect_threshold else None

    def _merge_results(
        self,
        result1: "RetrievalResult",
        result2: "RetrievalResult",
        limit: int,
    ) -> "RetrievalResult":
        """Merge two retrieval results, deduplicating and re-ranking."""
        from icd.retrieval.hybrid import RetrievalResult

        # Combine chunks with scores
        seen_ids = set()
        combined = []

        for chunks, scores in [(result1.chunks, result1.scores),
                                (result2.chunks, result2.scores)]:
            for chunk, score in zip(chunks, scores):
                if chunk.chunk_id not in seen_ids:
                    seen_ids.add(chunk.chunk_id)
                    combined.append((chunk, score))

        # Sort by score
        combined.sort(key=lambda x: x[1], reverse=True)

        # Take top limit
        combined = combined[:limit]

        # Compute new entropy
        new_scores = [s for _, s in combined]
        new_entropy = max(result1.entropy, result2.entropy)  # Conservative

        return RetrievalResult(
            chunks=[c for c, _ in combined],
            scores=new_scores,
            entropy=new_entropy,
            query=result1.query,
            metadata={
                "merged_from": 2,
                "original_count": len(result1.chunks),
                "reformulated_count": len(result2.chunks),
            },
        )


def create_crag_retriever(
    config: "Config",
    base_retriever: Any,
) -> CRAGRetriever:
    """
    Create a CRAG-wrapped retriever.

    Args:
        config: ICD configuration.
        base_retriever: The underlying retriever to wrap.

    Returns:
        CRAGRetriever instance.
    """
    return CRAGRetriever(config, base_retriever)
