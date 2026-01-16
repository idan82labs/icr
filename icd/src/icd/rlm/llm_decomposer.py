"""
LLM-based query decomposition for RLM.

Uses Claude to intelligently decompose complex queries into focused sub-queries.
Falls back to heuristic decomposition if no API key is available.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class QueryType(str, Enum):
    """Types of sub-queries for code understanding."""
    DEFINITION = "definition"
    USAGE = "usage"
    IMPLEMENTATION = "implementation"
    RELATED = "related"
    CONTRACT = "contract"
    TEST = "test"


@dataclass
class SubQuery:
    """A decomposed sub-query."""
    query: str
    query_type: QueryType
    focus_hint: str | None = None
    priority: int = 1


@dataclass
class DecompositionResult:
    """Result from query decomposition."""
    original_query: str
    sub_queries: list[SubQuery]
    reasoning: str
    used_llm: bool


class LLMDecomposer:
    """
    LLM-based query decomposer.

    Uses Claude to decompose complex queries into focused sub-queries.
    Caches results to avoid repeated API calls.
    """

    DECOMPOSITION_PROMPT = """You are a code search query decomposer. Given a user's question about a codebase, break it down into 2-4 focused sub-queries that would help find relevant code.

For each sub-query, specify:
1. The search query (concise, 3-10 words)
2. The type: definition, usage, implementation, related, contract, or test
3. A focus hint (optional file pattern or symbol type to prioritize)

User question: {query}

Respond in JSON format:
{{
  "reasoning": "Brief explanation of decomposition strategy",
  "sub_queries": [
    {{"query": "...", "type": "...", "focus_hint": "..."}},
    ...
  ]
}}

Rules:
- Keep sub-queries specific and searchable
- "definition" = find where X is defined
- "usage" = find where X is used/called
- "implementation" = find how X works internally
- "related" = find similar/connected code
- "contract" = find interfaces, types, schemas
- "test" = find test cases for X
- Return 2-4 sub-queries, prioritized by importance"""

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: Path | None = None,
        model: str = "claude-sonnet-4-20250514",
    ):
        """
        Initialize the LLM decomposer.

        Args:
            api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY env var.
            cache_dir: Directory for caching decomposition results.
            model: Claude model to use.
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.cache_dir = cache_dir or Path(".icr/cache/decompositions")
        self.model = model
        self._client = None

    def _get_client(self):
        """Lazily initialize Anthropic client."""
        if self._client is None and self.api_key:
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self.api_key)
            except ImportError:
                logger.warning("anthropic package not installed, using heuristic decomposition")
                return None
        return self._client

    def _cache_key(self, query: str) -> str:
        """Generate cache key for a query."""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()

    def _load_from_cache(self, query: str) -> DecompositionResult | None:
        """Load cached decomposition result."""
        cache_file = self.cache_dir / f"{self._cache_key(query)}.json"
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text())
                return DecompositionResult(
                    original_query=data["original_query"],
                    sub_queries=[
                        SubQuery(
                            query=sq["query"],
                            query_type=QueryType(sq["type"]),
                            focus_hint=sq.get("focus_hint"),
                            priority=sq.get("priority", 1),
                        )
                        for sq in data["sub_queries"]
                    ],
                    reasoning=data["reasoning"],
                    used_llm=data["used_llm"],
                )
            except Exception as e:
                logger.debug(f"Failed to load cache: {e}")
        return None

    def _save_to_cache(self, result: DecompositionResult) -> None:
        """Save decomposition result to cache."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / f"{self._cache_key(result.original_query)}.json"
        try:
            data = {
                "original_query": result.original_query,
                "sub_queries": [
                    {
                        "query": sq.query,
                        "type": sq.query_type.value,
                        "focus_hint": sq.focus_hint,
                        "priority": sq.priority,
                    }
                    for sq in result.sub_queries
                ],
                "reasoning": result.reasoning,
                "used_llm": result.used_llm,
            }
            cache_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug(f"Failed to save cache: {e}")

    async def decompose(self, query: str) -> DecompositionResult:
        """
        Decompose a query into sub-queries.

        Uses LLM if available, falls back to heuristics.

        Args:
            query: The user's query about the codebase.

        Returns:
            DecompositionResult with sub-queries.
        """
        # Check cache first
        cached = self._load_from_cache(query)
        if cached:
            logger.debug("Using cached decomposition", query=query[:50])
            return cached

        # Try LLM decomposition
        client = self._get_client()
        if client:
            try:
                result = await self._llm_decompose(query, client)
                self._save_to_cache(result)
                return result
            except Exception as e:
                logger.warning(f"LLM decomposition failed, using heuristics: {e}")

        # Fall back to heuristics
        result = self._heuristic_decompose(query)
        self._save_to_cache(result)
        return result

    async def _llm_decompose(self, query: str, client) -> DecompositionResult:
        """Use Claude to decompose the query."""
        import asyncio

        prompt = self.DECOMPOSITION_PROMPT.format(query=query)

        # Run sync client in thread pool
        def call_api():
            response = client.messages.create(
                model=self.model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        loop = asyncio.get_event_loop()
        response_text = await loop.run_in_executor(None, call_api)

        # Parse JSON response
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in response")

            sub_queries = []
            for i, sq in enumerate(data.get("sub_queries", [])):
                try:
                    query_type = QueryType(sq.get("type", "implementation"))
                except ValueError:
                    query_type = QueryType.IMPLEMENTATION

                sub_queries.append(SubQuery(
                    query=sq["query"],
                    query_type=query_type,
                    focus_hint=sq.get("focus_hint"),
                    priority=len(data["sub_queries"]) - i,  # Higher priority for earlier queries
                ))

            logger.info(
                "LLM decomposition complete",
                query=query[:50],
                num_sub_queries=len(sub_queries),
            )

            return DecompositionResult(
                original_query=query,
                sub_queries=sub_queries,
                reasoning=data.get("reasoning", "LLM decomposition"),
                used_llm=True,
            )

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            raise

    def _heuristic_decompose(self, query: str) -> DecompositionResult:
        """
        Heuristic-based query decomposition.

        Falls back to this when LLM is not available.
        """
        query_lower = query.lower()
        sub_queries = []
        reasoning_parts = []

        # Extract potential symbols (CamelCase, snake_case)
        # Filter out common English words that match CamelCase pattern
        stop_words = {"How", "What", "Where", "When", "Why", "Which", "Who", "The", "This", "That"}
        symbols = [
            s for s in re.findall(r'\b([A-Z][a-zA-Z0-9]+|[a-z]+_[a-z_]+)\b', query)
            if s not in stop_words
        ]

        # Determine query intent
        is_how_question = any(q in query_lower for q in ["how does", "how do", "how is", "how to"])
        is_where_question = any(q in query_lower for q in ["where is", "where are", "find"])
        is_what_question = any(q in query_lower for q in ["what is", "what are", "what does"])
        is_trace_question = any(q in query_lower for q in ["trace", "flow", "path", "lifecycle"])

        if symbols:
            main_symbol = symbols[0]

            # Definition query
            sub_queries.append(SubQuery(
                query=f"definition of {main_symbol}",
                query_type=QueryType.DEFINITION,
                focus_hint=None,
                priority=3,
            ))
            reasoning_parts.append(f"Find definition of '{main_symbol}'")

            if is_how_question or is_trace_question:
                # Implementation query
                sub_queries.append(SubQuery(
                    query=f"{main_symbol} implementation logic",
                    query_type=QueryType.IMPLEMENTATION,
                    priority=2,
                ))
                reasoning_parts.append("Find implementation details")

                # Usage query
                sub_queries.append(SubQuery(
                    query=f"calls to {main_symbol}",
                    query_type=QueryType.USAGE,
                    priority=1,
                ))
                reasoning_parts.append("Find usage patterns")

        elif is_how_question:
            # Extract topic
            topic = query_lower.replace("how does", "").replace("how do", "").replace("how is", "").replace("work", "").strip()

            sub_queries.append(SubQuery(
                query=f"{topic} implementation",
                query_type=QueryType.IMPLEMENTATION,
                priority=3,
            ))
            sub_queries.append(SubQuery(
                query=f"{topic} handler entry point",
                query_type=QueryType.DEFINITION,
                priority=2,
            ))
            reasoning_parts.append(f"Find implementation and entry points for '{topic}'")

        elif is_trace_question:
            # Extract what to trace
            topic = re.sub(r'trace|flow|path|lifecycle', '', query_lower).strip()

            sub_queries.append(SubQuery(
                query=f"{topic} entry point",
                query_type=QueryType.DEFINITION,
                priority=3,
            ))
            sub_queries.append(SubQuery(
                query=f"{topic} processing steps",
                query_type=QueryType.IMPLEMENTATION,
                priority=2,
            ))
            sub_queries.append(SubQuery(
                query=f"{topic} handler chain",
                query_type=QueryType.USAGE,
                priority=1,
            ))
            reasoning_parts.append(f"Trace flow of '{topic}'")

        else:
            # Generic decomposition
            sub_queries.append(SubQuery(
                query=query,
                query_type=QueryType.IMPLEMENTATION,
                priority=2,
            ))

            # Add contract search for complex queries
            if len(query.split()) > 5:
                sub_queries.append(SubQuery(
                    query=f"interface type {query.split()[0]}",
                    query_type=QueryType.CONTRACT,
                    priority=1,
                ))
            reasoning_parts.append("Generic search with contract awareness")

        logger.info(
            "Heuristic decomposition complete",
            query=query[:50],
            num_sub_queries=len(sub_queries),
        )

        return DecompositionResult(
            original_query=query,
            sub_queries=sub_queries,
            reasoning="; ".join(reasoning_parts) if reasoning_parts else "Heuristic decomposition",
            used_llm=False,
        )


# Convenience function
async def decompose_query(
    query: str,
    api_key: str | None = None,
    cache_dir: Path | None = None,
) -> DecompositionResult:
    """
    Decompose a query into sub-queries.

    Args:
        query: The user's query.
        api_key: Optional Anthropic API key.
        cache_dir: Optional cache directory.

    Returns:
        DecompositionResult with sub-queries.
    """
    decomposer = LLMDecomposer(api_key=api_key, cache_dir=cache_dir)
    return await decomposer.decompose(query)
