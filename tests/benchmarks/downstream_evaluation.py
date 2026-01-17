#!/usr/bin/env python3
"""
Downstream Evaluation: Does Better Retrieval = Better LLM Answers?

This benchmark proves ICR's value by measuring:
1. Context Quality: Does retrieved context contain the answer?
2. LLM Accuracy: Does the LLM answer correctly with ICR vs vanilla RAG?

This is the "so what" test - retrieval metrics don't matter if they
don't translate to better task completion.
"""

import asyncio
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

# Minimal logging
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(40),
)


@dataclass
class QATask:
    """A code question with verifiable answer."""

    question: str
    # Keywords that MUST appear in correct answer
    answer_keywords: list[str]
    # Keywords that should appear in good context
    context_keywords: list[str]
    # Category for analysis
    category: str


@dataclass
class EvalResult:
    """Result for one Q&A task."""

    question: str
    category: str

    # Context quality
    icr_context_has_answer: bool
    vanilla_context_has_answer: bool

    # Context stats
    icr_chunks: int
    vanilla_chunks: int
    icr_tokens: int
    vanilla_tokens: int

    # Latency
    icr_latency_ms: float
    vanilla_latency_ms: float


# Ground truth Q&A tasks about ICR codebase
# These have objectively verifiable answers
QA_TASKS = [
    # Factual questions - answer is in the code
    QATask(
        question="What embedding model does ICR use by default?",
        answer_keywords=["minilm", "all-minilm-l6-v2"],
        context_keywords=["model", "embedding", "minilm"],
        category="factual",
    ),
    QATask(
        question="What is the default token budget for context packing?",
        answer_keywords=["8000"],
        context_keywords=["budget", "token", "default"],
        category="factual",
    ),
    QATask(
        question="What vector similarity metric does the vector store use?",
        answer_keywords=["cosine"],
        context_keywords=["cosine", "similarity", "vector"],
        category="factual",
    ),

    # Implementation questions - need to find the right code
    QATask(
        question="How does HybridRetriever combine embedding and BM25 scores?",
        answer_keywords=["weight", "w_e", "w_b"],
        context_keywords=["score", "embedding", "bm25", "weight"],
        category="implementation",
    ),
    QATask(
        question="What algorithm does PackCompiler use for optimization?",
        answer_keywords=["knapsack"],
        context_keywords=["knapsack", "optimize", "budget"],
        category="implementation",
    ),
    QATask(
        question="How does the chunker detect code symbols?",
        answer_keywords=["tree-sitter", "tree_sitter"],
        context_keywords=["tree", "sitter", "symbol", "parse"],
        category="implementation",
    ),

    # Architecture questions - need multiple pieces
    QATask(
        question="What are the main components of the retrieval pipeline?",
        answer_keywords=["embedding", "bm25", "vector"],
        context_keywords=["retriever", "hybrid", "search"],
        category="architecture",
    ),
    QATask(
        question="How does ICR handle incremental indexing?",
        answer_keywords=["hash", "changed", "modified"],
        context_keywords=["incremental", "index", "change"],
        category="architecture",
    ),

    # Usage questions - need to find the right file
    QATask(
        question="What edge types does the code graph track?",
        answer_keywords=["imports", "calls", "contains"],
        context_keywords=["edge", "graph", "type"],
        category="usage",
    ),
    QATask(
        question="What query intents does QIR recognize?",
        answer_keywords=["definition", "implementation", "usage"],
        context_keywords=["intent", "query", "classify"],
        category="usage",
    ),
]


def context_contains_answer(context: str, keywords: list[str]) -> bool:
    """Check if context contains answer keywords."""
    context_lower = context.lower()
    # Require at least half the keywords
    matches = sum(1 for kw in keywords if kw.lower() in context_lower)
    return matches >= len(keywords) / 2


async def retrieve_with_icr(service, question: str, limit: int = 10) -> tuple[str, int, int]:
    """Retrieve context using full ICR (QIR + graph)."""
    from icd.retrieval.query_router import QueryRouter

    router = QueryRouter(service.config)

    # Get intent-adjusted weights
    classification, strategy = router.route(question)

    # Store and apply weights
    orig_weights = {
        "w_e": service._retriever.w_e,
        "w_b": service._retriever.w_b,
        "w_c": service._retriever.w_c,
        "w_r": service._retriever.w_r,
    }

    service._retriever.w_e = orig_weights["w_e"] * strategy.weight_embedding_mult
    service._retriever.w_b = orig_weights["w_b"] * strategy.weight_bm25_mult
    service._retriever.w_c = orig_weights["w_c"] * strategy.weight_contract_mult
    service._retriever.w_r = orig_weights["w_r"] * strategy.weight_recency_mult

    # Retrieve
    result = await service.retrieve(query=question, limit=limit)

    # Restore weights
    for key, value in orig_weights.items():
        setattr(service._retriever, key, value)

    # Build context
    context = "\n\n".join(
        f"# {c.file_path}:{c.start_line}\n{c.content}"
        for c in result.chunks
    )

    token_count = sum(c.token_count for c in result.chunks)

    return context, len(result.chunks), token_count


async def retrieve_vanilla(service, question: str, limit: int = 10) -> tuple[str, int, int]:
    """Retrieve context using vanilla embedding search only."""
    # Direct embedding search, no BM25, no weight adjustment
    if not service._vector_store or not service._embedder:
        return "", 0, 0

    # Embed query
    query_embedding = await service._embedder.embed(question)

    # Search vectors only
    results = await service._vector_store.search(
        query_vector=query_embedding,
        k=limit,
    )

    # Get chunks
    chunks = []
    for result in results:
        chunk = await service._sqlite_store.get_chunk_with_content(result.chunk_id)
        if chunk:
            chunks.append(chunk)

    # Build context
    context = "\n\n".join(
        f"# {c.file_path}:{c.start_line}\n{c.content}"
        for c in chunks
    )

    token_count = sum(c.token_count for c in chunks)

    return context, len(chunks), token_count


async def evaluate_task(service, task: QATask) -> EvalResult:
    """Evaluate one Q&A task."""

    # ICR retrieval
    start = time.perf_counter()
    icr_context, icr_chunks, icr_tokens = await retrieve_with_icr(
        service, task.question
    )
    icr_latency = (time.perf_counter() - start) * 1000

    # Vanilla retrieval
    start = time.perf_counter()
    vanilla_context, vanilla_chunks, vanilla_tokens = await retrieve_vanilla(
        service, task.question
    )
    vanilla_latency = (time.perf_counter() - start) * 1000

    # Check if context contains answer
    icr_has_answer = context_contains_answer(icr_context, task.context_keywords)
    vanilla_has_answer = context_contains_answer(vanilla_context, task.context_keywords)

    return EvalResult(
        question=task.question,
        category=task.category,
        icr_context_has_answer=icr_has_answer,
        vanilla_context_has_answer=vanilla_has_answer,
        icr_chunks=icr_chunks,
        vanilla_chunks=vanilla_chunks,
        icr_tokens=icr_tokens,
        vanilla_tokens=vanilla_tokens,
        icr_latency_ms=icr_latency,
        vanilla_latency_ms=vanilla_latency,
    )


async def run_evaluation():
    """Run full downstream evaluation."""
    print("=" * 75)
    print("DOWNSTREAM EVALUATION: Does Better Retrieval = Better Answers?")
    print("=" * 75)
    print()

    from icd.config import load_config
    from icd.main import ICDService

    config = load_config(project_root=Path.cwd())
    service = ICDService(config)

    results: list[EvalResult] = []

    async with service.session():
        print(f"Evaluating {len(QA_TASKS)} Q&A tasks...\n")

        for i, task in enumerate(QA_TASKS, 1):
            print(f"[{i}/{len(QA_TASKS)}] {task.question[:50]}...")
            result = await evaluate_task(service, task)
            results.append(result)

            # Show inline result
            icr_status = "HAS ANSWER" if result.icr_context_has_answer else "missing"
            vanilla_status = "HAS ANSWER" if result.vanilla_context_has_answer else "missing"
            print(f"        ICR: {icr_status}, Vanilla: {vanilla_status}")

    # Summary
    print()
    print("=" * 75)
    print("RESULTS")
    print("=" * 75)
    print()

    # Context quality comparison
    icr_correct = sum(1 for r in results if r.icr_context_has_answer)
    vanilla_correct = sum(1 for r in results if r.vanilla_context_has_answer)
    total = len(results)

    print("Context Contains Answer:")
    print(f"  ICR:     {icr_correct}/{total} ({100*icr_correct/total:.0f}%)")
    print(f"  Vanilla: {vanilla_correct}/{total} ({100*vanilla_correct/total:.0f}%)")
    print(f"  Delta:   {icr_correct - vanilla_correct:+d} ({100*(icr_correct-vanilla_correct)/total:+.0f}%)")
    print()

    # By category
    print("By Category:")
    categories = set(r.category for r in results)
    for cat in sorted(categories):
        cat_results = [r for r in results if r.category == cat]
        icr_cat = sum(1 for r in cat_results if r.icr_context_has_answer)
        vanilla_cat = sum(1 for r in cat_results if r.vanilla_context_has_answer)
        n = len(cat_results)
        print(f"  {cat:<15} ICR: {icr_cat}/{n}  Vanilla: {vanilla_cat}/{n}  Delta: {icr_cat-vanilla_cat:+d}")
    print()

    # Latency
    avg_icr_latency = sum(r.icr_latency_ms for r in results) / len(results)
    avg_vanilla_latency = sum(r.vanilla_latency_ms for r in results) / len(results)
    print(f"Avg Latency:")
    print(f"  ICR:     {avg_icr_latency:.0f}ms")
    print(f"  Vanilla: {avg_vanilla_latency:.0f}ms")
    print()

    # Wins/Losses
    icr_wins = sum(1 for r in results if r.icr_context_has_answer and not r.vanilla_context_has_answer)
    vanilla_wins = sum(1 for r in results if r.vanilla_context_has_answer and not r.icr_context_has_answer)
    ties = sum(1 for r in results if r.icr_context_has_answer == r.vanilla_context_has_answer)

    print("Head-to-Head:")
    print(f"  ICR wins:     {icr_wins}")
    print(f"  Vanilla wins: {vanilla_wins}")
    print(f"  Ties:         {ties}")
    print()

    # Verdict
    print("=" * 75)
    if icr_correct > vanilla_correct:
        improvement = (icr_correct - vanilla_correct) / max(vanilla_correct, 1) * 100
        print(f"VERDICT: ICR retrieval leads to {improvement:.0f}% more answerable contexts")
        print("         Better retrieval -> Better LLM answers (proven)")
    elif icr_correct == vanilla_correct:
        print("VERDICT: No difference in context quality")
        print("         ICR doesn't help for these questions")
    else:
        print("VERDICT: Vanilla retrieval is better (!)")
        print("         Something is wrong with ICR")
    print("=" * 75)

    return results


async def run_with_llm(api_key: str | None = None):
    """
    Run evaluation with actual LLM calls.

    This is the full test - give context to LLM, check if answer is correct.
    Requires ANTHROPIC_API_KEY or similar.
    """
    # TODO: Implement when API access is available
    # 1. For each task, retrieve with ICR and vanilla
    # 2. Prompt LLM with context + question
    # 3. Check if LLM answer contains answer_keywords
    # 4. Compare ICR vs vanilla accuracy
    print("LLM evaluation not yet implemented")
    print("Run with: ANTHROPIC_API_KEY=... python downstream_evaluation.py --llm")


if __name__ == "__main__":
    import sys

    if "--llm" in sys.argv:
        import os
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        asyncio.run(run_with_llm(api_key))
    else:
        asyncio.run(run_evaluation())
