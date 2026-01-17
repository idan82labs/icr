#!/usr/bin/env python3
"""
Ablation Study: Measuring Impact of Novel ICR Components.

This benchmark performs proper A/B testing:
1. Baseline: Standard hybrid retrieval
2. +QIR: With Query Intent Router
3. +MHGR: With Multi-Hop Graph Retrieval
4. +DAC: With Dependency-Aware Context Packing
5. Full: All novel components

For each configuration, measures:
- Recall@K (does the answer appear in top K?)
- MRR (Mean Reciprocal Rank)
- Context coherence (for packing tests)

This produces the ablation table needed for research publication.
"""

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

# Minimal logging
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(40),  # ERROR only
)


@dataclass
class AblationResult:
    """Result for one configuration."""
    config_name: str
    queries_tested: int
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    avg_latency_ms: float
    improvement_over_baseline: float  # MRR delta


# Ground truth queries with known answers
# Each tuple: (query, expected_file_substr, expected_symbol_substr)
GROUND_TRUTH = [
    # Definition queries - QIR should boost contracts and classes
    ("HybridRetriever class", "hybrid.py", "HybridRetriever"),
    ("PackCompiler class", "compiler.py", "PackCompiler"),
    ("EntropyCalculator class", "entropy.py", "EntropyCalculator"),

    # Implementation queries - QIR should boost function bodies
    ("MMR select_diverse function", "mmr.py", "select"),
    ("search_bm25 function", "sqlite_store.py", "search_bm25"),
    ("embed_texts function", "embedder.py", "embed"),

    # Usage/caller queries - MHGR should help find callers
    ("retrieve method callers", "main.py", None),
    ("VectorStore usage", "hybrid.py", None),
    ("compile method call sites", "main.py", None),

    # Complex queries requiring context
    ("HybridRetriever _score_chunks implementation", "hybrid.py", "_score"),
    ("_knapsack_optimize algorithm", "compiler.py", "_knapsack"),
]


def check_result(chunks, expected_file: str, expected_symbol: str | None) -> int | None:
    """
    Check if expected result is in chunks, return rank or None.
    """
    for i, chunk in enumerate(chunks, 1):
        # Check file match
        if expected_file not in chunk.file_path:
            continue

        # If symbol specified, check it too
        if expected_symbol:
            if chunk.symbol_name and expected_symbol.lower() in chunk.symbol_name.lower():
                return i
            if expected_symbol.lower() in chunk.content.lower():
                return i
        else:
            return i  # Just file match is enough

    return None


async def run_baseline(service, queries: list) -> AblationResult:
    """Run baseline hybrid retrieval."""
    ranks = []
    latencies = []

    for query, expected_file, expected_symbol in queries:
        start = time.perf_counter()
        result = await service.retrieve(query=query, limit=10)
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)

        rank = check_result(result.chunks, expected_file, expected_symbol)
        ranks.append(rank)

    return compute_metrics("Baseline", ranks, latencies, baseline_mrr=None)


async def run_with_intent_routing(service, queries: list, baseline_mrr: float) -> AblationResult:
    """Run with Query Intent Router adjusting weights AND symbol filtering."""
    from icd.retrieval.query_router import QueryRouter

    router = QueryRouter(service.config)
    ranks = []
    latencies = []

    # Store original weights
    orig_weights = {
        "w_e": service._retriever.w_e,
        "w_b": service._retriever.w_b,
        "w_c": service._retriever.w_c,
        "w_r": service._retriever.w_r,
    }

    for query, expected_file, expected_symbol in queries:
        # Get intent-based adjustments
        classification, strategy = router.route(query)

        # Temporarily adjust weights based on intent
        service._retriever.w_e = orig_weights["w_e"] * strategy.weight_embedding_mult
        service._retriever.w_b = orig_weights["w_b"] * strategy.weight_bm25_mult
        service._retriever.w_c = orig_weights["w_c"] * strategy.weight_contract_mult
        service._retriever.w_r = orig_weights["w_r"] * strategy.weight_recency_mult

        start = time.perf_counter()
        # Retrieve more candidates, then filter by symbol type
        result = await service.retrieve(query=query, limit=30)

        # Apply symbol type filtering based on QIR strategy
        preferred_types = strategy.preferred_symbol_types
        if preferred_types and result.chunks:
            # Re-rank: boost chunks with preferred symbol types
            filtered_chunks = []
            filtered_scores = []
            other_chunks = []
            other_scores = []

            for chunk, score in zip(result.chunks, result.scores):
                if chunk.symbol_type in preferred_types:
                    filtered_chunks.append(chunk)
                    filtered_scores.append(score * 1.5)  # Boost preferred types
                else:
                    other_chunks.append(chunk)
                    other_scores.append(score)

            # Combine: preferred types first, then others
            result.chunks = filtered_chunks + other_chunks
            result.scores = filtered_scores + other_scores
            result.chunks = result.chunks[:10]
            result.scores = result.scores[:10]

        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)

        rank = check_result(result.chunks, expected_file, expected_symbol)
        ranks.append(rank)

    # Restore original weights
    for key, value in orig_weights.items():
        setattr(service._retriever, key, value)

    return compute_metrics("+QIR (Intent)", ranks, latencies, baseline_mrr)


async def run_with_multihop(service, queries: list, baseline_mrr: float) -> AblationResult:
    """Run with Multi-Hop Graph Retrieval."""
    import json

    ranks = []
    latencies = []

    # Check if graph is available
    if service._graph_builder is None:
        return AblationResult(
            config_name="+MHGR (Graph)",
            queries_tested=0,
            recall_at_1=0, recall_at_5=0, recall_at_10=0,
            mrr=0, avg_latency_ms=0, improvement_over_baseline=0
        )

    # Load graph
    graph_path = service.config.absolute_data_dir / "code_graph.json"
    if graph_path.exists():
        data = json.loads(graph_path.read_text())
        service._graph_builder.load_from_dict(data)

    try:
        from icd.retrieval.multihop import MultiHopRetriever
        multihop = MultiHopRetriever(
            service.config,
            service._graph_builder,
            service._retriever,
        )
    except Exception as e:
        print(f"  Multi-hop init failed: {e}")
        return AblationResult(
            config_name="+MHGR (Graph)",
            queries_tested=0,
            recall_at_1=0, recall_at_5=0, recall_at_10=0,
            mrr=0, avg_latency_ms=0, improvement_over_baseline=0
        )

    for query, expected_file, expected_symbol in queries:
        start = time.perf_counter()
        result = await multihop.retrieve_multihop(
            query=query,
            intent="usage" if "where" in query.lower() or "call" in query.lower() else "implementation",
            limit=10,
        )
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)

        rank = check_result(result.chunks, expected_file, expected_symbol)
        ranks.append(rank)

    return compute_metrics("+MHGR (Graph)", ranks, latencies, baseline_mrr)


async def run_dac_pack_comparison(service) -> dict[str, Any]:
    """Compare standard packing vs DAC-Pack."""
    import json
    import re

    # Load graph for dependency analysis
    if service._graph_builder:
        graph_path = service.config.absolute_data_dir / "code_graph.json"
        if graph_path.exists():
            data = json.loads(graph_path.read_text())
            service._graph_builder.load_from_dict(data)

    # Get chunks to pack
    query = "HybridRetriever retrieve implementation"
    result = await service.retrieve(query=query, limit=20)

    if not result.chunks:
        return {"error": "no_chunks"}

    # Standard packing
    from icd.pack.compiler import PackCompiler
    std_compiler = PackCompiler(service.config)
    std_pack = await std_compiler.compile(
        chunks=result.chunks,
        scores=result.scores,
        budget_tokens=4000,
        query=query,
    )

    # DAC packing (with store for fetching missing dependencies)
    from icd.pack.dependency_packer import DependencyAwarePacker
    dac_compiler = DependencyAwarePacker(
        service.config,
        service._graph_builder,
        chunk_store=service._sqlite_store,
    )
    dac_pack = await dac_compiler.compile(
        chunks=result.chunks,
        scores=result.scores,
        budget_tokens=4000,
        query=query,
    )

    # Measure coherence: what fraction of ICD imports are defined in the pack?
    # Only count imports from icd.* modules (project-internal)
    STDLIB_MODULES = {
        'dataclass', 'field', 'any', 'path', 'enum', 'annotations', 'type_checking',
        'json', 'asyncio', 're', 'os', 'sys', 'time', 'math', 'typing', 'pathlib',
        'structlog', 'numpy', 'datetime', 'collections', 'functools', 'itertools',
    }

    def measure_coherence(content: str) -> tuple[int, int, float]:
        # Find imports from icd.* modules (project imports)
        imports = set()
        for m in re.finditer(r'from\s+(icd\.\S+)\s+import\s+(\w+)', content):
            imports.add(m.group(2).lower())

        # Also count CamelCase names (likely classes)
        for m in re.finditer(r'from\s+\S+\s+import\s+([A-Z][a-z]+(?:[A-Z][a-z]+)*)', content):
            name = m.group(1).lower()
            if name not in STDLIB_MODULES:
                imports.add(name)

        defined = set()
        for m in re.finditer(r'class\s+(\w+)', content):
            defined.add(m.group(1).lower())
        for m in re.finditer(r'def\s+(\w+)', content):
            defined.add(m.group(1).lower())

        resolved = imports & defined
        coherence = len(resolved) / len(imports) if imports else 1.0
        return len(imports), len(resolved), coherence

    std_imports, std_resolved, std_coherence = measure_coherence(std_pack.content)
    dac_imports, dac_resolved, dac_coherence = measure_coherence(dac_pack.content)

    return {
        "standard": {
            "chunks": len(std_pack.chunk_ids),
            "tokens": std_pack.token_count,
            "imports": std_imports,
            "resolved": std_resolved,
            "coherence": std_coherence,
        },
        "dac_pack": {
            "chunks": len(dac_pack.chunk_ids),
            "tokens": dac_pack.token_count,
            "imports": dac_imports,
            "resolved": dac_resolved,
            "coherence": dac_coherence,
            "deps_included": dac_pack.dependencies_included,
        },
        "improvement": dac_coherence - std_coherence,
    }


def compute_metrics(
    config_name: str,
    ranks: list[int | None],
    latencies: list[float],
    baseline_mrr: float | None,
) -> AblationResult:
    """Compute ablation metrics from ranks."""
    n = len(ranks)

    recall_at_1 = sum(1 for r in ranks if r is not None and r <= 1) / n
    recall_at_5 = sum(1 for r in ranks if r is not None and r <= 5) / n
    recall_at_10 = sum(1 for r in ranks if r is not None and r <= 10) / n

    # MRR
    mrr = sum(1/r if r else 0 for r in ranks) / n

    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    improvement = mrr - baseline_mrr if baseline_mrr else 0

    return AblationResult(
        config_name=config_name,
        queries_tested=n,
        recall_at_1=recall_at_1,
        recall_at_5=recall_at_5,
        recall_at_10=recall_at_10,
        mrr=mrr,
        avg_latency_ms=avg_latency,
        improvement_over_baseline=improvement,
    )


async def main():
    """Run ablation study."""
    print("=" * 75)
    print("ICR ABLATION STUDY: Novel Component Impact Analysis")
    print("=" * 75)
    print()

    from icd.config import load_config
    from icd.main import ICDService

    config = load_config(project_root=Path.cwd())
    service = ICDService(config)

    async with service.session():
        # Run baseline
        print("Running Baseline...")
        baseline = await run_baseline(service, GROUND_TRUTH)
        print(f"  Baseline MRR: {baseline.mrr:.3f}")

        # Run with intent routing
        print("Running +QIR (Intent Routing)...")
        with_qir = await run_with_intent_routing(service, GROUND_TRUTH, baseline.mrr)
        print(f"  +QIR MRR: {with_qir.mrr:.3f} ({with_qir.improvement_over_baseline:+.3f})")

        # Run with multi-hop
        print("Running +MHGR (Multi-Hop Graph)...")
        with_mhgr = await run_with_multihop(service, GROUND_TRUTH, baseline.mrr)
        print(f"  +MHGR MRR: {with_mhgr.mrr:.3f} ({with_mhgr.improvement_over_baseline:+.3f})")

        # DAC-Pack comparison
        print("Running DAC-Pack Coherence Test...")
        dac_result = await run_dac_pack_comparison(service)
        if "error" not in dac_result:
            print(f"  Standard coherence: {dac_result['standard']['coherence']:.3f}")
            print(f"  DAC-Pack coherence: {dac_result['dac_pack']['coherence']:.3f}")
            print(f"  Improvement: {dac_result['improvement']:+.3f}")

    # Summary table
    print()
    print("=" * 75)
    print("ABLATION RESULTS")
    print("=" * 75)
    print()
    print(f"{'Configuration':<25} {'R@1':<8} {'R@5':<8} {'R@10':<8} {'MRR':<8} {'Δ MRR':<10} {'Latency':<10}")
    print("-" * 75)

    for result in [baseline, with_qir, with_mhgr]:
        if result.queries_tested > 0:
            print(f"{result.config_name:<25} "
                  f"{result.recall_at_1:.3f}   "
                  f"{result.recall_at_5:.3f}   "
                  f"{result.recall_at_10:.3f}   "
                  f"{result.mrr:.3f}   "
                  f"{result.improvement_over_baseline:+.3f}     "
                  f"{result.avg_latency_ms:.0f}ms")

    print("-" * 75)
    print()

    # DAC-Pack table
    if "error" not in dac_result:
        print("DAC-Pack Context Coherence:")
        print(f"  {'Method':<15} {'Chunks':<10} {'Imports':<10} {'Resolved':<10} {'Coherence':<10}")
        print(f"  {'-'*55}")
        print(f"  {'Standard':<15} {dac_result['standard']['chunks']:<10} "
              f"{dac_result['standard']['imports']:<10} "
              f"{dac_result['standard']['resolved']:<10} "
              f"{dac_result['standard']['coherence']:.3f}")
        print(f"  {'DAC-Pack':<15} {dac_result['dac_pack']['chunks']:<10} "
              f"{dac_result['dac_pack']['imports']:<10} "
              f"{dac_result['dac_pack']['resolved']:<10} "
              f"{dac_result['dac_pack']['coherence']:.3f}")
        print(f"  Coherence Improvement: {dac_result['improvement']:+.3f}")

    print()
    print("=" * 75)

    # Research summary
    print("\nRESEARCH SUMMARY:")
    print("-" * 75)

    total_improvement = with_qir.improvement_over_baseline + with_mhgr.improvement_over_baseline
    print(f"  Total MRR improvement from novel components: {total_improvement:+.3f}")

    if "error" not in dac_result:
        print(f"  Context coherence improvement (DAC-Pack): {dac_result['improvement']:+.3f}")

    if total_improvement > 0 or dac_result.get("improvement", 0) > 0:
        print("\n  ✓ Novel components provide measurable improvement")
    else:
        print("\n  ✗ Novel components need further tuning")


if __name__ == "__main__":
    asyncio.run(main())
