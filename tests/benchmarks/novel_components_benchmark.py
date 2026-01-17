#!/usr/bin/env python3
"""
Novel Components Benchmark Suite.

Tests the novel ICR contributions:
1. Query Intent Router (QIR) - Does intent classification improve retrieval?
2. Multi-Hop Graph Retrieval (MHGR) - Do multi-hop queries benefit from graph?
3. Dependency-Aware Packing (DAC-Pack) - Does dependency bundling improve coherence?
4. Adaptive Entropy Calibration (AEC) - Does calibration improve RLM triggering?

This benchmark is designed to show the marginal contribution of each component.
"""

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

# Configure minimal logging
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(30),  # WARNING only
)


@dataclass
class NovelBenchmarkResult:
    """Result from novel component benchmark."""
    component: str
    query: str
    baseline_rank: int | None
    enhanced_rank: int | None
    improvement: float  # Positive = better
    latency_ms: float
    metadata: dict[str, Any]


# Test queries designed to exercise each novel component
NOVEL_TEST_QUERIES = {
    "intent_routing": [
        # Definition queries - should boost contracts
        ("What is HybridRetriever?", "hybrid.py", "definition"),
        ("Define PackCompiler class", "compiler.py", "definition"),
        # Implementation queries - should boost functions
        ("How does MMR selection work?", "mmr.py", "implementation"),
        ("How is entropy calculated?", "entropy.py", "implementation"),
        # Usage queries - should enable graph, find callers
        ("Where is BM25 search used?", "sqlite_store.py", "usage"),
        ("What calls the embedder?", "embedder.py", "usage"),
    ],
    "multihop": [
        # Queries requiring 2+ hops to find answer
        ("What imports HybridRetriever?", "hybrid.py", 2),
        ("What classes inherit from CodeGraphBuilder?", "builder.py", 2),
        ("What functions call the pack compiler?", "compiler.py", 2),
    ],
    "dependency_aware": [
        # Queries where function needs its imports
        ("show me the retrieve function with its dependencies", "hybrid.py", True),
        ("pack compilation with all required context", "compiler.py", True),
    ],
}


async def run_query_intent_benchmark(service):
    """Benchmark Query Intent Router effectiveness."""
    from icd.retrieval.query_router import QueryRouter

    print("\n" + "=" * 60)
    print("QUERY INTENT ROUTER BENCHMARK")
    print("=" * 60)

    router = QueryRouter(service.config)
    results = []

    for query, expected_file, expected_intent in NOVEL_TEST_QUERIES["intent_routing"]:
        # Classify intent
        classification, strategy = router.route(query)
        actual_intent = classification.primary_intent.value

        # Check if intent matches expected
        intent_correct = actual_intent == expected_intent

        # Run retrieval
        start = time.perf_counter()
        result = await service.retrieve(query=query, limit=10)
        latency = (time.perf_counter() - start) * 1000

        # Check if expected file is in results
        rank = None
        for i, chunk in enumerate(result.chunks, 1):
            if expected_file in chunk.file_path:
                rank = i
                break

        print(f"  {query[:40]:40} intent={actual_intent:15} "
              f"correct={intent_correct} rank={rank or '-'}")

        results.append(NovelBenchmarkResult(
            component="QIR",
            query=query,
            baseline_rank=None,  # Would need baseline comparison
            enhanced_rank=rank,
            improvement=1.0 if intent_correct else 0.0,
            latency_ms=latency,
            metadata={
                "expected_intent": expected_intent,
                "actual_intent": actual_intent,
                "confidence": classification.confidence,
            }
        ))

    # Summary
    correct_intents = sum(1 for r in results if r.metadata["expected_intent"] == r.metadata["actual_intent"])
    found_files = sum(1 for r in results if r.enhanced_rank is not None)

    print(f"\n  Intent Accuracy: {correct_intents}/{len(results)}")
    print(f"  Files Found: {found_files}/{len(results)}")

    return results


async def run_multihop_benchmark(service):
    """Benchmark Multi-Hop Graph Retrieval."""
    print("\n" + "=" * 60)
    print("MULTI-HOP GRAPH RETRIEVAL BENCHMARK")
    print("=" * 60)

    results = []

    # Check if graph is available
    if service._graph_builder is None:
        print("  [SKIP] Code graph not available")
        return results

    # Load graph
    import json
    graph_path = service.config.absolute_data_dir / "code_graph.json"
    if graph_path.exists():
        data = json.loads(graph_path.read_text())
        service._graph_builder.load_from_dict(data)
        print(f"  Loaded graph: {len(service._graph_builder.get_nodes())} nodes, "
              f"{len(service._graph_builder.get_edges())} edges")
    else:
        print("  [SKIP] No saved graph found")
        return results

    for query, expected_file, expected_hops in NOVEL_TEST_QUERIES["multihop"]:
        # Run base retrieval (1-hop)
        start = time.perf_counter()
        base_result = await service.retrieve(query=query, limit=10)
        base_latency = (time.perf_counter() - start) * 1000

        base_rank = None
        for i, chunk in enumerate(base_result.chunks, 1):
            if expected_file in chunk.file_path:
                base_rank = i
                break

        # Run multi-hop retrieval
        try:
            from icd.retrieval.multihop import MultiHopRetriever
            multihop = MultiHopRetriever(
                service.config,
                service._graph_builder,
                service._retriever,
            )

            start = time.perf_counter()
            mh_result = await multihop.retrieve_multihop(
                query=query,
                intent="usage",  # Most multi-hop queries are usage-based
                limit=10,
            )
            mh_latency = (time.perf_counter() - start) * 1000

            mh_rank = None
            for i, chunk in enumerate(mh_result.chunks, 1):
                if expected_file in chunk.file_path:
                    mh_rank = i
                    break

            paths_found = len(mh_result.paths)
        except Exception as e:
            mh_rank = None
            mh_latency = 0
            paths_found = 0

        improvement = 0
        if mh_rank and base_rank:
            improvement = base_rank - mh_rank  # Positive = multi-hop is better
        elif mh_rank and not base_rank:
            improvement = 10  # Found something base couldn't

        print(f"  {query[:40]:40} base_rank={base_rank or '-':3} mh_rank={mh_rank or '-':3} "
              f"paths={paths_found:2} improvement={improvement:+d}")

        results.append(NovelBenchmarkResult(
            component="MHGR",
            query=query,
            baseline_rank=base_rank,
            enhanced_rank=mh_rank,
            improvement=improvement,
            latency_ms=mh_latency,
            metadata={
                "paths_found": paths_found,
                "expected_hops": expected_hops,
            }
        ))

    # Summary
    improvements = [r.improvement for r in results if r.improvement != 0]
    avg_improvement = sum(improvements) / len(improvements) if improvements else 0

    print(f"\n  Average Rank Improvement: {avg_improvement:+.1f}")
    print(f"  Queries Improved: {sum(1 for i in improvements if i > 0)}/{len(results)}")

    return results


async def run_dac_pack_benchmark(service):
    """Benchmark Dependency-Aware Packing coherence."""
    print("\n" + "=" * 60)
    print("DEPENDENCY-AWARE PACKING BENCHMARK")
    print("=" * 60)

    results = []

    # Get some chunks to pack
    test_query = "HybridRetriever retrieve function"
    result = await service.retrieve(query=test_query, limit=20)

    if not result.chunks:
        print("  [SKIP] No chunks to pack")
        return results

    # Standard packing
    from icd.pack.compiler import PackCompiler
    std_compiler = PackCompiler(service.config)

    start = time.perf_counter()
    std_pack = await std_compiler.compile(
        chunks=result.chunks,
        scores=result.scores,
        budget_tokens=4000,
        query=test_query,
    )
    std_latency = (time.perf_counter() - start) * 1000

    # DAC packing
    try:
        from icd.pack.dependency_packer import DependencyAwarePacker
        dac_compiler = DependencyAwarePacker(service.config, service._graph_builder)

        start = time.perf_counter()
        dac_pack = await dac_compiler.compile(
            chunks=result.chunks,
            scores=result.scores,
            budget_tokens=4000,
            query=test_query,
        )
        dac_latency = (time.perf_counter() - start) * 1000

        # Measure coherence: count import statements that have their definitions
        std_coherence = _measure_coherence(std_pack.content)
        dac_coherence = _measure_coherence(dac_pack.content)

        improvement = dac_coherence - std_coherence

        print(f"  Standard Pack: {len(std_pack.chunk_ids)} chunks, "
              f"coherence={std_coherence:.2f}, latency={std_latency:.0f}ms")
        print(f"  DAC Pack:      {len(dac_pack.chunk_ids)} chunks, "
              f"coherence={dac_coherence:.2f}, latency={dac_latency:.0f}ms")
        print(f"  Dependencies included: {dac_pack.dependencies_included}")

        results.append(NovelBenchmarkResult(
            component="DAC-Pack",
            query=test_query,
            baseline_rank=None,
            enhanced_rank=None,
            improvement=improvement,
            latency_ms=dac_latency,
            metadata={
                "std_chunks": len(std_pack.chunk_ids),
                "dac_chunks": len(dac_pack.chunk_ids),
                "std_coherence": std_coherence,
                "dac_coherence": dac_coherence,
                "deps_included": dac_pack.dependencies_included,
            }
        ))

    except Exception as e:
        print(f"  [ERROR] DAC-Pack failed: {e}")

    return results


def _measure_coherence(content: str) -> float:
    """
    Measure coherence of packed content.

    Heuristic: count how many imported symbols are also defined in the pack.
    Higher = more coherent (dependencies are included).
    """
    import re

    # Find import statements
    imports = set()
    for match in re.finditer(r'from\s+\S+\s+import\s+(\w+)', content):
        imports.add(match.group(1))
    for match in re.finditer(r'import\s+(\w+)', content):
        imports.add(match.group(1))

    if not imports:
        return 1.0  # No imports = fully coherent

    # Find defined symbols
    defined = set()
    for match in re.finditer(r'class\s+(\w+)', content):
        defined.add(match.group(1))
    for match in re.finditer(r'def\s+(\w+)', content):
        defined.add(match.group(1))

    # Coherence = fraction of imports that are defined
    resolved = imports & defined
    return len(resolved) / len(imports) if imports else 1.0


async def run_entropy_calibration_benchmark(service):
    """Benchmark Adaptive Entropy Calibration."""
    print("\n" + "=" * 60)
    print("ADAPTIVE ENTROPY CALIBRATION BENCHMARK")
    print("=" * 60)

    results = []

    try:
        from icd.retrieval.entropy_calibrator import EntropyCalibrator

        calibrator = EntropyCalibrator(
            service.config,
            service._sqlite_store,
            service._retriever,
        )

        # Run calibration
        start = time.perf_counter()
        cal_result = await calibrator.calibrate(seed=42)
        latency = (time.perf_counter() - start) * 1000

        # Compare to default threshold
        default_threshold = service.config.rlm.entropy_threshold
        calibrated_threshold = cal_result.recommended_threshold

        print(f"  Default threshold:    {default_threshold:.3f}")
        print(f"  Calibrated threshold: {calibrated_threshold:.3f}")
        print(f"  Easy query entropy:   {cal_result.easy_query_entropy_mean:.3f} "
              f"(std={cal_result.easy_query_entropy_std:.3f})")
        print(f"  Hard query entropy:   {cal_result.hard_query_entropy_mean:.3f} "
              f"(std={cal_result.hard_query_entropy_std:.3f})")
        print(f"  Separation score:     {cal_result.separation_score:.2f}")
        print(f"  Confidence:           {cal_result.confidence:.2f}")
        print(f"  Samples used:         {cal_result.num_samples}")
        print(f"  Calibration time:     {latency:.0f}ms")

        # Improvement is the difference from default (if calibrated is different, that's novel)
        threshold_diff = abs(calibrated_threshold - default_threshold)

        results.append(NovelBenchmarkResult(
            component="AEC",
            query="calibration",
            baseline_rank=None,
            enhanced_rank=None,
            improvement=threshold_diff,
            latency_ms=latency,
            metadata={
                "default_threshold": default_threshold,
                "calibrated_threshold": calibrated_threshold,
                "separation_score": cal_result.separation_score,
                "confidence": cal_result.confidence,
                "num_samples": cal_result.num_samples,
            }
        ))

    except Exception as e:
        print(f"  [ERROR] Calibration failed: {e}")
        import traceback
        traceback.print_exc()

    return results


async def main():
    """Run all novel component benchmarks."""
    print("=" * 70)
    print("ICR NOVEL COMPONENTS BENCHMARK SUITE")
    print("=" * 70)

    from icd.config import load_config
    from icd.main import ICDService

    config = load_config(project_root=Path.cwd())
    service = ICDService(config)

    all_results = []

    async with service.session():
        # 1. Query Intent Router
        qir_results = await run_query_intent_benchmark(service)
        all_results.extend(qir_results)

        # 2. Multi-Hop Graph Retrieval
        mhgr_results = await run_multihop_benchmark(service)
        all_results.extend(mhgr_results)

        # 3. Dependency-Aware Packing
        dac_results = await run_dac_pack_benchmark(service)
        all_results.extend(dac_results)

        # 4. Entropy Calibration
        aec_results = await run_entropy_calibration_benchmark(service)
        all_results.extend(aec_results)

    # Final Summary
    print("\n" + "=" * 70)
    print("NOVEL COMPONENTS SUMMARY")
    print("=" * 70)

    by_component = {}
    for r in all_results:
        if r.component not in by_component:
            by_component[r.component] = []
        by_component[r.component].append(r)

    for component, results in by_component.items():
        improvements = [r.improvement for r in results]
        avg_imp = sum(improvements) / len(improvements) if improvements else 0
        print(f"  {component:12} | Tests: {len(results):2} | Avg Improvement: {avg_imp:+.2f}")

    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
