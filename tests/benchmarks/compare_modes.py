"""
ICR Mode Comparison Benchmark

Compares ICR performance with different feature configurations:
1. Basic (Hybrid only)
2. With CRAG
3. With Graph expansion
4. Full (CRAG + True RLM + Graph)
"""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "icd" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "ic-mcp" / "src"))


# Queries with expected file patterns
BENCHMARK_QUERIES = [
    ("HybridRetriever class implementation", "hybrid.py"),
    ("CRAG retrieval quality evaluation", "crag.py"),
    ("TrueRLMOrchestrator class", "true_rlm.py"),
    ("CrossEncoderReranker definition", "reranker.py"),
    ("CodeGraphBuilder class", "builder.py"),
    ("entropy calculation retrieval", "entropy.py"),
    ("pack compiler knapsack", "compiler.py"),
    ("embedding backend ONNX", "embedder.py"),
    ("file watcher indexing", "watcher.py"),
    ("MMR diversity selection", "mmr.py"),
]


@dataclass
class BenchmarkConfig:
    name: str
    crag_enabled: bool
    true_rlm_enabled: bool
    graph_expansion_enabled: bool


CONFIGS = [
    BenchmarkConfig("Basic (Hybrid only)", False, False, False),
    BenchmarkConfig("With CRAG", True, False, False),
    BenchmarkConfig("With Graph", False, False, True),
    BenchmarkConfig("Full Mode", True, True, True),
]


async def run_comparison():
    """Run comparison benchmark across configurations."""
    from ic_mcp.icd_bridge import ICDBridge, RetrievalConfig

    print("=" * 80)
    print("ICR MODE COMPARISON BENCHMARK")
    print("=" * 80)
    print(f"\nQueries: {len(BENCHMARK_QUERIES)}")
    print(f"Configurations: {len(CONFIGS)}")
    print()

    all_results = {}

    for config in CONFIGS:
        print(f"\n{'='*60}")
        print(f"Configuration: {config.name}")
        print(f"  CRAG: {config.crag_enabled}, True RLM: {config.true_rlm_enabled}, Graph: {config.graph_expansion_enabled}")
        print(f"{'='*60}")

        # Initialize bridge
        bridge = ICDBridge(
            project_root=PROJECT_ROOT,
            auto_index=False,
            watch_files=False,
        )
        bridge.config = RetrievalConfig(
            crag_enabled=config.crag_enabled,
            true_rlm_enabled=config.true_rlm_enabled,
            graph_expansion_enabled=config.graph_expansion_enabled,
        )
        await bridge.initialize()

        results = []
        for query, expected_pattern in BENCHMARK_QUERIES:
            start = time.perf_counter()
            try:
                result = await bridge.retrieve(
                    query=query,
                    k=10,
                    mode="auto",
                )
                latency_ms = (time.perf_counter() - start) * 1000

                # Check results
                found = False
                found_rank = -1
                for i, chunk in enumerate(result.chunks):
                    file_path = chunk.get("file_path", "")
                    if expected_pattern.lower() in file_path.lower():
                        if not found:
                            found = True
                            found_rank = i + 1

                results.append({
                    "query": query[:40],
                    "expected": expected_pattern,
                    "found": found,
                    "rank": found_rank if found else None,
                    "latency_ms": latency_ms,
                    "mode": result.metrics.mode if hasattr(result.metrics, 'mode') else "unknown",
                })

                status = f"rank {found_rank}" if found else "NOT FOUND"
                print(f"  {query[:40]:40s} -> {status:12s} ({latency_ms:.0f}ms)")

            except Exception as e:
                print(f"  {query[:40]:40s} -> ERROR: {e}")
                results.append({
                    "query": query[:40],
                    "expected": expected_pattern,
                    "found": False,
                    "rank": None,
                    "latency_ms": 0,
                    "mode": "error",
                })

        all_results[config.name] = results

    # Summary comparison
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print()

    # Header
    print(f"{'Configuration':<30} {'Found':>8} {'MRR':>8} {'Avg Rank':>10} {'Latency':>10}")
    print("-" * 70)

    for config_name, results in all_results.items():
        found_count = sum(1 for r in results if r["found"])
        mrr = sum(1/r["rank"] for r in results if r["rank"]) / len(results)
        avg_rank = sum(r["rank"] for r in results if r["rank"]) / found_count if found_count else 0
        avg_latency = sum(r["latency_ms"] for r in results) / len(results)

        found_pct = f"{found_count}/{len(results)}"
        print(f"{config_name:<30} {found_pct:>8} {mrr:>8.3f} {avg_rank:>10.1f} {avg_latency:>10.0f}ms")

    print("-" * 70)

    # Per-query comparison
    print("\nPer-Query Comparison (Rank, - means not found):")
    print("-" * 70)
    header = f"{'Query':<30}"
    for config in CONFIGS:
        header += f" {config.name[:12]:>12}"
    print(header)
    print("-" * 70)

    for i, (query, _) in enumerate(BENCHMARK_QUERIES):
        row = f"{query[:30]:<30}"
        for config in CONFIGS:
            result = all_results[config.name][i]
            rank = result["rank"] if result["rank"] else "-"
            row += f" {str(rank):>12}"
        print(row)

    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_comparison())
