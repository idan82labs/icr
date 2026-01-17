"""
Quick ICR Benchmark - Debug Version

A simpler benchmark that prints what's being retrieved vs what's expected.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "icd" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "ic-mcp" / "src"))


# Simple queries with expected file patterns
QUICK_QUERIES = [
    ("HybridRetriever class definition", "hybrid.py"),
    ("CRAG retrieval quality evaluation", "crag.py"),
    ("TrueRLMOrchestrator implementation", "true_rlm.py"),
    ("CrossEncoderReranker definition", "reranker.py"),
    ("CodeGraphBuilder class", "builder.py"),
    ("entropy calculation code", "entropy.py"),
    ("pack compiler implementation", "compiler.py"),
    ("embedding backend", "embedder.py"),
]


async def run_quick_benchmark():
    """Run a quick benchmark with debug output."""
    from ic_mcp.icd_bridge import ICDBridge, RetrievalConfig

    print("=" * 70)
    print("ICR Quick Benchmark")
    print("=" * 70)

    # Initialize bridge
    print("\nInitializing ICR...")
    bridge = ICDBridge(
        project_root=PROJECT_ROOT,
        auto_index=False,
        watch_files=False,
    )
    bridge.config = RetrievalConfig(
        crag_enabled=False,  # Disable CRAG for faster results
        true_rlm_enabled=False,  # Disable True RLM
        graph_expansion_enabled=False,  # Disable graph expansion
    )
    await bridge.initialize()

    print(f"Initialized. Running {len(QUICK_QUERIES)} queries.\n")

    results = []
    for query, expected_pattern in QUICK_QUERIES:
        print(f"Query: {query}")
        print(f"Expected pattern: {expected_pattern}")

        start = time.perf_counter()
        result = await bridge.retrieve(
            query=query,
            k=10,
            mode="pack",
            use_crag=False,
            use_graph_expansion=False,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        # Check results
        found = False
        found_rank = -1
        retrieved_files = []
        for i, chunk in enumerate(result.chunks):
            file_path = chunk.get("file_path", "")
            file_name = Path(file_path).name
            retrieved_files.append(file_name)
            if expected_pattern.lower() in file_path.lower():
                if not found:
                    found = True
                    found_rank = i + 1

        # Print results
        status = f"FOUND at rank {found_rank}" if found else "NOT FOUND"
        print(f"  Status: {status}")
        print(f"  Latency: {latency_ms:.0f}ms")
        print(f"  Top 5 files: {retrieved_files[:5]}")
        print()

        results.append({
            "query": query,
            "expected": expected_pattern,
            "found": found,
            "rank": found_rank if found else None,
            "latency_ms": latency_ms,
        })

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    found_count = sum(1 for r in results if r["found"])
    avg_rank = sum(r["rank"] for r in results if r["rank"]) / found_count if found_count else 0
    avg_latency = sum(r["latency_ms"] for r in results) / len(results)

    print(f"Queries: {len(results)}")
    print(f"Found: {found_count}/{len(results)} ({100*found_count/len(results):.0f}%)")
    print(f"Avg rank when found: {avg_rank:.1f}")
    print(f"Avg latency: {avg_latency:.0f}ms")

    # MRR calculation
    mrr = sum(1/r["rank"] for r in results if r["rank"]) / len(results)
    print(f"MRR: {mrr:.3f}")

    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_quick_benchmark())
