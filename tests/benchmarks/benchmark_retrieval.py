"""
ICR Retrieval Benchmarks

Comprehensive benchmark suite comparing:
1. ICR Full Mode (CRAG + True RLM + Graph)
2. ICR Basic Mode (Hybrid retrieval only)
3. Simple RAG baseline (semantic-only)
4. BM25 baseline (lexical-only)

Metrics:
- MRR (Mean Reciprocal Rank)
- NDCG@K (Normalized Discounted Cumulative Gain)
- Recall@K
- Precision@K
- Latency (p50, p95, p99)

Uses the ICR codebase itself as the evaluation corpus.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "icd" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "ic-mcp" / "src"))


# ==============================================================================
# Benchmark Data Types
# ==============================================================================

@dataclass
class BenchmarkQuery:
    """A benchmark query with ground truth."""
    query: str
    relevant_files: list[str]  # Ground truth: files that should be retrieved
    relevant_symbols: list[str]  # Ground truth: symbols that should be found
    difficulty: str  # "easy", "medium", "hard"
    category: str  # "conceptual", "symbol", "trace", "impact"


@dataclass
class RetrievalMetrics:
    """Metrics for a single retrieval run."""
    mrr: float  # Mean Reciprocal Rank (1/rank of first relevant)
    ndcg_5: float  # NDCG at 5
    ndcg_10: float  # NDCG at 10
    recall_5: float  # Recall at 5
    recall_10: float  # Recall at 10
    precision_5: float  # Precision at 5
    precision_10: float  # Precision at 10
    latency_ms: float  # Query latency
    chunks_retrieved: int
    entropy: float


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results for a configuration."""
    config_name: str
    queries_evaluated: int
    avg_mrr: float
    avg_ndcg_5: float
    avg_ndcg_10: float
    avg_recall_5: float
    avg_recall_10: float
    avg_precision_5: float
    avg_precision_10: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    per_category: dict[str, dict[str, float]] = field(default_factory=dict)
    per_difficulty: dict[str, dict[str, float]] = field(default_factory=dict)


# ==============================================================================
# Ground Truth Queries for ICR Codebase
# ==============================================================================

BENCHMARK_QUERIES = [
    # Conceptual queries (how does X work)
    BenchmarkQuery(
        query="How does the hybrid retrieval scoring work?",
        relevant_files=["icd/src/icd/retrieval/hybrid.py"],
        relevant_symbols=["HybridRetriever", "_compute_hybrid_scores"],
        difficulty="easy",
        category="conceptual",
    ),
    BenchmarkQuery(
        query="How does CRAG evaluate retrieval quality?",
        relevant_files=["icd/src/icd/retrieval/crag.py"],
        relevant_symbols=["CRAGRetriever", "RelevanceEvaluator", "_evaluate_quality"],
        difficulty="medium",
        category="conceptual",
    ),
    BenchmarkQuery(
        query="How does True RLM generate retrieval programs?",
        relevant_files=["icd/src/icd/rlm/true_rlm.py"],
        relevant_symbols=["TrueRLMOrchestrator", "_generate_program", "RLMProgram"],
        difficulty="hard",
        category="conceptual",
    ),
    BenchmarkQuery(
        query="How does entropy-based gating decide between pack and RLM mode?",
        relevant_files=["ic-mcp/src/ic_mcp/icd_bridge.py", "icd/src/icd/retrieval/entropy.py"],
        relevant_symbols=["EntropyCalculator", "entropy_threshold"],
        difficulty="medium",
        category="conceptual",
    ),

    # Symbol lookup queries
    BenchmarkQuery(
        query="Where is CrossEncoderReranker defined?",
        relevant_files=["icd/src/icd/retrieval/reranker.py"],
        relevant_symbols=["CrossEncoderReranker"],
        difficulty="easy",
        category="symbol",
    ),
    BenchmarkQuery(
        query="Find the CodeGraphBuilder class",
        relevant_files=["icd/src/icd/graph/builder.py"],
        relevant_symbols=["CodeGraphBuilder"],
        difficulty="easy",
        category="symbol",
    ),
    BenchmarkQuery(
        query="What is the GraphRetriever and where is it used?",
        relevant_files=["icd/src/icd/graph/traversal.py", "ic-mcp/src/ic_mcp/icd_bridge.py"],
        relevant_symbols=["GraphRetriever", "_graph_retriever"],
        difficulty="medium",
        category="symbol",
    ),

    # Trace/flow queries
    BenchmarkQuery(
        query="Trace the flow from user query to packed context output",
        relevant_files=[
            "ic-mcp/src/ic_mcp/tools/memory.py",
            "ic-mcp/src/ic_mcp/icd_bridge.py",
            "icd/src/icd/retrieval/hybrid.py",
        ],
        relevant_symbols=["memory_pack", "retrieve", "compile_pack"],
        difficulty="hard",
        category="trace",
    ),
    BenchmarkQuery(
        query="How does indexing flow from file to chunk to embedding?",
        relevant_files=[
            "icd/src/icd/main.py",
            "icd/src/icd/indexing/chunker.py",
            "icd/src/icd/indexing/embedder.py",
        ],
        relevant_symbols=["index_directory", "TreeSitterChunker", "EmbeddingBackend"],
        difficulty="hard",
        category="trace",
    ),

    # Impact analysis queries
    BenchmarkQuery(
        query="What would break if I change the Chunk dataclass?",
        relevant_files=[
            "icd/src/icd/retrieval/hybrid.py",
            "icd/src/icd/storage/sqlite_store.py",
            "ic-mcp/src/ic_mcp/icd_bridge.py",
        ],
        relevant_symbols=["Chunk", "chunks"],
        difficulty="hard",
        category="impact",
    ),
    BenchmarkQuery(
        query="What uses the entropy calculation?",
        relevant_files=[
            "icd/src/icd/retrieval/entropy.py",
            "icd/src/icd/retrieval/hybrid.py",
            "ic-mcp/src/ic_mcp/icd_bridge.py",
        ],
        relevant_symbols=["EntropyCalculator", "compute_entropy", "entropy"],
        difficulty="medium",
        category="impact",
    ),

    # Configuration queries
    BenchmarkQuery(
        query="How do I configure the embedding backend?",
        relevant_files=["icd/src/icd/config.py", "icd/src/icd/indexing/embedder.py"],
        relevant_symbols=["EmbeddingConfig", "EmbeddingBackend"],
        difficulty="easy",
        category="conceptual",
    ),
    BenchmarkQuery(
        query="What are the RLM configuration options?",
        relevant_files=["icd/src/icd/config.py", "ic-mcp/src/ic_mcp/icd_bridge.py"],
        relevant_symbols=["RLMConfig", "RetrievalConfig"],
        difficulty="easy",
        category="conceptual",
    ),
]


# ==============================================================================
# Metrics Calculation
# ==============================================================================

def calculate_mrr(retrieved_files: list[str], relevant_files: list[str]) -> float:
    """Calculate Mean Reciprocal Rank."""
    relevant_set = set(relevant_files)
    for rank, file_path in enumerate(retrieved_files, start=1):
        # Normalize paths for comparison
        normalized = normalize_path(file_path)
        if any(normalize_path(r) in normalized or normalized in normalize_path(r) for r in relevant_set):
            return 1.0 / rank
    return 0.0


def calculate_ndcg(retrieved_files: list[str], relevant_files: list[str], k: int) -> float:
    """Calculate NDCG at k."""
    relevant_set = set(normalize_path(f) for f in relevant_files)

    # Calculate DCG
    dcg = 0.0
    for i, file_path in enumerate(retrieved_files[:k]):
        normalized = normalize_path(file_path)
        # Binary relevance: 1 if relevant, 0 otherwise
        rel = 1.0 if any(r in normalized or normalized in r for r in relevant_set) else 0.0
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1) = 0

    # Calculate IDCG (ideal DCG)
    ideal_rels = [1.0] * min(k, len(relevant_files))
    ideal_rels.extend([0.0] * (k - len(ideal_rels)))
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))

    return dcg / idcg if idcg > 0 else 0.0


def calculate_recall(retrieved_files: list[str], relevant_files: list[str], k: int) -> float:
    """Calculate Recall at k."""
    if not relevant_files:
        return 1.0

    relevant_set = set(normalize_path(f) for f in relevant_files)
    retrieved_set = set(normalize_path(f) for f in retrieved_files[:k])

    # Count how many relevant items were retrieved
    found = 0
    for r in relevant_set:
        if any(r in ret or ret in r for ret in retrieved_set):
            found += 1

    return found / len(relevant_set)


def calculate_precision(retrieved_files: list[str], relevant_files: list[str], k: int) -> float:
    """Calculate Precision at k."""
    if k == 0:
        return 0.0

    relevant_set = set(normalize_path(f) for f in relevant_files)

    correct = 0
    for file_path in retrieved_files[:k]:
        normalized = normalize_path(file_path)
        if any(r in normalized or normalized in r for r in relevant_set):
            correct += 1

    return correct / k


def normalize_path(path: str) -> str:
    """Normalize path for comparison."""
    # Remove leading path components, keep from project root
    parts = Path(path).parts
    # Find icd or ic-mcp in path
    for i, part in enumerate(parts):
        if part in ("icd", "ic-mcp", "ic-claude", "tests"):
            return "/".join(parts[i:])
    return path


# ==============================================================================
# Benchmark Runners
# ==============================================================================

class BaseBenchmarkRunner:
    """Base class for benchmark runners."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.name = "base"

    async def setup(self) -> None:
        """Setup the retrieval system."""
        pass

    async def retrieve(self, query: str, k: int = 20) -> tuple[list[str], float, dict]:
        """
        Retrieve documents for a query.

        Returns:
            tuple of (file_paths, entropy, metadata)
        """
        raise NotImplementedError

    async def teardown(self) -> None:
        """Cleanup resources."""
        pass


class ICRFullBenchmarkRunner(BaseBenchmarkRunner):
    """ICR with all features enabled."""

    def __init__(self, project_root: Path):
        super().__init__(project_root)
        self.name = "ICR Full (CRAG + True RLM + Graph)"
        self._bridge = None

    async def setup(self) -> None:
        from ic_mcp.icd_bridge import ICDBridge, RetrievalConfig

        self._bridge = ICDBridge(
            project_root=self.project_root,
            auto_index=False,
            watch_files=False,
        )
        # Enable all features
        self._bridge.config = RetrievalConfig(
            crag_enabled=True,
            true_rlm_enabled=True,
            graph_expansion_enabled=True,
        )
        await self._bridge.initialize()

    async def retrieve(self, query: str, k: int = 20) -> tuple[list[str], float, dict]:
        result = await self._bridge.retrieve(
            query=query,
            k=k,
            mode="auto",
            use_crag=True,
            use_graph_expansion=True,
        )

        file_paths = [c["file_path"] for c in result.chunks]
        return file_paths, result.entropy, {
            "mode": result.metrics.mode,
            "crag_quality": result.metrics.crag_quality,
            "graph_nodes_expanded": result.metrics.graph_nodes_expanded,
        }

    async def teardown(self) -> None:
        if self._bridge:
            # Bridge doesn't have explicit teardown
            pass


class ICRBasicBenchmarkRunner(BaseBenchmarkRunner):
    """ICR with basic hybrid retrieval only."""

    def __init__(self, project_root: Path):
        super().__init__(project_root)
        self.name = "ICR Basic (Hybrid only)"
        self._bridge = None

    async def setup(self) -> None:
        from ic_mcp.icd_bridge import ICDBridge, RetrievalConfig

        self._bridge = ICDBridge(
            project_root=self.project_root,
            auto_index=False,
            watch_files=False,
        )
        # Disable advanced features
        self._bridge.config = RetrievalConfig(
            crag_enabled=False,
            true_rlm_enabled=False,
            graph_expansion_enabled=False,
        )
        await self._bridge.initialize()

    async def retrieve(self, query: str, k: int = 20) -> tuple[list[str], float, dict]:
        result = await self._bridge.retrieve(
            query=query,
            k=k,
            mode="pack",  # Force pack mode
            use_crag=False,
            use_graph_expansion=False,
        )

        file_paths = [c["file_path"] for c in result.chunks]
        return file_paths, result.entropy, {"mode": "pack"}

    async def teardown(self) -> None:
        pass


class SimpleRAGBenchmarkRunner(BaseBenchmarkRunner):
    """Simple semantic-only RAG baseline."""

    def __init__(self, project_root: Path):
        super().__init__(project_root)
        self.name = "Simple RAG (Semantic only)"
        self._retriever = None
        self._embedder = None

    async def setup(self) -> None:
        from icd.config import Config
        from icd.indexing.embedder import create_embedding_backend
        from icd.storage.vector_store import VectorStore
        from icd.storage.sqlite_store import SQLiteStore

        config = Config(project_root=self.project_root)

        # Load existing index
        self._sqlite = SQLiteStore(config)
        await self._sqlite.initialize()

        self._vectors = VectorStore(config)
        await self._vectors.initialize()

        self._embedder = create_embedding_backend(config)
        await self._embedder.initialize()

    async def retrieve(self, query: str, k: int = 20) -> tuple[list[str], float, dict]:
        # Embed query
        query_vec = await self._embedder.embed_single(query)

        # Search vectors only (no BM25, no boosts)
        chunk_ids, scores = await self._vectors.search(query_vec, k=k)

        # Get chunks
        file_paths = []
        for chunk_id in chunk_ids:
            chunk = await self._sqlite.get_chunk(chunk_id)
            if chunk:
                file_paths.append(chunk.file_path)

        # Simple entropy calculation
        if scores:
            probs = np.array(scores) / sum(scores)
            entropy = -sum(p * math.log(p + 1e-10) for p in probs)
        else:
            entropy = 0.0

        return file_paths, entropy, {"mode": "semantic_only"}

    async def teardown(self) -> None:
        if self._sqlite:
            await self._sqlite.close()


class BM25BenchmarkRunner(BaseBenchmarkRunner):
    """BM25-only baseline."""

    def __init__(self, project_root: Path):
        super().__init__(project_root)
        self.name = "BM25 (Lexical only)"
        self._sqlite = None

    async def setup(self) -> None:
        from icd.config import Config
        from icd.storage.sqlite_store import SQLiteStore

        config = Config(project_root=self.project_root)
        self._sqlite = SQLiteStore(config)
        await self._sqlite.initialize()

    async def retrieve(self, query: str, k: int = 20) -> tuple[list[str], float, dict]:
        # Search using FTS5
        results = await self._sqlite.search_fts(query, limit=k)

        file_paths = [r.file_path for r in results]

        # No meaningful entropy for BM25
        return file_paths, 0.0, {"mode": "bm25_only"}

    async def teardown(self) -> None:
        if self._sqlite:
            await self._sqlite.close()


# ==============================================================================
# Main Benchmark Runner
# ==============================================================================

async def run_benchmarks(
    project_root: Path,
    output_file: Path | None = None,
    verbose: bool = True,
) -> dict[str, BenchmarkResult]:
    """
    Run all benchmarks and return results.
    """
    results: dict[str, BenchmarkResult] = {}

    # Create runners
    runners = [
        ICRFullBenchmarkRunner(project_root),
        ICRBasicBenchmarkRunner(project_root),
        # SimpleRAGBenchmarkRunner(project_root),
        # BM25BenchmarkRunner(project_root),
    ]

    for runner in runners:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running benchmark: {runner.name}")
            print(f"{'='*60}")

        try:
            await runner.setup()
        except Exception as e:
            print(f"  ERROR: Setup failed - {e}")
            continue

        metrics_list: list[RetrievalMetrics] = []
        per_category: dict[str, list[RetrievalMetrics]] = {}
        per_difficulty: dict[str, list[RetrievalMetrics]] = {}

        for i, query in enumerate(BENCHMARK_QUERIES):
            if verbose:
                print(f"  [{i+1}/{len(BENCHMARK_QUERIES)}] {query.query[:50]}...")

            try:
                start_time = time.perf_counter()
                retrieved_files, entropy, metadata = await runner.retrieve(query.query)
                latency_ms = (time.perf_counter() - start_time) * 1000

                # Calculate metrics
                metrics = RetrievalMetrics(
                    mrr=calculate_mrr(retrieved_files, query.relevant_files),
                    ndcg_5=calculate_ndcg(retrieved_files, query.relevant_files, 5),
                    ndcg_10=calculate_ndcg(retrieved_files, query.relevant_files, 10),
                    recall_5=calculate_recall(retrieved_files, query.relevant_files, 5),
                    recall_10=calculate_recall(retrieved_files, query.relevant_files, 10),
                    precision_5=calculate_precision(retrieved_files, query.relevant_files, 5),
                    precision_10=calculate_precision(retrieved_files, query.relevant_files, 10),
                    latency_ms=latency_ms,
                    chunks_retrieved=len(retrieved_files),
                    entropy=entropy,
                )

                metrics_list.append(metrics)

                # Track by category and difficulty
                if query.category not in per_category:
                    per_category[query.category] = []
                per_category[query.category].append(metrics)

                if query.difficulty not in per_difficulty:
                    per_difficulty[query.difficulty] = []
                per_difficulty[query.difficulty].append(metrics)

                if verbose:
                    print(f"    MRR: {metrics.mrr:.3f}, NDCG@10: {metrics.ndcg_10:.3f}, "
                          f"Recall@10: {metrics.recall_10:.3f}, Latency: {metrics.latency_ms:.0f}ms")

            except Exception as e:
                print(f"    ERROR: {e}")
                continue

        await runner.teardown()

        # Aggregate results
        if metrics_list:
            latencies = [m.latency_ms for m in metrics_list]

            result = BenchmarkResult(
                config_name=runner.name,
                queries_evaluated=len(metrics_list),
                avg_mrr=sum(m.mrr for m in metrics_list) / len(metrics_list),
                avg_ndcg_5=sum(m.ndcg_5 for m in metrics_list) / len(metrics_list),
                avg_ndcg_10=sum(m.ndcg_10 for m in metrics_list) / len(metrics_list),
                avg_recall_5=sum(m.recall_5 for m in metrics_list) / len(metrics_list),
                avg_recall_10=sum(m.recall_10 for m in metrics_list) / len(metrics_list),
                avg_precision_5=sum(m.precision_5 for m in metrics_list) / len(metrics_list),
                avg_precision_10=sum(m.precision_10 for m in metrics_list) / len(metrics_list),
                latency_p50=np.percentile(latencies, 50),
                latency_p95=np.percentile(latencies, 95),
                latency_p99=np.percentile(latencies, 99),
                per_category={
                    cat: {
                        "mrr": sum(m.mrr for m in mlist) / len(mlist),
                        "ndcg_10": sum(m.ndcg_10 for m in mlist) / len(mlist),
                        "recall_10": sum(m.recall_10 for m in mlist) / len(mlist),
                    }
                    for cat, mlist in per_category.items()
                },
                per_difficulty={
                    diff: {
                        "mrr": sum(m.mrr for m in mlist) / len(mlist),
                        "ndcg_10": sum(m.ndcg_10 for m in mlist) / len(mlist),
                        "recall_10": sum(m.recall_10 for m in mlist) / len(mlist),
                    }
                    for diff, mlist in per_difficulty.items()
                },
            )

            results[runner.name] = result

    # Print summary
    if verbose:
        print_benchmark_summary(results)

    # Save to file
    if output_file:
        save_benchmark_results(results, output_file)

    return results


def print_benchmark_summary(results: dict[str, BenchmarkResult]) -> None:
    """Print a formatted summary of benchmark results."""
    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*80)

    # Header
    print(f"\n{'Configuration':<40} {'MRR':>8} {'NDCG@10':>8} {'Recall@10':>10} {'P50 (ms)':>10}")
    print("-" * 80)

    for name, result in results.items():
        print(f"{name:<40} {result.avg_mrr:>8.3f} {result.avg_ndcg_10:>8.3f} "
              f"{result.avg_recall_10:>10.3f} {result.latency_p50:>10.0f}")

    print("-" * 80)

    # Per-category breakdown
    print("\nPer-Category Results (NDCG@10):")
    print("-" * 80)

    categories = set()
    for result in results.values():
        categories.update(result.per_category.keys())

    header = f"{'Configuration':<30}"
    for cat in sorted(categories):
        header += f" {cat:>12}"
    print(header)
    print("-" * 80)

    for name, result in results.items():
        row = f"{name[:30]:<30}"
        for cat in sorted(categories):
            score = result.per_category.get(cat, {}).get("ndcg_10", 0.0)
            row += f" {score:>12.3f}"
        print(row)

    # Per-difficulty breakdown
    print("\nPer-Difficulty Results (NDCG@10):")
    print("-" * 80)

    difficulties = ["easy", "medium", "hard"]
    header = f"{'Configuration':<30}"
    for diff in difficulties:
        header += f" {diff:>12}"
    print(header)
    print("-" * 80)

    for name, result in results.items():
        row = f"{name[:30]:<30}"
        for diff in difficulties:
            score = result.per_difficulty.get(diff, {}).get("ndcg_10", 0.0)
            row += f" {score:>12.3f}"
        print(row)

    print("="*80)


def save_benchmark_results(results: dict[str, BenchmarkResult], output_file: Path) -> None:
    """Save benchmark results to JSON file."""
    data = {
        name: {
            "config_name": r.config_name,
            "queries_evaluated": r.queries_evaluated,
            "avg_mrr": r.avg_mrr,
            "avg_ndcg_5": r.avg_ndcg_5,
            "avg_ndcg_10": r.avg_ndcg_10,
            "avg_recall_5": r.avg_recall_5,
            "avg_recall_10": r.avg_recall_10,
            "avg_precision_5": r.avg_precision_5,
            "avg_precision_10": r.avg_precision_10,
            "latency_p50": r.latency_p50,
            "latency_p95": r.latency_p95,
            "latency_p99": r.latency_p99,
            "per_category": r.per_category,
            "per_difficulty": r.per_difficulty,
        }
        for name, r in results.items()
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(data, indent=2))
    print(f"\nResults saved to {output_file}")


# ==============================================================================
# CLI Entry Point
# ==============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ICR Retrieval Benchmarks")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "tests" / "benchmarks" / "results.json",
        help="Output file for results",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )

    args = parser.parse_args()

    asyncio.run(run_benchmarks(
        project_root=args.project_root,
        output_file=args.output,
        verbose=not args.quiet,
    ))
