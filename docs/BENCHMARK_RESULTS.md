# ICR Benchmark Results

## Executive Summary

Benchmark conducted on ICR's own codebase with 10 code retrieval queries.

| Configuration | Found | MRR | Avg Rank | Latency |
|---------------|-------|-----|----------|---------|
| **Basic (Hybrid only)** | 10/10 (100%) | **0.883** | **1.3** | **82ms** |
| With CRAG | 8/10 (80%) | 0.653 | 1.8 | 154ms |
| With Graph | 10/10 (100%) | 0.883 | 1.3 | 90ms |
| Full Mode | 8/10 (80%) | 0.653 | 1.8 | 177ms |

**Key Finding:** Basic hybrid retrieval currently outperforms advanced modes. CRAG's heuristic reformulation hurts precision; graph expansion shows no benefit with current chunk-node linkage.

---

## Honest Assessment

### What Works Well

1. **Hybrid Retrieval (Semantic + BM25)**
   - MRR of 0.883 is excellent for code retrieval
   - Avg rank 1.3 means correct result is usually first or second
   - 82ms latency is responsive
   - 100% recall on test queries

2. **AST-Aware Chunking**
   - Properly extracts functions/classes as semantic units
   - Symbol names in chunks help matching
   - Good token size balance (~500 tokens per chunk)

3. **Code Graph Construction**
   - Successfully built graph: 2,890 nodes, 4,972 edges
   - Captures imports, calls, inheritance relationships
   - Foundation for future improvements

### What Needs Work

1. **CRAG (Corrective RAG)**
   - **Problem:** Heuristic reformulation is poor
     - Reformulates "HybridRetriever class" → "definition of Hybrid" (loses precision)
     - Question marks cause BM25 syntax errors
   - **Problem:** Thresholds too aggressive
     - Good results marked as "incorrect" (avg_score=0.19 <= 0.3)
   - **Fix:** Need LLM-based reformulation or smarter heuristics

2. **Graph Expansion**
   - **Problem:** No benefit observed
   - **Root cause:** Chunk-to-node linkage not working
     - Log shows "No graph nodes found for retrieved chunks"
   - **Fix:** Need to link chunks to graph nodes during indexing

3. **True RLM**
   - Not triggered in tests (entropy threshold not exceeded)
   - When triggered manually, latency increases significantly
   - LLM calls required for full functionality

### Embedding Model Limitation

Using `all-MiniLM-L6-v2` (384 dimensions):
- Fast and efficient
- But limited code understanding
- Modern code embeddings (voyage-code-3, Nomic) would improve results

---

## Per-Query Results

| Query | Basic | CRAG | Graph | Full |
|-------|-------|------|-------|------|
| HybridRetriever class implementation | **rank 2** | NOT FOUND | **rank 2** | NOT FOUND |
| CRAG retrieval quality evaluation | **rank 1** | rank 5 | **rank 1** | rank 5 |
| TrueRLMOrchestrator class | **rank 1** | **rank 1** | **rank 1** | **rank 1** |
| CrossEncoderReranker definition | **rank 1** | **rank 1** | **rank 1** | **rank 1** |
| CodeGraphBuilder class | rank 3 | rank 3 | rank 3 | rank 3 |
| entropy calculation retrieval | **rank 1** | NOT FOUND | **rank 1** | NOT FOUND |
| pack compiler knapsack | **rank 1** | **rank 1** | **rank 1** | **rank 1** |
| embedding backend ONNX | **rank 1** | **rank 1** | **rank 1** | **rank 1** |
| file watcher indexing | **rank 1** | **rank 1** | **rank 1** | **rank 1** |
| MMR diversity selection | **rank 1** | **rank 1** | **rank 1** | **rank 1** |

---

## Comparison to Industry

### Estimated vs. Competitors

| Tool | Approach | Estimated Accuracy | Notes |
|------|----------|-------------------|-------|
| **ICR (Basic)** | Hybrid + AST chunking | ~70-80% | Good for conceptual queries |
| Cursor | Proprietary RAG | ~80-85% | Uses voyage-code-3, more context |
| Copilot | GPT + repo context | ~75-85% | Deep IDE integration |
| Continue.dev | Open, local | ~65-75% | Similar to ICR Basic |
| Cody | Sourcegraph | ~80-85% | Code graph advantages |

### Why ICR Could Beat Competitors

1. **Transparent scoring** - Users can see entropy, confidence metrics
2. **Budget-aware packing** - Optimal token usage for context
3. **Extensible** - CRAG/Graph/RLM can be tuned
4. **Privacy** - Fully local option

### Why ICR Currently Falls Short

1. **Small embedding model** - MiniLM vs voyage-code-3
2. **No fine-tuning** - Not trained on code specifically
3. **CRAG/Graph not tuned** - Current implementation hurts more than helps
4. **No IDE integration** - CLI-only currently

---

## Recommendations

### Immediate (High Impact, Low Effort)

1. **Disable CRAG by default** until reformulation is improved
2. **Fix query sanitization** - Strip question marks before BM25
3. **Lower entropy threshold** - Current 2.5 may be too high

### Short-Term

1. **Link chunks to graph nodes** during indexing
2. **Add voyage-code-3 support** for better code embeddings
3. **Improve CRAG heuristics** - Learn from failure patterns

### Medium-Term

1. **Benchmark on CodeSearchNet** - Standard evaluation
2. **Learn thresholds** - Replace magic numbers
3. **Cross-encoder reranking** - Currently implemented but not active

---

## Benchmark Methodology

- **Corpus:** ICR codebase itself (187 files, 937 chunks)
- **Queries:** 10 code retrieval queries with known answers
- **Metrics:** MRR (Mean Reciprocal Rank), Recall@10, Latency
- **Environment:** MacBook Air M1, Python 3.11, all-MiniLM-L6-v2

### Limitations

- Small benchmark set (10 queries)
- Self-evaluation (queries designed for ICR's codebase)
- No comparison to actual competitor outputs
- Heuristic mode only (no ANTHROPIC_API_KEY for LLM modes)

---

## Raw Output

```
COMPARISON SUMMARY
================================================================================

Configuration                     Found      MRR   Avg Rank    Latency
----------------------------------------------------------------------
Basic (Hybrid only)               10/10    0.883        1.3         82ms
With CRAG                          8/10    0.653        1.8        154ms
With Graph                        10/10    0.883        1.3         90ms
Full Mode                          8/10    0.653        1.8        177ms
----------------------------------------------------------------------
```
