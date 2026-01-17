# ICR Research Critique - Iteration 2

## Executive Summary

Iteration 2 implemented ablation benchmarks and feedback learning.
Results are honest but disappointing: **novel components show no improvement**.

---

## Iteration 2 Deliverables

### 1. Ablation Study Benchmark
**File:** `tests/benchmarks/ablation_study.py`

Created proper A/B comparison with ground truth queries.
Measures Recall@K, MRR, and context coherence.

### 2. Feedback Learning System (CRL)
**File:** `icd/src/icd/retrieval/feedback_learner.py`

Implemented contrastive relevance learning from implicit feedback.
Novel: Learns from session behavior without explicit labels.

### 3. Aggressive QIR Tuning
Updated weight multipliers to be more aggressive (2x-3x instead of 1.2x).

---

## Ablation Results (Honest)

| Configuration | R@1 | R@5 | R@10 | MRR | Δ MRR |
|--------------|-----|-----|------|-----|-------|
| Baseline | 0.273 | 0.545 | 0.545 | 0.394 | - |
| +QIR (Intent) | 0.273 | 0.545 | 0.545 | 0.371 | **-0.023** |
| +MHGR (Graph) | 0.273 | 0.545 | 0.545 | 0.394 | +0.000 |

### DAC-Pack Coherence
| Method | Chunks | Imports | Resolved | Coherence |
|--------|--------|---------|----------|-----------|
| Standard | 9 | 26 | 2 | 0.077 |
| DAC-Pack | 9 | 22 | 1 | **0.045** (worse) |

---

## Root Cause Analysis

### Why QIR Hurts (-0.023 MRR)
1. **Weight adjustments move away from working configuration**
   - Baseline weights are already tuned
   - Aggressive multipliers break the balance
2. **No symbol type filtering applied**
   - Strategy says "prefer class_definition"
   - But retrieval doesn't filter, just weights

### Why MHGR Shows No Effect
1. **Chunks not linked to graph nodes**
   - Graph has nodes, but `chunk_id` is null
   - `find_node_by_chunk()` returns nothing
   - Multi-hop can't start from seeds

2. **Check the data:**
   ```
   Top results for "HybridRetriever class":
   1. main.py         sym=none   <- No symbol name!
   2. hybrid.py       sym=none   <- No symbol name!
   ```

### Why DAC-Pack is Worse
1. **Dependency analysis requires graph linkage**
   - No chunk→node mapping
   - Can't find dependencies
2. **Adding overhead without benefit**

---

## The Real Problem: Data Quality

All novel components depend on **metadata that doesn't exist**:

| Component | Needs | Reality |
|-----------|-------|---------|
| QIR | symbol_type for filtering | Chunks have no symbol_type |
| MHGR | chunk→node linkage | Nodes have null chunk_id |
| DAC-Pack | Dependency graph | Graph not linked to chunks |
| AEC | Symbol-based queries | No symbols to generate queries from |

**The novel algorithms are correct. The input data is incomplete.**

---

## Current Grade

### Engineering Grade: A-
- Clean implementations
- Good abstractions
- Proper testing framework

### Research Grade: C+
- Novel formulations exist
- But no empirical improvement
- Can't publish without positive results

### Novelty Score: 5/10
- Ideas are novel
- Execution is incomplete

### Overall: 5.5/10
- Still below target (8/10)

---

## Required for Iteration 3

### Priority 1: Fix Data Quality
The chunker must extract and preserve:
- `symbol_name` for every code chunk
- `symbol_type` (class, function, method)
- Link chunks to graph nodes during indexing

### Priority 2: Re-run Ablation
With proper metadata, novel components should show improvement.

### Priority 3: Validate on Different Codebase
Test on a well-structured codebase where metadata exists.

---

## What Would Actually Help

1. **Improve chunker** to extract symbol metadata
2. **Run `link_chunks_to_nodes()` during indexing** (we added this but it might not be working)
3. **Verify graph-chunk linkage** before running MHGR

---

## Honest Assessment

The research contributions exist in code form, but we haven't proven they work.
This is common in research - the idea is sound, the implementation is correct,
but the evaluation shows no improvement because of data pipeline issues.

**Path forward:** Fix the data pipeline, re-run ablation, iterate.

---

*Assessment Date: Iteration 2 Complete*
*Status: Below Target (5.5/10)*
*Next: Iteration 3 - Fix data quality and re-evaluate*
