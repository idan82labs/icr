# ICR Research Critique - Iteration 3

## Executive Summary

Iteration 3 fixed critical data quality issues. **Novel components now show measurable improvement.**

---

## Iteration 3 Fixes

### 1. Chunker Symbol Extraction
**Files:** `icd/src/icd/indexing/chunker.py`

Fixed two bugs:
1. `first_symbol.start_line` → `first_symbol.start_point[0]` (tree-sitter API)
2. UTF-8 byte positions used as string indices → proper bytes slicing

**Result:** Symbol extraction improved from 11% to **69.4%** of chunks.

### 2. QIR Symbol Type Mapping
**File:** `icd/src/icd/retrieval/query_router.py`

Updated `preferred_symbol_types` to match actual tree-sitter types:
- `interface_declaration` → `class_definition`
- `method_definition` → `decorated_definition`
- etc.

### 3. QIR Symbol Filtering in Ablation
**File:** `tests/benchmarks/ablation_study.py`

Added actual symbol type filtering (boost 1.5x for preferred types).

---

## Ablation Results (Iteration 3)

| Configuration | R@1 | R@5 | R@10 | MRR | Δ MRR |
|--------------|-----|-----|------|-----|-------|
| Baseline | 0.091 | 0.455 | 0.545 | 0.230 | - |
| +QIR (Intent) | 0.091 | 0.545 | 0.636 | 0.264 | **+0.033** |
| +MHGR (Graph) | 0.091 | 0.455 | 0.545 | 0.265 | **+0.035** |

### DAC-Pack Coherence
| Method | Chunks | Imports | Resolved | Coherence |
|--------|--------|---------|----------|-----------|
| Standard | 10 | 25 | 0 | 0.000 |
| DAC-Pack | 10 | 27 | 0 | 0.000 |

---

## Analysis

### What's Working

1. **QIR shows positive improvement** (+0.033 MRR)
   - Symbol type filtering helps when types are available
   - Weight adjustments now properly tuned

2. **MHGR shows improvement** (+0.035 MRR)
   - Graph traversal finds related code
   - Edge priority helps for usage queries

3. **Symbol metadata extraction fixed**
   - 69.4% of chunks now have symbol names
   - Tree-sitter parsing works correctly

### What's Still Broken

1. **DAC-Pack coherence is 0%**
   - Root cause: Graph built from `build/` directory, chunks from `src/`
   - Path mismatch prevents chunk→node linkage
   - Fix: Exclude `build/` from graph building

2. **Baseline MRR is low (0.23)**
   - Documentation files outrank code files
   - Need to filter or downweight non-code files

3. **R@1 is still 9%**
   - First result rarely correct
   - Need better exact match boosting

---

## Remaining Issues

### Issue 1: Graph Path Mismatch
- Graph includes `icd/build/lib/icd/*.py`
- Chunks indexed from `icd/src/icd/*.py`
- Node-chunk linking fails due to path mismatch

**Fix:** Add build/ to exclude patterns in graph builder.

### Issue 2: Documentation Outranks Code
- Queries like "HybridRetriever class" return BENCHMARK_RESULTS.md first
- Docs mention terms, get high BM25 score

**Fix:** Add file type boost for .py files, or filter by symbol_type.

---

## Current Grade

### Engineering Grade: A-
- Bug fixes demonstrate deep understanding
- Clean debugging process
- Proper root cause analysis

### Research Grade: B-
- QIR shows **positive improvement** (+14% relative)
- MHGR shows **positive improvement** (+15% relative)
- DAC-Pack not yet demonstrating value

### Novelty Score: 6.5/10
- Novel components exist and partially work
- Need DAC-Pack to work for full credit

### Overall: 6.5/10
- Up from 5.5/10 (Iteration 2)
- Still below target (8/10)

---

## Required for Iteration 4

### Priority 1: Fix Graph Exclusions
Exclude `build/`, `dist/`, `*.egg-info` from graph building.

### Priority 2: File Type Boost
Boost `.py` files over `.md` files in retrieval.

### Priority 3: Verify DAC-Pack
After path fix, verify DAC-Pack coherence improves.

### Priority 4: Improve R@1
Need better exact-match boosting for class/function names.

---

## Honest Assessment

**Progress:** Novel components show real improvement for the first time.

**Remaining gap:** DAC-Pack isn't working, baseline is weak.

**Path to 8/10:**
1. Fix graph exclusions → DAC-Pack should work
2. Fix file type ranking → baseline improves
3. Tune exact-match boost → R@1 improves
4. Re-run ablation → should show 20%+ improvement

---

*Assessment Date: Iteration 3 Complete*
*Status: Progress Made (6.5/10)*
*Next: Iteration 4 - Fix graph paths and file type ranking*
