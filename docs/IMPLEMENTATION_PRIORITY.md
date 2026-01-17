# ICR Implementation Priority: Making It Real

## Immediate Actions (This Week)

### 1. Upgrade Embedding Model (2-3 hours)

**Current:** all-MiniLM-L6-v2 (384d, 2019)
**Target:** Nomic Embed Code or Jina-code-1.5b

```bash
# Install Nomic
pip install nomic

# Or Jina
pip install sentence-transformers
```

**Changes needed:**
- `icd/src/icd/embeddings/onnx_backend.py` - Add Nomic/Jina support
- `icd/src/icd/config.py` - Update default model
- `config/default_config.yaml` - Change model_name

**Expected gain:** +20-25% retrieval quality

### 2. AST-Aware Chunking (4-6 hours)

**Current:** Text sliding window
**Target:** Tree-sitter based function/class extraction

```bash
# Install tree-sitter
pip install tree-sitter-languages
```

**New file:** `icd/src/icd/indexing/ast_chunker.py`

**Expected gain:** +4-5% retrieval quality, better chunk coherence

### 3. Cross-Encoder Reranking (2-3 hours)

**Current:** None
**Target:** ms-marco-MiniLM reranker on top-K

```bash
pip install sentence-transformers
```

**New file:** `icd/src/icd/retrieval/reranker.py`

**Expected gain:** +5-10% precision

---

## Phase 2: Agentic Retrieval (Next 2 Weeks)

### 4. CRAG Correction Layer

Add retrieval quality evaluation:
- Score each retrieved chunk for relevance
- If average < threshold, reformulate query
- Combine original + reformulated results

### 5. AST-Derived Code Graph

Build dependency graph:
- Parse imports/exports
- Track function calls
- Index class inheritance
- Use for multi-hop retrieval

---

## Phase 3: True RLM (Following Month)

### 6. Context Externalization

Make codebase a variable, not prompt content:
- `CodebaseEnvironment` class with search/read methods
- Sandboxed execution of Claude-generated code
- Sub-LLM query capability

### 7. MCP Integration

Use Programmatic Tool Calling:
- Tools callable from code execution
- Single inference pass for multiple operations
- True RLM behavior in Claude Code

---

## Files to Create/Modify

| File | Action | Priority |
|------|--------|----------|
| `icd/src/icd/embeddings/nomic_backend.py` | Create | P0 |
| `icd/src/icd/indexing/ast_chunker.py` | Create | P0 |
| `icd/src/icd/retrieval/reranker.py` | Create | P1 |
| `icd/src/icd/retrieval/crag.py` | Create | P2 |
| `icd/src/icd/graph/code_graph.py` | Create | P2 |
| `icd/src/icd/rlm/true_rlm.py` | Create | P3 |
| `icd/src/icd/rlm/codebase_env.py` | Create | P3 |

---

## Benchmarking Plan

After each phase, measure on:

1. **CodeSearchNet** (Python subset)
   - MRR@10
   - Recall@100

2. **Internal test set**
   - Create 50 queries against this repo
   - Manual relevance judgments
   - Compare pack vs RLM modes

3. **Token efficiency**
   - Tokens used per query
   - Compare to full-context baseline

---

## Honest Marketing After Each Phase

### After P0 (Embeddings + Chunking):
> "ICR uses state-of-the-art code embeddings (Nomic Embed Code) and AST-aware chunking for high-quality semantic code search, running entirely on your machine."

### After P1 (Reranking + CRAG):
> "ICR implements agentic retrieval with automatic quality correction. If initial results don't match your query, it reformulates and tries again."

### After P3 (True RLM):
> "ICR implements true RLM: Claude explores your codebase programmatically, writing code to search and analyze without loading millions of lines into context. This matches the approach from the MIT RLM paper (arXiv:2512.24601)."
