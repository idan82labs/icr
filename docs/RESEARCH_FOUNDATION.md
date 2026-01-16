# ICR Research Foundation

This document explains the theoretical foundations underlying ICR's design decisions.

---

## Table of Contents

- [Introduction](#introduction)
- [Honest Assessment](#honest-assessment)
- [Prompt as Environment](#prompt-as-environment)
- [Bounded Tools and Variance Control](#bounded-tools-and-variance-control)
- [Non-Generative Aggregation](#non-generative-aggregation)
- [Mathematical Foundations](#mathematical-foundations)
- [Tail-Risk Engineering](#tail-risk-engineering)
- [References](#references)

---

## Introduction

ICR provides **intelligent context retrieval** for Claude Code through:
1. Hybrid search (semantic + BM25)
2. Budget-aware context packing (knapsack)
3. Query expansion when results are scattered (RLM boost)

---

## Honest Assessment

### What "RLM" Means in ICR

ICR uses "RLM" (Retrieval-augmented Language Model) to describe its query expansion feature. **This is NOT the same as canonical Recursive Language Models** (Zhang et al., 2024) which involve LLMs calling themselves recursively.

**ICR's RLM is:**
- Query decomposition based on heuristics (not LLM calls)
- Entropy-based gating to decide when to expand
- Score fusion to aggregate multiple retrieval passes
- Deterministic, not generative

**What it's closer to:**
- Multi-query expansion (traditional IR)
- Iterative retrieval (like FLARE/Self-RAG but without the LLM in the loop)

### ICR vs State-of-the-Art

| Feature | FLARE | Self-RAG | ICR |
|---------|-------|----------|-----|
| When to retrieve | Low-confidence tokens | Learned tokens | Entropy threshold |
| What to retrieve | Upcoming sentence prediction | On-demand | Heuristic decomposition |
| Self-critique | No | Yes | No |
| Training required | No | Yes | No |
| LLM in retrieval loop | Yes | Yes | **No** |

ICR's approach is simpler and doesn't require fine-tuned models, but it also lacks the adaptiveness of learned approaches.

---

## Design Philosophy

### Environment Model

ICR treats the codebase as an environment that Claude inspects through bounded operations:

```
Traditional LLM:
    f(prompt) -> response

ICR Paradigm:
    query = user_prompt
    results = hybrid_search(query)
    if high_entropy(results):
        sub_queries = decompose(query)
        for sq in sub_queries:
            results += hybrid_search(sq)
        results = aggregate(results)
    pack = knapsack_compile(results, budget)
    return pack
```

This is **not** true agent-environment interaction (no decision loop), but it provides:
- Bounded token usage per operation
- Deterministic results
- Traceable citations

### Why This Approach for Code?

| Property | Benefit |
|----------|---------|
| **Structure** | Symbol hierarchies enable targeted search |
| **Cross-references** | Dependencies provide context |
| **Redundancy** | Patterns repeat; one example often suffices |
| **Semantics** | Code has precise meaning; embedding captures intent |

---

## Prompt as Environment

### The Environment Model

In ICR, the "environment" (E) consists of:

```
E = {
    Repository: indexed source files
    Transcript: conversation history
    Diffs: recent changes
    Contracts: interfaces and types
    Memory: derived knowledge
}
```

### Environment Operations

The model interacts with E through bounded operations:

| Operation | Description | Bounds |
|-----------|-------------|--------|
| `env_search` | Query the environment | max 50 results |
| `env_peek` | View specific content | max 400 lines |
| `env_slice` | Extract symbol/range | max 1200 tokens |
| `env_aggregate` | Combine results | max 200 inputs |

### Benefits of Environment Abstraction

1. **Scalability**: E can grow without affecting model context
2. **Efficiency**: Only relevant information enters context
3. **Auditability**: All inspections are logged and traceable
4. **Reproducibility**: Deterministic operations enable debugging

---

## Bounded Tools and Variance Control

### Why Bounds Matter

Unbounded operations create **variance** in:
- **Latency**: Unpredictable response times
- **Cost**: Variable token consumption
- **Quality**: Inconsistent result relevance

### Variance Control Strategy

ICR implements **multi-level bounds**:

```
Level 1: Per-Operation Bounds
+-- env_peek: max 400 lines
+-- env_search: max 50 results
+-- memory_pack: max 12000 tokens

Level 2: Session Bounds
+-- RLM max steps: 12
+-- RLM max peek lines: 1200 cumulative
+-- RLM max candidates: 50

Level 3: Time Bounds
+-- Pack + Plan: 8 seconds wall clock
+-- Map-Reduce: 20 seconds wall clock
```

### Bounded Tool Design Principles

1. **Explicit Limits**: All parameters have documented min/max values
2. **Fail-Safe Defaults**: Conservative defaults prevent accidents
3. **Graceful Degradation**: Exceeding bounds triggers fallback, not failure
4. **Budget Transparency**: Remaining budget is always visible

### Example: Token Budget Enforcement

```python
def compile_pack(chunks: list[Chunk], budget: int) -> Pack:
    """
    Compile chunks into a pack within token budget.

    Uses knapsack optimization to maximize value within budget.
    """
    selected = []
    remaining = budget

    for chunk in sorted(chunks, key=lambda c: c.value / c.tokens, reverse=True):
        if chunk.tokens <= remaining:
            selected.append(chunk)
            remaining -= chunk.tokens

    return Pack(chunks=selected, tokens_used=budget - remaining)
```

---

## Non-Generative Aggregation

### The Problem with Generative Summarization

When combining multiple search results, generative summarization has issues:

1. **Hallucination Risk**: Model may add information not in sources
2. **Attribution Loss**: Hard to trace claims to specific sources
3. **Consistency**: Different runs may produce different summaries
4. **Cost**: Additional LLM calls increase latency and expense

### Non-Generative Operations

ICR uses **deterministic aggregation** operations:

| Operation | Description | Example |
|-----------|-------------|---------|
| `extract_regex` | Extract matching patterns | Find all TODO comments |
| `unique` | Deduplicate items | Unique function names |
| `sort` | Order by criteria | Sort by relevance score |
| `group_by` | Cluster by key | Group by file type |
| `count` | Count occurrences | Count usages of function |
| `top_k` | Select top items | Top 10 most similar |
| `join_on` | Combine on key | Join chunks with metadata |
| `diff_sets` | Set difference | New vs old functions |

### Benefits of Non-Generative Aggregation

1. **Trustworthy**: Results are exactly what's in the sources
2. **Traceable**: Every item links to its origin
3. **Reproducible**: Same inputs always produce same outputs
4. **Fast**: No LLM calls needed for aggregation

### When Generative Aggregation is Used

ICR reserves generative operations for:
- Final synthesis (user-facing response)
- RLM plan generation (internal reasoning)
- Map phase of map-reduce (with explicit prompts)

---

## Mathematical Foundations

### Hybrid Scoring Formula

ICR combines multiple signals into a unified relevance score:

```
score(chunk, query) = w_e * sim_embedding(chunk, query)
                    + w_b * score_bm25(chunk, query)
                    + w_r * decay_recency(chunk)
                    + w_c * is_contract(chunk)
                    + w_f * in_focus(chunk)
                    + w_p * is_pinned(chunk)
```

**Where:**
- `sim_embedding`: Cosine similarity of embeddings
- `score_bm25`: BM25 lexical relevance score
- `decay_recency`: Time-based decay function
- `is_contract`: Binary indicator for contracts/interfaces
- `in_focus`: Binary indicator for focus path membership
- `is_pinned`: Binary indicator for pinned items

**Default Weights:**

| Weight | Symbol | Default | Description |
|--------|--------|---------|-------------|
| w_e | Embedding | 0.4 | Semantic similarity |
| w_b | BM25 | 0.3 | Lexical matching |
| w_r | Recency | 0.1 | Recent modifications |
| w_c | Contract | 0.1 | Interface/type boost |
| w_f | Focus | 0.05 | Path-based priority |
| w_p | Pinned | 0.05 | User-pinned items |

### Recency Decay Function

```python
def recency_decay(chunk: Chunk, tau_days: float = 30.0) -> float:
    """
    Exponential decay based on modification time.

    decay = exp(-delta_days / tau)

    Where:
    - delta_days: days since last modification
    - tau: decay time constant (half-life ≈ 0.693 * tau)
    """
    import math
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    mtime = chunk.modified_at
    delta_days = (now - mtime).total_seconds() / 86400

    return math.exp(-delta_days / tau_days)
```

### MMR Diversity Selection

**Maximal Marginal Relevance (MMR)** balances relevance with diversity:

```
MMR(chunk) = lambda * sim(chunk, query) - (1 - lambda) * max(sim(chunk, selected))
```

**Where:**
- `lambda`: Trade-off parameter (0.7 default)
- `sim(chunk, query)`: Relevance to query
- `max(sim(chunk, selected))`: Maximum similarity to already-selected chunks

**Algorithm:**

```python
def mmr_select(candidates: list[Chunk], query_embedding: np.ndarray,
               k: int, lambda_: float = 0.7) -> list[Chunk]:
    """
    Select k chunks using MMR for diversity.
    """
    selected = []
    remaining = set(range(len(candidates)))

    for _ in range(k):
        if not remaining:
            break

        best_score = -float('inf')
        best_idx = None

        for idx in remaining:
            chunk = candidates[idx]

            # Relevance to query
            relevance = cosine_sim(chunk.embedding, query_embedding)

            # Maximum similarity to selected (diversity penalty)
            if selected:
                max_sim = max(
                    cosine_sim(chunk.embedding, s.embedding)
                    for s in selected
                )
            else:
                max_sim = 0

            # MMR score
            score = lambda_ * relevance - (1 - lambda_) * max_sim

            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is not None:
            selected.append(candidates[best_idx])
            remaining.remove(best_idx)

    return selected
```

### Retrieval Entropy Computation

**Retrieval entropy** measures uncertainty in the retrieval result:

```
H(scores) = -sum(p_i * log(p_i))

Where p_i = softmax(scores / temperature)
```

**Interpretation:**
- **Low entropy** (< 2.5): Peaked distribution, clear retrieval signal -> Pack mode
- **High entropy** (>= 2.5): Flat distribution, uncertain retrieval -> RLM mode

**Implementation:**

```python
import numpy as np

def retrieval_entropy(scores: list[float], temperature: float = 1.0) -> float:
    """
    Compute Shannon entropy of score distribution.

    Parameters:
    - scores: Relevance scores from retrieval
    - temperature: Softmax temperature (higher = more uniform)

    Returns:
    - entropy: Bits of uncertainty in the distribution
    """
    scores = np.array(scores)

    # Softmax to convert to probabilities
    scores_scaled = scores / temperature
    exp_scores = np.exp(scores_scaled - np.max(scores_scaled))  # Numerical stability
    probs = exp_scores / exp_scores.sum()

    # Shannon entropy
    entropy = -np.sum(probs * np.log2(probs + 1e-10))

    return float(entropy)
```

### Knapsack Optimization

Pack compilation uses **0/1 knapsack** to maximize value within token budget:

```
maximize: sum(value_i * x_i)
subject to: sum(tokens_i * x_i) <= budget
            x_i in {0, 1}
```

**Greedy Approximation (used for speed):**

```python
def knapsack_pack(chunks: list[Chunk], budget: int) -> list[Chunk]:
    """
    Greedy knapsack approximation for pack compilation.

    Uses value/weight ratio for selection order.
    Provides 1/2 approximation guarantee.
    """
    # Sort by value density (value per token)
    sorted_chunks = sorted(
        chunks,
        key=lambda c: c.score / c.token_count,
        reverse=True
    )

    selected = []
    remaining_budget = budget

    for chunk in sorted_chunks:
        if chunk.token_count <= remaining_budget:
            selected.append(chunk)
            remaining_budget -= chunk.token_count

    return selected
```

**Value Function:**

```python
def chunk_value(chunk: Chunk, query_context: QueryContext) -> float:
    """
    Compute value of including a chunk in the pack.

    Combines relevance score with strategic factors.
    """
    base_value = chunk.score  # From hybrid scoring

    # Boost for contracts (they explain structure)
    if chunk.is_contract:
        base_value *= 1.5

    # Boost for pinned items (user deemed important)
    if chunk.is_pinned:
        base_value *= 2.0

    # Penalty for redundancy with already-selected
    redundancy_penalty = compute_redundancy(chunk, query_context.selected)
    base_value *= (1.0 - 0.5 * redundancy_penalty)

    return base_value
```

---

## Tail-Risk Engineering

### The Problem of Tail Cases

In production systems, the 99th percentile matters as much as the median:

| Metric | P50 (Typical) | P95 (Stress) | P99 (Tail) |
|--------|---------------|--------------|------------|
| Latency | 70ms | 150ms | 500ms |
| Tokens | 2000 | 6000 | 12000 |
| Results | 15 | 40 | 50 |

### ICR's Tail-Risk Mitigations

1. **Hard Timeouts**: Wall-clock limits on all operations
2. **Budget Caps**: Absolute maxima that cannot be exceeded
3. **Graceful Fallback**: Exceed threshold -> switch to simpler mode
4. **Progressive Enhancement**: Start conservative, expand if within budget

### Timeout Hierarchy

```python
TIMEOUT_CONFIG = {
    # Per-operation timeouts
    "embedding_local": 100,      # ms
    "embedding_remote": 5000,    # ms
    "ann_lookup": 100,           # ms
    "fts_query": 100,            # ms
    "pack_compile": 500,         # ms

    # Session timeouts
    "pack_mode_total": 8000,     # ms
    "rlm_mode_total": 20000,     # ms

    # Hard abort (no recovery)
    "hard_abort": 30000,         # ms
}
```

### Fallback Strategy

```python
def execute_with_fallback(query: str, context: Context) -> Result:
    """
    Execute query with progressive fallback on resource exhaustion.
    """
    try:
        # Attempt full retrieval
        mode = select_mode(query, context)

        if mode == "rlm":
            try:
                return execute_rlm(query, context, timeout=20000)
            except TimeoutError:
                # Fallback to pack mode
                logger.warning("RLM timeout, falling back to pack mode")
                return execute_pack(query, context, timeout=8000)

        else:
            return execute_pack(query, context, timeout=8000)

    except TimeoutError:
        # Final fallback: return cached/stale results
        logger.error("All modes timed out, returning cached results")
        return get_cached_results(query, context)
```

---

## References

### Primary Research

1. **Recursive Language Models (RLM)** - The foundational paper proposing prompts as external environments for language model interaction.

2. **Maximal Marginal Relevance (MMR)** - Carbonell & Goldstein, 1998. "The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries."

3. **HNSW** - Malkov & Yashunin, 2018. "Efficient and Robust Approximate Nearest Neighbor using Hierarchical Navigable Small World Graphs."

4. **BM25** - Robertson & Zaragoza, 2009. "The Probabilistic Relevance Framework: BM25 and Beyond."

### Supporting Work

5. **Sentence Transformers** - Reimers & Gurevych, 2019. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks."

6. **Tree-sitter** - Brunsfeld et al. "Tree-sitter: An incremental parsing system for programming tools."

7. **Model Context Protocol (MCP)** - Anthropic. Specification for tool integration with language models.

### Implementation References

8. **hnswlib** - Python bindings for the HNSW algorithm.

9. **SQLite FTS5** - Full-text search extension for SQLite.

10. **ONNX Runtime** - Cross-platform inference for ONNX models.

---

## Further Reading

- [ARCHITECTURE.md](ARCHITECTURE.md): System architecture and component details
- [API_REFERENCE.md](API_REFERENCE.md): Complete tool documentation
- [CONFIGURATION.md](CONFIGURATION.md): Tuning parameters explained
