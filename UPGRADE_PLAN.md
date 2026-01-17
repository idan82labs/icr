# ICR Upgrade Plan: From B+ Engineering to A-Grade Research

## Current Assessment (Iteration 0)
- **Engineering Grade**: B+
- **Research Grade**: C-
- **Novelty Score**: 3/10
- **Overall**: 5/10

## Target
- **Engineering Grade**: A
- **Research Grade**: A
- **Novelty Score**: 8+/10
- **Overall**: 8+/10

---

## Critique Summary (What Makes It "Meh")

1. **Embedding**: MiniLM-L6 not trained on code
2. **CRAG**: Rule-based filtering, not LLM evaluation
3. **True RLM**: Falls back to regex heuristics
4. **Graph**: Underutilized, 1-hop only
5. **No Query Understanding**: Bag-of-words treatment
6. **No Learning**: Static system, no feedback loop
7. **Knapsack**: Ignores chunk dependencies
8. **Entropy Threshold**: Magic number, not calibrated

---

## Novel Contributions Plan

### Contribution 1: Dependency-Aware Context Packing (DAC-Pack)
**Novelty: HIGH** - No existing system models chunk dependencies in context packing

**Problem**: Current knapsack assumes chunks are independent. But:
- A function is useless without its imports
- A method needs its class definition
- A type usage needs the type definition

**Solution**: Extend knapsack to **Precedence-Constrained Knapsack**:
```
maximize: Σ u_i · x_i
subject to:
  Σ c_i · x_i ≤ B           (budget constraint)
  x_j ≤ x_i for all (i,j) ∈ D  (dependency constraint: if j selected, i must be)
```

**Algorithm**:
1. Build dependency graph from code graph
2. Compute transitive closure of dependencies
3. For each chunk, compute "bundle cost" = chunk + all dependencies
4. Solve modified knapsack with bundled items
5. Use greedy approximation for efficiency

**Implementation**: New `dependency_packer.py`

---

### Contribution 2: Adaptive Entropy Calibration (AEC)
**Novelty: MEDIUM-HIGH** - Entropy thresholds are always hand-tuned

**Problem**: `entropy_threshold = 2.5` is arbitrary. Different:
- Codebases have different entropy distributions
- Query types have different expected entropies
- Embedding models produce different score distributions

**Solution**: Per-project calibration using **percentile-based thresholds**:
```python
# During indexing, sample queries and compute entropy distribution
# Set threshold at p75 of "easy" queries (where top result is correct)
calibrated_threshold = percentile(easy_query_entropies, 75)
```

**Algorithm**:
1. Generate synthetic queries from indexed symbols
2. Run retrieval, check if symbol appears in top-k
3. Build entropy histogram for "easy" vs "hard" queries
4. Set threshold at decision boundary

**Implementation**: New `entropy_calibrator.py`

---

### Contribution 3: Multi-Hop Graph Retrieval (MHGR)
**Novelty: MEDIUM** - Existing but not in code retrieval context

**Problem**: Current graph expansion is 1-hop. Real questions need:
- "What calls the function that implements this interface?" (2-hop)
- "What are all the downstream effects of changing this type?" (transitive)

**Solution**: **Query-Guided Multi-Hop Traversal**:
```python
# For query "how is X used", traverse:
# X -> callers(X) -> callers(callers(X)) with decay
# Score nodes by: relevance_to_query * (decay ^ hops)
```

**Algorithm**:
1. Parse query intent (definition, usage, impact)
2. Start from seed nodes (initial retrieval)
3. Expand based on intent:
   - "definition" → follow CONTAINS, INHERITS edges backward
   - "usage" → follow CALLS, REFERENCES edges forward
   - "impact" → follow all edges forward transitively
4. Apply hop decay and re-rank

**Implementation**: Upgrade `graph/retriever.py`

---

### Contribution 4: Query Intent Router (QIR)
**Novelty: MEDIUM** - Intent classification for code queries

**Problem**: All queries go through same pipeline. But:
- "What is X?" → needs definitions, contracts
- "How does X work?" → needs implementations
- "Where is X used?" → needs call sites, references
- "Why does X do Y?" → needs comments, commits, docs

**Solution**: **Lightweight Intent Classifier**:
```python
intents = {
    "definition": ["what is", "define", "class", "interface"],
    "implementation": ["how does", "implement", "algorithm", "logic"],
    "usage": ["where", "used", "called", "referenced"],
    "explanation": ["why", "reason", "purpose", "design"],
}
# Route to specialized retrieval strategies
```

**Algorithm**:
1. Rule-based intent detection (fast, no LLM needed)
2. Adjust retrieval weights based on intent:
   - definition → boost contracts, reduce recency
   - implementation → boost functions, reduce contracts
   - usage → enable graph traversal, boost call sites
3. Adjust CRAG thresholds per intent

**Implementation**: New `query_router.py`

---

### Contribution 5: Contrastive Relevance Learning (CRL)
**Novelty: HIGH** - Learning from implicit feedback in code retrieval

**Problem**: No learning from user behavior. We could learn from:
- Which chunks were in successful context (user solved problem)
- Which queries led to follow-up queries (failed retrieval)

**Solution**: **Implicit Feedback Collection + Contrastive Learning**:
```python
# Collect: (query, selected_chunks, outcome)
# Positive: chunks in context when user succeeded
# Negative: chunks that were retrieved but not used

# Learn adjustment weights:
relevance_adjustment[chunk_type][query_intent] = learned_weight
```

**Algorithm**:
1. Log all retrievals with context
2. Track session outcomes (implicit: no follow-up = success)
3. Build contrastive pairs: (query, positive_chunk, negative_chunk)
4. Learn per-category adjustments using simple logistic regression
5. Apply as multipliers to retrieval scores

**Implementation**: New `feedback_learner.py`

---

### Contribution 6: Code-Native Embedding Upgrade
**Novelty: LOW** (but necessary) - Use modern code embeddings

**Problem**: MiniLM-L6 is a 2021 NLP model, not trained on code.

**Solution**: Switch default to **Nomic-Embed** or add **CodeSage**:
- Nomic: 768D, 8K context, Apache 2.0, trained on code
- CodeSage: Specifically designed for code retrieval

**Implementation**: Update `embedder.py` defaults

---

## Implementation Priority

| Component | Novelty | Impact | Effort | Priority |
|-----------|---------|--------|--------|----------|
| DAC-Pack | HIGH | HIGH | MEDIUM | 1 |
| Query Router | MEDIUM | HIGH | LOW | 2 |
| Multi-Hop Graph | MEDIUM | MEDIUM | MEDIUM | 3 |
| Entropy Calibration | MED-HIGH | MEDIUM | LOW | 4 |
| Feedback Learning | HIGH | HIGH | HIGH | 5 |
| Embedding Upgrade | LOW | HIGH | LOW | 6 |

---

## Success Metrics

### Quantitative
- Benchmark recall: 10/10 maintained
- MRR improvement: >0.90 (from 0.883)
- Novel feature coverage: 4+ genuinely new techniques
- Dependency-aware packing: Measurable improvement on multi-file queries

### Qualitative
- Can explain each component's novelty
- Would pass peer review scrutiny
- Addresses all critique points
- No "marketing" claims without substance

---

## Iteration Plan

### Iteration 1: Core Novel Components
- [ ] DAC-Pack (dependency-aware packing)
- [ ] Query Intent Router
- [ ] Embedding upgrade to Nomic

### Iteration 2: Advanced Features
- [ ] Multi-Hop Graph Retrieval
- [ ] Entropy Calibration

### Iteration 3: Learning System
- [ ] Feedback collection infrastructure
- [ ] Contrastive relevance learning

### Review After Each Iteration
- Run benchmarks
- Self-critique with research standards
- Grade honestly
- Identify remaining gaps
