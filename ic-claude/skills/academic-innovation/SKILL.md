# Academic Innovation Advisor: AI/LLM Engineering (2026)

A research-informed skill for evaluating and designing genuinely innovative AI systems.

---

## Purpose

This skill helps you:
1. **Evaluate claims** - Is this approach actually novel or well-established?
2. **Find real innovation opportunities** - What gaps exist in current research?
3. **Design defensible systems** - Build things that survive academic scrutiny
4. **Avoid marketing pitfalls** - Don't oversell standard techniques

---

## 2026 State of the Art

### RAG Techniques (Established → Novel)

| Technique | Status | Description |
|-----------|--------|-------------|
| Vector + BM25 Hybrid | **Standard** | Industry baseline since 2024 |
| Query Decomposition | **Standard** | Well-documented in LlamaIndex, Haystack |
| Cross-encoder Reranking | **Standard** | Expected in production systems |
| CRAG (Corrective) | **Emerging** | Retrieval quality evaluation + correction |
| Self-RAG | **Emerging** | Reflection tokens for self-critique |
| FLARE | **Emerging** | Forward-looking confidence-based retrieval |
| Speculative RAG | **Emerging** | Parallel drafts with verification |
| GraphRAG | **Emerging** | Knowledge graphs + community summaries |
| True RLM | **Novel** | Context externalization + recursive LLM calls |
| Agentic RAG | **Novel** | Autonomous agents with reason-act-observe loops |

### Code-Specific Embeddings (Weak → Strong)

| Model | License | Quality | Notes |
|-------|---------|---------|-------|
| all-MiniLM-L6-v2 | Apache 2.0 | Weak (~55%) | 2019, general-purpose |
| CodeBERT/GraphCodeBERT | MIT | Weak | Obsolete, use as baseline only |
| UniXcoder | MIT | Moderate | Dated but functional |
| Jina-code-1.5b | Apache 2.0 | Strong (~78%) | Good size/quality tradeoff |
| **Nomic Embed Code** | Apache 2.0 | **Strong (81.7%)** | Best open-source option |
| Voyage-code-3 | Proprietary | Very Strong (80.8%) | API-only |
| CodeXEmbed 7B | CC-BY-NC | SOTA | Non-commercial only |

### Code Chunking (What Works)

| Approach | Impact | Recommendation |
|----------|--------|----------------|
| Text sliding window | Baseline | Avoid for code |
| AST-aware (tree-sitter) | **+4-5%** | **Use this** |
| Function-level | Good | When functions fit token limit |
| Semantic (LLM-based) | Variable | Expensive, marginal gains |

---

## Innovation Evaluation Framework

### Question 1: Is This Technique Novel?

**Check against established baselines:**
```
If technique is in {hybrid search, query decomposition, MMR diversity,
   BM25, embedding similarity, top-k retrieval, sliding window chunking}:
    → NOT NOVEL (standard IR since 2020s)

If technique is in {CRAG, Self-RAG, FLARE, GraphRAG}:
    → EMERGING (documented but not ubiquitous)

If technique involves {LLM in retrieval loop, context externalization,
   recursive LLM calls, learned confidence thresholds}:
    → POTENTIALLY NOVEL (verify against recent papers)
```

### Question 2: What Would Reviewers Say?

**ACL/EMNLP/NeurIPS rejection patterns:**
- "This is engineering + known methods without novel contributions"
- "Missing comparison to [FLARE/Self-RAG/GraphRAG]"
- "No evaluation on standard benchmarks"
- "The proposed method is a straightforward combination of existing techniques"

**To pass review, you need:**
1. Clear novel contribution (not just combination)
2. Comparison to recent strong baselines (2024-2025)
3. Evaluation on established benchmarks
4. Ablation studies showing each component's contribution

### Question 3: What Are Real Innovation Opportunities?

**Under-explored areas (2026):**

1. **Code-specific RAG benchmarks** - CodeRAG-Bench and CoIR are new; room for contribution
2. **Learned retrieval decisions** - Most systems use heuristics, not learned thresholds
3. **Static analysis integration** - Tabnine's SEM-RAG approach is not widely replicated
4. **True RLM for code** - The academic RLM paper focuses on text, not code
5. **Cross-repository retrieval** - Most tools are single-repo; multi-repo is harder
6. **Evaluation of code RAG** - How do you measure if retrieved code actually helps?

---

## Terminology Audit

### Dangerous Terms (Invite Scrutiny)

| Term | Problem | Safer Alternative |
|------|---------|-------------------|
| "Novel" | Requires proof | "Combines established techniques" |
| "State-of-the-art" | Requires benchmarks | "Competitive with recent work" |
| "Research-grade" | Implies rigor | "Production-ready" or "Well-engineered" |
| "RLM" | Academic term with specific meaning | "Iterative retrieval" or explain exactly |
| "Intelligent" | Vague | Describe the specific mechanism |

### Safe Claims

- "Uses hybrid semantic + lexical search" (accurate, verifiable)
- "Implements budget-aware context packing via 0/1 knapsack" (specific)
- "Runs entirely locally with no data egress" (differentiator)
- "Benchmarked on CodeSearchNet with X% MRR" (quantified)

---

## Design Principles for Genuine Innovation

### Principle 1: Start from Benchmarks

Before claiming innovation, know your baselines:
- **CodeSearchNet**: Standard code retrieval benchmark
- **CodeRAG-Bench**: RAG for code tasks (2024)
- **CoIR**: Code Information Retrieval benchmark (ACL 2025)
- **RepoEval**: Repository-level code completion
- **SWE-Bench**: Software engineering tasks

### Principle 2: Implement Strong Baselines First

Don't innovate on a weak foundation:
1. Use modern embeddings (Nomic, Jina, Voyage)
2. Use AST-aware chunking
3. Implement reranking
4. Then add your innovation on top

### Principle 3: Measure Everything

Innovation claims require evidence:
- Ablation: What happens if you remove each component?
- Comparison: How does it compare to published baselines?
- Error analysis: Where does it fail? Why?

### Principle 4: Be Specific About Contributions

**Weak:** "We propose a novel retrieval system"
**Strong:** "We show that AST-derived code graphs improve retrieval by 15% over embedding-only baselines on RepoEval, with 40% of gains coming from call-graph traversal"

---

## True RLM: The Real Deal

The academic RLM paper (arXiv:2512.24601) defines specific requirements:

### Required for "True RLM" Claim

1. **Context Externalization**
   - Data is NOT in the LLM prompt
   - Data is stored as a variable in an environment
   - LLM accesses data programmatically

2. **LLM-Driven Exploration**
   - LLM writes code to explore data
   - LLM decides search strategy (not hardcoded)
   - Emergent patterns (filtering, chunking, stitching)

3. **Recursive Sub-Calls**
   - `llm_query(prompt)` primitive available
   - LLM can call sub-LLMs on data snippets
   - Results aggregated programmatically

### NOT True RLM

- Query expansion with heuristic templates
- Multiple retrieval rounds with fixed logic
- LLM rewriting queries but not exploring data
- Entropy-based gating without LLM reasoning

### Minimal True RLM Implementation

```python
# This is the minimum to honestly claim "RLM-inspired"
class MinimalRLM:
    def __init__(self, data, llm_client):
        self.data = data  # NOT in prompt
        self.client = llm_client

    def query(self, question: str) -> str:
        # LLM writes exploration code
        code = self.client.generate(
            system="Write Python to explore `data` and answer the question. "
                   "Use llm_query(prompt) for sub-questions. "
                   "Store answer in `result`.",
            user=question
        )

        # Execute with data and llm_query available
        env = {
            'data': self.data,
            'llm_query': lambda p: self.client.generate(user=p),
            'result': None
        }
        exec(code, env)

        return env['result']
```

---

## Quick Reference: What's Novel in 2026?

### Novel (Defensible Innovation)
- True RLM for code (context externalization + recursive calls)
- Learned retrieval confidence thresholds (not heuristics)
- Static analysis (SEM-RAG) integration
- Cross-repository code understanding
- Agentic retrieval with self-correction loops

### Emerging (Differentiation, Not Innovation)
- GraphRAG with AST-derived graphs
- CRAG-style retrieval correction
- Speculative RAG for code generation
- FLARE-style confidence-based retrieval

### Standard (Expected, Not Differentiating)
- Hybrid vector + BM25 search
- Query decomposition
- Cross-encoder reranking
- MMR diversity selection
- AST-aware chunking
- Modern code embeddings

---

## Usage

When evaluating or designing a system, ask:

1. **Is this actually novel?** Check the established/emerging/novel lists above.

2. **Can I prove it?** Do I have benchmarks, ablations, comparisons?

3. **What would a reviewer say?** Would ACL/EMNLP accept this as a contribution?

4. **Am I using loaded terms?** Audit for "novel", "intelligent", "state-of-the-art", "RLM"

5. **What's my honest value proposition?** Privacy? Convenience? Speed? Accuracy?

---

## References

### Foundational Papers
- FLARE: "Active Retrieval Augmented Generation" (EMNLP 2023)
- Self-RAG: "Self-Reflective Retrieval Augmented Generation" (ICLR 2024)
- CRAG: "Corrective Retrieval Augmented Generation" (ICLR 2025)
- GraphRAG: Microsoft Research (2024)
- RLM: "Recursive Language Models" arXiv:2512.24601 (2025)
- Speculative RAG: "Enhancing RAG through Drafting" (ICLR 2025)

### Code-Specific
- CodeXEmbed: arXiv:2411.12644 (2024)
- CoIR Benchmark: ACL 2025
- CodeRAG-Bench: code-rag-bench.github.io
- cAST (AST Chunking): EMNLP 2025 Findings

### Industry Analysis
- Cursor architecture (Turbopuffer, tree-sitter)
- GitHub Copilot semantic indexing
- Tabnine SEM-RAG
- Sourcegraph Cody graph-based retrieval
