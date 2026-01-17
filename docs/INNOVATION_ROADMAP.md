# ICR Innovation Roadmap: From Standard RAG to True RLM

**Date:** January 2026
**Based on:** Comprehensive academic and industry research

---

## Executive Summary

This document outlines the path to make ICR genuinely innovative, not just well-engineered. Based on extensive research into cutting-edge RAG techniques (FLARE, Self-RAG, GraphRAG, CRAG, Speculative RAG) and the academic RLM paper (arXiv:2512.24601), we identify three tiers of innovation:

| Tier | Effort | Impact | Description |
|------|--------|--------|-------------|
| **Tier 1** | Low | High | Foundation fixes (embeddings, chunking) |
| **Tier 2** | Medium | High | Agentic retrieval (adaptive, corrective) |
| **Tier 3** | High | Very High | True RLM (context externalization, recursive calls) |

---

## Current State: What ICR Actually Does

```
Query → Embedding → Vector + BM25 → Top-K → Entropy Check →
  If high: Heuristic sub-queries → Aggregate
  → Knapsack Pack → Output
```

**Problems:**
1. Weak embedding model (MiniLM-L6-v2, 2019, 384d)
2. Text-based chunking (not AST-aware)
3. "RLM" is just query expansion, not recursive LLM calls
4. Magic thresholds (0.3, 0.7) with no principled basis
5. No reranking stage
6. No graph/dependency awareness

---

## Tier 1: Foundation Fixes (Low Effort, High Impact)

### 1.1 Upgrade Embedding Model

**Current:** all-MiniLM-L6-v2 (384d, 2019, 58.0 MTEB)
**Target:** Nomic Embed Code (Apache 2.0, 81.7% on Python CodeSearchNet)

| Model | License | Python Score | Dims | Local |
|-------|---------|-------------|------|-------|
| MiniLM-L6 (current) | Apache 2.0 | ~55% | 384 | Yes |
| **Nomic Embed Code** | Apache 2.0 | **81.7%** | 768 | Yes |
| Jina-code-1.5b | Apache 2.0 | ~78% | 1024 | Yes |
| Voyage-code-3 | Proprietary | 80.8% | 2048 | No |

**Implementation:**
```python
# config.yaml
embedding:
  model_name: nomic-ai/nomic-embed-code  # or jina-ai/jina-embeddings-v3
  dimension: 768  # or 1024 for Jina
  use_onnx: true  # if available
```

**Expected Impact:** +20-25% retrieval quality

### 1.2 AST-Aware Chunking with Tree-sitter

**Current:** Text-based sliding window
**Target:** AST-boundary respecting chunks

**Research shows:** +4-5% improvement on RepoEval and CrossCodeEval

**Implementation:**
```python
from tree_sitter_languages import get_parser

def chunk_by_ast(code: str, language: str, max_tokens: int = 512) -> list[Chunk]:
    parser = get_parser(language)
    tree = parser.parse(code.encode())

    chunks = []
    for node in tree.root_node.children:
        if node.type in ['function_definition', 'class_definition', 'method_definition']:
            chunk_text = code[node.start_byte:node.end_byte]
            if token_count(chunk_text) <= max_tokens:
                chunks.append(Chunk(
                    content=chunk_text,
                    symbol_name=extract_name(node),
                    symbol_type=node.type,
                    start_line=node.start_point[0],
                    end_line=node.end_point[0],
                ))
            else:
                # Recursively chunk large functions
                chunks.extend(chunk_node_recursive(node, code, max_tokens))
    return chunks
```

**Expected Impact:** +4-5% retrieval quality, better chunk coherence

### 1.3 Learned Entropy Thresholds

**Current:** Hardcoded 0.3 and 0.7
**Target:** Calibrated thresholds based on query complexity

**Approach:** Use a lightweight classifier or calibrated confidence scores

```python
# Instead of:
if entropy > 0.7:
    use_rlm = True

# Use:
from sklearn.calibration import CalibratedClassifierCV

class QueryComplexityClassifier:
    """Predicts query complexity: simple, moderate, complex."""

    def predict(self, query: str, initial_entropy: float) -> str:
        features = self.extract_features(query, initial_entropy)
        # Features: query length, entity count, question type, entropy
        return self.model.predict(features)  # simple/moderate/complex
```

**Expected Impact:** Better mode selection, fewer false triggers

---

## Tier 2: Agentic Retrieval (Medium Effort, High Impact)

### 2.1 Implement CRAG (Corrective RAG)

**What it does:** Evaluates retrieval quality, corrects when wrong

```python
class CorrectiveRetriever:
    async def retrieve_with_correction(self, query: str) -> RetrievalResult:
        # Step 1: Initial retrieval
        initial = await self.retriever.retrieve(query)

        # Step 2: Evaluate each result (lightweight classifier)
        evaluations = []
        for chunk in initial.chunks[:10]:
            score = self.relevance_evaluator.score(query, chunk.content)
            evaluations.append((chunk, score))

        # Step 3: Classify overall quality
        avg_relevance = mean([e[1] for e in evaluations])

        if avg_relevance > 0.7:  # Correct - use as is
            return initial
        elif avg_relevance < 0.3:  # Incorrect - try alternative
            return await self.fallback_retrieval(query)
        else:  # Ambiguous - combine
            fallback = await self.fallback_retrieval(query)
            return self.merge_results(initial, fallback)

    async def fallback_retrieval(self, query: str) -> RetrievalResult:
        # Try documentation search, different query formulation
        reformulated = await self.reformulate_query(query)
        return await self.retriever.retrieve(reformulated)
```

**Expected Impact:** +10-20% accuracy on difficult queries (per CRAG benchmarks)

### 2.2 Implement Cross-Encoder Reranking

**Current:** No reranking
**Target:** Cross-encoder reranking of top-K results

```python
from sentence_transformers import CrossEncoder

class RerankedRetriever:
    def __init__(self):
        self.bi_encoder = load_embedding_model()
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    async def retrieve(self, query: str, k: int = 20, final_k: int = 10):
        # Step 1: Bi-encoder retrieval (fast, approximate)
        candidates = await self.bi_encoder_retrieve(query, k=k * 3)

        # Step 2: Cross-encoder reranking (slow, precise)
        pairs = [(query, c.content) for c in candidates]
        scores = self.cross_encoder.predict(pairs)

        # Step 3: Return top-k after reranking
        reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [c for c, s in reranked[:final_k]]
```

**Expected Impact:** +5-10% precision (standard in production RAG systems)

### 2.3 Implement GraphRAG (AST-Derived)

**What it does:** Uses code structure (imports, calls, inheritance) for retrieval

**Research finding:** AST-derived graphs achieve 94% of LLM-extracted quality at fraction of cost

```python
import networkx as nx
from tree_sitter_languages import get_parser

class CodeGraph:
    def __init__(self, repo_path: Path):
        self.graph = nx.DiGraph()
        self.build_graph(repo_path)

    def build_graph(self, repo_path: Path):
        for file_path in repo_path.rglob("*.py"):
            self.index_file(file_path)

    def index_file(self, file_path: Path):
        code = file_path.read_text()
        tree = self.parser.parse(code.encode())

        # Extract: imports, function definitions, class definitions, calls
        for node in self.walk_tree(tree.root_node):
            if node.type == 'import_statement':
                self.add_import_edge(file_path, node)
            elif node.type == 'function_definition':
                self.add_function_node(file_path, node)
            elif node.type == 'call':
                self.add_call_edge(file_path, node)

    def retrieve_with_graph(self, query: str, entry_points: list[str]) -> list[Chunk]:
        # Start from entry points (initial vector search results)
        visited = set()
        relevant = []

        for entry in entry_points:
            # Traverse graph: imports, callers, callees
            for neighbor in self.graph.neighbors(entry):
                if neighbor not in visited:
                    visited.add(neighbor)
                    relevant.append(self.get_chunk(neighbor))

        return relevant
```

**Expected Impact:** +15% on complex multi-file queries (per GraphRAG benchmarks)

---

## Tier 3: True RLM (High Effort, Very High Impact)

### 3.1 What True RLM Means

The academic RLM paper (arXiv:2512.24601) defines RLM as:

1. **Context Externalization**: The codebase is NOT in the prompt - it's a Python variable
2. **LLM-Driven Exploration**: Claude writes code to search/filter the codebase
3. **Recursive Sub-Calls**: `llm_query()` primitive for reasoning over snippets
4. **Emergent Strategies**: Model learns to filter, chunk, and stitch

**Current ICR "RLM":**
```
Query → Heuristic sub-queries → Multiple retrievals → Aggregate
```

**True RLM:**
```
Query → Claude writes Python code → Code explores codebase →
Sub-LLM calls on snippets → Programmatic aggregation → Answer
```

### 3.2 Implementation Architecture

```python
class TrueRLM:
    """
    True RLM implementation for code retrieval.

    The codebase is NOT in the prompt. Claude writes code to explore it.
    """

    def __init__(self, repo_path: Path, llm_client):
        self.repo = CodebaseEnvironment(repo_path)
        self.client = llm_client

    async def query(self, user_query: str) -> str:
        # System prompt explains the environment
        system = """
        You have access to a Python REPL environment with a codebase.

        Available functions:
        - repo.search(pattern: str) -> list[File]: grep for patterns
        - repo.read(path: str) -> str: read a file
        - repo.list_files(glob: str) -> list[str]: list files matching glob
        - repo.get_symbol(name: str) -> Chunk: get a function/class by name
        - llm_query(prompt: str) -> str: ask a sub-question about specific code

        Write Python code to answer the user's question.
        Store your final answer in: answer = "..."

        IMPORTANT: Do NOT try to load the entire codebase. Use search and
        targeted reads. The codebase may be millions of lines.
        """

        # Get Claude's exploration code
        response = await self.client.messages.create(
            model="claude-sonnet-4-5-20250514",
            system=system,
            messages=[{"role": "user", "content": user_query}]
        )

        code = extract_python_code(response)

        # Execute in sandboxed environment
        result = await self.execute_in_sandbox(code)

        return result.get('answer', 'No answer generated')

    async def execute_in_sandbox(self, code: str) -> dict:
        """Execute Claude's code in a sandboxed environment."""

        # Globals available to the code
        env = {
            'repo': self.repo,
            'llm_query': self._sub_llm_query,
            'answer': None,
        }

        # Execute (with timeout and memory limits)
        exec(code, env)

        return {'answer': env.get('answer')}

    async def _sub_llm_query(self, prompt: str) -> str:
        """Sub-LLM call for reasoning over specific code snippets."""
        response = await self.client.messages.create(
            model="claude-haiku-3-5-20241022",  # Use cheaper model for sub-calls
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
```

### 3.3 CodebaseEnvironment Implementation

```python
class CodebaseEnvironment:
    """
    The codebase as an explorable environment.

    This is NOT loaded into the LLM context. The LLM writes code to explore it.
    """

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.index = self._build_index()

    def search(self, pattern: str, max_results: int = 20) -> list[SearchResult]:
        """Grep-like search for patterns."""
        import subprocess
        result = subprocess.run(
            ['rg', '-l', '-m', str(max_results), pattern, str(self.repo_path)],
            capture_output=True, text=True
        )
        return [SearchResult(path=p) for p in result.stdout.strip().split('\n') if p]

    def read(self, path: str, max_lines: int = 500) -> str:
        """Read a file (with limits to prevent context explosion)."""
        full_path = self.repo_path / path
        if not full_path.exists():
            return f"Error: File not found: {path}"

        content = full_path.read_text()
        lines = content.split('\n')

        if len(lines) > max_lines:
            return '\n'.join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"
        return content

    def get_symbol(self, name: str) -> Chunk | None:
        """Get a specific function/class by name from the index."""
        return self.index.get_symbol(name)

    def list_files(self, glob_pattern: str = "**/*.py") -> list[str]:
        """List files matching a glob pattern."""
        return [str(p.relative_to(self.repo_path))
                for p in self.repo_path.glob(glob_pattern)]
```

### 3.4 Example: True RLM in Action

**Query:** "How does authentication work in this codebase?"

**Claude's generated exploration code:**
```python
# Step 1: Find auth-related files
auth_files = repo.search("authenticate|authorization|login|jwt")
print(f"Found {len(auth_files)} auth-related files")

# Step 2: Analyze the main auth module
main_auth = None
for f in auth_files:
    if 'auth' in f.path and 'test' not in f.path:
        main_auth = f.path
        break

if main_auth:
    auth_code = repo.read(main_auth)
    auth_analysis = llm_query(f"""
    Analyze this authentication code and describe:
    1. What authentication method is used?
    2. Where are credentials validated?
    3. What tokens/sessions are created?

    Code:
    {auth_code[:3000]}
    """)
else:
    auth_analysis = "No main auth module found"

# Step 3: Find auth usage patterns
usage_files = repo.search("@requires_auth|authenticate\\(|login\\(")
usage_summary = f"Auth is used in {len(usage_files)} files"

# Final answer
answer = f"""
## Authentication System Analysis

### Main Module
{auth_analysis}

### Usage
{usage_summary}

### Key Files
{chr(10).join('- ' + f.path for f in auth_files[:5])}
"""
```

**Why this is TRUE RLM:**
1. The codebase (millions of lines) is NOT in the prompt
2. Claude decides what to search for ("authenticate|authorization|...")
3. Claude decides which file to analyze deeply
4. Claude makes a sub-LLM call for detailed analysis
5. Claude programmatically assembles the answer

### 3.5 Integration with MCP

Claude Code's MCP architecture supports this via **Programmatic Tool Calling**:

```python
# MCP tool definition
tools = [
    {
        "type": "code_execution_20250825",
        "name": "code_execution"
    },
    {
        "name": "repo_search",
        "description": "Search repository for patterns",
        "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}}},
        "allowed_callers": ["code_execution_20250825"]  # Callable from code!
    },
    {
        "name": "repo_read",
        "description": "Read a file from the repository",
        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
        "allowed_callers": ["code_execution_20250825"]
    },
    {
        "name": "sub_llm_query",
        "description": "Ask a sub-question about specific code",
        "input_schema": {"type": "object", "properties": {"prompt": {"type": "string"}}},
        "allowed_callers": ["code_execution_20250825"]
    }
]
```

**Benefit:** Claude writes code that calls tools, avoiding multiple inference passes.

---

## Comparison: Current vs. Proposed

| Aspect | Current ICR | Tier 1 | Tier 2 | Tier 3 (True RLM) |
|--------|-------------|--------|--------|-------------------|
| Embedding | MiniLM-L6 (2019) | Nomic Code (2024) | Same | Same |
| Chunking | Text sliding | AST-aware | Same | Same |
| Query handling | Heuristic expansion | Same | Adaptive/CRAG | LLM-written code |
| Graph awareness | None | None | AST-derived graph | Same |
| LLM in loop | No | No | Reranking | Exploration + sub-calls |
| Context management | All in prompt | Same | Same | Externalized |

---

## Implementation Priority

### Phase 1: Foundation (2-3 weeks)
- [ ] Upgrade to Nomic Embed Code (or Jina-code-1.5b)
- [ ] Implement AST-aware chunking with tree-sitter
- [ ] Add cross-encoder reranking
- [ ] Benchmark on CodeSearchNet

### Phase 2: Agentic (3-4 weeks)
- [ ] Implement CRAG correction layer
- [ ] Build AST-derived code graph
- [ ] Implement graph-enhanced retrieval
- [ ] Add adaptive retrieval (query complexity classification)

### Phase 3: True RLM (4-6 weeks)
- [ ] Build CodebaseEnvironment
- [ ] Implement sandboxed code execution
- [ ] Add sub-LLM query capability
- [ ] Integrate with MCP via Programmatic Tool Calling
- [ ] Benchmark on LongBench-v2 CodeQA

---

## Success Metrics

| Metric | Current (Estimated) | Target |
|--------|---------------------|--------|
| CodeSearchNet Python MRR | ~55% | 75%+ |
| Complex query accuracy | ~40% | 65%+ |
| RepoEval Pass@1 | Unknown | 50%+ |
| Token efficiency (vs full context) | 1x | 0.3x |

---

## Honest Claims After Implementation

### After Tier 1:
> "ICR uses state-of-the-art code embeddings and AST-aware chunking for high-quality semantic code search."

### After Tier 2:
> "ICR implements agentic retrieval with automatic correction and graph-based code understanding."

### After Tier 3:
> "ICR implements true RLM: Claude explores your codebase programmatically, making targeted queries and recursive reasoning calls without loading the full codebase into context."

---

## References

1. Zhang, Kraska, Khattab. "Recursive Language Models" arXiv:2512.24601 (2025)
2. Jiang et al. "Active Retrieval Augmented Generation" (FLARE) EMNLP 2023
3. Asai et al. "Self-RAG" ICLR 2024 Oral
4. Yan et al. "Corrective Retrieval Augmented Generation" ICLR 2025
5. Microsoft Research. "GraphRAG" 2024
6. Nomic AI. "Nomic Embed Code" 2025
7. ACL 2025. "cAST: AST-Aware Code Chunking"
