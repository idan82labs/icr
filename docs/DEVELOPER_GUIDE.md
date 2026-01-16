# ICR Developer Guide

This guide provides everything developers need to contribute to ICR, including development environment setup, project structure, coding standards, and contribution guidelines.

---

## Table of Contents

- [Development Environment Setup](#development-environment-setup)
- [Project Structure](#project-structure)
- [Component Overview](#component-overview)
- [How to Add New Tools](#how-to-add-new-tools)
- [Testing Guidelines](#testing-guidelines)
- [Code Style Requirements](#code-style-requirements)
- [Contributing Guidelines](#contributing-guidelines)
- [Release Process](#release-process)

---

## Development Environment Setup

### Prerequisites

- **Python 3.10+**: Required for all components
- **Git**: Version control
- **Make** (optional): Build automation
- **Docker** (optional): For containerized testing

### Clone the Repository

```bash
git clone https://github.com/icr/icr.git
cd icr
```

### Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate (Unix/macOS)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

### Install Dependencies

```bash
# Install all packages in development mode
pip install -e "./icd[dev]"
pip install -e "./ic-mcp[dev]"

# Or use make
make install-dev
```

### Install Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Set up hooks
pre-commit install
```

### Verify Setup

```bash
# Run tests
pytest

# Run linters
ruff check .
mypy .

# Run formatter check
black --check .
```

### IDE Configuration

#### VS Code

Recommended extensions:
- Python (Microsoft)
- Pylance
- Ruff
- Black Formatter

Settings (`.vscode/settings.json`):
```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
    "python.analysis.typeCheckingMode": "strict",
    "editor.formatOnSave": true,
    "[python]": {
        "editor.defaultFormatter": "ms-python.black-formatter"
    },
    "ruff.enable": true
}
```

#### PyCharm

1. Set Project Interpreter to `.venv`
2. Enable Ruff integration
3. Enable Black as external tool
4. Set line length to 100

---

## Project Structure

```
icr/
+-- icd/                        # Daemon (data plane)
|   +-- src/
|   |   +-- icd/
|   |       +-- __init__.py
|   |       +-- main.py         # Entry point
|   |       +-- config.py       # Configuration
|   |       +-- storage/        # Storage layer
|   |       +-- indexing/       # Indexing components
|   |       +-- retrieval/      # Retrieval components
|   |       +-- pack/           # Pack compilation
|   |       +-- rlm/            # RLM execution
|   |       +-- metrics/        # Telemetry
|   +-- tests/
|   +-- pyproject.toml
|
+-- ic-mcp/                     # MCP Server (tool plane)
|   +-- src/
|   |   +-- ic_mcp/
|   |       +-- __init__.py
|   |       +-- server.py       # MCP server main
|   |       +-- tools/          # Tool implementations
|   |       +-- schemas/        # Input/output schemas
|   |       +-- transport/      # Transport layer
|   +-- tests/
|   +-- pyproject.toml
|
+-- ic-claude/                  # Claude Code plugin (behavior plane)
|   +-- plugin.json             # Plugin manifest
|   +-- commands/               # Command definitions
|   +-- hooks/                  # Hook configurations
|   +-- scripts/                # Hook scripts
|
+-- docs/                       # Documentation
+-- examples/                   # Example usage
+-- scripts/                    # Development scripts
+-- Makefile                    # Build automation
+-- README.md                   # Project readme
```

---

## Component Overview

### icd (Daemon)

The daemon handles all data operations. Key modules:

| Module | Responsibility |
|--------|----------------|
| `storage/sqlite_store.py` | SQLite operations, FTS5 indexing |
| `storage/vector_store.py` | HNSW vector index management |
| `indexing/watcher.py` | File system watching with debouncing |
| `indexing/chunker.py` | Tree-sitter based code chunking |
| `indexing/embedder.py` | Embedding generation (local/remote) |
| `retrieval/hybrid.py` | Combined semantic + lexical retrieval |
| `retrieval/mmr.py` | Diversity-aware result selection |
| `pack/compiler.py` | Knapsack-based pack assembly |
| `rlm/planner.py` | RLM plan generation |

### ic-mcp (MCP Server)

The MCP server exposes tools to Claude Code. Key modules:

| Module | Responsibility |
|--------|----------------|
| `server.py` | MCP protocol handling, tool dispatch |
| `tools/memory.py` | memory_pack, memory_pin, etc. |
| `tools/env.py` | env_search, env_peek, etc. |
| `tools/project.py` | project_map, project_impact, etc. |
| `tools/rlm.py` | rlm_plan, rlm_map_reduce |
| `schemas/inputs.py` | Pydantic input validation |
| `schemas/outputs.py` | Pydantic output serialization |

### ic-claude (Plugin)

The plugin integrates with Claude Code:

| File | Purpose |
|------|---------|
| `plugin.json` | Plugin manifest and metadata |
| `hooks/hooks.json` | Hook configuration |
| `scripts/ic-hook-*.py` | Hook implementation scripts |
| `commands/ic.md` | /ic command definition |

---

## How to Add New Tools

### Step 1: Define Input Schema

Create or update `ic-mcp/src/ic_mcp/schemas/inputs.py`:

```python
from pydantic import BaseModel, Field
from typing import Literal

class MyNewToolInput(BaseModel):
    """Input schema for my_new_tool."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The query to process",
    )
    mode: Literal["fast", "thorough"] = Field(
        default="fast",
        description="Processing mode",
    )
    limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum results (1-100)",
    )
```

### Step 2: Define Output Schema

Create or update `ic-mcp/src/ic_mcp/schemas/outputs.py`:

```python
from pydantic import BaseModel, Field
from typing import List

class MyNewToolResult(BaseModel):
    """Single result from my_new_tool."""
    id: str
    content: str
    score: float

class MyNewToolOutput(BaseModel):
    """Output schema for my_new_tool."""

    results: List[MyNewToolResult] = Field(
        description="List of results",
    )
    total_count: int = Field(
        description="Total matches found",
    )
    truncated: bool = Field(
        default=False,
        description="Whether results were truncated",
    )
```

### Step 3: Implement the Tool

Create `ic-mcp/src/ic_mcp/tools/my_module.py`:

```python
"""Implementation of my_new_tool."""

from ic_mcp.schemas.inputs import MyNewToolInput
from ic_mcp.schemas.outputs import MyNewToolOutput, MyNewToolResult

async def my_new_tool(input: MyNewToolInput) -> MyNewToolOutput:
    """
    Execute my_new_tool.

    This tool does X, Y, and Z within bounded constraints.

    Args:
        input: Validated input parameters

    Returns:
        MyNewToolOutput with results

    Raises:
        ValueError: If input is invalid
        TimeoutError: If execution exceeds time limit
    """
    # Implementation here
    results = []

    # ... process query ...

    # Enforce limit
    if len(results) > input.limit:
        results = results[:input.limit]
        truncated = True
    else:
        truncated = False

    return MyNewToolOutput(
        results=[
            MyNewToolResult(id=r.id, content=r.content, score=r.score)
            for r in results
        ],
        total_count=len(results),
        truncated=truncated,
    )
```

### Step 4: Register the Tool

Update `ic-mcp/src/ic_mcp/server.py`:

```python
from ic_mcp.tools.my_module import my_new_tool

TOOLS = {
    # ... existing tools ...
    "my_new_tool": {
        "handler": my_new_tool,
        "input_schema": MyNewToolInput,
        "output_schema": MyNewToolOutput,
        "description": "Description of what this tool does",
    },
}
```

### Step 5: Add Tests

Create `ic-mcp/tests/test_my_new_tool.py`:

```python
import pytest
from ic_mcp.tools.my_module import my_new_tool
from ic_mcp.schemas.inputs import MyNewToolInput

@pytest.mark.asyncio
async def test_my_new_tool_basic():
    """Test basic functionality."""
    input = MyNewToolInput(
        query="test query",
        mode="fast",
        limit=10,
    )
    result = await my_new_tool(input)

    assert result.total_count >= 0
    assert len(result.results) <= 10

@pytest.mark.asyncio
async def test_my_new_tool_limit_enforcement():
    """Test that limit is enforced."""
    input = MyNewToolInput(
        query="common term",
        limit=5,
    )
    result = await my_new_tool(input)

    assert len(result.results) <= 5

@pytest.mark.asyncio
async def test_my_new_tool_invalid_input():
    """Test input validation."""
    with pytest.raises(ValueError):
        MyNewToolInput(
            query="",  # Empty query should fail
            limit=10,
        )
```

### Step 6: Update Documentation

Add tool documentation to `docs/API_REFERENCE.md`.

---

## Testing Guidelines

### Test Categories

| Category | Location | Purpose |
|----------|----------|---------|
| Unit Tests | `*/tests/unit/` | Test individual functions |
| Integration Tests | `*/tests/integration/` | Test component interactions |
| Acceptance Tests | `tests/acceptance/` | End-to-end scenarios |

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=icd --cov=ic_mcp --cov-report=html

# Run specific test file
pytest icd/tests/test_chunker.py

# Run specific test
pytest icd/tests/test_chunker.py::test_symbol_chunking

# Run with verbose output
pytest -v

# Run only fast tests
pytest -m "not slow"
```

### Writing Good Tests

```python
import pytest
from unittest.mock import Mock, patch

class TestChunker:
    """Tests for the code chunker."""

    @pytest.fixture
    def sample_python_code(self):
        """Fixture providing sample Python code."""
        return '''
def hello(name: str) -> str:
    """Say hello."""
    return f"Hello, {name}!"

class Greeter:
    """A greeter class."""

    def greet(self, name: str) -> str:
        return hello(name)
'''

    def test_chunks_by_symbol(self, sample_python_code, chunker):
        """Test that chunker respects symbol boundaries."""
        chunks = chunker.chunk(sample_python_code, language="python")

        # Should have separate chunks for function and class
        assert len(chunks) >= 2

        # Function chunk should include docstring
        func_chunk = next(c for c in chunks if "hello" in c.symbol_name)
        assert "Say hello" in func_chunk.content

    def test_respects_max_tokens(self, chunker):
        """Test that chunks don't exceed max token limit."""
        large_code = "x = 1\n" * 10000

        chunks = chunker.chunk(large_code, language="python")

        for chunk in chunks:
            assert chunk.token_count <= chunker.config.max_tokens

    @pytest.mark.slow
    def test_large_file_performance(self, chunker, large_codebase):
        """Test chunking performance on large files."""
        import time

        start = time.time()
        chunks = chunker.chunk(large_codebase, language="python")
        elapsed = time.time() - start

        assert elapsed < 5.0  # Should complete within 5 seconds
```

### Test Coverage Requirements

- **Minimum Coverage**: 80%
- **Critical Paths**: 95%
- **New Code**: Must include tests

---

## Code Style Requirements

### Python Style Guide

We follow PEP 8 with these specifics:

- **Line Length**: 100 characters
- **Quotes**: Double quotes for strings
- **Imports**: Sorted with isort
- **Formatting**: Black formatter

### Type Hints

All code must be fully typed:

```python
# Good
def process_chunks(
    chunks: list[Chunk],
    config: ProcessConfig,
    *,
    timeout: float = 30.0,
) -> ProcessResult:
    """Process chunks with configuration."""
    ...

# Bad - missing types
def process_chunks(chunks, config, timeout=30.0):
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def hybrid_search(
    query: str,
    k: int = 20,
    weights: SearchWeights | None = None,
) -> list[SearchResult]:
    """
    Execute hybrid semantic + lexical search.

    Combines embedding similarity with BM25 scores, applies MMR
    for diversity, and returns the top-k results.

    Args:
        query: The search query string.
        k: Number of results to return.
        weights: Optional custom scoring weights.

    Returns:
        List of SearchResult objects, sorted by relevance.

    Raises:
        ValueError: If query is empty.
        IndexError: If index is not initialized.

    Example:
        >>> results = hybrid_search("authentication", k=10)
        >>> for r in results:
        ...     print(f"{r.path}: {r.score:.2f}")
    """
    ...
```

### Error Handling

```python
# Good - specific exceptions with context
class ChunkingError(Exception):
    """Error during code chunking."""
    pass

def chunk_file(path: Path) -> list[Chunk]:
    if not path.exists():
        raise FileNotFoundError(f"Cannot chunk non-existent file: {path}")

    try:
        content = path.read_text()
    except UnicodeDecodeError as e:
        raise ChunkingError(f"Cannot decode {path}: {e}") from e

    # ... process ...

# Bad - bare exceptions
def chunk_file(path):
    try:
        content = open(path).read()
        # ...
    except:
        return []
```

### Logging

Use structured logging:

```python
import structlog

logger = structlog.get_logger()

def process_file(path: Path) -> None:
    logger.info("processing_file", path=str(path))

    try:
        result = do_processing(path)
        logger.info(
            "file_processed",
            path=str(path),
            chunks=len(result.chunks),
            duration_ms=result.duration_ms,
        )
    except ProcessingError as e:
        logger.error(
            "processing_failed",
            path=str(path),
            error=str(e),
            error_type=type(e).__name__,
        )
        raise
```

### Async Guidelines

```python
# Good - use asyncio primitives
async def fetch_embeddings(texts: list[str]) -> list[np.ndarray]:
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_one(session, text) for text in texts]
        return await asyncio.gather(*tasks)

# Good - proper timeout handling
async def search_with_timeout(query: str) -> list[Result]:
    try:
        return await asyncio.wait_for(
            do_search(query),
            timeout=10.0,
        )
    except asyncio.TimeoutError:
        logger.warning("search_timeout", query=query)
        return []

# Bad - blocking in async context
async def bad_search(query: str) -> list[Result]:
    import time
    time.sleep(1)  # Blocks the event loop!
    return do_search(query)
```

---

## Contributing Guidelines

### Getting Started

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Write/update tests
5. Submit a pull request

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation
- `refactor/description` - Code refactoring

### Commit Messages

Follow Conventional Commits:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance

Examples:
```
feat(retrieval): add MMR diversity selection

Implement Maximal Marginal Relevance algorithm for diverse
result selection. Uses configurable lambda parameter.

Closes #123
```

```
fix(indexing): handle binary files gracefully

Skip binary files during indexing instead of crashing.
Added is_binary_file() check using magic numbers.

Fixes #456
```

### Pull Request Process

1. **Before Submitting**:
   - Run all tests: `pytest`
   - Run linters: `ruff check . && mypy .`
   - Run formatter: `black .`
   - Update documentation if needed

2. **PR Description**:
   - Describe what the change does
   - Link related issues
   - Include testing notes
   - Add screenshots if UI-related

3. **Review Process**:
   - At least one approval required
   - All CI checks must pass
   - Address review feedback

4. **After Merge**:
   - Delete the feature branch
   - Verify deployment (if applicable)

### Code Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests are included and passing
- [ ] Documentation is updated
- [ ] No security vulnerabilities
- [ ] Performance is acceptable
- [ ] Error handling is appropriate
- [ ] Logging is sufficient

---

## Release Process

### Version Numbering

We use Semantic Versioning (SemVer):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Steps

1. **Prepare Release**:
   ```bash
   # Update version numbers
   ./scripts/bump-version.py 1.2.0

   # Update CHANGELOG.md
   # ... add release notes ...

   # Commit
   git commit -am "Release v1.2.0"
   ```

2. **Create Tag**:
   ```bash
   git tag -a v1.2.0 -m "Release v1.2.0"
   git push origin v1.2.0
   ```

3. **Build Packages**:
   ```bash
   make build
   ```

4. **Publish**:
   ```bash
   make publish
   ```

5. **Verify**:
   ```bash
   pip install icr==1.2.0
   icr doctor
   ```

---

## Next Steps

- [ARCHITECTURE.md](ARCHITECTURE.md): System architecture
- [API_REFERENCE.md](API_REFERENCE.md): Tool API documentation
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md): Debugging guide
