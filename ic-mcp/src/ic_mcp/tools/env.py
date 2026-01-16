"""
Environment exploration tools for IC-MCP.

These tools provide safe, bounded access to the codebase:
- env_search: Search across repository, transcripts, diffs, contracts
- env_peek: View specific lines of a file
- env_slice: Extract symbols or ranges from files
- env_aggregate: Perform aggregation operations on data
"""

import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from ic_mcp.schemas.inputs import (
    EnvAggregateInput,
    EnvPeekInput,
    EnvSearchInput,
    EnvSliceInput,
)
from ic_mcp.schemas.outputs import (
    AggregateResult,
    EnvAggregateOutput,
    EnvPeekOutput,
    EnvSearchOutput,
    EnvSliceOutput,
    Evidence,
    PaginationInfo,
    SearchResult,
    Span,
)
from ic_mcp.schemas.validation import (
    create_pagination_cursor,
    parse_pagination_cursor,
    truncate_to_token_budget,
)

logger = logging.getLogger(__name__)


class EnvTools:
    """
    Environment exploration tools.

    These tools provide safe, bounded access to the codebase for
    searching, viewing, and analyzing code.
    """

    def __init__(self, repo_root: str | None = None) -> None:
        """
        Initialize environment tools.

        Args:
            repo_root: Optional default repository root path
        """
        self.repo_root = Path(repo_root) if repo_root else None

    def _get_repo_root(self, path: str | None = None) -> Path:
        """Get the repository root, using path hint if available."""
        if path:
            # Try to find .git directory walking up from path
            p = Path(path)
            if p.is_absolute():
                current = p if p.is_dir() else p.parent
            else:
                current = Path.cwd() / p
                current = current if current.is_dir() else current.parent

            while current != current.parent:
                if (current / ".git").exists():
                    return current
                current = current.parent

        if self.repo_root:
            return self.repo_root

        # Default to current working directory
        return Path.cwd()

    def _generate_source_id(self, path: str) -> str:
        """Generate a deterministic source ID from a path."""
        import hashlib

        return f"S{hashlib.sha256(path.encode()).hexdigest()[:12]}"

    def _search_files(
        self,
        query: str,
        repo_root: Path,
        path_prefix: str | None = None,
        language: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """
        Search files in the repository.

        Uses simple text search. In production, this would integrate with
        a proper search index (e.g., ripgrep, tree-sitter, or vector search).
        """
        results: list[dict[str, Any]] = []

        # Map language to extensions
        language_extensions: dict[str, set[str]] = {
            "python": {".py", ".pyx", ".pyi"},
            "typescript": {".ts", ".tsx"},
            "javascript": {".js", ".jsx", ".mjs"},
            "go": {".go"},
            "rust": {".rs"},
            "java": {".java"},
            "c": {".c", ".h"},
            "cpp": {".cpp", ".hpp", ".cc", ".hh"},
            "ruby": {".rb"},
            "php": {".php"},
        }

        allowed_extensions = language_extensions.get(language.lower()) if language else None

        # Patterns to exclude
        exclude_dirs = {
            ".git", "node_modules", "__pycache__", ".venv", "venv",
            "dist", "build", ".next", ".nuxt", "target", "vendor",
        }

        query_lower = query.lower()
        query_pattern = re.compile(re.escape(query), re.IGNORECASE)

        try:
            for root, dirs, files in os.walk(repo_root):
                # Skip excluded directories
                dirs[:] = [d for d in dirs if d not in exclude_dirs]

                for file in files:
                    file_path = Path(root) / file
                    rel_path = str(file_path.relative_to(repo_root))

                    # Check path prefix filter
                    if path_prefix and not rel_path.startswith(path_prefix):
                        continue

                    # Check language filter
                    if allowed_extensions and file_path.suffix.lower() not in allowed_extensions:
                        continue

                    # Skip binary files
                    if file_path.suffix.lower() in {".pyc", ".pyo", ".so", ".dylib", ".dll"}:
                        continue

                    try:
                        content = file_path.read_text(errors="ignore")

                        # Search for matches
                        matches = list(query_pattern.finditer(content))
                        if not matches:
                            continue

                        # Get context around first match
                        first_match = matches[0]
                        start = max(0, first_match.start() - 100)
                        end = min(len(content), first_match.end() + 100)
                        snippet = content[start:end]

                        # Find line numbers
                        lines_before = content[:first_match.start()].count("\n")
                        start_line = lines_before + 1
                        match_lines = content[first_match.start():first_match.end()].count("\n")
                        end_line = start_line + match_lines

                        # Calculate score based on match count and position
                        score = min(1.0, len(matches) * 0.1 + 0.3)

                        # Boost if query is in filename
                        if query_lower in file.lower():
                            score = min(1.0, score + 0.3)

                        results.append({
                            "source_id": self._generate_source_id(rel_path),
                            "path": rel_path,
                            "score": score,
                            "snippet": snippet,
                            "span": {"start_line": start_line, "end_line": end_line},
                            "match_count": len(matches),
                        })

                    except Exception as e:
                        logger.debug(f"Error reading {file_path}: {e}")

        except Exception as e:
            logger.error(f"Error searching repository: {e}")

        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)

        total = len(results)
        return results[offset:offset + limit], total

    async def env_search(
        self,
        input_data: EnvSearchInput,
        request_id: UUID,
    ) -> EnvSearchOutput:
        """
        Search across the environment.

        Searches repository files, transcripts, diffs, and contracts
        based on the specified scope.

        Args:
            input_data: Validated input parameters
            request_id: Unique request identifier

        Returns:
            EnvSearchOutput with search results
        """
        logger.info(f"env_search: query={input_data.query[:50]}, scope={input_data.scope}")

        offset, _ = parse_pagination_cursor(input_data.cursor)
        results: list[SearchResult] = []
        evidence: list[Evidence] = []
        explanation = None

        repo_root = self._get_repo_root(input_data.path_prefix)

        if input_data.scope in ("repo", "all"):
            file_results, total = self._search_files(
                input_data.query,
                repo_root,
                input_data.path_prefix,
                input_data.language,
                input_data.limit,
                offset,
            )

            for r in file_results:
                span = Span(**r["span"]) if r.get("span") else None
                results.append(
                    SearchResult(
                        source_id=r["source_id"],
                        path=r["path"],
                        score=r["score"],
                        snippet=r["snippet"],
                        span=span,
                        highlights=[],
                    )
                )

                evidence.append(
                    Evidence(
                        source_id=r["source_id"],
                        source_type="file",
                        path=r["path"],
                        repo_rev="working-tree",
                        span=span,
                        content=r["snippet"][:200],
                    )
                )

        if input_data.scope in ("transcript", "all"):
            # Transcript search would query a transcript store
            # For now, this is a placeholder
            if input_data.explain:
                explanation = (explanation or "") + "Transcript search not yet implemented. "

        if input_data.scope in ("diffs", "all"):
            # Diff search would query git history
            # For now, this is a placeholder
            if input_data.explain:
                explanation = (explanation or "") + "Diff search not yet implemented. "

        if input_data.scope in ("contracts", "all"):
            # Contract search would query contract definitions
            # For now, this is a placeholder
            if input_data.explain:
                explanation = (explanation or "") + "Contract search not yet implemented. "

        if input_data.explain and not explanation:
            explanation = (
                f"Searched {input_data.scope} scope with query '{input_data.query}'. "
                f"Found {len(results)} results using text matching."
            )

        cursor = create_pagination_cursor(offset, input_data.limit, len(results) + offset + 1)
        has_more = len(results) == input_data.limit

        output = EnvSearchOutput(
            request_id=request_id,
            results=results,
            pagination=PaginationInfo(
                cursor=cursor if has_more else None,
                has_more=has_more,
                total_count=None,
            ),
            explanation=explanation,
            evidence=evidence,
        )

        # Truncate if necessary
        output_dict = output.model_dump()
        truncated_dict, _ = truncate_to_token_budget(
            output_dict,
            truncatable_fields=["results", "evidence"],
        )

        return EnvSearchOutput.model_validate(truncated_dict)

    async def env_peek(
        self,
        input_data: EnvPeekInput,
        request_id: UUID,
    ) -> EnvPeekOutput:
        """
        Peek at specific lines of a file.

        Provides bounded access to file content with line number limits.

        Args:
            input_data: Validated input parameters
            request_id: Unique request identifier

        Returns:
            EnvPeekOutput with file content
        """
        logger.info(
            f"env_peek: path={input_data.path}, "
            f"lines={input_data.start_line}-{input_data.end_line}"
        )

        path = Path(input_data.path)
        if not path.is_absolute():
            repo_root = self._get_repo_root(input_data.path)
            path = repo_root / input_data.path

        # Read file
        try:
            content = path.read_text(errors="ignore")
            lines = content.splitlines(keepends=True)
            total_lines = len(lines)
        except FileNotFoundError:
            # Return empty result for non-existent file
            return EnvPeekOutput(
                request_id=request_id,
                path=input_data.path,
                content="",
                start_line=input_data.start_line,
                end_line=input_data.start_line,
                total_lines=0,
                truncated=False,
                evidence=Evidence(
                    source_id=self._generate_source_id(input_data.path),
                    source_type="file",
                    path=input_data.path,
                    repo_rev="working-tree",
                    content="File not found",
                ),
            )
        except Exception as e:
            logger.error(f"Error reading file {path}: {e}")
            raise

        # Calculate actual line range
        start_idx = max(0, input_data.start_line - 1)
        end_idx = min(len(lines), input_data.end_line)

        # Apply max_lines limit
        requested_lines = end_idx - start_idx
        truncated = requested_lines > input_data.max_lines
        if truncated:
            end_idx = start_idx + input_data.max_lines

        # Extract content with line numbers
        selected_lines = lines[start_idx:end_idx]
        numbered_content = ""
        for i, line in enumerate(selected_lines):
            line_num = start_idx + i + 1
            numbered_content += f"{line_num:6d}  {line}"

        # Ensure newline at end
        if numbered_content and not numbered_content.endswith("\n"):
            numbered_content += "\n"

        # Create evidence
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        evidence = Evidence(
            source_id=self._generate_source_id(input_data.path),
            source_type="file",
            path=input_data.path,
            repo_rev="working-tree",
            span=Span(start_line=start_idx + 1, end_line=end_idx),
            mtime=mtime,
            content=numbered_content[:500],
        )

        return EnvPeekOutput(
            request_id=request_id,
            path=input_data.path,
            content=numbered_content,
            start_line=start_idx + 1,
            end_line=end_idx,
            total_lines=total_lines,
            truncated=truncated,
            evidence=evidence,
        )

    async def env_slice(
        self,
        input_data: EnvSliceInput,
        request_id: UUID,
    ) -> EnvSliceOutput:
        """
        Extract a slice from a file.

        Can extract by symbol name or line range, with optional context lines.

        Args:
            input_data: Validated input parameters
            request_id: Unique request identifier

        Returns:
            EnvSliceOutput with sliced content
        """
        logger.info(f"env_slice: path={input_data.path}, symbol={input_data.symbol}")

        path = Path(input_data.path)
        if not path.is_absolute():
            repo_root = self._get_repo_root(input_data.path)
            path = repo_root / input_data.path

        # Read file
        try:
            content = path.read_text(errors="ignore")
            lines = content.splitlines(keepends=True)
        except FileNotFoundError:
            raise ValueError(f"File not found: {input_data.path}")
        except Exception as e:
            logger.error(f"Error reading file {path}: {e}")
            raise

        start_line: int
        end_line: int
        symbol_found = None

        if input_data.symbol:
            # Find symbol in file
            symbol_span = self._find_symbol(content, lines, input_data.symbol)
            if symbol_span is None:
                raise ValueError(f"Symbol '{input_data.symbol}' not found in {input_data.path}")

            start_line, end_line = symbol_span
            symbol_found = input_data.symbol

        elif input_data.start_line is not None:
            start_line = input_data.start_line
            end_line = input_data.end_line or start_line

        else:
            raise ValueError("Either symbol or start_line must be provided")

        # Apply context lines
        context_start = max(1, start_line - input_data.context_lines)
        context_end = min(len(lines), end_line + input_data.context_lines)

        # Extract content
        start_idx = context_start - 1
        end_idx = context_end
        selected_lines = lines[start_idx:end_idx]

        # Format with line numbers
        numbered_content = ""
        for i, line in enumerate(selected_lines):
            line_num = context_start + i
            # Mark the actual slice vs context
            marker = " " if context_start + i < start_line or context_start + i > end_line else ">"
            numbered_content += f"{marker}{line_num:5d}  {line}"

        if numbered_content and not numbered_content.endswith("\n"):
            numbered_content += "\n"

        # Create evidence
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        evidence = Evidence(
            source_id=self._generate_source_id(input_data.path),
            source_type="file",
            path=input_data.path,
            repo_rev="working-tree",
            span=Span(start_line=start_line, end_line=end_line),
            mtime=mtime,
            content=numbered_content[:500],
        )

        return EnvSliceOutput(
            request_id=request_id,
            path=input_data.path,
            content=numbered_content,
            symbol=symbol_found,
            span=Span(start_line=start_line, end_line=end_line),
            context_before=start_line - context_start,
            context_after=context_end - end_line,
            evidence=evidence,
        )

    def _find_symbol(
        self,
        content: str,
        lines: list[str],
        symbol: str,
    ) -> tuple[int, int] | None:
        """
        Find a symbol (function, class, etc.) in file content.

        This is a simplified implementation. In production, this would
        use tree-sitter or a proper parser for accurate symbol extraction.
        """
        # Common patterns for symbol definitions
        patterns = [
            # Python function/class
            rf"^(\s*)(def|class|async\s+def)\s+{re.escape(symbol)}\s*[\(:]",
            # JavaScript/TypeScript function
            rf"^(\s*)(function|const|let|var|export\s+function|export\s+const)\s+{re.escape(symbol)}\s*[=\(]",
            # Go function
            rf"^func\s+(?:\([^)]+\)\s+)?{re.escape(symbol)}\s*\(",
            # Rust function
            rf"^(\s*)(pub\s+)?(fn|struct|enum|trait)\s+{re.escape(symbol)}",
            # Java/C++ method/class
            rf"^(\s*)(public|private|protected|static|\s)*\s*(class|void|int|string|bool|auto)\s+{re.escape(symbol)}\s*[\({{]",
        ]

        for i, line in enumerate(lines):
            for pattern in patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    start_line = i + 1

                    # Find the end of the symbol definition
                    end_line = self._find_block_end(lines, i)
                    return start_line, end_line

        return None

    def _find_block_end(self, lines: list[str], start_idx: int) -> int:
        """
        Find the end of a code block starting at start_idx.

        Uses simple brace/indent matching.
        """
        if start_idx >= len(lines):
            return start_idx + 1

        start_line = lines[start_idx]

        # Detect indentation-based (Python) or brace-based
        if "{" in start_line:
            # Brace-based block
            brace_count = 0
            for i in range(start_idx, len(lines)):
                line = lines[i]
                brace_count += line.count("{") - line.count("}")
                if brace_count <= 0 and i > start_idx:
                    return i + 1
            return len(lines)

        else:
            # Indentation-based (Python style)
            base_indent = len(start_line) - len(start_line.lstrip())

            for i in range(start_idx + 1, len(lines)):
                line = lines[i]
                # Skip empty lines and comments
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue

                current_indent = len(line) - len(line.lstrip())
                if current_indent <= base_indent and stripped:
                    return i

            return len(lines)

    async def env_aggregate(
        self,
        input_data: EnvAggregateInput,
        request_id: UUID,
    ) -> EnvAggregateOutput:
        """
        Perform aggregation operations on input data.

        Supports various operations like extract_regex, unique, sort,
        group_by, count, top_k, join_on, and diff_sets.

        Args:
            input_data: Validated input parameters
            request_id: Unique request identifier

        Returns:
            EnvAggregateOutput with aggregation results
        """
        logger.info(f"env_aggregate: op={input_data.op}, inputs={len(input_data.inputs)}")

        results: list[AggregateResult] = []
        truncated = False

        if input_data.op == "extract_regex":
            pattern = input_data.params.get("pattern", ".*")
            try:
                regex = re.compile(pattern)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern: {e}")

            for inp in input_data.inputs:
                matches = regex.findall(inp)
                for match in matches:
                    results.append(AggregateResult(value=match))

            if len(results) > input_data.limit:
                results = results[:input_data.limit]
                truncated = True

        elif input_data.op == "unique":
            seen: set[str] = set()
            for inp in input_data.inputs:
                if inp not in seen:
                    seen.add(inp)
                    results.append(AggregateResult(value=inp))
                    if len(results) >= input_data.limit:
                        truncated = True
                        break

        elif input_data.op == "sort":
            reverse = input_data.params.get("reverse", False)
            key = input_data.params.get("key")

            sorted_inputs = sorted(input_data.inputs, reverse=reverse)
            for inp in sorted_inputs[:input_data.limit]:
                results.append(AggregateResult(value=inp))

            truncated = len(input_data.inputs) > input_data.limit

        elif input_data.op == "group_by":
            key_pattern = input_data.params.get("key_pattern", r"^([^/]+)")
            try:
                regex = re.compile(key_pattern)
            except re.error as e:
                raise ValueError(f"Invalid key pattern: {e}")

            groups: dict[str, list[str]] = {}
            for inp in input_data.inputs:
                match = regex.search(inp)
                key = match.group(1) if match else "_other"
                if key not in groups:
                    groups[key] = []
                groups[key].append(inp)

            for key, values in list(groups.items())[:input_data.limit]:
                results.append(AggregateResult(
                    key=key,
                    value=values,
                    count=len(values),
                ))

            truncated = len(groups) > input_data.limit

        elif input_data.op == "count":
            from collections import Counter

            counts = Counter(input_data.inputs)
            for value, count in counts.most_common(input_data.limit):
                results.append(AggregateResult(value=value, count=count))

            truncated = len(counts) > input_data.limit

        elif input_data.op == "top_k":
            k = input_data.params.get("k", 10)
            # Assume inputs are (value, score) pairs as JSON strings
            scored: list[tuple[str, float]] = []
            for inp in input_data.inputs:
                try:
                    import json

                    data = json.loads(inp)
                    if isinstance(data, dict) and "score" in data:
                        scored.append((data.get("value", inp), data["score"]))
                    else:
                        scored.append((inp, 0.0))
                except (json.JSONDecodeError, TypeError):
                    scored.append((inp, 0.0))

            scored.sort(key=lambda x: x[1], reverse=True)
            for value, score in scored[:min(k, input_data.limit)]:
                results.append(AggregateResult(value=value, count=int(score * 100)))

        elif input_data.op == "join_on":
            separator = input_data.params.get("separator", "\n")
            joined = separator.join(input_data.inputs[:input_data.limit])
            results.append(AggregateResult(value=joined))
            truncated = len(input_data.inputs) > input_data.limit

        elif input_data.op == "diff_sets":
            # Expect two sets in params: "set_a" and "set_b" as input indices
            set_a_indices = input_data.params.get("set_a", [])
            set_b_indices = input_data.params.get("set_b", [])

            set_a = {input_data.inputs[i] for i in set_a_indices if i < len(input_data.inputs)}
            set_b = {input_data.inputs[i] for i in set_b_indices if i < len(input_data.inputs)}

            only_a = set_a - set_b
            only_b = set_b - set_a
            both = set_a & set_b

            results.append(AggregateResult(key="only_a", value=list(only_a)[:input_data.limit]))
            results.append(AggregateResult(key="only_b", value=list(only_b)[:input_data.limit]))
            results.append(AggregateResult(key="both", value=list(both)[:input_data.limit]))

        else:
            raise ValueError(f"Unknown aggregation operation: {input_data.op}")

        return EnvAggregateOutput(
            request_id=request_id,
            op=input_data.op,
            results=results,
            total_inputs=len(input_data.inputs),
            truncated=truncated,
        )
