"""
Project analysis tools for IC-MCP.

These tools provide project-level analysis capabilities:
- project_map: Generate project structure map
- project_symbol_search: Search for symbols across the project
- project_impact: Analyze impact of changes
- project_commands: Discover project commands
"""

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from ic_mcp.schemas.inputs import (
    ProjectCommandsInput,
    ProjectImpactInput,
    ProjectMapInput,
    ProjectSymbolSearchInput,
)
from ic_mcp.schemas.outputs import (
    Evidence,
    FileNode,
    ImpactEdge,
    ImpactNode,
    PaginationInfo,
    ProjectCommand,
    ProjectCommandsOutput,
    ProjectImpactOutput,
    ProjectMapOutput,
    ProjectSymbolSearchOutput,
    Span,
    SymbolResult,
)
from ic_mcp.schemas.validation import (
    create_pagination_cursor,
    parse_pagination_cursor,
    truncate_to_token_budget,
)

logger = logging.getLogger(__name__)


# Language detection by extension
LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".pyx": "python",
    ".pyi": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".scala": "scala",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
    ".c": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".swift": "swift",
    ".m": "objective-c",
    ".r": "r",
    ".R": "r",
    ".sql": "sql",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".md": "markdown",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".xml": "xml",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".less": "less",
}


class ProjectTools:
    """
    Project analysis tools.

    These tools provide project-level analysis for understanding
    codebase structure, symbols, and change impact.
    """

    def __init__(self) -> None:
        """Initialize project tools."""
        pass

    def _detect_language(self, path: Path) -> str | None:
        """Detect programming language from file extension."""
        return LANGUAGE_MAP.get(path.suffix.lower())

    def _should_exclude_dir(self, name: str) -> bool:
        """Check if directory should be excluded."""
        exclude_dirs = {
            ".git", "node_modules", "__pycache__", ".venv", "venv",
            "dist", "build", ".next", ".nuxt", "target", "vendor",
            ".idea", ".vscode", ".cache", "coverage", ".pytest_cache",
            ".mypy_cache", ".ruff_cache", "egg-info", ".eggs",
        }
        return name in exclude_dirs or name.endswith(".egg-info")

    def _matches_pattern(self, path: str, pattern: str) -> bool:
        """Check if path matches a glob pattern."""
        import fnmatch

        return fnmatch.fnmatch(path, pattern)

    def _generate_source_id(self, path: str) -> str:
        """Generate a deterministic source ID from a path."""
        import hashlib

        return f"S{hashlib.sha256(path.encode()).hexdigest()[:12]}"

    async def project_map(
        self,
        input_data: ProjectMapInput,
        request_id: UUID,
    ) -> ProjectMapOutput:
        """
        Generate a project structure map.

        Creates a hierarchical view of the project with optional statistics.

        Args:
            input_data: Validated input parameters
            request_id: Unique request identifier

        Returns:
            ProjectMapOutput with project tree
        """
        logger.info(f"project_map: repo_root={input_data.repo_root}, depth={input_data.depth}")

        repo_root = Path(input_data.repo_root)
        if not repo_root.exists():
            raise ValueError(f"Repository root does not exist: {input_data.repo_root}")

        languages: dict[str, int] = {}
        total_files = 0
        total_dirs = 0

        def build_tree(
            path: Path,
            current_depth: int,
            max_depth: int,
        ) -> FileNode | None:
            """Recursively build the file tree."""
            nonlocal total_files, total_dirs, languages

            rel_path = str(path.relative_to(repo_root))
            if rel_path == ".":
                rel_path = ""

            name = path.name or repo_root.name

            # Check exclusion patterns
            for pattern in input_data.exclude_patterns:
                if self._matches_pattern(rel_path, pattern):
                    return None

            # Check inclusion patterns (if specified)
            if input_data.include_patterns:
                matches_include = any(
                    self._matches_pattern(rel_path, p)
                    for p in input_data.include_patterns
                )
                if not matches_include and path.is_file():
                    return None

            if path.is_file():
                total_files += 1

                # Detect language
                lang = self._detect_language(path)
                if lang:
                    languages[lang] = languages.get(lang, 0) + 1

                # Get file stats if requested
                size_bytes = None
                line_count = None

                if input_data.include_stats:
                    try:
                        stat = path.stat()
                        size_bytes = stat.st_size

                        # Count lines for text files
                        if lang or path.suffix.lower() in {".txt", ".md", ".rst"}:
                            try:
                                content = path.read_text(errors="ignore")
                                line_count = content.count("\n") + 1
                            except Exception:
                                pass
                    except Exception:
                        pass

                return FileNode(
                    path=rel_path,
                    type="file",
                    name=name,
                    children=[],
                    size_bytes=size_bytes,
                    line_count=line_count,
                    language=lang,
                )

            elif path.is_dir():
                # Check if we should exclude this directory
                if self._should_exclude_dir(name):
                    return None

                total_dirs += 1

                children: list[FileNode] = []

                if current_depth < max_depth:
                    try:
                        for child in sorted(path.iterdir()):
                            child_node = build_tree(child, current_depth + 1, max_depth)
                            if child_node:
                                children.append(child_node)
                    except PermissionError:
                        pass

                return FileNode(
                    path=rel_path,
                    type="directory",
                    name=name,
                    children=children,
                )

            return None

        tree = build_tree(repo_root, 0, input_data.depth)

        if tree is None:
            tree = FileNode(
                path="",
                type="directory",
                name=repo_root.name,
                children=[],
            )

        output = ProjectMapOutput(
            request_id=request_id,
            repo_root=str(repo_root),
            tree=tree,
            total_files=total_files,
            total_directories=total_dirs,
            languages=languages,
        )

        # Truncate if necessary
        output_dict = output.model_dump()
        truncated_dict, _ = truncate_to_token_budget(
            output_dict,
            truncatable_fields=["tree"],
        )

        return ProjectMapOutput.model_validate(truncated_dict)

    async def project_symbol_search(
        self,
        input_data: ProjectSymbolSearchInput,
        request_id: UUID,
    ) -> ProjectSymbolSearchOutput:
        """
        Search for symbols across the project.

        Finds functions, classes, methods, variables, types, and interfaces.

        Args:
            input_data: Validated input parameters
            request_id: Unique request identifier

        Returns:
            ProjectSymbolSearchOutput with matching symbols
        """
        logger.info(f"project_symbol_search: query={input_data.query}")

        repo_root = Path(input_data.repo_root)
        if not repo_root.exists():
            raise ValueError(f"Repository root does not exist: {input_data.repo_root}")

        offset, _ = parse_pagination_cursor(input_data.cursor)
        results: list[SymbolResult] = []

        # Symbol patterns by language
        symbol_patterns: dict[str, list[tuple[str, str]]] = {
            "python": [
                (r"^\s*def\s+(\w+)\s*\(", "function"),
                (r"^\s*async\s+def\s+(\w+)\s*\(", "function"),
                (r"^\s*class\s+(\w+)", "class"),
            ],
            "typescript": [
                (r"^\s*(?:export\s+)?function\s+(\w+)", "function"),
                (r"^\s*(?:export\s+)?(?:const|let)\s+(\w+)\s*=\s*(?:async\s*)?\(", "function"),
                (r"^\s*(?:export\s+)?class\s+(\w+)", "class"),
                (r"^\s*(?:export\s+)?interface\s+(\w+)", "interface"),
                (r"^\s*(?:export\s+)?type\s+(\w+)", "type"),
            ],
            "javascript": [
                (r"^\s*(?:export\s+)?function\s+(\w+)", "function"),
                (r"^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(", "function"),
                (r"^\s*(?:export\s+)?class\s+(\w+)", "class"),
            ],
            "go": [
                (r"^func\s+(\w+)\s*\(", "function"),
                (r"^func\s+\([^)]+\)\s+(\w+)\s*\(", "method"),
                (r"^type\s+(\w+)\s+struct", "class"),
                (r"^type\s+(\w+)\s+interface", "interface"),
            ],
            "rust": [
                (r"^\s*(?:pub\s+)?fn\s+(\w+)", "function"),
                (r"^\s*(?:pub\s+)?struct\s+(\w+)", "class"),
                (r"^\s*(?:pub\s+)?enum\s+(\w+)", "type"),
                (r"^\s*(?:pub\s+)?trait\s+(\w+)", "interface"),
            ],
            "java": [
                (r"^\s*(?:public|private|protected)?\s*(?:static\s+)?(?:\w+\s+)?(\w+)\s*\(", "function"),
                (r"^\s*(?:public|private|protected)?\s*class\s+(\w+)", "class"),
                (r"^\s*(?:public|private|protected)?\s*interface\s+(\w+)", "interface"),
            ],
        }

        query_lower = input_data.query.lower()
        all_results: list[dict[str, Any]] = []

        def search_file(file_path: Path) -> None:
            """Search a single file for symbols."""
            rel_path = str(file_path.relative_to(repo_root))
            lang = self._detect_language(file_path)

            if not lang:
                return

            # Filter by language if specified
            if input_data.languages and lang not in input_data.languages:
                return

            patterns = symbol_patterns.get(lang, [])
            if not patterns:
                return

            try:
                content = file_path.read_text(errors="ignore")
                lines = content.splitlines()
            except Exception:
                return

            for i, line in enumerate(lines):
                for pattern, symbol_type in patterns:
                    # Filter by symbol type if specified
                    if input_data.symbol_types and symbol_type not in input_data.symbol_types:
                        continue

                    match = re.match(pattern, line)
                    if match:
                        name = match.group(1)

                        # Check if matches query
                        if query_lower not in name.lower():
                            continue

                        # Calculate score
                        if name.lower() == query_lower:
                            score = 1.0
                        elif name.lower().startswith(query_lower):
                            score = 0.9
                        else:
                            score = 0.7

                        # Extract signature (simplified)
                        signature = line.strip()

                        # Extract docstring (simplified)
                        docstring = None
                        if i + 1 < len(lines):
                            next_line = lines[i + 1].strip()
                            if next_line.startswith('"""') or next_line.startswith("'''"):
                                docstring = next_line.strip("\"'")
                            elif next_line.startswith("//") or next_line.startswith("/*"):
                                docstring = next_line.lstrip("/*/ ")

                        all_results.append({
                            "name": name,
                            "qualified_name": f"{rel_path}:{name}",
                            "symbol_type": symbol_type,
                            "path": rel_path,
                            "span": {"start_line": i + 1, "end_line": i + 1},
                            "signature": signature,
                            "docstring": docstring,
                            "score": score,
                        })

        # Walk repository
        for root, dirs, files in os.walk(repo_root):
            dirs[:] = [d for d in dirs if not self._should_exclude_dir(d)]

            for file in files:
                file_path = Path(root) / file
                search_file(file_path)

        # Sort by score and apply pagination
        all_results.sort(key=lambda x: x["score"], reverse=True)
        total = len(all_results)
        page_results = all_results[offset:offset + input_data.limit]

        for r in page_results:
            results.append(
                SymbolResult(
                    name=r["name"],
                    qualified_name=r["qualified_name"],
                    symbol_type=r["symbol_type"],
                    path=r["path"],
                    span=Span(**r["span"]),
                    signature=r["signature"],
                    docstring=r["docstring"],
                    score=r["score"],
                )
            )

        cursor = create_pagination_cursor(offset, input_data.limit, total)
        has_more = offset + len(page_results) < total

        return ProjectSymbolSearchOutput(
            request_id=request_id,
            results=results,
            pagination=PaginationInfo(
                cursor=cursor if has_more else None,
                has_more=has_more,
                total_count=total,
            ),
        )

    async def project_impact(
        self,
        input_data: ProjectImpactInput,
        request_id: UUID,
    ) -> ProjectImpactOutput:
        """
        Analyze the impact of file changes.

        Builds a dependency graph showing what might be affected by changes.

        Args:
            input_data: Validated input parameters
            request_id: Unique request identifier

        Returns:
            ProjectImpactOutput with impact graph
        """
        logger.info(f"project_impact: changed_paths={len(input_data.changed_paths)}")

        nodes: list[ImpactNode] = []
        edges: list[ImpactEdge] = []
        evidence: list[Evidence] = []
        node_ids: set[str] = set()

        def add_node(
            path: str,
            node_type: str,
            label: str | None = None,
        ) -> str:
            """Add a node to the graph if not already present."""
            node_id = self._generate_source_id(path)
            if node_id not in node_ids:
                node_ids.add(node_id)
                nodes.append(
                    ImpactNode(
                        id=node_id,
                        type=node_type,
                        label=label or Path(path).name,
                        path=path,
                    )
                )
            return node_id

        def add_edge(
            from_id: str,
            to_id: str,
            edge_type: str,
        ) -> None:
            """Add an edge to the graph."""
            if len(edges) >= input_data.max_edges:
                return

            edges.append(
                ImpactEdge(
                    from_node=from_id,
                    to_node=to_id,
                    type=edge_type,
                    evidence_source_id=from_id,
                )
            )

        # Analyze each changed file
        for changed_path in input_data.changed_paths[:input_data.max_nodes]:
            path = Path(changed_path)

            # Determine node type
            node_type = self._infer_node_type(path)
            from_id = add_node(changed_path, node_type)

            # Try to read the file and find imports/dependencies
            try:
                if path.exists():
                    content = path.read_text(errors="ignore")

                    # Find imports based on language
                    lang = self._detect_language(path)
                    imports = self._extract_imports(content, lang)

                    for imp in imports[:20]:  # Limit imports per file
                        if len(nodes) >= input_data.max_nodes:
                            break

                        to_type = self._infer_node_type(Path(imp))
                        to_id = add_node(imp, to_type, Path(imp).name)
                        add_edge(from_id, to_id, "imports")

                    # Add evidence
                    evidence.append(
                        Evidence(
                            source_id=from_id,
                            source_type="file",
                            path=changed_path,
                            repo_rev="working-tree",
                            mtime=datetime.fromtimestamp(
                                path.stat().st_mtime, tz=timezone.utc
                            ),
                            content=content[:200],
                        )
                    )

            except Exception as e:
                logger.debug(f"Error analyzing {changed_path}: {e}")

        # Determine depth reached
        depth_reached = min(2, len(input_data.changed_paths))

        return ProjectImpactOutput(
            request_id=request_id,
            nodes=nodes,
            edges=edges,
            root_paths=input_data.changed_paths[:input_data.max_nodes],
            depth_reached=depth_reached,
            truncated=len(nodes) >= input_data.max_nodes or len(edges) >= input_data.max_edges,
            evidence=evidence,
        )

    def _infer_node_type(self, path: Path) -> str:
        """Infer the node type from the path."""
        name = path.name.lower()
        parts = [p.lower() for p in path.parts]

        # Test files
        if "test" in name or "spec" in name or "tests" in parts or "__tests__" in parts:
            return "test"

        # Frontend components
        if any(d in parts for d in ["components", "pages", "views", "screens"]):
            return "fe_component"

        # Frontend clients
        if "client" in name or "api" in parts:
            return "fe_client"

        # Backend handlers
        if any(d in parts for d in ["handlers", "controllers", "routes", "api"]):
            return "be_handler"

        # Contracts/interfaces
        if "contract" in name or "interface" in name or "types" in parts:
            return "contract"

        # Shared types
        if "shared" in parts or "common" in parts:
            return "shared_type"

        # Endpoints
        if "endpoint" in name or "route" in name:
            return "endpoint"

        return "file"

    def _extract_imports(self, content: str, lang: str | None) -> list[str]:
        """Extract import statements from code."""
        imports: list[str] = []

        if lang == "python":
            # Python imports
            for match in re.finditer(r"^(?:from|import)\s+([\w.]+)", content, re.MULTILINE):
                imports.append(match.group(1).replace(".", "/") + ".py")

        elif lang in ("typescript", "javascript"):
            # JS/TS imports
            for match in re.finditer(
                r"(?:import|from)\s+['\"]([^'\"]+)['\"]", content
            ):
                imp = match.group(1)
                if not imp.startswith("."):
                    continue  # Skip node_modules
                imports.append(imp)

        elif lang == "go":
            # Go imports
            for match in re.finditer(r'"([^"]+)"', content):
                imp = match.group(1)
                if "/" in imp:  # Skip standard library
                    imports.append(imp)

        elif lang == "rust":
            # Rust imports
            for match in re.finditer(r"^use\s+([\w:]+)", content, re.MULTILINE):
                imports.append(match.group(1).replace("::", "/") + ".rs")

        return imports

    async def project_commands(
        self,
        input_data: ProjectCommandsInput,
        request_id: UUID,
    ) -> ProjectCommandsOutput:
        """
        Discover project commands from config files.

        Examines package.json, Makefile, pyproject.toml, etc.

        Args:
            input_data: Validated input parameters
            request_id: Unique request identifier

        Returns:
            ProjectCommandsOutput with discovered commands
        """
        logger.info(f"project_commands: repo_root={input_data.repo_root}")

        repo_root = Path(input_data.repo_root)
        if not repo_root.exists():
            raise ValueError(f"Repository root does not exist: {input_data.repo_root}")

        commands: list[ProjectCommand] = []
        config_files: list[str] = []

        # Check package.json
        package_json = repo_root / "package.json"
        if package_json.exists():
            config_files.append("package.json")
            try:
                data = json.loads(package_json.read_text())
                scripts = data.get("scripts", {})
                for name, cmd in scripts.items():
                    cmd_type = self._classify_npm_script(name)
                    if input_data.command_type == "all" or cmd_type == input_data.command_type:
                        commands.append(
                            ProjectCommand(
                                name=name,
                                command=f"npm run {name}",
                                type=cmd_type,
                                source="package.json",
                                description=f"Runs: {cmd}",
                            )
                        )
            except Exception as e:
                logger.debug(f"Error reading package.json: {e}")

        # Check Makefile
        makefile = repo_root / "Makefile"
        if makefile.exists():
            config_files.append("Makefile")
            try:
                content = makefile.read_text()
                for match in re.finditer(r"^(\w+):", content, re.MULTILINE):
                    name = match.group(1)
                    if name.startswith("_") or name in ("all", "default"):
                        continue

                    cmd_type = self._classify_make_target(name)
                    if input_data.command_type == "all" or cmd_type == input_data.command_type:
                        commands.append(
                            ProjectCommand(
                                name=name,
                                command=f"make {name}",
                                type=cmd_type,
                                source="Makefile",
                            )
                        )
            except Exception as e:
                logger.debug(f"Error reading Makefile: {e}")

        # Check pyproject.toml
        pyproject = repo_root / "pyproject.toml"
        if pyproject.exists():
            config_files.append("pyproject.toml")
            try:
                import tomllib  # Python 3.11+

                data = tomllib.loads(pyproject.read_text())

                # Check for scripts
                scripts = data.get("project", {}).get("scripts", {})
                for name in scripts:
                    cmd_type = self._classify_python_script(name)
                    if input_data.command_type == "all" or cmd_type == input_data.command_type:
                        commands.append(
                            ProjectCommand(
                                name=name,
                                command=name,
                                type=cmd_type,
                                source="pyproject.toml",
                            )
                        )

                # Check for pytest
                if "tool" in data and "pytest" in data["tool"]:
                    if input_data.command_type in ("all", "test"):
                        commands.append(
                            ProjectCommand(
                                name="test",
                                command="pytest",
                                type="test",
                                source="pyproject.toml",
                            )
                        )

                # Check for ruff
                if "tool" in data and "ruff" in data["tool"]:
                    if input_data.command_type in ("all", "lint"):
                        commands.append(
                            ProjectCommand(
                                name="lint",
                                command="ruff check .",
                                type="lint",
                                source="pyproject.toml",
                            )
                        )
                        commands.append(
                            ProjectCommand(
                                name="format",
                                command="ruff format .",
                                type="format",
                                source="pyproject.toml",
                            )
                        )

            except ImportError:
                # tomllib not available (Python < 3.11)
                pass
            except Exception as e:
                logger.debug(f"Error reading pyproject.toml: {e}")

        # Check for common files
        if (repo_root / "requirements.txt").exists():
            config_files.append("requirements.txt")
            if input_data.command_type in ("all", "build"):
                commands.append(
                    ProjectCommand(
                        name="install",
                        command="pip install -r requirements.txt",
                        type="build",
                        source="requirements.txt",
                    )
                )

        if (repo_root / "Cargo.toml").exists():
            config_files.append("Cargo.toml")
            if input_data.command_type in ("all", "build"):
                commands.append(
                    ProjectCommand(
                        name="build",
                        command="cargo build",
                        type="build",
                        source="Cargo.toml",
                    )
                )
            if input_data.command_type in ("all", "test"):
                commands.append(
                    ProjectCommand(
                        name="test",
                        command="cargo test",
                        type="test",
                        source="Cargo.toml",
                    )
                )

        if (repo_root / "go.mod").exists():
            config_files.append("go.mod")
            if input_data.command_type in ("all", "build"):
                commands.append(
                    ProjectCommand(
                        name="build",
                        command="go build ./...",
                        type="build",
                        source="go.mod",
                    )
                )
            if input_data.command_type in ("all", "test"):
                commands.append(
                    ProjectCommand(
                        name="test",
                        command="go test ./...",
                        type="test",
                        source="go.mod",
                    )
                )

        return ProjectCommandsOutput(
            request_id=request_id,
            commands=commands,
            config_files=config_files,
        )

    def _classify_npm_script(self, name: str) -> str:
        """Classify an npm script by name."""
        name_lower = name.lower()

        if "test" in name_lower or "jest" in name_lower or "vitest" in name_lower:
            return "test"
        if "lint" in name_lower or "eslint" in name_lower:
            return "lint"
        if "format" in name_lower or "prettier" in name_lower:
            return "format"
        if "build" in name_lower or "compile" in name_lower:
            return "build"
        if name_lower in ("start", "dev", "serve"):
            return "run"

        return "other"

    def _classify_make_target(self, name: str) -> str:
        """Classify a Makefile target by name."""
        name_lower = name.lower()

        if "test" in name_lower:
            return "test"
        if "lint" in name_lower or "check" in name_lower:
            return "lint"
        if "format" in name_lower or "fmt" in name_lower:
            return "format"
        if "build" in name_lower or "compile" in name_lower:
            return "build"
        if "run" in name_lower or "serve" in name_lower:
            return "run"

        return "other"

    def _classify_python_script(self, name: str) -> str:
        """Classify a Python script by name."""
        name_lower = name.lower()

        if "test" in name_lower:
            return "test"
        if "lint" in name_lower:
            return "lint"
        if "format" in name_lower:
            return "format"

        return "run"
