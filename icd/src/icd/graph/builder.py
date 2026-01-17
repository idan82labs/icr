"""
Code graph builder using tree-sitter AST.

Extracts structural relationships from code:
- Imports/exports
- Function calls
- Class inheritance
- Type references

Reference: "GraphCodeBERT" (Guo et al., 2020), "CodeTF" (Nguyen et al., 2023)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from icd.config import Config
    from icd.storage.sqlite_store import SQLiteStore

logger = structlog.get_logger(__name__)


class NodeType(str, Enum):
    """Types of nodes in the code graph."""

    FILE = "file"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    INTERFACE = "interface"
    TYPE = "type"
    MODULE = "module"
    VARIABLE = "variable"


class EdgeType(str, Enum):
    """Types of edges in the code graph."""

    IMPORTS = "imports"  # File imports module
    EXPORTS = "exports"  # Module exports symbol
    CALLS = "calls"  # Function calls function
    INHERITS = "inherits"  # Class inherits from class
    IMPLEMENTS = "implements"  # Class implements interface
    USES_TYPE = "uses_type"  # Symbol uses type
    CONTAINS = "contains"  # File/class contains symbol
    REFERENCES = "references"  # Generic reference


@dataclass
class GraphNode:
    """Node in the code graph."""

    node_id: str
    node_type: NodeType
    name: str
    file_path: str
    start_line: int
    end_line: int
    chunk_id: str | None = None  # Link to retrieval chunk
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    """Edge in the code graph."""

    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


# Language-specific AST patterns for relationship extraction
LANGUAGE_PATTERNS = {
    "python": {
        "import_patterns": [
            "import_statement",
            "import_from_statement",
        ],
        "class_patterns": ["class_definition"],
        "function_patterns": ["function_definition", "decorated_definition"],
        "call_patterns": ["call"],
        "inheritance_field": "argument_list",  # class Foo(Bar):
    },
    "typescript": {
        "import_patterns": ["import_statement"],
        "export_patterns": ["export_statement"],
        "class_patterns": ["class_declaration"],
        "function_patterns": ["function_declaration", "method_definition", "arrow_function"],
        "call_patterns": ["call_expression"],
        "interface_patterns": ["interface_declaration"],
        "type_patterns": ["type_alias_declaration"],
        "inheritance_field": "class_heritage",
    },
    "javascript": {
        "import_patterns": ["import_statement"],
        "export_patterns": ["export_statement"],
        "class_patterns": ["class_declaration"],
        "function_patterns": ["function_declaration", "method_definition", "arrow_function"],
        "call_patterns": ["call_expression"],
        "inheritance_field": "class_heritage",
    },
    "go": {
        "import_patterns": ["import_declaration"],
        "function_patterns": ["function_declaration", "method_declaration"],
        "call_patterns": ["call_expression"],
        "type_patterns": ["type_declaration"],
    },
    "rust": {
        "import_patterns": ["use_declaration"],
        "function_patterns": ["function_item"],
        "call_patterns": ["call_expression"],
        "struct_patterns": ["struct_item"],
        "trait_patterns": ["trait_item", "impl_item"],
    },
}


class CodeGraphBuilder:
    """
    Builds code graphs from AST analysis.

    The graph captures structural relationships for multi-hop retrieval:
    - If searching for "authentication", find auth.py
    - Graph shows auth.py imports UserService
    - UserService implements AuthProvider interface
    - Multi-hop retrieval pulls in all related code
    """

    def __init__(self, config: "Config") -> None:
        self.config = config
        self._parsers: dict[str, Any] = {}
        self._graph_nodes: dict[str, GraphNode] = {}
        self._graph_edges: list[GraphEdge] = []

        # Symbol table: fully qualified name -> node_id
        self._symbol_table: dict[str, str] = {}

    def _get_parser(self, language: str) -> Any | None:
        """Get or create a tree-sitter parser."""
        if language in self._parsers:
            return self._parsers[language]

        try:
            import tree_sitter

            # Language module mapping
            module_map = {
                "python": "tree_sitter_python",
                "typescript": "tree_sitter_typescript",
                "javascript": "tree_sitter_javascript",
                "go": "tree_sitter_go",
                "rust": "tree_sitter_rust",
            }

            module_name = module_map.get(language)
            if not module_name:
                return None

            lang_module = __import__(module_name)
            lang_func = getattr(lang_module, "language", None)

            if not lang_func:
                return None

            ts_language = tree_sitter.Language(lang_func())
            parser = tree_sitter.Parser(ts_language)

            self._parsers[language] = parser
            return parser

        except Exception as e:
            logger.debug("Failed to create parser", language=language, error=str(e))
            return None

    def build_from_files(self, files: list[tuple[Path, str, str]]) -> None:
        """
        Build graph from list of (path, content, language) tuples.

        This is a two-pass algorithm:
        1. First pass: Extract all symbols and create nodes
        2. Second pass: Resolve references and create edges
        """
        # Pass 1: Extract symbols
        for path, content, language in files:
            self._extract_symbols(path, content, language)

        # Pass 2: Resolve references
        for path, content, language in files:
            self._extract_relationships(path, content, language)

        logger.info(
            "Code graph built",
            nodes=len(self._graph_nodes),
            edges=len(self._graph_edges),
        )

    def _extract_symbols(self, path: Path, content: str, language: str) -> None:
        """Extract symbol definitions and create nodes."""
        str_path = str(path)

        # Create file node
        file_node_id = f"file:{str_path}"
        self._graph_nodes[file_node_id] = GraphNode(
            node_id=file_node_id,
            node_type=NodeType.FILE,
            name=path.name,
            file_path=str_path,
            start_line=1,
            end_line=content.count("\n") + 1,
        )

        parser = self._get_parser(language)
        if not parser:
            return

        patterns = LANGUAGE_PATTERNS.get(language, {})

        try:
            tree = parser.parse(content.encode("utf-8"))
            root = tree.root_node

            # Extract classes
            for pattern in patterns.get("class_patterns", []):
                self._extract_class_nodes(root, pattern, str_path, content, language)

            # Extract functions
            for pattern in patterns.get("function_patterns", []):
                self._extract_function_nodes(root, pattern, str_path, content, language)

            # Extract interfaces (TypeScript)
            for pattern in patterns.get("interface_patterns", []):
                self._extract_interface_nodes(root, pattern, str_path, content, language)

            # Extract types
            for pattern in patterns.get("type_patterns", []):
                self._extract_type_nodes(root, pattern, str_path, content, language)

        except Exception as e:
            logger.debug("Symbol extraction failed", path=str_path, error=str(e))

    def _extract_class_nodes(
        self, root: Any, pattern: str, file_path: str, content: str, language: str
    ) -> None:
        """Extract class definitions."""
        for node in self._find_nodes_by_type(root, pattern):
            name = self._get_node_name(node, content)
            if not name:
                continue

            node_id = f"class:{file_path}:{name}"
            fqn = f"{file_path}:{name}"

            self._graph_nodes[node_id] = GraphNode(
                node_id=node_id,
                node_type=NodeType.CLASS,
                name=name,
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
            )
            self._symbol_table[fqn] = node_id
            self._symbol_table[name] = node_id  # Short name for local resolution

            # Contains edge from file
            self._graph_edges.append(
                GraphEdge(
                    source_id=f"file:{file_path}",
                    target_id=node_id,
                    edge_type=EdgeType.CONTAINS,
                )
            )

            # Extract methods within class
            self._extract_methods_from_class(node, node_id, file_path, content, language)

    def _extract_methods_from_class(
        self, class_node: Any, class_id: str, file_path: str, content: str, language: str
    ) -> None:
        """Extract method definitions within a class."""
        method_patterns = ["function_definition", "method_definition"]

        for pattern in method_patterns:
            for node in self._find_nodes_by_type(class_node, pattern):
                name = self._get_node_name(node, content)
                if not name:
                    continue

                # Get class name from class_id
                class_name = class_id.split(":")[-1]
                node_id = f"method:{file_path}:{class_name}.{name}"

                self._graph_nodes[node_id] = GraphNode(
                    node_id=node_id,
                    node_type=NodeType.METHOD,
                    name=name,
                    file_path=file_path,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                )

                # Contains edge from class
                self._graph_edges.append(
                    GraphEdge(
                        source_id=class_id,
                        target_id=node_id,
                        edge_type=EdgeType.CONTAINS,
                    )
                )

    def _extract_function_nodes(
        self, root: Any, pattern: str, file_path: str, content: str, language: str
    ) -> None:
        """Extract top-level function definitions."""
        for node in self._find_nodes_by_type(root, pattern):
            # Skip if inside a class
            if self._is_inside_class(node, language):
                continue

            name = self._get_node_name(node, content)
            if not name:
                continue

            node_id = f"function:{file_path}:{name}"
            fqn = f"{file_path}:{name}"

            self._graph_nodes[node_id] = GraphNode(
                node_id=node_id,
                node_type=NodeType.FUNCTION,
                name=name,
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
            )
            self._symbol_table[fqn] = node_id
            self._symbol_table[name] = node_id

            # Contains edge from file
            self._graph_edges.append(
                GraphEdge(
                    source_id=f"file:{file_path}",
                    target_id=node_id,
                    edge_type=EdgeType.CONTAINS,
                )
            )

    def _extract_interface_nodes(
        self, root: Any, pattern: str, file_path: str, content: str, language: str
    ) -> None:
        """Extract interface definitions (TypeScript)."""
        for node in self._find_nodes_by_type(root, pattern):
            name = self._get_node_name(node, content)
            if not name:
                continue

            node_id = f"interface:{file_path}:{name}"

            self._graph_nodes[node_id] = GraphNode(
                node_id=node_id,
                node_type=NodeType.INTERFACE,
                name=name,
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
            )
            self._symbol_table[name] = node_id

    def _extract_type_nodes(
        self, root: Any, pattern: str, file_path: str, content: str, language: str
    ) -> None:
        """Extract type alias definitions."""
        for node in self._find_nodes_by_type(root, pattern):
            name = self._get_node_name(node, content)
            if not name:
                continue

            node_id = f"type:{file_path}:{name}"

            self._graph_nodes[node_id] = GraphNode(
                node_id=node_id,
                node_type=NodeType.TYPE,
                name=name,
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
            )
            self._symbol_table[name] = node_id

    def _extract_relationships(self, path: Path, content: str, language: str) -> None:
        """Extract relationships (imports, calls, inheritance)."""
        str_path = str(path)
        parser = self._get_parser(language)
        if not parser:
            return

        patterns = LANGUAGE_PATTERNS.get(language, {})

        try:
            tree = parser.parse(content.encode("utf-8"))
            root = tree.root_node

            # Extract imports
            for pattern in patterns.get("import_patterns", []):
                self._extract_imports(root, pattern, str_path, content, language)

            # Extract function calls
            for pattern in patterns.get("call_patterns", []):
                self._extract_calls(root, pattern, str_path, content, language)

            # Extract inheritance
            self._extract_inheritance(root, str_path, content, language, patterns)

        except Exception as e:
            logger.debug("Relationship extraction failed", path=str_path, error=str(e))

    def _extract_imports(
        self, root: Any, pattern: str, file_path: str, content: str, language: str
    ) -> None:
        """Extract import statements."""
        file_node_id = f"file:{file_path}"

        for node in self._find_nodes_by_type(root, pattern):
            # Extract imported module/symbol names
            import_text = content[node.start_byte : node.end_byte]

            if language == "python":
                # Handle: import foo, from foo import bar
                imported = self._parse_python_import(import_text)
            elif language in ("typescript", "javascript"):
                # Handle: import { foo } from './bar'
                imported = self._parse_js_import(import_text)
            else:
                imported = []

            for module_name in imported:
                # Try to resolve to a known symbol
                target_id = self._resolve_symbol(module_name, file_path)
                if target_id:
                    self._graph_edges.append(
                        GraphEdge(
                            source_id=file_node_id,
                            target_id=target_id,
                            edge_type=EdgeType.IMPORTS,
                            metadata={"import_text": import_text},
                        )
                    )

    def _extract_calls(
        self, root: Any, pattern: str, file_path: str, content: str, language: str
    ) -> None:
        """Extract function calls."""
        for node in self._find_nodes_by_type(root, pattern):
            # Get the function being called
            callee_name = self._get_callee_name(node, content)
            if not callee_name:
                continue

            # Get the containing function/method
            caller_id = self._get_containing_symbol(node, file_path, content)
            if not caller_id:
                continue

            # Try to resolve callee to a known symbol
            target_id = self._resolve_symbol(callee_name, file_path)
            if target_id and target_id != caller_id:  # Avoid self-references
                self._graph_edges.append(
                    GraphEdge(
                        source_id=caller_id,
                        target_id=target_id,
                        edge_type=EdgeType.CALLS,
                    )
                )

    def _extract_inheritance(
        self, root: Any, file_path: str, content: str, language: str, patterns: dict
    ) -> None:
        """Extract class inheritance relationships."""
        for class_pattern in patterns.get("class_patterns", []):
            for node in self._find_nodes_by_type(root, class_pattern):
                class_name = self._get_node_name(node, content)
                if not class_name:
                    continue

                class_id = f"class:{file_path}:{class_name}"

                # Find base classes
                base_classes = self._get_base_classes(node, content, language)

                for base_name in base_classes:
                    target_id = self._resolve_symbol(base_name, file_path)
                    if target_id:
                        self._graph_edges.append(
                            GraphEdge(
                                source_id=class_id,
                                target_id=target_id,
                                edge_type=EdgeType.INHERITS,
                            )
                        )

    def _find_nodes_by_type(self, root: Any, node_type: str) -> list[Any]:
        """Recursively find all nodes of a given type."""
        results = []
        if root.type == node_type:
            results.append(root)
        for child in root.children:
            results.extend(self._find_nodes_by_type(child, node_type))
        return results

    def _get_node_name(self, node: Any, content: str) -> str | None:
        """Extract name from a definition node."""
        for child in node.children:
            if child.type in ("identifier", "name", "property_identifier", "type_identifier"):
                return content[child.start_byte : child.end_byte]
            # Handle decorated definitions
            if child.type in ("function_definition", "class_definition"):
                return self._get_node_name(child, content)
        return None

    def _get_callee_name(self, call_node: Any, content: str) -> str | None:
        """Extract the function name being called."""
        for child in call_node.children:
            if child.type in ("identifier", "property_identifier"):
                return content[child.start_byte : child.end_byte]
            if child.type == "member_expression":
                # Get the property being accessed (e.g., obj.method -> method)
                for sub in child.children:
                    if sub.type == "property_identifier":
                        return content[sub.start_byte : sub.end_byte]
            if child.type == "attribute":
                # Python: obj.method
                for sub in child.children:
                    if sub.type == "identifier":
                        return content[sub.start_byte : sub.end_byte]
        return None

    def _get_containing_symbol(self, node: Any, file_path: str, content: str) -> str | None:
        """Find the containing function/method for a node."""
        current = node.parent
        while current:
            if current.type in (
                "function_definition",
                "method_definition",
                "function_declaration",
            ):
                name = self._get_node_name(current, content)
                if name:
                    # Check if inside a class
                    class_node = self._find_parent_class(current)
                    if class_node:
                        class_name = self._get_node_name(class_node, content)
                        if class_name:
                            return f"method:{file_path}:{class_name}.{name}"
                    return f"function:{file_path}:{name}"
            current = current.parent
        return None

    def _find_parent_class(self, node: Any) -> Any | None:
        """Find parent class node if any."""
        current = node.parent
        while current:
            if current.type in ("class_definition", "class_declaration"):
                return current
            current = current.parent
        return None

    def _is_inside_class(self, node: Any, language: str) -> bool:
        """Check if node is inside a class definition."""
        return self._find_parent_class(node) is not None

    def _resolve_symbol(self, name: str, current_file: str) -> str | None:
        """Resolve a symbol name to a node ID."""
        # Try fully qualified name first
        fqn = f"{current_file}:{name}"
        if fqn in self._symbol_table:
            return self._symbol_table[fqn]

        # Try short name (for imports from other files)
        if name in self._symbol_table:
            return self._symbol_table[name]

        return None

    def _get_base_classes(self, class_node: Any, content: str, language: str) -> list[str]:
        """Extract base class names from class definition."""
        bases = []

        if language == "python":
            # Look for argument_list child (Python: class Foo(Bar, Baz))
            for child in class_node.children:
                if child.type == "argument_list":
                    for arg in child.children:
                        if arg.type == "identifier":
                            bases.append(content[arg.start_byte : arg.end_byte])

        elif language in ("typescript", "javascript"):
            # Look for class_heritage (extends Foo)
            for child in class_node.children:
                if child.type == "class_heritage":
                    for sub in child.children:
                        if sub.type == "extends_clause":
                            for name_node in sub.children:
                                if name_node.type in ("identifier", "type_identifier"):
                                    bases.append(content[name_node.start_byte : name_node.end_byte])

        return bases

    def _parse_python_import(self, import_text: str) -> list[str]:
        """Parse Python import statement."""
        imported = []

        # from X import Y, Z
        from_match = re.match(r"from\s+([\w.]+)\s+import\s+(.+)", import_text)
        if from_match:
            module = from_match.group(1)
            imported.append(module)
            # Add individual imports
            names = from_match.group(2).split(",")
            for name in names:
                name = name.strip().split(" as ")[0].strip()
                if name and name != "*":
                    imported.append(name)
            return imported

        # import X, Y
        import_match = re.match(r"import\s+(.+)", import_text)
        if import_match:
            modules = import_match.group(1).split(",")
            for mod in modules:
                mod = mod.strip().split(" as ")[0].strip()
                if mod:
                    imported.append(mod)

        return imported

    def _parse_js_import(self, import_text: str) -> list[str]:
        """Parse JavaScript/TypeScript import statement."""
        imported = []

        # import { X, Y } from './module'
        named_match = re.search(r"\{([^}]+)\}", import_text)
        if named_match:
            names = named_match.group(1).split(",")
            for name in names:
                name = name.strip().split(" as ")[0].strip()
                if name:
                    imported.append(name)

        # import X from './module'
        default_match = re.match(r"import\s+(\w+)\s+from", import_text)
        if default_match:
            imported.append(default_match.group(1))

        # Extract module path
        path_match = re.search(r"from\s+['\"]([^'\"]+)['\"]", import_text)
        if path_match:
            module_path = path_match.group(1)
            # Convert relative path to potential file path
            imported.append(module_path.split("/")[-1])

        return imported

    def get_nodes(self) -> dict[str, GraphNode]:
        """Get all graph nodes."""
        return self._graph_nodes.copy()

    def get_edges(self) -> list[GraphEdge]:
        """Get all graph edges."""
        return self._graph_edges.copy()

    def get_neighbors(self, node_id: str, edge_types: list[EdgeType] | None = None) -> list[str]:
        """Get neighbor node IDs for a given node."""
        neighbors = []
        for edge in self._graph_edges:
            if edge.source_id == node_id:
                if edge_types is None or edge.edge_type in edge_types:
                    neighbors.append(edge.target_id)
            elif edge.target_id == node_id:
                if edge_types is None or edge.edge_type in edge_types:
                    neighbors.append(edge.source_id)
        return neighbors

    def find_node_by_chunk(self, chunk_id: str) -> GraphNode | None:
        """Find a graph node by its associated chunk ID."""
        for node in self._graph_nodes.values():
            if node.chunk_id == chunk_id:
                return node
        return None

    def link_chunks_to_nodes(self, chunks: list[Any]) -> None:
        """Link chunk IDs to graph nodes based on file/line overlap."""
        for chunk in chunks:
            file_path = chunk.file_path
            start_line = chunk.start_line
            end_line = chunk.end_line

            # Find overlapping nodes
            for node in self._graph_nodes.values():
                if node.file_path != file_path:
                    continue

                # Check line overlap
                if (
                    node.start_line <= end_line
                    and node.end_line >= start_line
                ):
                    # Prefer exact symbol matches
                    if (
                        chunk.symbol_name
                        and chunk.symbol_name == node.name
                    ):
                        node.chunk_id = chunk.chunk_id
                        break
                    elif node.chunk_id is None:
                        node.chunk_id = chunk.chunk_id

    def to_networkx(self) -> Any:
        """Export to NetworkX DiGraph for advanced analysis."""
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("networkx required for graph export")

        G = nx.DiGraph()

        # Add nodes
        for node_id, node in self._graph_nodes.items():
            G.add_node(
                node_id,
                node_type=node.node_type.value,
                name=node.name,
                file_path=node.file_path,
                start_line=node.start_line,
                end_line=node.end_line,
                chunk_id=node.chunk_id,
            )

        # Add edges
        for edge in self._graph_edges:
            G.add_edge(
                edge.source_id,
                edge.target_id,
                edge_type=edge.edge_type.value,
                weight=edge.weight,
            )

        return G
