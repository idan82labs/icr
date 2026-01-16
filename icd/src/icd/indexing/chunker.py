"""
Symbol-based code chunking with tree-sitter.

Chunks code into semantic units (functions, classes, methods) while
respecting token limits and maintaining stable content-based IDs.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

try:
    import tiktoken
except ImportError:
    tiktoken = None  # type: ignore

if TYPE_CHECKING:
    from icd.config import Config

logger = structlog.get_logger(__name__)


# Language to tree-sitter module mapping
LANGUAGE_MODULES = {
    ".py": ("tree_sitter_python", "python"),
    ".js": ("tree_sitter_javascript", "javascript"),
    ".jsx": ("tree_sitter_javascript", "javascript"),
    ".ts": ("tree_sitter_typescript", "typescript"),
    ".tsx": ("tree_sitter_typescript", "tsx"),
    ".go": ("tree_sitter_go", "go"),
    ".rs": ("tree_sitter_rust", "rust"),
    ".java": ("tree_sitter_java", "java"),
    ".c": ("tree_sitter_c", "c"),
    ".cpp": ("tree_sitter_cpp", "cpp"),
    ".h": ("tree_sitter_c", "c"),
    ".hpp": ("tree_sitter_cpp", "cpp"),
}

# Symbol types to extract by language
SYMBOL_TYPES = {
    "python": [
        "function_definition",
        "class_definition",
        "decorated_definition",
    ],
    "javascript": [
        "function_declaration",
        "class_declaration",
        "method_definition",
        "arrow_function",
        "function",
    ],
    "typescript": [
        "function_declaration",
        "class_declaration",
        "method_definition",
        "arrow_function",
        "interface_declaration",
        "type_alias_declaration",
    ],
    "tsx": [
        "function_declaration",
        "class_declaration",
        "method_definition",
        "arrow_function",
        "interface_declaration",
        "type_alias_declaration",
    ],
    "go": [
        "function_declaration",
        "method_declaration",
        "type_declaration",
    ],
    "rust": [
        "function_item",
        "impl_item",
        "struct_item",
        "enum_item",
        "trait_item",
    ],
    "java": [
        "method_declaration",
        "class_declaration",
        "interface_declaration",
    ],
    "c": [
        "function_definition",
        "struct_specifier",
    ],
    "cpp": [
        "function_definition",
        "class_specifier",
        "struct_specifier",
    ],
}


@dataclass
class Chunk:
    """Represents a code chunk."""

    chunk_id: str
    file_path: str
    content: str
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    symbol_name: str | None
    symbol_type: str | None
    language: str
    token_count: int
    is_contract: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class TokenCounter:
    """Token counter with fallback."""

    def __init__(self, model: str = "cl100k_base") -> None:
        """Initialize token counter."""
        self._encoder = None
        self._model = model

        if tiktoken:
            try:
                self._encoder = tiktoken.get_encoding(model)
            except Exception:
                pass

    def count(self, text: str) -> int:
        """Count tokens in text."""
        if self._encoder:
            return len(self._encoder.encode(text))

        # Fallback: rough estimate based on characters
        # Average of ~4 characters per token for code
        return len(text) // 4


class Chunker:
    """
    Symbol-based code chunker using tree-sitter.

    Features:
    - Language-aware parsing
    - Symbol extraction (functions, classes, methods)
    - Token-based size limits
    - Stable content-hash IDs
    - Fallback to line-based chunking
    """

    def __init__(self, config: "Config") -> None:
        """
        Initialize the chunker.

        Args:
            config: ICD configuration.
        """
        self.config = config
        self.min_tokens = config.chunking.min_tokens
        self.target_tokens = config.chunking.target_tokens
        self.max_tokens = config.chunking.max_tokens
        self.overlap_tokens = config.chunking.overlap_tokens
        self.preserve_symbols = config.chunking.preserve_symbols

        self._token_counter = TokenCounter()
        self._parsers: dict[str, Any] = {}
        self._languages: dict[str, Any] = {}

    def _get_parser(self, language: str) -> Any | None:
        """Get or create a tree-sitter parser for a language."""
        if language in self._parsers:
            return self._parsers[language]

        try:
            import tree_sitter

            # Find the language module
            module_name = None
            for ext, (mod, lang) in LANGUAGE_MODULES.items():
                if lang == language:
                    module_name = mod
                    break

            if not module_name:
                return None

            # Import the language module
            lang_module = __import__(module_name)
            lang_func = getattr(lang_module, "language", None)

            if not lang_func:
                return None

            # Create parser
            ts_language = tree_sitter.Language(lang_func())
            parser = tree_sitter.Parser(ts_language)

            self._parsers[language] = parser
            self._languages[language] = ts_language

            return parser

        except Exception as e:
            logger.debug(
                "Failed to create parser",
                language=language,
                error=str(e),
            )
            return None

    def _detect_language(self, path: Path) -> str:
        """Detect language from file extension."""
        ext = path.suffix.lower()
        if ext in LANGUAGE_MODULES:
            return LANGUAGE_MODULES[ext][1]
        return "text"

    def _generate_chunk_id(self, file_path: str, content: str) -> str:
        """Generate a stable chunk ID based on content."""
        combined = f"{file_path}:{hashlib.sha256(content.encode()).hexdigest()[:8]}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def chunk_file(self, path: Path, content: str) -> list[Chunk]:
        """
        Chunk a file into semantic units.

        Args:
            path: File path.
            content: File content.

        Returns:
            List of chunks.
        """
        str_path = str(path)
        language = self._detect_language(path)

        # Try symbol-based chunking first
        if self.preserve_symbols and language != "text":
            parser = self._get_parser(language)
            if parser:
                chunks = self._chunk_with_treesitter(
                    str_path,
                    content,
                    language,
                    parser,
                )
                if chunks:
                    return chunks

        # Fallback to line-based chunking
        return self._chunk_by_lines(str_path, content, language)

    def _chunk_with_treesitter(
        self,
        file_path: str,
        content: str,
        language: str,
        parser: Any,
    ) -> list[Chunk]:
        """Chunk code using tree-sitter for symbol extraction."""
        try:
            tree = parser.parse(content.encode("utf-8"))
            root_node = tree.root_node

            # Extract symbols
            symbol_types = SYMBOL_TYPES.get(language, [])
            symbols = self._extract_symbols(root_node, symbol_types, content)

            if not symbols:
                return []

            chunks = []
            content_lines = content.split("\n")

            for symbol in symbols:
                node, symbol_name, symbol_type = symbol

                # Get symbol content
                start_line = node.start_point[0]
                end_line = node.end_point[0]
                start_byte = node.start_byte
                end_byte = node.end_byte

                symbol_content = content[start_byte:end_byte]
                token_count = self._token_counter.count(symbol_content)

                # Handle oversized symbols
                if token_count > self.max_tokens:
                    # Split the symbol into smaller chunks
                    sub_chunks = self._split_large_symbol(
                        file_path,
                        symbol_content,
                        start_line,
                        start_byte,
                        language,
                        symbol_name,
                        symbol_type,
                    )
                    chunks.extend(sub_chunks)
                elif token_count >= self.min_tokens:
                    # Create chunk from symbol
                    chunk = Chunk(
                        chunk_id=self._generate_chunk_id(file_path, symbol_content),
                        file_path=file_path,
                        content=symbol_content,
                        start_line=start_line + 1,  # 1-indexed
                        end_line=end_line + 1,
                        start_byte=start_byte,
                        end_byte=end_byte,
                        symbol_name=symbol_name,
                        symbol_type=symbol_type,
                        language=language,
                        token_count=token_count,
                    )
                    chunks.append(chunk)

            # Handle gaps between symbols (imports, module-level code)
            chunks.extend(
                self._chunk_gaps(file_path, content, language, symbols)
            )

            # Sort by start position
            chunks.sort(key=lambda c: c.start_byte)

            return chunks

        except Exception as e:
            logger.debug(
                "Tree-sitter parsing failed",
                file_path=file_path,
                error=str(e),
            )
            return []

    def _extract_symbols(
        self,
        node: Any,
        symbol_types: list[str],
        content: str,
    ) -> list[tuple[Any, str | None, str]]:
        """Recursively extract symbols from AST."""
        symbols = []

        if node.type in symbol_types:
            # Extract symbol name
            symbol_name = self._get_symbol_name(node, content)
            symbols.append((node, symbol_name, node.type))
        else:
            # Recurse into children
            for child in node.children:
                symbols.extend(
                    self._extract_symbols(child, symbol_types, content)
                )

        return symbols

    def _get_symbol_name(self, node: Any, content: str) -> str | None:
        """Extract the name of a symbol from its AST node."""
        # Look for name/identifier child nodes
        for child in node.children:
            if child.type in ("identifier", "name", "property_identifier"):
                return content[child.start_byte : child.end_byte]

            # For decorated definitions, look deeper
            if child.type in ("function_definition", "class_definition"):
                return self._get_symbol_name(child, content)

        return None

    def _split_large_symbol(
        self,
        file_path: str,
        content: str,
        base_line: int,
        base_byte: int,
        language: str,
        symbol_name: str | None,
        symbol_type: str,
    ) -> list[Chunk]:
        """Split a large symbol into smaller chunks."""
        chunks = []
        lines = content.split("\n")
        current_chunk_lines: list[str] = []
        current_start_line = 0
        current_start_byte = 0

        for i, line in enumerate(lines):
            current_chunk_lines.append(line)
            chunk_content = "\n".join(current_chunk_lines)
            token_count = self._token_counter.count(chunk_content)

            if token_count >= self.target_tokens or i == len(lines) - 1:
                if token_count >= self.min_tokens or i == len(lines) - 1:
                    # Calculate byte offset
                    start_byte = base_byte + current_start_byte
                    end_byte = start_byte + len(chunk_content.encode("utf-8"))

                    chunk = Chunk(
                        chunk_id=self._generate_chunk_id(file_path, chunk_content),
                        file_path=file_path,
                        content=chunk_content,
                        start_line=base_line + current_start_line + 1,
                        end_line=base_line + i + 1,
                        start_byte=start_byte,
                        end_byte=end_byte,
                        symbol_name=symbol_name,
                        symbol_type=symbol_type,
                        language=language,
                        token_count=token_count,
                    )
                    chunks.append(chunk)

                    # Start new chunk with overlap
                    if i < len(lines) - 1:
                        overlap_lines = self._get_overlap_lines(
                            current_chunk_lines
                        )
                        current_chunk_lines = overlap_lines
                        current_start_line = i - len(overlap_lines) + 1
                        current_start_byte = sum(
                            len(l.encode("utf-8")) + 1
                            for l in lines[: current_start_line]
                        )

        return chunks

    def _get_overlap_lines(self, lines: list[str]) -> list[str]:
        """Get overlap lines based on token count."""
        if not lines or self.overlap_tokens == 0:
            return []

        overlap = []
        token_count = 0

        for line in reversed(lines):
            line_tokens = self._token_counter.count(line)
            if token_count + line_tokens > self.overlap_tokens:
                break
            overlap.insert(0, line)
            token_count += line_tokens

        return overlap

    def _chunk_gaps(
        self,
        file_path: str,
        content: str,
        language: str,
        symbols: list[tuple[Any, str | None, str]],
    ) -> list[Chunk]:
        """Chunk code that falls between symbols (imports, etc.)."""
        if not symbols:
            return []

        chunks = []
        lines = content.split("\n")

        # Sort symbols by position
        sorted_symbols = sorted(symbols, key=lambda s: s[0].start_byte)

        # Check for content before first symbol
        if sorted_symbols:
            first_symbol = sorted_symbols[0][0]
            if first_symbol.start_line > 0:
                gap_content = content[: first_symbol.start_byte].rstrip()
                if gap_content:
                    token_count = self._token_counter.count(gap_content)
                    if token_count >= self.min_tokens:
                        chunks.append(
                            Chunk(
                                chunk_id=self._generate_chunk_id(
                                    file_path, gap_content
                                ),
                                file_path=file_path,
                                content=gap_content,
                                start_line=1,
                                end_line=first_symbol.start_point[0],
                                start_byte=0,
                                end_byte=first_symbol.start_byte,
                                symbol_name=None,
                                symbol_type="module_header",
                                language=language,
                                token_count=token_count,
                            )
                        )

        return chunks

    def _chunk_by_lines(
        self,
        file_path: str,
        content: str,
        language: str,
    ) -> list[Chunk]:
        """Fallback line-based chunking."""
        chunks = []
        lines = content.split("\n")

        current_lines: list[str] = []
        current_start_line = 0
        current_start_byte = 0

        for i, line in enumerate(lines):
            current_lines.append(line)
            chunk_content = "\n".join(current_lines)
            token_count = self._token_counter.count(chunk_content)

            # Check for natural break points
            is_break = self._is_natural_break(line, lines[i + 1] if i + 1 < len(lines) else "")

            if (token_count >= self.target_tokens and is_break) or i == len(lines) - 1:
                if token_count >= self.min_tokens or i == len(lines) - 1:
                    end_byte = current_start_byte + len(chunk_content.encode("utf-8"))

                    chunk = Chunk(
                        chunk_id=self._generate_chunk_id(file_path, chunk_content),
                        file_path=file_path,
                        content=chunk_content,
                        start_line=current_start_line + 1,
                        end_line=i + 1,
                        start_byte=current_start_byte,
                        end_byte=end_byte,
                        symbol_name=None,
                        symbol_type="text_block",
                        language=language,
                        token_count=token_count,
                    )
                    chunks.append(chunk)

                    # Start new chunk with overlap
                    if i < len(lines) - 1:
                        overlap_lines = self._get_overlap_lines(current_lines)
                        current_lines = overlap_lines
                        current_start_line = i - len(overlap_lines) + 1
                        current_start_byte = sum(
                            len(l.encode("utf-8")) + 1
                            for l in lines[: current_start_line]
                        )

        return chunks

    def _is_natural_break(self, current_line: str, next_line: str) -> bool:
        """Check if this is a natural break point for chunking."""
        # Empty line
        if not current_line.strip():
            return True

        # End of block (less indentation)
        current_indent = len(current_line) - len(current_line.lstrip())
        next_indent = len(next_line) - len(next_line.lstrip()) if next_line else 0

        if next_indent < current_indent and next_line.strip():
            return True

        # Comment line followed by code
        if current_line.strip().startswith(("#", "//", "*", "/*")):
            if next_line.strip() and not next_line.strip().startswith(
                ("#", "//", "*", "/*")
            ):
                return True

        return False

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return self._token_counter.count(text)
