"""
Markdown pack formatter with citations.

Formats compiled packs into well-structured markdown with
proper code blocks, citations, and metadata.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from icd.pack.compiler import PackItem

logger = structlog.get_logger(__name__)


# Language to markdown fence mapping
LANGUAGE_FENCES = {
    "python": "python",
    "javascript": "javascript",
    "typescript": "typescript",
    "tsx": "tsx",
    "jsx": "jsx",
    "go": "go",
    "rust": "rust",
    "java": "java",
    "c": "c",
    "cpp": "cpp",
    "text": "",
}


class PackFormatter:
    """
    Markdown formatter for context packs.

    Features:
    - Proper code fence languages
    - File path headers
    - Citation markers
    - Symbol metadata
    - Contract/pinned indicators
    """

    def __init__(
        self,
        include_metadata: bool = True,
        include_citations: bool = True,
        max_content_lines: int = 100,
    ) -> None:
        """
        Initialize the formatter.

        Args:
            include_metadata: Include file/symbol metadata.
            include_citations: Include citation references.
            max_content_lines: Maximum lines per chunk.
        """
        self.include_metadata = include_metadata
        self.include_citations = include_citations
        self.max_content_lines = max_content_lines

    def format(
        self,
        items: list["PackItem"],
        citations: dict[str, str],
        query: str | None = None,
    ) -> str:
        """
        Format pack items into markdown.

        Args:
            items: Selected pack items.
            citations: Citation mapping.
            query: Original query.

        Returns:
            Formatted markdown string.
        """
        parts: list[str] = []

        # Header
        if query:
            parts.append(f"# Context for: {query}\n")

        # Group by file for better organization
        file_groups: dict[str, list[tuple["PackItem", int]]] = {}
        for i, item in enumerate(items, 1):
            file_path = item.chunk.file_path
            if file_path not in file_groups:
                file_groups[file_path] = []
            file_groups[file_path].append((item, i))

        # Format each file group
        for file_path, group_items in file_groups.items():
            parts.append(self._format_file_group(file_path, group_items))

        # Citation reference section
        if self.include_citations and citations:
            parts.append(self._format_citations(citations))

        return "\n".join(parts)

    def _format_file_group(
        self,
        file_path: str,
        items: list[tuple["PackItem", int]],
    ) -> str:
        """Format a group of chunks from the same file."""
        parts: list[str] = []

        # File header
        parts.append(f"## {file_path}\n")

        # Sort by line number
        items.sort(key=lambda x: x[0].chunk.start_line)

        for item, cite_num in items:
            chunk = item.chunk
            content = chunk.content

            # Chunk header with metadata
            header_parts = []

            if self.include_citations:
                header_parts.append(f"[{cite_num}]")

            if chunk.symbol_name:
                header_parts.append(f"**{chunk.symbol_name}**")

            if chunk.symbol_type:
                header_parts.append(f"({chunk.symbol_type})")

            header_parts.append(f"L{chunk.start_line}-{chunk.end_line}")

            # Indicators
            indicators = []
            if chunk.is_contract:
                indicators.append("contract")
            if chunk.is_pinned:
                indicators.append("pinned")

            if indicators:
                header_parts.append(f"[{', '.join(indicators)}]")

            if self.include_metadata and header_parts:
                parts.append(" ".join(header_parts))
                parts.append("")

            # Code block
            fence_lang = LANGUAGE_FENCES.get(chunk.language, "")
            parts.append(f"```{fence_lang}")

            # Limit content lines if needed
            lines = content.split("\n")
            if len(lines) > self.max_content_lines:
                lines = lines[: self.max_content_lines]
                lines.append(f"... ({len(content.split(chr(10))) - self.max_content_lines} more lines)")

            parts.append("\n".join(lines))
            parts.append("```\n")

        return "\n".join(parts)

    def _format_citations(self, citations: dict[str, str]) -> str:
        """Format the citation reference section."""
        parts = ["---", "### References\n"]

        for cite_key, cite_value in citations.items():
            parts.append(f"- {cite_key} {cite_value}")

        return "\n".join(parts)

    def format_compact(
        self,
        items: list["PackItem"],
    ) -> str:
        """
        Format pack items in a compact format.

        Optimized for token efficiency.

        Args:
            items: Selected pack items.

        Returns:
            Compact formatted string.
        """
        parts: list[str] = []

        for item in items:
            chunk = item.chunk

            # Compact header
            header = f"# {chunk.file_path}:{chunk.start_line}"
            if chunk.symbol_name:
                header += f" ({chunk.symbol_name})"
            parts.append(header)

            # Code without markdown fence
            parts.append(chunk.content)
            parts.append("")

        return "\n".join(parts)

    def format_xml(
        self,
        items: list["PackItem"],
        query: str | None = None,
    ) -> str:
        """
        Format pack items as XML for structured LLM prompts.

        Args:
            items: Selected pack items.
            query: Original query.

        Returns:
            XML formatted string.
        """
        parts: list[str] = ["<context>"]

        if query:
            parts.append(f"  <query>{self._escape_xml(query)}</query>")

        parts.append("  <chunks>")

        for i, item in enumerate(items, 1):
            chunk = item.chunk
            parts.append(f'    <chunk id="{i}">')
            parts.append(f"      <file>{self._escape_xml(chunk.file_path)}</file>")
            parts.append(f"      <lines>{chunk.start_line}-{chunk.end_line}</lines>")

            if chunk.symbol_name:
                parts.append(
                    f"      <symbol>{self._escape_xml(chunk.symbol_name)}</symbol>"
                )

            if chunk.symbol_type:
                parts.append(f"      <type>{chunk.symbol_type}</type>")

            parts.append(f"      <language>{chunk.language}</language>")

            if chunk.is_contract:
                parts.append("      <contract>true</contract>")

            if chunk.is_pinned:
                parts.append("      <pinned>true</pinned>")

            parts.append(
                f"      <content><![CDATA[{chunk.content}]]></content>"
            )
            parts.append("    </chunk>")

        parts.append("  </chunks>")
        parts.append("</context>")

        return "\n".join(parts)

    def _escape_xml(self, text: str) -> str:
        """Escape special XML characters."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
        )


class DiffFormatter:
    """
    Format pack differences for context updates.

    Useful for streaming scenarios where only changes need
    to be communicated.
    """

    def __init__(self) -> None:
        """Initialize the diff formatter."""
        self._previous_ids: set[str] = set()

    def format_diff(
        self,
        items: list["PackItem"],
        full_format: bool = False,
    ) -> str:
        """
        Format only the changes since last call.

        Args:
            items: Current pack items.
            full_format: Force full format regardless of changes.

        Returns:
            Formatted diff or full content.
        """
        current_ids = {item.chunk.chunk_id for item in items}

        if full_format or not self._previous_ids:
            self._previous_ids = current_ids
            formatter = PackFormatter()
            return formatter.format(items, {})

        # Find additions and removals
        added = current_ids - self._previous_ids
        removed = self._previous_ids - current_ids

        parts: list[str] = []

        if removed:
            parts.append("### Removed from context:")
            for chunk_id in removed:
                parts.append(f"- {chunk_id[:8]}...")

        if added:
            parts.append("\n### Added to context:")
            added_items = [i for i in items if i.chunk.chunk_id in added]
            for item in added_items:
                chunk = item.chunk
                parts.append(f"\n**{chunk.file_path}:{chunk.start_line}**")
                if chunk.symbol_name:
                    parts.append(f"*{chunk.symbol_name}*")
                parts.append(f"```{chunk.language}")
                # Show preview
                preview = chunk.content[:500]
                if len(chunk.content) > 500:
                    preview += "\n..."
                parts.append(preview)
                parts.append("```")

        self._previous_ids = current_ids

        if not parts:
            return "No changes to context."

        return "\n".join(parts)

    def reset(self) -> None:
        """Reset diff tracking."""
        self._previous_ids.clear()


def format_for_prompt(
    items: list["PackItem"],
    query: str,
    format_type: str = "markdown",
) -> str:
    """
    Convenience function to format pack for LLM prompts.

    Args:
        items: Pack items.
        query: Original query.
        format_type: Format type (markdown, compact, xml).

    Returns:
        Formatted string.
    """
    formatter = PackFormatter()

    if format_type == "compact":
        return formatter.format_compact(items)
    elif format_type == "xml":
        return formatter.format_xml(items, query)
    else:
        citations = {
            f"[{i}]": f"{item.chunk.file_path}:{item.chunk.start_line}"
            for i, item in enumerate(items, 1)
        }
        return formatter.format(items, citations, query)
