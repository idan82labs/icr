"""
Memory tools for IC-MCP.

These tools manage context memory for intelligent code retrieval:
- memory_pack: Pack relevant context for a prompt
- memory_pin: Pin sources to always include
- memory_unpin: Remove pins
- memory_list: List memory items
- memory_get: Get a specific memory item
- memory_stats: Get memory statistics
"""

import hashlib
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from ic_mcp.schemas.inputs import (
    MemoryGetInput,
    MemoryListInput,
    MemoryPackInput,
    MemoryPinInput,
    MemoryStatsInput,
    MemoryUnpinInput,
)
from ic_mcp.schemas.outputs import (
    Evidence,
    MemoryGetOutput,
    MemoryItem,
    MemoryListOutput,
    MemoryPackOutput,
    MemoryPinOutput,
    MemoryStatsOutput,
    MemoryUnpinOutput,
    PaginationInfo,
    SourceInfo,
    Span,
)
from ic_mcp.schemas.validation import (
    count_tokens,
    create_pagination_cursor,
    parse_pagination_cursor,
    truncate_string_to_tokens,
    truncate_to_token_budget,
)

logger = logging.getLogger(__name__)


class MemoryStore:
    """
    In-memory store for memory items.

    In production, this would be backed by a persistent store like SQLite
    or a vector database. This implementation provides the interface and
    basic functionality for development and testing.
    """

    def __init__(self) -> None:
        """Initialize the memory store."""
        self._items: dict[str, dict[str, Any]] = {}
        self._pins: dict[str, dict[str, Any]] = {}
        self._created_at = datetime.now(timezone.utc)

    def add_item(
        self,
        source_id: str,
        path: str,
        source_type: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add or update a memory item."""
        now = datetime.now(timezone.utc)
        item = {
            "source_id": source_id,
            "path": path,
            "source_type": source_type,
            "content": content,
            "metadata": metadata or {},
            "indexed_at": now,
            "last_accessed": now,
            "size_bytes": len(content.encode()) if content else 0,
        }
        self._items[source_id] = item
        return item

    def get_item(self, source_id: str) -> dict[str, Any] | None:
        """Get a memory item by ID."""
        item = self._items.get(source_id)
        if item:
            item["last_accessed"] = datetime.now(timezone.utc)
        return item

    def list_items(
        self,
        filter_type: str = "all",
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """List memory items with optional filtering."""
        items = list(self._items.values())

        if filter_type == "pinned":
            items = [i for i in items if i["source_id"] in self._pins]
        elif filter_type == "recent":
            # Sort by last_accessed descending
            items = sorted(items, key=lambda x: x["last_accessed"], reverse=True)
        elif filter_type == "stale":
            # Items not accessed in the last hour
            threshold = datetime.now(timezone.utc).timestamp() - 3600
            items = [
                i
                for i in items
                if i["last_accessed"].timestamp() < threshold
            ]

        total = len(items)
        items = items[offset : offset + limit]

        return items, total

    def pin(
        self,
        source_id: str,
        path: str,
        label: str | None = None,
        ttl_seconds: int | None = None,
    ) -> dict[str, Any]:
        """Pin a source."""
        now = datetime.now(timezone.utc)
        expires_at = None
        if ttl_seconds:
            from datetime import timedelta

            expires_at = now + timedelta(seconds=ttl_seconds)

        pin_info = {
            "source_id": source_id,
            "path": path,
            "label": label,
            "pinned_at": now,
            "expires_at": expires_at,
        }
        self._pins[source_id] = pin_info

        # Ensure the item exists in the store
        if source_id not in self._items:
            self.add_item(source_id, path, "file")

        return pin_info

    def unpin(self, source_id: str) -> bool:
        """Unpin a source. Returns True if it was pinned."""
        was_pinned = source_id in self._pins
        self._pins.pop(source_id, None)
        return was_pinned

    def is_pinned(self, source_id: str) -> bool:
        """Check if a source is pinned."""
        pin = self._pins.get(source_id)
        if not pin:
            return False

        # Check expiration
        if pin.get("expires_at"):
            if datetime.now(timezone.utc) > pin["expires_at"]:
                self._pins.pop(source_id)
                return False

        return True

    def get_pin_info(self, source_id: str) -> dict[str, Any] | None:
        """Get pin information for a source."""
        if self.is_pinned(source_id):
            return self._pins.get(source_id)
        return None

    def get_stats(self, include_breakdown: bool = False) -> dict[str, Any]:
        """Get memory statistics."""
        stats = {
            "total_items": len(self._items),
            "pinned_count": sum(1 for sid in self._items if self.is_pinned(sid)),
            "total_size_bytes": sum(i.get("size_bytes", 0) for i in self._items.values()),
            "index_freshness": self._created_at,
        }

        if include_breakdown:
            breakdown: dict[str, int] = {}
            for item in self._items.values():
                source_type = item.get("source_type", "unknown")
                breakdown[source_type] = breakdown.get(source_type, 0) + 1
            stats["breakdown"] = breakdown

        return stats


# Global memory store instance
_memory_store: MemoryStore | None = None


def get_memory_store() -> MemoryStore:
    """Get or create the global memory store."""
    global _memory_store
    if _memory_store is None:
        _memory_store = MemoryStore()
    return _memory_store


class MemoryTools:
    """
    Memory tools for context management.

    These tools provide intelligent context retrieval and management
    for the ICR system.
    """

    def __init__(self) -> None:
        """Initialize memory tools."""
        self.store = get_memory_store()

    def _generate_source_id(self, path: str) -> str:
        """Generate a deterministic source ID from a path."""
        return f"S{hashlib.sha256(path.encode()).hexdigest()[:12]}"

    def _calculate_entropy(self, content: str) -> float:
        """
        Calculate information entropy of content.

        Higher entropy indicates more diverse/complex content.
        """
        import math
        from collections import Counter

        if not content:
            return 0.0

        # Count character frequencies
        counter = Counter(content)
        total = len(content)

        # Calculate Shannon entropy
        entropy = 0.0
        for count in counter.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        # Normalize to 0-1 range (max entropy for ASCII is ~7 bits)
        return min(entropy / 7.0, 1.0)

    def _should_use_rlm(self, entropy: float, prompt: str) -> tuple[bool, list[str]]:
        """
        Determine if RLM mode should be used based on entropy and prompt.

        Returns (should_use_rlm, reason_codes)
        """
        reasons = []

        # High entropy suggests complex, diverse codebase
        if entropy > 0.7:
            reasons.append("HIGH_ENTROPY")

        # Long prompts with many requirements suggest need for RLM
        prompt_tokens = count_tokens(prompt)
        if prompt_tokens > 500:
            reasons.append("COMPLEX_PROMPT")

        # Check for keywords suggesting multi-step analysis
        rlm_keywords = [
            "all", "every", "across", "throughout", "comprehensive",
            "trace", "follow", "dependency", "impact", "analyze"
        ]
        prompt_lower = prompt.lower()
        if any(kw in prompt_lower for kw in rlm_keywords):
            reasons.append("MULTI_STEP_KEYWORDS")

        return len(reasons) >= 2, reasons

    async def memory_pack(
        self,
        input_data: MemoryPackInput,
        request_id: UUID,
    ) -> MemoryPackOutput:
        """
        Pack relevant context for a prompt.

        This is the primary tool for context retrieval. It analyzes the prompt
        and repo to determine the most relevant sources, respecting the token
        budget.

        Args:
            input_data: Validated input parameters
            request_id: Unique request identifier

        Returns:
            MemoryPackOutput with packed context
        """
        logger.info(f"memory_pack: mode={input_data.mode}, budget={input_data.budget_tokens}")

        # Collect sources from the repository
        sources: list[dict[str, Any]] = []
        evidence: list[Evidence] = []
        warnings: list[str] = []

        repo_root = Path(input_data.repo_root)
        if not repo_root.exists():
            warnings.append(f"Repository root does not exist: {input_data.repo_root}")
        else:
            # Scan repository for relevant files
            sources = self._scan_repository(
                repo_root,
                input_data.focus_paths,
                input_data.k,
            )

        # Calculate entropy from collected sources
        all_content = "\n".join(s.get("content", "") for s in sources if s.get("content"))
        entropy = self._calculate_entropy(all_content)

        # Determine mode
        if input_data.mode == "auto":
            use_rlm, gating_reasons = self._should_use_rlm(entropy, input_data.prompt)
            mode_resolved = "rlm" if use_rlm else "pack"
        else:
            mode_resolved = "rlm" if input_data.mode == "rlm" else "pack"
            gating_reasons = [f"USER_SPECIFIED_{mode_resolved.upper()}"]

        # Filter to pinned only if requested
        if input_data.pinned_only:
            sources = [s for s in sources if self.store.is_pinned(s["source_id"])]

        # Score and rank sources based on prompt relevance
        scored_sources = self._score_sources(sources, input_data.prompt)

        # Take top k sources
        top_sources = scored_sources[: input_data.k]

        # Build the context pack within budget
        pack_markdown, budget_used, pack_evidence = self._build_pack(
            top_sources,
            input_data.budget_tokens,
            input_data.prompt,
        )

        evidence.extend(pack_evidence)

        # Calculate confidence based on coverage and relevance
        confidence = self._calculate_confidence(
            top_sources,
            budget_used,
            input_data.budget_tokens,
        )

        # Create source info for output
        source_info = [
            SourceInfo(
                source_id=s["source_id"],
                path=s["path"],
                score=s.get("score", 0.5),
            )
            for s in top_sources[:10]  # Limit to top 10 in output
        ]

        output = MemoryPackOutput(
            request_id=request_id,
            mode_resolved=mode_resolved,
            pack_markdown=pack_markdown,
            confidence=confidence,
            budget_used_tokens=budget_used,
            entropy=entropy,
            gating_reason_codes=gating_reasons,
            top_sources=source_info,
            evidence=evidence,
            warnings=warnings,
        )

        # Truncate if necessary
        output_dict = output.model_dump()
        truncated_dict, was_truncated = truncate_to_token_budget(
            output_dict,
            truncatable_fields=["pack_markdown", "evidence", "warnings"],
        )

        if was_truncated:
            return MemoryPackOutput.model_validate(truncated_dict)

        return output

    def _scan_repository(
        self,
        repo_root: Path,
        focus_paths: list[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Scan repository for relevant files."""
        sources: list[dict[str, Any]] = []

        # Define patterns to include
        include_extensions = {
            ".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs",
            ".java", ".kt", ".scala", ".rb", ".php", ".cs",
            ".c", ".cpp", ".h", ".hpp", ".swift", ".m",
            ".json", ".yaml", ".yml", ".toml", ".md",
        }

        # Patterns to exclude
        exclude_dirs = {
            ".git", "node_modules", "__pycache__", ".venv", "venv",
            "dist", "build", ".next", ".nuxt", "target", "vendor",
        }

        def should_include(path: Path) -> bool:
            """Check if path should be included."""
            # Check exclusions
            for part in path.parts:
                if part in exclude_dirs:
                    return False

            # Check extensions
            if path.suffix.lower() not in include_extensions:
                return False

            # Check focus paths if specified
            if focus_paths:
                rel_path = str(path.relative_to(repo_root))
                return any(rel_path.startswith(fp) for fp in focus_paths)

            return True

        # Walk the repository
        try:
            for root, dirs, files in os.walk(repo_root):
                # Skip excluded directories
                dirs[:] = [d for d in dirs if d not in exclude_dirs]

                for file in files:
                    if len(sources) >= limit * 3:  # Collect more than needed for ranking
                        break

                    file_path = Path(root) / file
                    if should_include(file_path):
                        try:
                            content = file_path.read_text(errors="ignore")
                            rel_path = str(file_path.relative_to(repo_root))
                            source_id = self._generate_source_id(rel_path)

                            source = {
                                "source_id": source_id,
                                "path": rel_path,
                                "content": content,
                                "size_bytes": len(content.encode()),
                                "mtime": datetime.fromtimestamp(
                                    file_path.stat().st_mtime, tz=timezone.utc
                                ),
                            }
                            sources.append(source)

                            # Add to memory store
                            self.store.add_item(
                                source_id,
                                rel_path,
                                "file",
                                content,
                                {"mtime": source["mtime"]},
                            )

                        except Exception as e:
                            logger.warning(f"Failed to read {file_path}: {e}")

        except Exception as e:
            logger.error(f"Error scanning repository: {e}")

        return sources

    def _score_sources(
        self,
        sources: list[dict[str, Any]],
        prompt: str,
    ) -> list[dict[str, Any]]:
        """Score sources based on relevance to the prompt."""
        prompt_lower = prompt.lower()
        prompt_words = set(prompt_lower.split())

        for source in sources:
            score = 0.0

            path = source.get("path", "").lower()
            content = source.get("content", "").lower()

            # Path relevance
            path_words = set(path.replace("/", " ").replace("_", " ").replace("-", " ").split())
            path_overlap = len(prompt_words & path_words) / max(len(prompt_words), 1)
            score += path_overlap * 0.3

            # Content relevance (simple word overlap)
            content_words = set(content.split()[:1000])  # Limit for performance
            content_overlap = len(prompt_words & content_words) / max(len(prompt_words), 1)
            score += content_overlap * 0.5

            # Recency boost
            mtime = source.get("mtime")
            if mtime:
                age_hours = (datetime.now(timezone.utc) - mtime).total_seconds() / 3600
                recency_score = max(0, 1 - (age_hours / (24 * 7)))  # Decay over a week
                score += recency_score * 0.1

            # Pin boost
            if self.store.is_pinned(source["source_id"]):
                score += 0.2

            source["score"] = min(score, 1.0)

        # Sort by score descending
        return sorted(sources, key=lambda x: x.get("score", 0), reverse=True)

    def _build_pack(
        self,
        sources: list[dict[str, Any]],
        budget_tokens: int,
        prompt: str,
    ) -> tuple[str, int, list[Evidence]]:
        """Build the context pack within the token budget."""
        pack_parts: list[str] = []
        evidence: list[Evidence] = []
        used_tokens = 0

        # Build sources summary for visibility
        sources_summary = "\n".join(
            f"  - {s['path']} (score: {s.get('score', 0):.2f})"
            for s in sources[:10]
        )

        # Add header with sources visibility
        header = f"""# ICR Context Pack

**Query:** {prompt}

**Sources Retrieved ({len(sources)} files):**
{sources_summary}

---

"""
        header_tokens = count_tokens(header)
        used_tokens += header_tokens
        pack_parts.append(header)

        for source in sources:
            content = source.get("content", "")
            if not content:
                continue

            # Format source section
            section_header = f"## {source['path']}\n\n```\n"
            section_footer = "\n```\n\n"
            overhead_tokens = count_tokens(section_header + section_footer)

            # Calculate available tokens for content
            available = budget_tokens - used_tokens - overhead_tokens - 100  # Buffer
            if available <= 0:
                break

            # Truncate content if needed
            content_truncated, was_truncated = truncate_string_to_tokens(
                content,
                available,
            )

            section = section_header + content_truncated + section_footer
            section_tokens = count_tokens(section)

            if used_tokens + section_tokens > budget_tokens:
                break

            pack_parts.append(section)
            used_tokens += section_tokens

            # Add evidence
            evidence.append(
                Evidence(
                    source_id=source["source_id"],
                    source_type="file",
                    path=source["path"],
                    repo_rev="working-tree",
                    mtime=source.get("mtime"),
                    content=content_truncated[:500] if was_truncated else content[:500],
                )
            )

        pack_markdown = "".join(pack_parts)
        return pack_markdown, used_tokens, evidence

    def _calculate_confidence(
        self,
        sources: list[dict[str, Any]],
        used_tokens: int,
        budget_tokens: int,
    ) -> float:
        """Calculate confidence score for the pack."""
        if not sources:
            return 0.0

        # Average source score
        avg_score = sum(s.get("score", 0) for s in sources) / len(sources)

        # Budget utilization
        utilization = min(used_tokens / budget_tokens, 1.0)

        # Combined confidence
        confidence = (avg_score * 0.6) + (utilization * 0.4)
        return round(min(confidence, 1.0), 3)

    async def memory_pin(
        self,
        input_data: MemoryPinInput,
        request_id: UUID,
    ) -> MemoryPinOutput:
        """
        Pin a source to always include in context packs.

        Args:
            input_data: Validated input parameters
            request_id: Unique request identifier

        Returns:
            MemoryPinOutput with pin confirmation
        """
        logger.info(f"memory_pin: source_id={input_data.source_id}")

        pin_info = self.store.pin(
            input_data.source_id,
            input_data.path,
            input_data.label,
            input_data.ttl_seconds,
        )

        return MemoryPinOutput(
            request_id=request_id,
            source_id=pin_info["source_id"],
            path=pin_info["path"],
            label=pin_info.get("label"),
            pinned_at=pin_info["pinned_at"],
            expires_at=pin_info.get("expires_at"),
        )

    async def memory_unpin(
        self,
        input_data: MemoryUnpinInput,
        request_id: UUID,
    ) -> MemoryUnpinOutput:
        """
        Remove a pin from a source.

        Args:
            input_data: Validated input parameters
            request_id: Unique request identifier

        Returns:
            MemoryUnpinOutput with unpin confirmation
        """
        logger.info(f"memory_unpin: source_id={input_data.source_id}")

        was_pinned = self.store.unpin(input_data.source_id)

        return MemoryUnpinOutput(
            request_id=request_id,
            source_id=input_data.source_id,
            was_pinned=was_pinned,
        )

    async def memory_list(
        self,
        input_data: MemoryListInput,
        request_id: UUID,
    ) -> MemoryListOutput:
        """
        List memory items with optional filtering.

        Args:
            input_data: Validated input parameters
            request_id: Unique request identifier

        Returns:
            MemoryListOutput with items and pagination
        """
        logger.info(f"memory_list: filter={input_data.filter_type}, limit={input_data.limit}")

        offset, _ = parse_pagination_cursor(input_data.cursor)

        items, total = self.store.list_items(
            input_data.filter_type,
            input_data.limit,
            offset,
        )

        memory_items = [
            MemoryItem(
                source_id=item["source_id"],
                path=item["path"],
                source_type=item.get("source_type", "file"),
                pinned=self.store.is_pinned(item["source_id"]),
                label=self.store.get_pin_info(item["source_id"]).get("label")
                if self.store.is_pinned(item["source_id"])
                else None,
                last_accessed=item.get("last_accessed"),
                indexed_at=item.get("indexed_at"),
                size_bytes=item.get("size_bytes"),
            )
            for item in items
        ]

        cursor = create_pagination_cursor(offset, input_data.limit, total)
        has_more = cursor is not None

        return MemoryListOutput(
            request_id=request_id,
            items=memory_items,
            pagination=PaginationInfo(
                cursor=cursor,
                has_more=has_more,
                total_count=total,
            ),
        )

    async def memory_get(
        self,
        input_data: MemoryGetInput,
        request_id: UUID,
    ) -> MemoryGetOutput:
        """
        Get a specific memory item.

        Args:
            input_data: Validated input parameters
            request_id: Unique request identifier

        Returns:
            MemoryGetOutput with item details
        """
        logger.info(f"memory_get: source_id={input_data.source_id}")

        item = self.store.get_item(input_data.source_id)

        if item is None:
            # Return empty result for non-existent item
            return MemoryGetOutput(
                request_id=request_id,
                source_id=input_data.source_id,
                path="",
                source_type="file",
                content=None,
                evidence=None,
                pinned=False,
                label=None,
            )

        content = item.get("content") if input_data.include_content else None

        evidence = Evidence(
            source_id=item["source_id"],
            source_type=item.get("source_type", "file"),
            path=item["path"],
            repo_rev="working-tree",
            mtime=item.get("metadata", {}).get("mtime"),
            indexed_at=item.get("indexed_at"),
            content=content[:500] if content else None,
        )

        pin_info = self.store.get_pin_info(input_data.source_id)

        return MemoryGetOutput(
            request_id=request_id,
            source_id=item["source_id"],
            path=item["path"],
            source_type=item.get("source_type", "file"),
            content=content,
            evidence=evidence,
            pinned=pin_info is not None,
            label=pin_info.get("label") if pin_info else None,
        )

    async def memory_stats(
        self,
        input_data: MemoryStatsInput,
        request_id: UUID,
    ) -> MemoryStatsOutput:
        """
        Get memory statistics.

        Args:
            input_data: Validated input parameters
            request_id: Unique request identifier

        Returns:
            MemoryStatsOutput with statistics
        """
        logger.info(f"memory_stats: include_breakdown={input_data.include_breakdown}")

        stats = self.store.get_stats(input_data.include_breakdown)

        return MemoryStatsOutput(
            request_id=request_id,
            total_items=stats["total_items"],
            pinned_count=stats["pinned_count"],
            total_size_bytes=stats["total_size_bytes"],
            index_freshness=stats.get("index_freshness"),
            breakdown=stats.get("breakdown"),
        )
