"""
Memory tools for IC-MCP.

These tools manage context memory for intelligent code retrieval:
- memory_pack: Pack relevant context for a prompt
- memory_pin: Pin sources to always include
- memory_unpin: Remove pins
- memory_list: List memory items
- memory_get: Get a specific memory item
- memory_stats: Get memory statistics

The memory_pack tool supports RLM (Recursive Language Model) mode which
automatically activates when initial retrieval has high entropy, decomposing
complex queries into focused sub-queries for better context gathering.
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

# Flag to track if ICD bridge is available
_icd_bridge_available: bool | None = None


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

        Supports RLM (Recursive Language Model) mode which automatically
        activates when initial retrieval has high entropy (scattered results),
        decomposing complex queries into focused sub-queries.

        Args:
            input_data: Validated input parameters
            request_id: Unique request identifier

        Returns:
            MemoryPackOutput with packed context
        """
        logger.info(f"memory_pack: mode={input_data.mode}, budget={input_data.budget_tokens}")

        repo_root = Path(input_data.repo_root)
        warnings: list[str] = []

        if not repo_root.exists():
            warnings.append(f"Repository root does not exist: {input_data.repo_root}")
            return MemoryPackOutput(
                request_id=request_id,
                mode_resolved="pack",
                pack_markdown="",
                confidence=0.0,
                budget_used_tokens=0,
                entropy=0.0,
                gating_reason_codes=["NO_REPO"],
                top_sources=[],
                evidence=[],
                warnings=warnings,
            )

        # Try to use ICD bridge with RLM support
        icd_result = await self._try_icd_retrieval(
            prompt=input_data.prompt,
            repo_root=repo_root,
            k=input_data.k,
            mode=input_data.mode,
            focus_paths=input_data.focus_paths,
        )

        if icd_result is not None:
            # ICD retrieval succeeded - use those results
            sources = icd_result["sources"]
            entropy = icd_result["entropy"]
            mode_resolved = icd_result["mode"]
            gating_reasons = icd_result["gating_reasons"]
            sub_query_info = icd_result.get("sub_query_info", [])

            # Add RLM info to warnings if RLM was used
            if mode_resolved == "rlm" and sub_query_info:
                warnings.append(f"RLM mode: executed {len(sub_query_info)} sub-queries")
        else:
            # Fallback to basic file scanning
            warnings.append("Using basic retrieval (ICD index not available)")
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

            sub_query_info = []

        # Filter to pinned only if requested
        if input_data.pinned_only:
            sources = [s for s in sources if self.store.is_pinned(s.get("source_id", ""))]

        # Score and rank sources based on prompt relevance (for fallback mode)
        if not icd_result:
            scored_sources = self._score_sources(sources, input_data.prompt)
        else:
            scored_sources = sources  # ICD already scored

        # Take top k sources
        top_sources = scored_sources[: input_data.k]

        # Pre-calculate confidence to include in pack header
        # We need a rough estimate of budget_used first
        estimated_budget = min(
            sum(len(s.get("content", "")) // 4 for s in top_sources),
            input_data.budget_tokens
        )

        # Calculate confidence based on coverage, relevance, and entropy
        confidence = self._calculate_confidence(
            top_sources,
            estimated_budget,
            input_data.budget_tokens,
            entropy=entropy,
        )
        confidence_level = self._get_confidence_level(confidence)

        # Build the context pack within budget
        pack_markdown, budget_used, pack_evidence = self._build_pack_with_rlm_info(
            top_sources,
            input_data.budget_tokens,
            input_data.prompt,
            mode_resolved,
            sub_query_info,
            used_llm=icd_result.get("used_llm", False) if icd_result else False,
            llm_reasoning=icd_result.get("llm_reasoning", "") if icd_result else "",
            entropy=entropy,
            confidence=confidence,
            confidence_level=confidence_level,
        )

        # Recalculate confidence with actual budget_used
        confidence = self._calculate_confidence(
            top_sources,
            budget_used,
            input_data.budget_tokens,
            entropy=entropy,
        )

        # Create source info for output
        source_info = [
            SourceInfo(
                source_id=s.get("source_id", s.get("chunk_id", "")),
                path=s.get("path", s.get("file_path", "")),
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
            evidence=pack_evidence,
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

    async def _try_icd_retrieval(
        self,
        prompt: str,
        repo_root: Path,
        k: int,
        mode: str,
        focus_paths: list[str],
    ) -> dict[str, Any] | None:
        """
        Try to use ICD bridge for retrieval with RLM support.

        Returns None if ICD is not available.
        """
        global _icd_bridge_available

        # Quick check if we already know ICD isn't available
        if _icd_bridge_available is False:
            return None

        try:
            from ic_mcp.icd_bridge import retrieve_with_rlm

            result = await retrieve_with_rlm(
                query=prompt,
                project_root=repo_root,
                k=k,
                mode=mode,
                focus_paths=focus_paths,
            )

            # Check if bridge actually worked
            if result.metrics.mode == "fallback":
                _icd_bridge_available = False
                return None

            _icd_bridge_available = True

            # Convert chunks to source format
            sources = []
            for i, chunk in enumerate(result.chunks):
                score = result.scores[i] if i < len(result.scores) else 0.5
                sources.append({
                    "source_id": chunk.get("chunk_id", ""),
                    "path": chunk.get("file_path", ""),
                    "content": chunk.get("content", ""),
                    "score": score,
                    "start_line": chunk.get("start_line", 0),
                    "end_line": chunk.get("end_line", 0),
                    "symbol_name": chunk.get("symbol_name"),
                    "is_contract": chunk.get("is_contract", False),
                })

            # Build gating reasons
            gating_reasons = []
            if result.metrics.mode == "rlm":
                gating_reasons.append("HIGH_ENTROPY")
                gating_reasons.append(f"RLM_{result.metrics.iterations}_ITERATIONS")
                if result.metrics.used_llm_decomposition:
                    gating_reasons.append("LLM_DECOMPOSITION")
            else:
                gating_reasons.append("LOW_ENTROPY")

            return {
                "sources": sources,
                "entropy": result.entropy,
                "mode": result.metrics.mode,
                "gating_reasons": gating_reasons,
                "sub_query_info": result.sub_query_results,
                "used_llm": result.metrics.used_llm_decomposition,
                "llm_reasoning": result.metrics.llm_reasoning,
            }

        except ImportError:
            _icd_bridge_available = False
            logger.debug("ICD bridge not available")
            return None
        except Exception as e:
            logger.warning(f"ICD retrieval failed: {e}")
            return None

    def _build_pack_with_rlm_info(
        self,
        sources: list[dict[str, Any]],
        budget_tokens: int,
        prompt: str,
        mode: str,
        sub_query_info: list[dict[str, Any]],
        used_llm: bool = False,
        llm_reasoning: str = "",
        entropy: float = 0.0,
        confidence: float = 0.0,
        confidence_level: str = "",
    ) -> tuple[str, int, list[Evidence]]:
        """Build the context pack with RLM information in header."""
        pack_parts: list[str] = []
        evidence: list[Evidence] = []
        used_tokens = 0

        # Build sources summary for visibility
        sources_summary = "\n".join(
            f"  - {s.get('path', s.get('file_path', 'unknown'))} (score: {s.get('score', 0):.2f})"
            for s in sources[:10]
        )

        # Build RLM info section if applicable
        rlm_section = ""
        if mode == "rlm" and sub_query_info:
            sub_query_lines = "\n".join(
                f"  {i+1}. \"{sq['query']}\" ({sq['type']}) → {sq['results']} results"
                for i, sq in enumerate(sub_query_info)
            )
            decomposition_method = "LLM (Claude)" if used_llm else "Heuristic"
            reasoning_line = f"\n**Reasoning:** {llm_reasoning}" if llm_reasoning else ""
            rlm_section = f"""
**Mode:** RLM (high entropy detected - iterative retrieval)
**Decomposition:** {decomposition_method}{reasoning_line}
**Sub-queries executed:**
{sub_query_lines}
"""
        else:
            rlm_section = f"\n**Mode:** Pack (direct retrieval)\n"

        # Build confidence display
        confidence_display = f"**Confidence:** {confidence:.2f} ({confidence_level})"

        # Add header with sources visibility, RLM info, and confidence
        header = f"""# ICR Context Pack

**Query:** {prompt}
{rlm_section}
{confidence_display}

**Sources Retrieved ({len(sources)} chunks):**
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

            path = source.get("path", source.get("file_path", "unknown"))
            start_line = source.get("start_line", 1)
            end_line = source.get("end_line", start_line)
            symbol_name = source.get("symbol_name")

            # Format source section with line info
            symbol_info = f" ({symbol_name})" if symbol_name else ""
            section_header = f"## {path}:{start_line}-{end_line}{symbol_info}\n\n```\n"
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
                    source_id=source.get("source_id", source.get("chunk_id", "")),
                    source_type="chunk" if "chunk_id" in source else "file",
                    path=path,
                    repo_rev="working-tree",
                    mtime=source.get("mtime"),
                    content=content_truncated[:500] if was_truncated else content[:500],
                )
            )

        pack_markdown = "".join(pack_parts)
        return pack_markdown, used_tokens, evidence

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
        entropy: float = 0.0,
    ) -> float:
        """Calculate confidence score for the pack."""
        if not sources:
            return 0.0

        # Source relevance: average of top 5 scores
        top_scores = [s.get("score", 0) for s in sources[:5]]
        source_relevance = sum(top_scores) / len(top_scores) if top_scores else 0

        # Entropy factor: lower entropy = higher confidence (normalize to 0-1)
        entropy_factor = max(0, 1 - (entropy / 5.0))

        # Budget utilization: higher = better (found enough content)
        budget_util = min(used_tokens / budget_tokens, 1.0) if budget_tokens else 0

        # Weighted average
        confidence = (source_relevance * 0.5) + (entropy_factor * 0.3) + (budget_util * 0.2)
        return round(min(confidence, 1.0), 3)

    def _get_confidence_level(self, confidence: float) -> str:
        """Get human-readable confidence level description."""
        if confidence >= 0.75:
            return "High - clear matches found"
        elif confidence >= 0.5:
            return "Medium - results may need refinement"
        else:
            return "Low - consider more specific query"

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
