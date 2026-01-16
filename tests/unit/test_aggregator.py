"""
Unit tests for the non-generative aggregation module.

Tests cover:
- Deterministic operations only
- No LLM calls for aggregation
- Set operations
- Regex extraction
- Sorting and grouping
- Count and top-k operations
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

import pytest


# ==============================================================================
# Aggregation Operation Types
# ==============================================================================

AggregateOp = Literal[
    "extract_regex",
    "unique",
    "sort",
    "group_by",
    "count",
    "top_k",
    "join_on",
    "diff_sets",
]


@dataclass
class AggregateResult:
    """Result of an aggregation operation."""

    operation: AggregateOp
    items: list[Any]
    count: int
    metadata: dict[str, Any]


# ==============================================================================
# Aggregation Functions (Non-Generative)
# ==============================================================================

def aggregate(
    op: AggregateOp,
    inputs: list[str],
    params: dict[str, Any] | None = None,
    limit: int = 100,
) -> AggregateResult:
    """
    Perform non-generative aggregation on inputs.

    IMPORTANT: All operations are deterministic and do NOT use LLM.

    Args:
        op: Aggregation operation to perform
        inputs: List of input strings
        params: Operation-specific parameters
        limit: Maximum number of results

    Returns:
        AggregateResult with operation results
    """
    params = params or {}

    if op == "extract_regex":
        return _extract_regex(inputs, params, limit)
    elif op == "unique":
        return _unique(inputs, limit)
    elif op == "sort":
        return _sort(inputs, params, limit)
    elif op == "group_by":
        return _group_by(inputs, params, limit)
    elif op == "count":
        return _count(inputs)
    elif op == "top_k":
        return _top_k(inputs, params, limit)
    elif op == "join_on":
        return _join_on(inputs, params, limit)
    elif op == "diff_sets":
        return _diff_sets(inputs, params)
    else:
        raise ValueError(f"Unknown operation: {op}")


def _extract_regex(inputs: list[str], params: dict, limit: int) -> AggregateResult:
    """Extract matches using regex pattern."""
    pattern = params.get("pattern", ".*")
    group = params.get("group", 0)

    matches = []
    regex = re.compile(pattern)

    for text in inputs:
        for match in regex.finditer(text):
            try:
                if group == 0:
                    matches.append(match.group())
                else:
                    matches.append(match.group(group))
            except IndexError:
                continue

            if len(matches) >= limit:
                break
        if len(matches) >= limit:
            break

    return AggregateResult(
        operation="extract_regex",
        items=matches[:limit],
        count=len(matches),
        metadata={"pattern": pattern, "group": group},
    )


def _unique(inputs: list[str], limit: int) -> AggregateResult:
    """Get unique values preserving order."""
    seen = set()
    unique_items = []

    for item in inputs:
        if item not in seen:
            seen.add(item)
            unique_items.append(item)
            if len(unique_items) >= limit:
                break

    return AggregateResult(
        operation="unique",
        items=unique_items,
        count=len(unique_items),
        metadata={"original_count": len(inputs)},
    )


def _sort(inputs: list[str], params: dict, limit: int) -> AggregateResult:
    """Sort inputs."""
    reverse = params.get("reverse", False)
    key = params.get("key")  # Optional key function name

    if key == "length":
        sorted_items = sorted(inputs, key=len, reverse=reverse)
    elif key == "numeric":
        # Try to extract numbers for sorting
        def numeric_key(s):
            nums = re.findall(r'\d+', s)
            return int(nums[0]) if nums else 0
        sorted_items = sorted(inputs, key=numeric_key, reverse=reverse)
    else:
        sorted_items = sorted(inputs, reverse=reverse)

    return AggregateResult(
        operation="sort",
        items=sorted_items[:limit],
        count=len(sorted_items),
        metadata={"reverse": reverse, "key": key},
    )


def _group_by(inputs: list[str], params: dict, limit: int) -> AggregateResult:
    """Group inputs by pattern."""
    pattern = params.get("pattern", ".*")
    regex = re.compile(pattern)

    groups: dict[str, list[str]] = {}

    for item in inputs:
        match = regex.search(item)
        key = match.group() if match else "_other"
        if key not in groups:
            groups[key] = []
        groups[key].append(item)

    # Convert to list of tuples
    items = [(k, v) for k, v in sorted(groups.items())][:limit]

    return AggregateResult(
        operation="group_by",
        items=items,
        count=len(groups),
        metadata={"pattern": pattern},
    )


def _count(inputs: list[str]) -> AggregateResult:
    """Count occurrences of each input."""
    counts: dict[str, int] = {}

    for item in inputs:
        counts[item] = counts.get(item, 0) + 1

    # Sort by count descending
    items = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    return AggregateResult(
        operation="count",
        items=items,
        count=len(counts),
        metadata={"total_items": len(inputs)},
    )


def _top_k(inputs: list[str], params: dict, limit: int) -> AggregateResult:
    """Get top-k items by frequency."""
    k = min(params.get("k", 10), limit)

    counts: dict[str, int] = {}
    for item in inputs:
        counts[item] = counts.get(item, 0) + 1

    # Sort by count descending
    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:k]

    return AggregateResult(
        operation="top_k",
        items=[item for item, _ in sorted_items],
        count=len(sorted_items),
        metadata={"k": k, "counts": dict(sorted_items)},
    )


def _join_on(inputs: list[str], params: dict, limit: int) -> AggregateResult:
    """Join strings with delimiter."""
    delimiter = params.get("delimiter", ", ")

    joined = delimiter.join(inputs[:limit])

    return AggregateResult(
        operation="join_on",
        items=[joined],
        count=1,
        metadata={"delimiter": delimiter, "joined_count": min(len(inputs), limit)},
    )


def _diff_sets(inputs: list[str], params: dict) -> AggregateResult:
    """Compute set difference."""
    # Expect inputs to be two lists encoded as JSON or split by separator
    separator = params.get("separator", "|||")
    split_idx = None

    for i, item in enumerate(inputs):
        if item == separator:
            split_idx = i
            break

    if split_idx is None:
        # Assume first half is set A, second half is set B
        split_idx = len(inputs) // 2

    set_a = set(inputs[:split_idx])
    set_b = set(inputs[split_idx + 1:] if inputs[split_idx] == separator else inputs[split_idx:])

    diff = sorted(set_a - set_b)

    return AggregateResult(
        operation="diff_sets",
        items=diff,
        count=len(diff),
        metadata={"set_a_size": len(set_a), "set_b_size": len(set_b)},
    )


# ==============================================================================
# Deterministic Operation Tests
# ==============================================================================

class TestDeterministicOperations:
    """Tests ensuring all operations are deterministic."""

    def test_operations_are_deterministic(self):
        """Test that running same operation twice gives same result."""
        inputs = ["c", "a", "b", "a", "c", "c"]

        result1 = aggregate("count", inputs)
        result2 = aggregate("count", inputs)

        assert result1.items == result2.items
        assert result1.count == result2.count

    def test_sort_deterministic(self):
        """Test sort is deterministic."""
        inputs = ["banana", "apple", "cherry"]

        for _ in range(10):
            result = aggregate("sort", inputs)
            assert result.items == ["apple", "banana", "cherry"]

    def test_unique_preserves_order(self):
        """Test unique preserves first occurrence order."""
        inputs = ["b", "a", "c", "a", "b"]

        result = aggregate("unique", inputs)

        assert result.items == ["b", "a", "c"]


# ==============================================================================
# No LLM Calls Tests
# ==============================================================================

class TestNoLLMCalls:
    """Tests ensuring no LLM/generative calls are made."""

    def test_extract_regex_no_generation(self):
        """Test regex extraction doesn't generate content."""
        inputs = ["function foo()", "function bar()", "const x = 1"]

        result = aggregate("extract_regex", inputs, {"pattern": r"function (\w+)"})

        # Results must be substrings of inputs
        for item in result.items:
            assert any(item in inp for inp in inputs)

    def test_count_no_generation(self):
        """Test count doesn't generate new values."""
        inputs = ["a", "b", "a"]

        result = aggregate("count", inputs)

        # All keys must be from inputs
        for key, _ in result.items:
            assert key in inputs

    def test_group_by_no_generation(self):
        """Test group_by doesn't generate group names beyond pattern."""
        inputs = ["file1.py", "file2.py", "doc.md"]

        result = aggregate("group_by", inputs, {"pattern": r"\.\w+$"})

        # Group names must be pattern matches or "_other"
        for group_name, _ in result.items:
            assert group_name == "_other" or group_name.startswith(".")


# ==============================================================================
# Regex Extraction Tests
# ==============================================================================

class TestRegexExtraction:
    """Tests for regex extraction operation."""

    def test_extract_simple_pattern(self):
        """Test simple pattern extraction."""
        inputs = ["Hello world", "Hello there", "Goodbye world"]

        result = aggregate("extract_regex", inputs, {"pattern": r"Hello"})

        assert result.items == ["Hello", "Hello"]

    def test_extract_with_groups(self):
        """Test extraction with capture groups."""
        inputs = ["def foo():", "def bar():", "class Baz:"]

        result = aggregate("extract_regex", inputs, {"pattern": r"def (\w+)", "group": 1})

        assert result.items == ["foo", "bar"]

    def test_extract_no_matches(self):
        """Test extraction with no matches."""
        inputs = ["abc", "def", "ghi"]

        result = aggregate("extract_regex", inputs, {"pattern": r"\d+"})

        assert result.items == []
        assert result.count == 0

    def test_extract_multiple_matches_per_input(self):
        """Test multiple matches in single input."""
        inputs = ["a1 b2 c3"]

        result = aggregate("extract_regex", inputs, {"pattern": r"\w\d"})

        assert result.items == ["a1", "b2", "c3"]

    def test_extract_respects_limit(self):
        """Test extraction respects limit."""
        inputs = ["a1 a2 a3 a4 a5"]

        result = aggregate("extract_regex", inputs, {"pattern": r"a\d"}, limit=3)

        assert len(result.items) == 3


# ==============================================================================
# Unique Operation Tests
# ==============================================================================

class TestUniqueOperation:
    """Tests for unique operation."""

    def test_unique_basic(self):
        """Test basic deduplication."""
        inputs = ["a", "b", "a", "c", "b"]

        result = aggregate("unique", inputs)

        assert result.items == ["a", "b", "c"]

    def test_unique_all_same(self):
        """Test unique with all same values."""
        inputs = ["x", "x", "x"]

        result = aggregate("unique", inputs)

        assert result.items == ["x"]

    def test_unique_all_different(self):
        """Test unique with all different values."""
        inputs = ["a", "b", "c"]

        result = aggregate("unique", inputs)

        assert result.items == ["a", "b", "c"]

    def test_unique_empty(self):
        """Test unique with empty input."""
        result = aggregate("unique", [])

        assert result.items == []


# ==============================================================================
# Sort Operation Tests
# ==============================================================================

class TestSortOperation:
    """Tests for sort operation."""

    def test_sort_alphabetical(self):
        """Test alphabetical sorting."""
        inputs = ["cherry", "apple", "banana"]

        result = aggregate("sort", inputs)

        assert result.items == ["apple", "banana", "cherry"]

    def test_sort_reverse(self):
        """Test reverse sorting."""
        inputs = ["a", "b", "c"]

        result = aggregate("sort", inputs, {"reverse": True})

        assert result.items == ["c", "b", "a"]

    def test_sort_by_length(self):
        """Test sorting by length."""
        inputs = ["aa", "a", "aaa"]

        result = aggregate("sort", inputs, {"key": "length"})

        assert result.items == ["a", "aa", "aaa"]

    def test_sort_numeric(self):
        """Test numeric sorting."""
        inputs = ["item10", "item2", "item1"]

        result = aggregate("sort", inputs, {"key": "numeric"})

        assert result.items == ["item1", "item2", "item10"]


# ==============================================================================
# Group By Operation Tests
# ==============================================================================

class TestGroupByOperation:
    """Tests for group_by operation."""

    def test_group_by_extension(self):
        """Test grouping by file extension."""
        inputs = ["a.py", "b.py", "c.js", "d.ts"]

        result = aggregate("group_by", inputs, {"pattern": r"\.\w+$"})

        groups = dict(result.items)
        assert len(groups[".py"]) == 2
        assert len(groups[".js"]) == 1
        assert len(groups[".ts"]) == 1

    def test_group_by_no_match(self):
        """Test grouping with non-matching items."""
        inputs = ["file.py", "noextension"]

        result = aggregate("group_by", inputs, {"pattern": r"\.\w+$"})

        groups = dict(result.items)
        assert "noextension" in groups.get("_other", [])


# ==============================================================================
# Count Operation Tests
# ==============================================================================

class TestCountOperation:
    """Tests for count operation."""

    def test_count_basic(self):
        """Test basic counting."""
        inputs = ["a", "b", "a", "a", "b"]

        result = aggregate("count", inputs)

        counts = dict(result.items)
        assert counts["a"] == 3
        assert counts["b"] == 2

    def test_count_sorted_by_frequency(self):
        """Test counts are sorted by frequency."""
        inputs = ["rare", "common", "common", "common"]

        result = aggregate("count", inputs)

        # First item should be most common
        assert result.items[0][0] == "common"
        assert result.items[0][1] == 3


# ==============================================================================
# Top-K Operation Tests
# ==============================================================================

class TestTopKOperation:
    """Tests for top_k operation."""

    def test_top_k_basic(self):
        """Test basic top-k."""
        inputs = ["a", "a", "a", "b", "b", "c"]

        result = aggregate("top_k", inputs, {"k": 2})

        assert result.items == ["a", "b"]

    def test_top_k_all(self):
        """Test top-k when k >= unique items."""
        inputs = ["a", "b"]

        result = aggregate("top_k", inputs, {"k": 10})

        assert len(result.items) == 2


# ==============================================================================
# Join Operation Tests
# ==============================================================================

class TestJoinOperation:
    """Tests for join_on operation."""

    def test_join_default_delimiter(self):
        """Test join with default delimiter."""
        inputs = ["a", "b", "c"]

        result = aggregate("join_on", inputs)

        assert result.items[0] == "a, b, c"

    def test_join_custom_delimiter(self):
        """Test join with custom delimiter."""
        inputs = ["a", "b", "c"]

        result = aggregate("join_on", inputs, {"delimiter": " | "})

        assert result.items[0] == "a | b | c"

    def test_join_empty(self):
        """Test join with empty input."""
        result = aggregate("join_on", [])

        assert result.items[0] == ""


# ==============================================================================
# Set Difference Tests
# ==============================================================================

class TestSetDifference:
    """Tests for diff_sets operation."""

    def test_diff_basic(self):
        """Test basic set difference."""
        inputs = ["a", "b", "c", "|||", "b", "d"]

        result = aggregate("diff_sets", inputs)

        assert set(result.items) == {"a", "c"}

    def test_diff_empty_result(self):
        """Test set difference with empty result."""
        inputs = ["a", "b", "|||", "a", "b", "c"]

        result = aggregate("diff_sets", inputs)

        assert result.items == []

    def test_diff_no_separator(self):
        """Test set difference without separator."""
        inputs = ["a", "b", "c", "d"]  # First half: a, b; Second half: c, d

        result = aggregate("diff_sets", inputs)

        assert set(result.items) == {"a", "b"}


# ==============================================================================
# Limit Handling Tests
# ==============================================================================

class TestLimitHandling:
    """Tests for limit parameter handling."""

    def test_limit_respected(self):
        """Test that limit is respected."""
        inputs = list("abcdefghij")

        result = aggregate("unique", inputs, limit=5)

        assert len(result.items) == 5

    def test_limit_greater_than_results(self):
        """Test limit greater than available results."""
        inputs = ["a", "b", "c"]

        result = aggregate("unique", inputs, limit=100)

        assert len(result.items) == 3


# ==============================================================================
# Edge Cases Tests
# ==============================================================================

class TestAggregatorEdgeCases:
    """Tests for edge cases in aggregation."""

    def test_empty_inputs(self):
        """Test operations with empty inputs."""
        for op in ["unique", "sort", "count", "top_k"]:
            result = aggregate(op, [])
            assert result.count == 0

    def test_single_input(self):
        """Test operations with single input."""
        result = aggregate("unique", ["single"])
        assert result.items == ["single"]

    def test_unicode_inputs(self):
        """Test operations with unicode inputs."""
        inputs = ["hello", "world", "hej"]

        result = aggregate("unique", inputs)

        assert len(result.items) == 3

    def test_empty_string_inputs(self):
        """Test operations with empty strings."""
        inputs = ["", "a", "", "b"]

        result = aggregate("unique", inputs)

        assert "" in result.items

    def test_special_characters(self):
        """Test operations with special characters."""
        inputs = ["a.b", "a*b", "a+b"]

        result = aggregate("unique", inputs)

        assert len(result.items) == 3

    def test_invalid_operation(self):
        """Test invalid operation raises error."""
        with pytest.raises(ValueError, match="Unknown operation"):
            aggregate("invalid_op", ["a"])
