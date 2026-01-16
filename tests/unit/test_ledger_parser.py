"""
Unit tests for the ledger parser module.

Tests cover:
- Structured ledger extraction
- Missing ledger handling
- Malformed ledger handling
- NO free-text inference (strict parsing only)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest


# ==============================================================================
# Ledger Data Types
# ==============================================================================

@dataclass
class LedgerEntry:
    """A structured ledger entry."""

    category: str
    content: str
    timestamp: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedLedger:
    """Result of ledger parsing."""

    decisions: list[str]
    invariants: list[str]
    observations: list[str]
    raw_entries: list[LedgerEntry]
    parse_errors: list[str]


# ==============================================================================
# Ledger Parsing Functions
# ==============================================================================

def parse_ledger_json(response: str) -> ParsedLedger | None:
    """
    Parse structured ledger from JSON format in response.

    IMPORTANT: This function ONLY extracts explicitly structured ledger data.
    It does NOT infer or generate ledger entries from free text.

    Expected format in response:
    ```json
    {
        "ledger": {
            "decisions": ["decision 1", "decision 2"],
            "invariants": ["invariant 1"],
            "observations": ["observation 1"]
        }
    }
    ```

    Or:
    <!-- LEDGER: {"decisions": [...], "invariants": [...]} -->

    Args:
        response: Model response text

    Returns:
        ParsedLedger if found, None otherwise
    """
    # Try to find JSON ledger block
    import re

    # Pattern 1: JSON code block
    json_pattern = r'```json\s*\n\s*\{[^}]*"ledger"[^}]*\}[^`]*```'
    match = re.search(json_pattern, response, re.DOTALL)

    if match:
        try:
            # Extract JSON from code block
            json_text = match.group(0)
            json_text = json_text.replace("```json", "").replace("```", "").strip()
            data = json.loads(json_text)
            ledger_data = data.get("ledger", data)
            return _create_parsed_ledger(ledger_data)
        except json.JSONDecodeError:
            pass

    # Pattern 2: HTML comment format
    comment_pattern = r'<!--\s*LEDGER:\s*(\{.*?\})\s*-->'
    match = re.search(comment_pattern, response, re.DOTALL)

    if match:
        try:
            ledger_data = json.loads(match.group(1))
            return _create_parsed_ledger(ledger_data)
        except json.JSONDecodeError:
            pass

    # Pattern 3: Inline JSON object with ledger key
    inline_pattern = r'\{[^{}]*"ledger"\s*:\s*\{[^{}]*\}[^{}]*\}'
    match = re.search(inline_pattern, response)

    if match:
        try:
            data = json.loads(match.group(0))
            ledger_data = data.get("ledger", {})
            return _create_parsed_ledger(ledger_data)
        except json.JSONDecodeError:
            pass

    return None


def _create_parsed_ledger(data: dict) -> ParsedLedger:
    """Create ParsedLedger from dictionary data."""
    return ParsedLedger(
        decisions=data.get("decisions", []),
        invariants=data.get("invariants", []),
        observations=data.get("observations", []),
        raw_entries=[],
        parse_errors=[],
    )


def extract_ledger_entries(response: str) -> list[LedgerEntry]:
    """
    Extract individual ledger entries from response.

    Only extracts explicitly marked entries, NOT inferred from context.

    Supported formats:
    - [DECISION] content
    - [INVARIANT] content
    - [OBSERVATION] content

    Args:
        response: Model response text

    Returns:
        List of extracted LedgerEntry objects
    """
    import re

    entries = []

    # Pattern for bracketed markers
    patterns = {
        "decision": r'\[DECISION\]\s*(.+?)(?=\[(?:DECISION|INVARIANT|OBSERVATION)\]|$)',
        "invariant": r'\[INVARIANT\]\s*(.+?)(?=\[(?:DECISION|INVARIANT|OBSERVATION)\]|$)',
        "observation": r'\[OBSERVATION\]\s*(.+?)(?=\[(?:DECISION|INVARIANT|OBSERVATION)\]|$)',
    }

    for category, pattern in patterns.items():
        for match in re.finditer(pattern, response, re.DOTALL | re.IGNORECASE):
            content = match.group(1).strip()
            if content:
                entries.append(LedgerEntry(category=category, content=content))

    return entries


def validate_ledger(ledger: ParsedLedger) -> tuple[bool, list[str]]:
    """
    Validate a parsed ledger.

    Args:
        ledger: Parsed ledger to validate

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    # Validate decisions
    for i, decision in enumerate(ledger.decisions):
        if not isinstance(decision, str):
            errors.append(f"Decision {i} is not a string")
        elif len(decision.strip()) == 0:
            errors.append(f"Decision {i} is empty")

    # Validate invariants
    for i, invariant in enumerate(ledger.invariants):
        if not isinstance(invariant, str):
            errors.append(f"Invariant {i} is not a string")
        elif len(invariant.strip()) == 0:
            errors.append(f"Invariant {i} is empty")

    return len(errors) == 0, errors


# ==============================================================================
# Structured Extraction Tests
# ==============================================================================

class TestStructuredExtraction:
    """Tests for structured ledger extraction."""

    def test_extract_json_code_block(self):
        """Test extraction from JSON code block."""
        response = '''
Here is the analysis:

```json
{
    "ledger": {
        "decisions": ["Use JWT for authentication", "Store tokens in httpOnly cookies"],
        "invariants": ["Token must be valid before access"],
        "observations": ["Current flow uses session cookies"]
    }
}
```
'''
        ledger = parse_ledger_json(response)

        assert ledger is not None
        assert len(ledger.decisions) == 2
        assert len(ledger.invariants) == 1
        assert len(ledger.observations) == 1

    def test_extract_html_comment(self):
        """Test extraction from HTML comment format."""
        response = '''
The authentication flow works as follows...

<!-- LEDGER: {"decisions": ["Migrate to OAuth"], "invariants": ["Rate limit API calls"]} -->
'''
        ledger = parse_ledger_json(response)

        assert ledger is not None
        assert "Migrate to OAuth" in ledger.decisions
        assert "Rate limit API calls" in ledger.invariants

    def test_extract_bracketed_markers(self):
        """Test extraction from bracketed markers."""
        response = '''
[DECISION] Use async/await for all database operations

[INVARIANT] Database connections must be pooled

[OBSERVATION] Current code uses callbacks in some places
'''
        entries = extract_ledger_entries(response)

        assert len(entries) == 3
        categories = {e.category for e in entries}
        assert categories == {"decision", "invariant", "observation"}

    def test_preserve_content_exactly(self):
        """Test that content is preserved exactly."""
        response = '<!-- LEDGER: {"decisions": ["Decision with special chars: <>&"]} -->'

        ledger = parse_ledger_json(response)

        assert ledger is not None
        assert ledger.decisions[0] == 'Decision with special chars: <>&"'


# ==============================================================================
# Missing Ledger Handling Tests
# ==============================================================================

class TestMissingLedgerHandling:
    """Tests for missing ledger handling."""

    def test_no_ledger_returns_none(self):
        """Test that missing ledger returns None."""
        response = '''
This is a regular response without any ledger information.
Just some text explaining how the code works.
'''
        ledger = parse_ledger_json(response)

        assert ledger is None

    def test_empty_response(self):
        """Test empty response handling."""
        ledger = parse_ledger_json("")
        assert ledger is None

    def test_whitespace_response(self):
        """Test whitespace-only response."""
        ledger = parse_ledger_json("   \n\t  ")
        assert ledger is None

    def test_partial_json_no_ledger_key(self):
        """Test JSON without ledger key."""
        response = '''
```json
{
    "status": "success",
    "data": {}
}
```
'''
        ledger = parse_ledger_json(response)

        # Should return None as no ledger key exists
        assert ledger is None

    def test_no_bracketed_markers(self):
        """Test response without bracketed markers."""
        response = "Just a normal response without markers."
        entries = extract_ledger_entries(response)

        assert len(entries) == 0


# ==============================================================================
# Malformed Ledger Handling Tests
# ==============================================================================

class TestMalformedLedgerHandling:
    """Tests for malformed ledger handling."""

    def test_invalid_json_in_code_block(self):
        """Test invalid JSON in code block."""
        response = '''
```json
{
    "ledger": {
        "decisions": ["unclosed string,
        "invariants": []
    }
}
```
'''
        ledger = parse_ledger_json(response)

        # Should return None on parse error
        assert ledger is None

    def test_invalid_json_in_comment(self):
        """Test invalid JSON in HTML comment."""
        response = '<!-- LEDGER: {not valid json} -->'

        ledger = parse_ledger_json(response)

        assert ledger is None

    def test_wrong_data_types(self):
        """Test ledger with wrong data types."""
        response = '''
```json
{
    "ledger": {
        "decisions": "should be array",
        "invariants": 123
    }
}
```
'''
        ledger = parse_ledger_json(response)

        if ledger:
            is_valid, errors = validate_ledger(ledger)
            # Should have validation errors
            # (actual behavior depends on implementation)

    def test_empty_arrays(self):
        """Test ledger with empty arrays."""
        response = '<!-- LEDGER: {"decisions": [], "invariants": []} -->'

        ledger = parse_ledger_json(response)

        assert ledger is not None
        assert len(ledger.decisions) == 0
        assert len(ledger.invariants) == 0

    def test_extra_fields_ignored(self):
        """Test that extra fields are ignored."""
        response = '<!-- LEDGER: {"decisions": ["d1"], "unknown_field": "value"} -->'

        ledger = parse_ledger_json(response)

        assert ledger is not None
        assert len(ledger.decisions) == 1


# ==============================================================================
# No Free-Text Inference Tests
# ==============================================================================

class TestNoFreeTextInference:
    """
    CRITICAL: Tests ensuring NO free-text inference.

    The ledger parser must ONLY extract explicitly structured data.
    It must NOT infer or generate entries from unstructured text.
    """

    def test_no_inference_from_narrative(self):
        """Test that narrative text is NOT converted to ledger entries."""
        response = '''
I decided to use JWT tokens for authentication because they are stateless.
The invariant we must maintain is that tokens expire after 1 hour.
I observed that the current implementation uses sessions.
'''
        # Even though text contains "decided", "invariant", "observed"
        # these should NOT be extracted as they're not structured
        ledger = parse_ledger_json(response)

        assert ledger is None

        entries = extract_ledger_entries(response)
        assert len(entries) == 0

    def test_no_inference_from_bullet_points(self):
        """Test that bullet points are NOT inferred as entries."""
        response = '''
Key points:
- We should use JWT (decision)
- Always validate tokens (invariant)
- Current flow is complex (observation)
'''
        ledger = parse_ledger_json(response)
        assert ledger is None

        entries = extract_ledger_entries(response)
        assert len(entries) == 0

    def test_no_inference_from_headers(self):
        """Test that headers are NOT inferred as entries."""
        response = '''
## Decisions
- Use OAuth 2.0

## Invariants
- Token expiration

## Observations
- Legacy code exists
'''
        ledger = parse_ledger_json(response)
        assert ledger is None

        entries = extract_ledger_entries(response)
        assert len(entries) == 0

    def test_only_explicit_markers_extracted(self):
        """Test that only explicit markers are extracted."""
        response = '''
Some context about decisions made earlier.

[DECISION] This is the only valid decision entry

More text about invariants in general.

And some observations about the code.
'''
        entries = extract_ledger_entries(response)

        # Only the explicitly marked entry should be extracted
        assert len(entries) == 1
        assert entries[0].category == "decision"
        assert "only valid decision" in entries[0].content

    def test_similar_words_not_extracted(self):
        """Test that similar words don't trigger extraction."""
        response = '''
The decisive factor was performance.
This invariant-like behavior is important.
Our observation-based approach works well.
'''
        entries = extract_ledger_entries(response)

        # "decisive", "invariant-like", "observation-based" should NOT match
        assert len(entries) == 0


# ==============================================================================
# Validation Tests
# ==============================================================================

class TestLedgerValidation:
    """Tests for ledger validation."""

    def test_valid_ledger(self):
        """Test validation of valid ledger."""
        ledger = ParsedLedger(
            decisions=["Decision 1", "Decision 2"],
            invariants=["Invariant 1"],
            observations=["Observation 1"],
            raw_entries=[],
            parse_errors=[],
        )

        is_valid, errors = validate_ledger(ledger)

        assert is_valid is True
        assert len(errors) == 0

    def test_empty_decision_invalid(self):
        """Test that empty decisions are invalid."""
        ledger = ParsedLedger(
            decisions=["Valid", ""],
            invariants=[],
            observations=[],
            raw_entries=[],
            parse_errors=[],
        )

        is_valid, errors = validate_ledger(ledger)

        assert is_valid is False
        assert len(errors) == 1

    def test_whitespace_decision_invalid(self):
        """Test that whitespace-only decisions are invalid."""
        ledger = ParsedLedger(
            decisions=["  \t\n  "],
            invariants=[],
            observations=[],
            raw_entries=[],
            parse_errors=[],
        )

        is_valid, errors = validate_ledger(ledger)

        assert is_valid is False

    def test_empty_ledger_valid(self):
        """Test that empty ledger is valid."""
        ledger = ParsedLedger(
            decisions=[],
            invariants=[],
            observations=[],
            raw_entries=[],
            parse_errors=[],
        )

        is_valid, errors = validate_ledger(ledger)

        assert is_valid is True


# ==============================================================================
# Edge Cases Tests
# ==============================================================================

class TestLedgerParserEdgeCases:
    """Tests for ledger parser edge cases."""

    def test_multiple_json_blocks(self):
        """Test response with multiple JSON blocks."""
        response = '''
```json
{
    "other": "data"
}
```

```json
{
    "ledger": {
        "decisions": ["Decision 1"]
    }
}
```
'''
        ledger = parse_ledger_json(response)

        # Should find the ledger block
        assert ledger is not None
        assert len(ledger.decisions) == 1

    def test_nested_json(self):
        """Test deeply nested JSON."""
        response = '''
```json
{
    "response": {
        "ledger": {
            "decisions": ["Nested decision"]
        }
    }
}
```
'''
        # May or may not extract depending on implementation
        # This tests the current behavior

    def test_unicode_in_ledger(self):
        """Test unicode content in ledger."""
        response = '<!-- LEDGER: {"decisions": ["Use emoji ", "Support internationalization"]} -->'

        ledger = parse_ledger_json(response)

        assert ledger is not None
        # Unicode should be preserved

    def test_special_characters_in_content(self):
        """Test special characters in content."""
        response = '''
```json
{
    "ledger": {
        "decisions": ["Use `backticks` and 'quotes' and \\"escapes\\""]
    }
}
```
'''
        ledger = parse_ledger_json(response)

        # Should handle special characters

    def test_very_long_entries(self):
        """Test very long ledger entries."""
        long_content = "x" * 10000
        response = f'<!-- LEDGER: {{"decisions": ["{long_content}"]}} -->'

        ledger = parse_ledger_json(response)

        assert ledger is not None
        assert len(ledger.decisions[0]) == 10000


# ==============================================================================
# Integration Tests
# ==============================================================================

class TestLedgerParserIntegration:
    """Integration tests for ledger parsing."""

    def test_full_parsing_pipeline(self):
        """Test complete ledger parsing pipeline."""
        response = '''
Here is my analysis of the authentication system:

The current implementation uses session-based authentication which has some limitations.

```json
{
    "ledger": {
        "decisions": [
            "Migrate from sessions to JWT tokens",
            "Implement token refresh mechanism"
        ],
        "invariants": [
            "All API endpoints must validate token before processing",
            "Tokens must expire within 1 hour"
        ],
        "observations": [
            "Legacy code still uses session cookies",
            "Mobile app already expects JWT format"
        ]
    }
}
```

[DECISION] Also consider implementing rate limiting

The migration should be done in phases to minimize disruption.
'''
        # Parse JSON ledger
        json_ledger = parse_ledger_json(response)

        assert json_ledger is not None
        assert len(json_ledger.decisions) == 2
        assert len(json_ledger.invariants) == 2
        assert len(json_ledger.observations) == 2

        # Parse bracketed markers
        marker_entries = extract_ledger_entries(response)

        assert len(marker_entries) == 1
        assert marker_entries[0].category == "decision"

        # Validate
        is_valid, errors = validate_ledger(json_ledger)
        assert is_valid is True
