"""
JSON Schema validation utilities and token budget management.

This module provides utilities for:
- Input/output validation against Pydantic models
- Token counting and budget enforcement
- Response truncation to meet output limits
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

# Lazy load tiktoken to avoid import overhead
_tokenizer = None


def _get_tokenizer() -> Any:
    """Get or create the tiktoken tokenizer (lazy loading)."""
    global _tokenizer
    if _tokenizer is None:
        import tiktoken

        # Use cl100k_base which is used by GPT-4 and Claude models
        _tokenizer = tiktoken.get_encoding("cl100k_base")
    return _tokenizer


@dataclass
class TokenBudget:
    """Token budget configuration for tool responses."""

    soft_limit: int = 8000  # Soft limit - try to stay under
    hard_limit: int = 25000  # Hard limit - must not exceed

    def __post_init__(self) -> None:
        """Validate budget configuration."""
        if self.soft_limit > self.hard_limit:
            raise ValueError("soft_limit cannot exceed hard_limit")
        if self.soft_limit < 100:
            raise ValueError("soft_limit must be at least 100")
        if self.hard_limit < 100:
            raise ValueError("hard_limit must be at least 100")


# Default token budget per PRD requirements
DEFAULT_TOKEN_BUDGET = TokenBudget(soft_limit=8000, hard_limit=25000)


def count_tokens(text: str) -> int:
    """
    Count tokens in a string using tiktoken.

    Args:
        text: The text to count tokens for

    Returns:
        Number of tokens
    """
    tokenizer = _get_tokenizer()
    return len(tokenizer.encode(text))


def estimate_json_tokens(obj: Any) -> int:
    """
    Estimate token count for a JSON-serializable object.

    Args:
        obj: Object to estimate tokens for

    Returns:
        Estimated token count
    """
    json_str = json.dumps(obj, default=str)
    return count_tokens(json_str)


T = TypeVar("T", bound=BaseModel)


def validate_input(model_class: type[T], data: dict[str, Any]) -> T:
    """
    Validate input data against a Pydantic model.

    Args:
        model_class: The Pydantic model class to validate against
        data: Input data dictionary

    Returns:
        Validated model instance

    Raises:
        ValueError: If validation fails with detailed error message
    """
    try:
        return model_class.model_validate(data)
    except ValidationError as e:
        errors = []
        for error in e.errors():
            loc = ".".join(str(x) for x in error["loc"])
            msg = error["msg"]
            errors.append(f"{loc}: {msg}")
        raise ValueError(f"Input validation failed: {'; '.join(errors)}") from e


def validate_output(model_class: type[T], data: dict[str, Any]) -> T:
    """
    Validate output data against a Pydantic model.

    Args:
        model_class: The Pydantic model class to validate against
        data: Output data dictionary

    Returns:
        Validated model instance

    Raises:
        ValueError: If validation fails
    """
    try:
        return model_class.model_validate(data)
    except ValidationError as e:
        logger.error(f"Output validation failed: {e}")
        raise ValueError(f"Output validation failed: {e}") from e


def truncate_string_to_tokens(
    text: str,
    max_tokens: int,
    suffix: str = "\n... [truncated]",
) -> tuple[str, bool]:
    """
    Truncate a string to fit within a token budget.

    Args:
        text: Text to truncate
        max_tokens: Maximum tokens allowed
        suffix: Suffix to append if truncated

    Returns:
        Tuple of (truncated_text, was_truncated)
    """
    tokenizer = _get_tokenizer()
    tokens = tokenizer.encode(text)

    if len(tokens) <= max_tokens:
        return text, False

    # Account for suffix tokens
    suffix_tokens = len(tokenizer.encode(suffix))
    target_tokens = max_tokens - suffix_tokens

    if target_tokens <= 0:
        return suffix, True

    # Truncate tokens and decode
    truncated_tokens = tokens[:target_tokens]
    truncated_text = tokenizer.decode(truncated_tokens)

    return truncated_text + suffix, True


def truncate_list_to_tokens(
    items: list[Any],
    max_tokens: int,
    item_serializer: Any | None = None,
) -> tuple[list[Any], bool, int]:
    """
    Truncate a list to fit within a token budget.

    Args:
        items: List of items to truncate
        max_tokens: Maximum tokens allowed
        item_serializer: Optional function to serialize items for token counting

    Returns:
        Tuple of (truncated_list, was_truncated, token_count)
    """
    if not items:
        return [], False, 0

    serializer = item_serializer or (lambda x: json.dumps(x, default=str))

    result: list[Any] = []
    total_tokens = 2  # Account for [] brackets

    for item in items:
        item_str = serializer(item)
        item_tokens = count_tokens(item_str) + 1  # +1 for comma

        if total_tokens + item_tokens > max_tokens:
            return result, True, total_tokens

        result.append(item)
        total_tokens += item_tokens

    return result, False, total_tokens


def truncate_to_token_budget(
    response: dict[str, Any],
    budget: TokenBudget | None = None,
    truncatable_fields: list[str] | None = None,
) -> tuple[dict[str, Any], bool]:
    """
    Truncate a response dictionary to fit within token budget.

    This function intelligently truncates fields in priority order:
    1. First tries to truncate explicitly specified truncatable_fields
    2. Falls back to truncating large string/list fields

    Args:
        response: Response dictionary to truncate
        budget: Token budget (defaults to DEFAULT_TOKEN_BUDGET)
        truncatable_fields: List of field names that can be truncated (in priority order)

    Returns:
        Tuple of (truncated_response, was_truncated)
    """
    budget = budget or DEFAULT_TOKEN_BUDGET
    truncatable_fields = truncatable_fields or []

    # Check if we're under budget
    current_tokens = estimate_json_tokens(response)
    if current_tokens <= budget.soft_limit:
        return response, False

    logger.info(
        f"Response exceeds soft limit ({current_tokens} > {budget.soft_limit}), truncating"
    )

    result = response.copy()
    was_truncated = False

    # Try truncating specified fields first
    for field in truncatable_fields:
        if field not in result:
            continue

        value = result[field]

        if isinstance(value, str):
            # Calculate how much we need to reduce
            target_tokens = budget.soft_limit - (current_tokens - count_tokens(value))
            if target_tokens > 0:
                result[field], truncated = truncate_string_to_tokens(value, target_tokens)
                if truncated:
                    was_truncated = True

        elif isinstance(value, list):
            target_tokens = budget.soft_limit - (
                current_tokens - estimate_json_tokens(value)
            )
            if target_tokens > 0:
                result[field], truncated, _ = truncate_list_to_tokens(value, target_tokens)
                if truncated:
                    was_truncated = True

        current_tokens = estimate_json_tokens(result)
        if current_tokens <= budget.soft_limit:
            break

    # If still over hard limit, aggressively truncate
    if current_tokens > budget.hard_limit:
        logger.warning(
            f"Response exceeds hard limit ({current_tokens} > {budget.hard_limit}), "
            "aggressive truncation required"
        )

        # Find and truncate the largest field
        for field, value in sorted(
            result.items(),
            key=lambda x: estimate_json_tokens(x[1]),
            reverse=True,
        ):
            if field in ("ok", "request_id", "error"):
                continue  # Don't truncate essential fields

            if isinstance(value, str):
                result[field] = "[content truncated due to size limits]"
                was_truncated = True
            elif isinstance(value, list):
                result[field] = value[:5] if len(value) > 5 else value
                was_truncated = True

            current_tokens = estimate_json_tokens(result)
            if current_tokens <= budget.hard_limit:
                break

    return result, was_truncated


def create_pagination_cursor(
    offset: int,
    limit: int,
    total: int | None = None,
) -> str | None:
    """
    Create a pagination cursor for the next page.

    Args:
        offset: Current offset
        limit: Page size
        total: Total count if known

    Returns:
        Cursor string or None if no more pages
    """
    import base64

    next_offset = offset + limit

    # Check if there are more results
    if total is not None and next_offset >= total:
        return None

    cursor_data = {"offset": next_offset, "limit": limit}
    cursor_json = json.dumps(cursor_data)
    return base64.b64encode(cursor_json.encode()).decode()


def parse_pagination_cursor(cursor: str | None) -> tuple[int, int]:
    """
    Parse a pagination cursor.

    Args:
        cursor: Cursor string or None

    Returns:
        Tuple of (offset, limit)
    """
    import base64

    if cursor is None:
        return 0, 50  # Default values

    try:
        cursor_json = base64.b64decode(cursor.encode()).decode()
        cursor_data = json.loads(cursor_json)
        return cursor_data.get("offset", 0), cursor_data.get("limit", 50)
    except Exception as e:
        logger.warning(f"Failed to parse cursor: {e}")
        return 0, 50


def get_json_schema(model_class: type[BaseModel]) -> dict[str, Any]:
    """
    Get JSON Schema for a Pydantic model.

    Args:
        model_class: Pydantic model class

    Returns:
        JSON Schema dictionary
    """
    return model_class.model_json_schema()
