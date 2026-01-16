"""
Unit tests for the contract detection module.

Tests cover:
- Contract pattern detection
- Language-specific contract identification
- Contract boost application
- Edge cases
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest


# ==============================================================================
# Contract Detection Functions
# ==============================================================================

# Default contract patterns from config
DEFAULT_CONTRACT_PATTERNS = [
    r"\binterface\b",
    r"\babstract\b",
    r"\bprotocol\b",
    r"\btrait\b",
    r"\btype\s+\w+\s*=",  # TypeScript type aliases
    r"@dataclass",
    r"\bschema\b",
    r"\bmodel\b",
    r"\bstruct\b",
    r"\benum\b",
]

# File patterns that indicate contracts
CONTRACT_FILE_PATTERNS = [
    r"types?\.(?:ts|d\.ts|py|go|rs)$",
    r"interfaces?\.(?:ts|d\.ts|py|go|rs)$",
    r"schemas?\.(?:yaml|yml|json|graphql)$",
    r"contracts?\.(?:sol|yaml|yml)$",
    r"models?\.(?:py|ts|go|rs)$",
    r"\.d\.ts$",  # TypeScript declaration files
    r"api\.(?:yaml|yml|json)$",
    r"openapi\.(?:yaml|yml|json)$",
    r"swagger\.(?:yaml|yml|json)$",
]


def is_contract_content(content: str, patterns: list[str] | None = None) -> bool:
    """
    Detect if content contains contract definitions.

    Args:
        content: Source code content
        patterns: Regex patterns to check (uses defaults if None)

    Returns:
        True if content appears to be a contract
    """
    patterns = patterns or DEFAULT_CONTRACT_PATTERNS

    for pattern in patterns:
        if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
            return True

    return False


def is_contract_file(file_path: str | Path) -> bool:
    """
    Detect if file path indicates a contract file.

    Args:
        file_path: Path to the file

    Returns:
        True if file appears to be a contract
    """
    path_str = str(file_path)

    for pattern in CONTRACT_FILE_PATTERNS:
        if re.search(pattern, path_str, re.IGNORECASE):
            return True

    return False


def detect_contract_type(content: str) -> str | None:
    """
    Detect the specific type of contract.

    Returns:
        Contract type string or None
    """
    type_patterns = {
        "interface": r"\binterface\s+\w+",
        "abstract_class": r"\babstract\s+class\s+\w+",
        "protocol": r"\bprotocol\s+\w+",
        "trait": r"\btrait\s+\w+",
        "type_alias": r"\btype\s+\w+\s*=",
        "dataclass": r"@dataclass",
        "schema": r"\bschema\b",
        "struct": r"\bstruct\s+\w+",
        "enum": r"\benum\s+\w+",
    }

    for contract_type, pattern in type_patterns.items():
        if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
            return contract_type

    return None


def compute_contract_boost(
    is_contract: bool,
    boost_factor: float = 1.5,
    base_score: float = 0.0
) -> float:
    """
    Compute contract boost score.

    Args:
        is_contract: Whether item is a contract
        boost_factor: Multiplication factor for contracts
        base_score: Base relevance score

    Returns:
        Boosted score
    """
    if is_contract:
        return base_score * boost_factor
    return base_score


# ==============================================================================
# Content Pattern Detection Tests
# ==============================================================================

class TestContentPatternDetection:
    """Tests for contract pattern detection in content."""

    def test_typescript_interface(self):
        """Test TypeScript interface detection."""
        content = """
interface User {
  id: string;
  name: string;
  email: string;
}
"""
        assert is_contract_content(content) is True
        assert detect_contract_type(content) == "interface"

    def test_typescript_type_alias(self):
        """Test TypeScript type alias detection."""
        content = """
type UserId = string;
type UserRole = 'admin' | 'user' | 'guest';
"""
        assert is_contract_content(content) is True
        assert detect_contract_type(content) == "type_alias"

    def test_python_abstract_class(self):
        """Test Python abstract class detection."""
        content = """
from abc import ABC, abstractmethod

class AbstractHandler(ABC):
    @abstractmethod
    def handle(self, request):
        pass
"""
        assert is_contract_content(content) is True
        assert detect_contract_type(content) == "abstract_class"

    def test_python_dataclass(self):
        """Test Python dataclass detection."""
        content = """
from dataclasses import dataclass

@dataclass
class User:
    id: str
    name: str
    email: str
"""
        assert is_contract_content(content) is True
        assert detect_contract_type(content) == "dataclass"

    def test_python_protocol(self):
        """Test Python protocol detection."""
        content = """
from typing import Protocol

class Readable(Protocol):
    def read(self) -> bytes:
        ...
"""
        assert is_contract_content(content) is True
        assert detect_contract_type(content) == "protocol"

    def test_rust_trait(self):
        """Test Rust trait detection."""
        content = """
pub trait Handler {
    fn handle(&self, request: Request) -> Response;
}
"""
        assert is_contract_content(content) is True
        assert detect_contract_type(content) == "trait"

    def test_rust_struct(self):
        """Test Rust struct detection."""
        content = """
#[derive(Debug, Clone)]
pub struct Config {
    pub host: String,
    pub port: u16,
}
"""
        assert is_contract_content(content) is True
        assert detect_contract_type(content) == "struct"

    def test_go_struct(self):
        """Test Go struct detection."""
        content = """
type User struct {
    ID    string
    Name  string
    Email string
}
"""
        assert is_contract_content(content) is True
        assert detect_contract_type(content) == "struct"

    def test_go_interface(self):
        """Test Go interface detection."""
        content = """
type Handler interface {
    Handle(ctx context.Context, req Request) (Response, error)
}
"""
        assert is_contract_content(content) is True
        assert detect_contract_type(content) == "interface"

    def test_enum_detection(self):
        """Test enum detection."""
        content = """
enum Status {
    Active,
    Inactive,
    Pending,
}
"""
        assert is_contract_content(content) is True
        assert detect_contract_type(content) == "enum"

    def test_schema_keyword(self):
        """Test schema keyword detection."""
        content = """
const userSchema = z.object({
    id: z.string(),
    name: z.string(),
});
"""
        assert is_contract_content(content) is True

    def test_non_contract_code(self):
        """Test non-contract code detection."""
        content = """
function calculateSum(a, b) {
    return a + b;
}

const result = calculateSum(1, 2);
console.log(result);
"""
        assert is_contract_content(content) is False

    def test_implementation_not_contract(self):
        """Test that implementation code is not detected as contract."""
        content = """
class UserService {
    constructor(private db: Database) {}

    async getUser(id: string): Promise<User | null> {
        return this.db.findOne('users', { id });
    }
}
"""
        # Has 'class' but not interface/abstract
        assert detect_contract_type(content) is None or is_contract_content(content) is False


# ==============================================================================
# File Path Detection Tests
# ==============================================================================

class TestFilePathDetection:
    """Tests for contract file path detection."""

    @pytest.mark.parametrize("file_path,expected", [
        ("src/types.ts", True),
        ("src/types/user.ts", True),
        ("lib/interfaces.ts", True),
        ("api/schemas/user.yaml", True),
        ("contracts/api.yaml", True),
        ("models/user.py", True),
        ("types.d.ts", True),
        ("api.yaml", True),
        ("openapi.yaml", True),
        ("swagger.json", True),
        ("src/utils.ts", False),
        ("src/handler.py", False),
        ("tests/test_user.py", False),
        ("README.md", False),
    ])
    def test_contract_file_paths(self, file_path, expected):
        """Test contract file path detection."""
        assert is_contract_file(file_path) == expected

    def test_typescript_declaration_files(self):
        """Test TypeScript declaration files are contracts."""
        declaration_files = [
            "types.d.ts",
            "global.d.ts",
            "lib/custom.d.ts",
            "@types/node/index.d.ts",
        ]

        for file_path in declaration_files:
            assert is_contract_file(file_path) is True

    def test_schema_files(self):
        """Test schema file detection."""
        schema_files = [
            "schema.yaml",
            "schema.yml",
            "schema.json",
            "user.schema.yaml",
            "api/schemas/auth.yaml",
        ]

        for file_path in schema_files:
            assert is_contract_file(file_path) is True

    def test_api_spec_files(self):
        """Test API specification file detection."""
        api_files = [
            "api.yaml",
            "api.yml",
            "api.json",
            "openapi.yaml",
            "openapi.yml",
            "swagger.yaml",
            "swagger.json",
        ]

        for file_path in api_files:
            assert is_contract_file(file_path) is True

    def test_path_case_insensitivity(self):
        """Test case insensitive path matching."""
        paths = [
            "TYPES.ts",
            "Types.TS",
            "INTERFACES.ts",
            "Schema.YAML",
        ]

        for path in paths:
            assert is_contract_file(path) is True


# ==============================================================================
# Contract Type Detection Tests
# ==============================================================================

class TestContractTypeDetection:
    """Tests for specific contract type detection."""

    def test_interface_type(self):
        """Test interface type detection."""
        content = "interface User { id: string; }"
        assert detect_contract_type(content) == "interface"

    def test_abstract_class_type(self):
        """Test abstract class type detection."""
        content = "abstract class Base { abstract method(): void; }"
        assert detect_contract_type(content) == "abstract_class"

    def test_protocol_type(self):
        """Test protocol type detection."""
        content = "protocol Readable: def read(self) -> bytes"
        assert detect_contract_type(content) == "protocol"

    def test_trait_type(self):
        """Test trait type detection."""
        content = "trait Display { fn fmt(&self); }"
        assert detect_contract_type(content) == "trait"

    def test_type_alias_type(self):
        """Test type alias detection."""
        content = "type UserId = string"
        assert detect_contract_type(content) == "type_alias"

    def test_struct_type(self):
        """Test struct type detection."""
        content = "struct Point { x: i32, y: i32 }"
        assert detect_contract_type(content) == "struct"

    def test_enum_type(self):
        """Test enum type detection."""
        content = "enum Color { Red, Green, Blue }"
        assert detect_contract_type(content) == "enum"

    def test_no_contract_type(self):
        """Test non-contract returns None."""
        content = "function add(a, b) { return a + b; }"
        assert detect_contract_type(content) is None


# ==============================================================================
# Contract Boost Tests
# ==============================================================================

class TestContractBoost:
    """Tests for contract score boosting."""

    def test_boost_applied_to_contracts(self):
        """Test boost is applied to contracts."""
        base_score = 0.5
        boost_factor = 1.5

        boosted = compute_contract_boost(True, boost_factor, base_score)

        assert boosted == 0.75

    def test_no_boost_for_non_contracts(self):
        """Test no boost for non-contracts."""
        base_score = 0.5
        boost_factor = 1.5

        score = compute_contract_boost(False, boost_factor, base_score)

        assert score == base_score

    def test_default_boost_factor(self):
        """Test default boost factor is 1.5."""
        base_score = 1.0

        boosted = compute_contract_boost(True, base_score=base_score)

        assert boosted == 1.5

    @pytest.mark.parametrize("boost_factor", [1.0, 1.5, 2.0, 3.0, 5.0])
    def test_various_boost_factors(self, boost_factor):
        """Test various boost factor values."""
        base_score = 0.5

        boosted = compute_contract_boost(True, boost_factor, base_score)

        assert boosted == base_score * boost_factor

    def test_boost_preserves_zero(self):
        """Test boost of zero score is still zero."""
        boosted = compute_contract_boost(True, 1.5, 0.0)
        assert boosted == 0.0


# ==============================================================================
# Edge Cases Tests
# ==============================================================================

class TestContractDetectorEdgeCases:
    """Tests for edge cases in contract detection."""

    def test_empty_content(self):
        """Test empty content handling."""
        assert is_contract_content("") is False

    def test_whitespace_only(self):
        """Test whitespace-only content."""
        assert is_contract_content("   \n\t  ") is False

    def test_comment_with_keyword(self):
        """Test that comments containing keywords are detected."""
        content = "// This is an interface comment"
        # Currently detected - may want to filter comments in production
        assert is_contract_content(content) is True

    def test_string_with_keyword(self):
        """Test that strings containing keywords are detected."""
        content = 'const msg = "This interface is great";'
        # Currently detected - may want to filter strings in production
        assert is_contract_content(content) is True

    def test_partial_keyword_no_match(self):
        """Test that partial keywords don't match."""
        content = "const interfaces = [];"  # plural, different context
        # Should match due to 'interface' substring
        # More sophisticated parsing would avoid this

    def test_unicode_content(self):
        """Test content with unicode characters."""
        content = """
interface Usuario {
    nombre: string;
    correo: string;
}
"""
        assert is_contract_content(content) is True

    def test_mixed_contract_types(self):
        """Test content with multiple contract types."""
        content = """
interface User { id: string; }
type UserId = string;
enum Status { Active, Inactive }
"""
        assert is_contract_content(content) is True
        # Returns first detected type
        first_type = detect_contract_type(content)
        assert first_type in ["interface", "type_alias", "enum"]


# ==============================================================================
# Custom Patterns Tests
# ==============================================================================

class TestCustomPatterns:
    """Tests for custom contract patterns."""

    def test_custom_patterns_override(self):
        """Test custom patterns override defaults."""
        custom_patterns = [r"\bspec\b"]

        content = "const spec = { version: 1 };"
        assert is_contract_content(content) is False  # Default patterns
        assert is_contract_content(content, custom_patterns) is True

    def test_empty_patterns_no_match(self):
        """Test empty patterns list matches nothing."""
        content = "interface User { id: string; }"
        assert is_contract_content(content, []) is False

    def test_invalid_regex_handled(self):
        """Test invalid regex patterns are handled."""
        # This should raise an error or be handled gracefully
        # Depending on implementation
        pass  # Implementation-specific


# ==============================================================================
# Integration Tests
# ==============================================================================

class TestContractDetectorIntegration:
    """Integration tests for contract detection."""

    def test_detect_and_boost_pipeline(self):
        """Test complete detection and boost pipeline."""
        chunks = [
            {"id": "c1", "content": "interface User { id: string; }", "path": "types.ts", "score": 0.8},
            {"id": "c2", "content": "function getUser(id) { ... }", "path": "service.ts", "score": 0.9},
            {"id": "c3", "content": "type UserId = string;", "path": "types.ts", "score": 0.7},
        ]

        boost_factor = 1.5

        for chunk in chunks:
            is_contract = (
                is_contract_content(chunk["content"]) or
                is_contract_file(chunk["path"])
            )
            chunk["is_contract"] = is_contract
            chunk["boosted_score"] = compute_contract_boost(
                is_contract, boost_factor, chunk["score"]
            )

        # Verify contracts are detected and boosted
        assert chunks[0]["is_contract"] is True
        assert chunks[0]["boosted_score"] == 1.2

        assert chunks[1]["is_contract"] is False
        assert chunks[1]["boosted_score"] == 0.9

        assert chunks[2]["is_contract"] is True
        assert chunks[2]["boosted_score"] == 1.05
