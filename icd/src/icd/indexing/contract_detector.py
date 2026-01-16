"""
Heuristic contract detection for code chunks.

Identifies interface definitions, type declarations, abstract classes,
and other boundary-defining code elements that are crucial for
understanding system architecture.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from icd.config import Config
    from icd.indexing.chunker import Chunk
    from icd.storage.contract_store import Contract, ContractType

logger = structlog.get_logger(__name__)


# Language-specific contract patterns
CONTRACT_PATTERNS = {
    "python": {
        "abstract_class": [
            r"class\s+\w+\s*\([^)]*ABC[^)]*\)",
            r"@abstractmethod",
            r"from\s+abc\s+import",
        ],
        "protocol": [
            r"class\s+\w+\s*\([^)]*Protocol[^)]*\)",
            r"typing\.Protocol",
            r"typing_extensions\.Protocol",
        ],
        "dataclass": [
            r"@dataclass",
            r"@dataclasses\.dataclass",
        ],
        "type_alias": [
            r":\s*TypeAlias\s*=",
            r"=\s*TypeVar\s*\(",
            r"=\s*NewType\s*\(",
        ],
        "schema": [
            r"class\s+\w+\s*\([^)]*BaseModel[^)]*\)",  # Pydantic
            r"class\s+\w+\s*\([^)]*Schema[^)]*\)",
            r"@schema",
        ],
        "enum": [
            r"class\s+\w+\s*\([^)]*Enum[^)]*\)",
            r"class\s+\w+\s*\([^)]*IntEnum[^)]*\)",
            r"class\s+\w+\s*\([^)]*StrEnum[^)]*\)",
        ],
    },
    "typescript": {
        "interface": [
            r"^\s*(?:export\s+)?interface\s+\w+",
        ],
        "type_alias": [
            r"^\s*(?:export\s+)?type\s+\w+\s*=",
        ],
        "abstract_class": [
            r"^\s*(?:export\s+)?abstract\s+class\s+\w+",
        ],
        "enum": [
            r"^\s*(?:export\s+)?(?:const\s+)?enum\s+\w+",
        ],
    },
    "javascript": {
        "schema": [
            r"const\s+\w+Schema\s*=",
            r"\.schema\s*\(",
            r"Joi\.",
            r"yup\.",
            r"zod\.",
        ],
    },
    "go": {
        "interface": [
            r"^\s*type\s+\w+\s+interface\s*\{",
        ],
        "struct": [
            r"^\s*type\s+\w+\s+struct\s*\{",
        ],
    },
    "rust": {
        "trait": [
            r"^\s*(?:pub\s+)?trait\s+\w+",
        ],
        "struct": [
            r"^\s*(?:pub\s+)?struct\s+\w+",
        ],
        "enum": [
            r"^\s*(?:pub\s+)?enum\s+\w+",
        ],
    },
    "java": {
        "interface": [
            r"^\s*(?:public\s+)?interface\s+\w+",
        ],
        "abstract_class": [
            r"^\s*(?:public\s+)?abstract\s+class\s+\w+",
        ],
        "enum": [
            r"^\s*(?:public\s+)?enum\s+\w+",
        ],
    },
    "c": {
        "struct": [
            r"^\s*(?:typedef\s+)?struct\s+\w*\s*\{",
        ],
        "enum": [
            r"^\s*(?:typedef\s+)?enum\s+\w*\s*\{",
        ],
    },
    "cpp": {
        "interface": [
            r"class\s+\w+\s*\{[^}]*virtual\s+\w+\s*\([^)]*\)\s*=\s*0",
        ],
        "abstract_class": [
            r"class\s+\w+.*\{[^}]*virtual.*=\s*0",
        ],
        "struct": [
            r"^\s*(?:template\s*<[^>]*>\s*)?struct\s+\w+",
        ],
        "enum": [
            r"^\s*enum\s+(?:class\s+)?\w+",
        ],
    },
}

# Generic patterns that apply to multiple languages
GENERIC_CONTRACT_INDICATORS = [
    r"interface",
    r"abstract",
    r"protocol",
    r"trait",
    r"contract",
    r"schema",
    r"typedef",
    r"@api",
    r"@public",
    r"@export",
]


@dataclass
class ContractMatch:
    """Result of contract pattern matching."""

    contract_type: str
    confidence: float
    pattern: str
    match_text: str


class ContractDetector:
    """
    Heuristic-based contract detector.

    Features:
    - Language-aware pattern matching
    - Multiple contract types
    - Confidence scoring
    - Signature extraction
    """

    def __init__(self, config: "Config") -> None:
        """
        Initialize the contract detector.

        Args:
            config: ICD configuration.
        """
        self.config = config
        self.enabled = config.contract.enabled
        self.custom_patterns = config.contract.patterns
        self.boost_factor = config.contract.boost_factor

        # Compile patterns
        self._compiled_patterns: dict[str, dict[str, list[re.Pattern]]] = {}
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile all regex patterns."""
        for language, type_patterns in CONTRACT_PATTERNS.items():
            self._compiled_patterns[language] = {}
            for contract_type, patterns in type_patterns.items():
                self._compiled_patterns[language][contract_type] = [
                    re.compile(p, re.MULTILINE) for p in patterns
                ]

        # Compile generic patterns
        self._generic_patterns = [
            re.compile(p, re.IGNORECASE) for p in GENERIC_CONTRACT_INDICATORS
        ]

        # Compile custom patterns
        self._custom_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.custom_patterns
        ]

    def is_contract(self, chunk: "Chunk") -> bool:
        """
        Determine if a chunk represents a contract.

        Args:
            chunk: Code chunk to analyze.

        Returns:
            True if the chunk is a contract.
        """
        if not self.enabled:
            return False

        matches = self.detect_contracts(chunk)
        return len(matches) > 0

    def detect_contracts(self, chunk: "Chunk") -> list[ContractMatch]:
        """
        Detect contract patterns in a chunk.

        Args:
            chunk: Code chunk to analyze.

        Returns:
            List of contract matches.
        """
        if not self.enabled:
            return []

        matches: list[ContractMatch] = []
        content = chunk.content
        language = chunk.language

        # Check language-specific patterns
        if language in self._compiled_patterns:
            for contract_type, patterns in self._compiled_patterns[language].items():
                for pattern in patterns:
                    match = pattern.search(content)
                    if match:
                        matches.append(
                            ContractMatch(
                                contract_type=contract_type,
                                confidence=0.9,
                                pattern=pattern.pattern,
                                match_text=match.group(0),
                            )
                        )

        # Check generic patterns (lower confidence)
        for pattern in self._generic_patterns:
            match = pattern.search(content)
            if match:
                # Avoid duplicates
                if not any(m.match_text == match.group(0) for m in matches):
                    matches.append(
                        ContractMatch(
                            contract_type="unknown",
                            confidence=0.6,
                            pattern=pattern.pattern,
                            match_text=match.group(0),
                        )
                    )

        # Check custom patterns
        for pattern in self._custom_patterns:
            match = pattern.search(content)
            if match:
                if not any(m.match_text == match.group(0) for m in matches):
                    matches.append(
                        ContractMatch(
                            contract_type="custom",
                            confidence=0.7,
                            pattern=pattern.pattern,
                            match_text=match.group(0),
                        )
                    )

        # Consider symbol type
        if chunk.symbol_type in (
            "interface_declaration",
            "type_alias_declaration",
            "trait_item",
            "abstract_class",
        ):
            if not matches:
                matches.append(
                    ContractMatch(
                        contract_type=self._normalize_symbol_type(chunk.symbol_type),
                        confidence=0.95,
                        pattern="symbol_type",
                        match_text=chunk.symbol_type,
                    )
                )

        return matches

    def _normalize_symbol_type(self, symbol_type: str) -> str:
        """Normalize tree-sitter symbol type to contract type."""
        mappings = {
            "interface_declaration": "interface",
            "type_alias_declaration": "type_alias",
            "trait_item": "trait",
            "abstract_class": "abstract_class",
            "struct_item": "struct",
            "struct_specifier": "struct",
            "enum_item": "enum",
            "class_definition": "class",
        }
        return mappings.get(symbol_type, "unknown")

    def extract_signature(self, chunk: "Chunk") -> str | None:
        """
        Extract the contract signature from a chunk.

        Args:
            chunk: Code chunk.

        Returns:
            Signature string or None.
        """
        content = chunk.content
        lines = content.split("\n")

        # Get first non-empty, non-comment line(s)
        signature_lines: list[str] = []
        in_signature = False
        brace_count = 0

        for line in lines:
            stripped = line.strip()

            # Skip empty lines and comments at the start
            if not in_signature:
                if not stripped or stripped.startswith(("#", "//", "/*", "*")):
                    continue
                in_signature = True

            if in_signature:
                signature_lines.append(line)

                # Track braces
                brace_count += line.count("{") + line.count("(")
                brace_count -= line.count("}") + line.count(")")

                # Stop at opening brace or colon (Python)
                if "{" in line or (line.rstrip().endswith(":") and chunk.language == "python"):
                    break

                # Limit signature length
                if len(signature_lines) > 5:
                    break

        if signature_lines:
            return "\n".join(signature_lines).strip()

        return None

    def extract_dependencies(self, chunk: "Chunk") -> list[str]:
        """
        Extract dependencies (extends, implements) from a chunk.

        Args:
            chunk: Code chunk.

        Returns:
            List of dependency names.
        """
        content = chunk.content
        language = chunk.language
        dependencies: list[str] = []

        # Language-specific patterns
        patterns = {
            "python": [
                r"class\s+\w+\s*\(([^)]+)\)",  # Python inheritance
            ],
            "typescript": [
                r"extends\s+([\w,\s]+)",
                r"implements\s+([\w,\s]+)",
            ],
            "javascript": [
                r"extends\s+(\w+)",
            ],
            "java": [
                r"extends\s+(\w+)",
                r"implements\s+([\w,\s]+)",
            ],
            "go": [
                r"//\s*implements:\s*([\w,\s]+)",  # Go convention
            ],
            "rust": [
                r"impl\s+(\w+)\s+for",
                r":\s*([\w\s+]+)(?:\s*\{|$)",  # Trait bounds
            ],
        }

        if language in patterns:
            for pattern in patterns[language]:
                matches = re.findall(pattern, content)
                for match in matches:
                    # Split on comma and clean up
                    names = [n.strip() for n in match.split(",")]
                    dependencies.extend(n for n in names if n and n != "object")

        return list(set(dependencies))

    def create_contract(self, chunk: "Chunk") -> "Contract | None":
        """
        Create a Contract object from a chunk.

        Args:
            chunk: Code chunk identified as a contract.

        Returns:
            Contract object or None.
        """
        if not chunk.is_contract:
            return None

        from icd.storage.contract_store import Contract, ContractType

        # Detect contract type
        matches = self.detect_contracts(chunk)
        if not matches:
            return None

        # Use highest confidence match
        best_match = max(matches, key=lambda m: m.confidence)

        # Map to ContractType enum
        type_mapping = {
            "interface": ContractType.INTERFACE,
            "abstract_class": ContractType.ABSTRACT_CLASS,
            "protocol": ContractType.PROTOCOL,
            "trait": ContractType.TRAIT,
            "type_alias": ContractType.TYPE_ALIAS,
            "schema": ContractType.SCHEMA,
            "dataclass": ContractType.DATACLASS,
            "struct": ContractType.STRUCT,
            "enum": ContractType.ENUM,
            "api_endpoint": ContractType.API_ENDPOINT,
        }

        contract_type = type_mapping.get(
            best_match.contract_type, ContractType.UNKNOWN
        )

        # Generate contract ID
        contract_id = hashlib.sha256(
            f"{chunk.file_path}:{chunk.symbol_name or ''}:{chunk.start_line}".encode()
        ).hexdigest()[:16]

        return Contract(
            contract_id=contract_id,
            chunk_id=chunk.chunk_id,
            name=chunk.symbol_name or f"contract_{contract_id[:8]}",
            contract_type=contract_type,
            file_path=chunk.file_path,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            language=chunk.language,
            signature=self.extract_signature(chunk),
            dependencies=self.extract_dependencies(chunk),
            metadata={
                "confidence": best_match.confidence,
                "pattern": best_match.pattern,
            },
        )

    def get_contract_boost(self, chunk: "Chunk") -> float:
        """
        Get the retrieval boost factor for a contract chunk.

        Args:
            chunk: Code chunk.

        Returns:
            Boost factor (1.0 for non-contracts).
        """
        if not chunk.is_contract:
            return 1.0

        matches = self.detect_contracts(chunk)
        if not matches:
            return 1.0

        # Scale boost by confidence
        max_confidence = max(m.confidence for m in matches)
        return 1.0 + (self.boost_factor - 1.0) * max_confidence
