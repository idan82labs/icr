"""
Contract store for indexing and retrieving code contracts.

Contracts are interface definitions, type declarations, abstract classes,
and other boundary-defining code elements that are crucial for understanding
system architecture.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from icd.config import Config
    from icd.storage.sqlite_store import SQLiteStore

logger = structlog.get_logger(__name__)


class ContractType(str, Enum):
    """Types of code contracts."""

    INTERFACE = "interface"
    ABSTRACT_CLASS = "abstract_class"
    PROTOCOL = "protocol"
    TRAIT = "trait"
    TYPE_ALIAS = "type_alias"
    SCHEMA = "schema"
    DATACLASS = "dataclass"
    STRUCT = "struct"
    ENUM = "enum"
    API_ENDPOINT = "api_endpoint"
    UNKNOWN = "unknown"


@dataclass
class Contract:
    """Represents a code contract."""

    contract_id: str
    chunk_id: str
    name: str
    contract_type: ContractType
    file_path: str
    start_line: int
    end_line: int
    language: str
    signature: str | None = None
    dependencies: list[str] = field(default_factory=list)
    implementors: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


class ContractStore:
    """
    Store for code contracts with dependency tracking.

    Features:
    - Contract type classification
    - Dependency graph tracking
    - Implementor tracking
    - Efficient lookup by type and name
    """

    # Additional SQL schema for contracts
    SCHEMA = """
    -- Contracts table
    CREATE TABLE IF NOT EXISTS contracts (
        contract_id TEXT PRIMARY KEY,
        chunk_id TEXT NOT NULL,
        name TEXT NOT NULL,
        contract_type TEXT NOT NULL,
        file_path TEXT NOT NULL,
        start_line INTEGER NOT NULL,
        end_line INTEGER NOT NULL,
        language TEXT NOT NULL,
        signature TEXT,
        created_at TEXT NOT NULL,
        metadata TEXT DEFAULT '{}',
        FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id)
    );

    -- Contract dependencies (what this contract depends on)
    CREATE TABLE IF NOT EXISTS contract_dependencies (
        contract_id TEXT NOT NULL,
        dependency_name TEXT NOT NULL,
        PRIMARY KEY (contract_id, dependency_name),
        FOREIGN KEY (contract_id) REFERENCES contracts(contract_id)
    );

    -- Contract implementors (what implements this contract)
    CREATE TABLE IF NOT EXISTS contract_implementors (
        contract_id TEXT NOT NULL,
        implementor_chunk_id TEXT NOT NULL,
        PRIMARY KEY (contract_id, implementor_chunk_id),
        FOREIGN KEY (contract_id) REFERENCES contracts(contract_id),
        FOREIGN KEY (implementor_chunk_id) REFERENCES chunks(chunk_id)
    );

    -- Indexes
    CREATE INDEX IF NOT EXISTS idx_contracts_chunk_id ON contracts(chunk_id);
    CREATE INDEX IF NOT EXISTS idx_contracts_name ON contracts(name);
    CREATE INDEX IF NOT EXISTS idx_contracts_type ON contracts(contract_type);
    CREATE INDEX IF NOT EXISTS idx_contracts_file_path ON contracts(file_path);
    """

    def __init__(
        self,
        config: "Config",
        sqlite_store: "SQLiteStore",
    ) -> None:
        """
        Initialize contract store.

        Args:
            config: ICD configuration.
            sqlite_store: SQLite store for persistence.
        """
        self.config = config
        self.sqlite_store = sqlite_store
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the contract store schema."""
        logger.info("Initializing contract store")

        if self.sqlite_store._db:
            await self.sqlite_store._db.executescript(self.SCHEMA)

        logger.info("Contract store initialized")

    async def store_contract(
        self,
        contract: Contract,
    ) -> str:
        """
        Store a contract.

        Args:
            contract: Contract to store.

        Returns:
            Contract ID.
        """
        db = self.sqlite_store._db
        if not db:
            raise RuntimeError("Database not initialized")

        import json

        async with self._lock:
            await db.execute(
                """
                INSERT OR REPLACE INTO contracts (
                    contract_id, chunk_id, name, contract_type,
                    file_path, start_line, end_line, language,
                    signature, created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    contract.contract_id,
                    contract.chunk_id,
                    contract.name,
                    contract.contract_type.value,
                    contract.file_path,
                    contract.start_line,
                    contract.end_line,
                    contract.language,
                    contract.signature,
                    contract.created_at.isoformat(),
                    json.dumps(contract.metadata),
                ),
            )

            # Store dependencies
            for dep in contract.dependencies:
                await db.execute(
                    """
                    INSERT OR IGNORE INTO contract_dependencies
                    (contract_id, dependency_name) VALUES (?, ?)
                    """,
                    (contract.contract_id, dep),
                )

            # Store implementors
            for impl in contract.implementors:
                await db.execute(
                    """
                    INSERT OR IGNORE INTO contract_implementors
                    (contract_id, implementor_chunk_id) VALUES (?, ?)
                    """,
                    (contract.contract_id, impl),
                )

        return contract.contract_id

    async def get_contract(self, contract_id: str) -> Contract | None:
        """
        Get a contract by ID.

        Args:
            contract_id: Contract identifier.

        Returns:
            Contract or None if not found.
        """
        db = self.sqlite_store._db
        if not db:
            raise RuntimeError("Database not initialized")

        import json

        async with db.execute(
            "SELECT * FROM contracts WHERE contract_id = ?",
            (contract_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None

            columns = [d[0] for d in cursor.description]
            data = dict(zip(columns, row))

            # Get dependencies
            async with db.execute(
                "SELECT dependency_name FROM contract_dependencies WHERE contract_id = ?",
                (contract_id,),
            ) as dep_cursor:
                deps = [r[0] for r in await dep_cursor.fetchall()]

            # Get implementors
            async with db.execute(
                "SELECT implementor_chunk_id FROM contract_implementors WHERE contract_id = ?",
                (contract_id,),
            ) as impl_cursor:
                impls = [r[0] for r in await impl_cursor.fetchall()]

            return Contract(
                contract_id=data["contract_id"],
                chunk_id=data["chunk_id"],
                name=data["name"],
                contract_type=ContractType(data["contract_type"]),
                file_path=data["file_path"],
                start_line=data["start_line"],
                end_line=data["end_line"],
                language=data["language"],
                signature=data.get("signature"),
                dependencies=deps,
                implementors=impls,
                created_at=datetime.fromisoformat(data["created_at"]),
                metadata=json.loads(data.get("metadata", "{}")),
            )

    async def get_contract_by_chunk(self, chunk_id: str) -> Contract | None:
        """
        Get a contract by its chunk ID.

        Args:
            chunk_id: Chunk identifier.

        Returns:
            Contract or None if not found.
        """
        db = self.sqlite_store._db
        if not db:
            raise RuntimeError("Database not initialized")

        async with db.execute(
            "SELECT contract_id FROM contracts WHERE chunk_id = ?",
            (chunk_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return await self.get_contract(row[0])
            return None

    async def get_contracts_by_type(
        self,
        contract_type: ContractType,
        limit: int = 100,
    ) -> list[Contract]:
        """
        Get contracts by type.

        Args:
            contract_type: Type of contracts to retrieve.
            limit: Maximum number of results.

        Returns:
            List of contracts.
        """
        db = self.sqlite_store._db
        if not db:
            raise RuntimeError("Database not initialized")

        async with db.execute(
            "SELECT contract_id FROM contracts WHERE contract_type = ? LIMIT ?",
            (contract_type.value, limit),
        ) as cursor:
            rows = await cursor.fetchall()

        contracts = []
        for row in rows:
            contract = await self.get_contract(row[0])
            if contract:
                contracts.append(contract)

        return contracts

    async def get_contracts_by_name(
        self,
        name: str,
        fuzzy: bool = False,
    ) -> list[Contract]:
        """
        Get contracts by name.

        Args:
            name: Contract name to search.
            fuzzy: Enable fuzzy matching.

        Returns:
            List of matching contracts.
        """
        db = self.sqlite_store._db
        if not db:
            raise RuntimeError("Database not initialized")

        if fuzzy:
            pattern = f"%{name}%"
            sql = "SELECT contract_id FROM contracts WHERE name LIKE ?"
        else:
            pattern = name
            sql = "SELECT contract_id FROM contracts WHERE name = ?"

        async with db.execute(sql, (pattern,)) as cursor:
            rows = await cursor.fetchall()

        contracts = []
        for row in rows:
            contract = await self.get_contract(row[0])
            if contract:
                contracts.append(contract)

        return contracts

    async def get_contracts_by_file(self, file_path: str) -> list[Contract]:
        """
        Get all contracts in a file.

        Args:
            file_path: File path.

        Returns:
            List of contracts.
        """
        db = self.sqlite_store._db
        if not db:
            raise RuntimeError("Database not initialized")

        async with db.execute(
            "SELECT contract_id FROM contracts WHERE file_path = ?",
            (file_path,),
        ) as cursor:
            rows = await cursor.fetchall()

        contracts = []
        for row in rows:
            contract = await self.get_contract(row[0])
            if contract:
                contracts.append(contract)

        return contracts

    async def get_all_contract_chunk_ids(self) -> list[str]:
        """Get all chunk IDs that are contracts."""
        db = self.sqlite_store._db
        if not db:
            raise RuntimeError("Database not initialized")

        async with db.execute("SELECT chunk_id FROM contracts") as cursor:
            rows = await cursor.fetchall()
            return [row[0] for row in rows]

    async def add_implementor(
        self,
        contract_id: str,
        implementor_chunk_id: str,
    ) -> None:
        """
        Add an implementor to a contract.

        Args:
            contract_id: Contract identifier.
            implementor_chunk_id: Chunk ID of the implementor.
        """
        db = self.sqlite_store._db
        if not db:
            raise RuntimeError("Database not initialized")

        await db.execute(
            """
            INSERT OR IGNORE INTO contract_implementors
            (contract_id, implementor_chunk_id) VALUES (?, ?)
            """,
            (contract_id, implementor_chunk_id),
        )

    async def get_implementors(self, contract_id: str) -> list[str]:
        """
        Get all implementor chunk IDs for a contract.

        Args:
            contract_id: Contract identifier.

        Returns:
            List of implementor chunk IDs.
        """
        db = self.sqlite_store._db
        if not db:
            raise RuntimeError("Database not initialized")

        async with db.execute(
            "SELECT implementor_chunk_id FROM contract_implementors WHERE contract_id = ?",
            (contract_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [row[0] for row in rows]

    async def get_dependencies(self, contract_id: str) -> list[str]:
        """
        Get all dependencies for a contract.

        Args:
            contract_id: Contract identifier.

        Returns:
            List of dependency names.
        """
        db = self.sqlite_store._db
        if not db:
            raise RuntimeError("Database not initialized")

        async with db.execute(
            "SELECT dependency_name FROM contract_dependencies WHERE contract_id = ?",
            (contract_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [row[0] for row in rows]

    async def delete_contract(self, contract_id: str) -> bool:
        """
        Delete a contract.

        Args:
            contract_id: Contract identifier.

        Returns:
            True if deleted.
        """
        db = self.sqlite_store._db
        if not db:
            raise RuntimeError("Database not initialized")

        async with self._lock:
            # Delete related records first
            await db.execute(
                "DELETE FROM contract_dependencies WHERE contract_id = ?",
                (contract_id,),
            )
            await db.execute(
                "DELETE FROM contract_implementors WHERE contract_id = ?",
                (contract_id,),
            )

            # Delete contract
            cursor = await db.execute(
                "DELETE FROM contracts WHERE contract_id = ?",
                (contract_id,),
            )

            return cursor.rowcount > 0

    async def delete_contracts_by_file(self, file_path: str) -> int:
        """
        Delete all contracts in a file.

        Args:
            file_path: File path.

        Returns:
            Number of deleted contracts.
        """
        db = self.sqlite_store._db
        if not db:
            raise RuntimeError("Database not initialized")

        # Get contract IDs first
        async with db.execute(
            "SELECT contract_id FROM contracts WHERE file_path = ?",
            (file_path,),
        ) as cursor:
            rows = await cursor.fetchall()

        count = 0
        for row in rows:
            if await self.delete_contract(row[0]):
                count += 1

        return count

    async def find_related_contracts(
        self,
        chunk_id: str,
        max_depth: int = 2,
    ) -> list[Contract]:
        """
        Find contracts related to a chunk through dependencies and implementations.

        Args:
            chunk_id: Chunk identifier.
            max_depth: Maximum traversal depth.

        Returns:
            List of related contracts.
        """
        db = self.sqlite_store._db
        if not db:
            raise RuntimeError("Database not initialized")

        visited: set[str] = set()
        to_visit: list[tuple[str, int]] = [(chunk_id, 0)]
        related: list[Contract] = []

        while to_visit:
            current_id, depth = to_visit.pop(0)

            if current_id in visited or depth > max_depth:
                continue

            visited.add(current_id)

            # Check if this chunk is a contract
            contract = await self.get_contract_by_chunk(current_id)
            if contract:
                related.append(contract)

                # Add implementors to visit
                for impl in contract.implementors:
                    if impl not in visited:
                        to_visit.append((impl, depth + 1))

            # Check if this chunk implements any contracts
            async with db.execute(
                "SELECT contract_id FROM contract_implementors WHERE implementor_chunk_id = ?",
                (current_id,),
            ) as cursor:
                rows = await cursor.fetchall()
                for row in rows:
                    impl_contract = await self.get_contract(row[0])
                    if impl_contract and impl_contract.chunk_id not in visited:
                        to_visit.append((impl_contract.chunk_id, depth + 1))

        return related

    async def get_stats(self) -> dict[str, Any]:
        """Get contract store statistics."""
        db = self.sqlite_store._db
        if not db:
            raise RuntimeError("Database not initialized")

        stats: dict[str, Any] = {}

        async with db.execute("SELECT COUNT(*) FROM contracts") as cursor:
            row = await cursor.fetchone()
            stats["total_contracts"] = row[0] if row else 0

        async with db.execute(
            "SELECT contract_type, COUNT(*) FROM contracts GROUP BY contract_type"
        ) as cursor:
            rows = await cursor.fetchall()
            stats["by_type"] = {row[0]: row[1] for row in rows}

        async with db.execute(
            "SELECT COUNT(*) FROM contract_dependencies"
        ) as cursor:
            row = await cursor.fetchone()
            stats["total_dependencies"] = row[0] if row else 0

        async with db.execute(
            "SELECT COUNT(*) FROM contract_implementors"
        ) as cursor:
            row = await cursor.fetchone()
            stats["total_implementors"] = row[0] if row else 0

        return stats
