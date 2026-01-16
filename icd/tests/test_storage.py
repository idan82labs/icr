"""
Tests for storage modules.
"""

from __future__ import annotations

import numpy as np
import pytest

from icd.config import Config


class TestSQLiteStore:
    """Tests for SQLiteStore."""

    @pytest.mark.asyncio
    async def test_initialize(self, sqlite_store):
        """Test store initialization."""
        stats = await sqlite_store.get_stats()
        assert stats["total_chunks"] == 0
        assert stats["total_files"] == 0

    @pytest.mark.asyncio
    async def test_store_and_retrieve_chunk(self, sqlite_store):
        """Test storing and retrieving a chunk."""
        chunk_id = await sqlite_store.store_chunk(
            chunk_id="test_chunk_1",
            file_path="/path/to/file.py",
            content="def hello(): pass",
            start_line=1,
            end_line=1,
            start_byte=0,
            end_byte=18,
            symbol_name="hello",
            symbol_type="function_definition",
            language="python",
            token_count=5,
        )

        assert chunk_id == "test_chunk_1"

        metadata = await sqlite_store.get_chunk(chunk_id)
        assert metadata is not None
        assert metadata.file_path == "/path/to/file.py"
        assert metadata.symbol_name == "hello"
        assert metadata.language == "python"

        content = await sqlite_store.get_chunk_content(chunk_id)
        assert content == "def hello(): pass"

    @pytest.mark.asyncio
    async def test_batch_store(self, sqlite_store):
        """Test batch chunk storage."""
        chunks = [
            {
                "chunk_id": f"batch_chunk_{i}",
                "file_path": "/path/to/file.py",
                "content": f"def func_{i}(): pass",
                "start_line": i,
                "end_line": i,
                "start_byte": 0,
                "end_byte": 20,
                "symbol_name": f"func_{i}",
                "symbol_type": "function_definition",
                "language": "python",
                "token_count": 5,
            }
            for i in range(10)
        ]

        chunk_ids = await sqlite_store.store_chunks_batch(chunks)
        assert len(chunk_ids) == 10

        stats = await sqlite_store.get_stats()
        assert stats["total_chunks"] == 10

    @pytest.mark.asyncio
    async def test_bm25_search(self, sqlite_store):
        """Test BM25 search."""
        # Store some chunks
        await sqlite_store.store_chunk(
            chunk_id="search_1",
            file_path="/path/to/users.py",
            content="def get_user_by_id(user_id): return users.get(user_id)",
            start_line=1,
            end_line=1,
            start_byte=0,
            end_byte=50,
            symbol_name="get_user_by_id",
            symbol_type="function_definition",
            language="python",
            token_count=10,
        )

        await sqlite_store.store_chunk(
            chunk_id="search_2",
            file_path="/path/to/products.py",
            content="def get_product_by_id(product_id): return products.get(product_id)",
            start_line=1,
            end_line=1,
            start_byte=0,
            end_byte=60,
            symbol_name="get_product_by_id",
            symbol_type="function_definition",
            language="python",
            token_count=10,
        )

        # Search for user-related content
        results = await sqlite_store.search_bm25("user", limit=10)
        assert len(results) >= 1
        assert results[0].chunk_id == "search_1"

    @pytest.mark.asyncio
    async def test_file_tracking(self, sqlite_store):
        """Test file tracking."""
        from datetime import datetime

        await sqlite_store.track_file(
            file_path="/path/to/tracked.py",
            content_hash="abc123",
            size_bytes=1000,
            modified_at=datetime.utcnow(),
            chunk_count=5,
            language="python",
        )

        record = await sqlite_store.get_file_record("/path/to/tracked.py")
        assert record is not None
        assert record.content_hash == "abc123"
        assert record.chunk_count == 5

    @pytest.mark.asyncio
    async def test_pinned_status(self, sqlite_store):
        """Test pinned chunk status."""
        await sqlite_store.store_chunk(
            chunk_id="pin_test",
            file_path="/path/to/file.py",
            content="important code",
            start_line=1,
            end_line=1,
            start_byte=0,
            end_byte=14,
            symbol_name=None,
            symbol_type=None,
            language="python",
            token_count=2,
        )

        # Initially not pinned
        metadata = await sqlite_store.get_chunk("pin_test")
        assert not metadata.is_pinned

        # Pin the chunk
        result = await sqlite_store.set_pinned("pin_test", True)
        assert result is True

        # Verify pinned
        metadata = await sqlite_store.get_chunk("pin_test")
        assert metadata.is_pinned


class TestVectorStore:
    """Tests for VectorStore."""

    @pytest.mark.asyncio
    async def test_initialize(self, vector_store):
        """Test store initialization."""
        stats = await vector_store.get_stats()
        assert stats["num_vectors"] == 0
        assert stats["dimension"] == 384

    @pytest.mark.asyncio
    async def test_add_and_search(self, vector_store, sample_embeddings):
        """Test adding vectors and searching."""
        # Add vectors
        for i, embedding in enumerate(sample_embeddings):
            await vector_store.add_vector(f"chunk_{i}", embedding)

        stats = await vector_store.get_stats()
        assert stats["num_vectors"] == 10

        # Search
        query_vector = sample_embeddings[0]
        results = await vector_store.search(query_vector, k=5)

        assert len(results) == 5
        assert results[0].chunk_id == "chunk_0"  # Should be exact match
        assert results[0].score > 0.99  # High similarity for exact match

    @pytest.mark.asyncio
    async def test_batch_add(self, vector_store):
        """Test batch vector addition."""
        np.random.seed(42)
        vectors = np.random.randn(20, 384).astype(np.float32)
        chunk_ids = [f"batch_{i}" for i in range(20)]

        internal_ids = await vector_store.add_vectors_batch(chunk_ids, vectors)
        assert len(internal_ids) == 20

        stats = await vector_store.get_stats()
        assert stats["num_vectors"] == 20

    @pytest.mark.asyncio
    async def test_float16_storage(self, vector_store, sample_embeddings):
        """Test float16 storage and float32 retrieval."""
        await vector_store.add_vector("f16_test", sample_embeddings[0])

        # Retrieve vector
        retrieved = await vector_store.get_vector("f16_test")
        assert retrieved is not None
        assert retrieved.dtype == np.float32

        # Should be close to original (some precision loss from float16)
        np.testing.assert_array_almost_equal(
            retrieved,
            sample_embeddings[0],
            decimal=2,  # float16 precision
        )

    @pytest.mark.asyncio
    async def test_delete_vector(self, vector_store, sample_embeddings):
        """Test vector deletion."""
        await vector_store.add_vector("delete_test", sample_embeddings[0])

        assert await vector_store.contains("delete_test")

        result = await vector_store.delete_vector("delete_test")
        assert result is True

    @pytest.mark.asyncio
    async def test_similarity_computation(self, vector_store):
        """Test similarity computation between stored vectors."""
        np.random.seed(42)

        # Create two similar vectors
        base = np.random.randn(384).astype(np.float32)
        similar = base + np.random.randn(384).astype(np.float32) * 0.1
        different = np.random.randn(384).astype(np.float32)

        await vector_store.add_vector("base", base)
        await vector_store.add_vector("similar", similar)
        await vector_store.add_vector("different", different)

        sim_to_similar = await vector_store.compute_similarity("base", "similar")
        sim_to_different = await vector_store.compute_similarity("base", "different")

        assert sim_to_similar > sim_to_different


class TestContractStore:
    """Tests for ContractStore."""

    @pytest.mark.asyncio
    async def test_store_and_retrieve_contract(self, test_config, sqlite_store):
        """Test storing and retrieving contracts."""
        from icd.storage.contract_store import Contract, ContractStore, ContractType

        contract_store = ContractStore(test_config, sqlite_store)
        await contract_store.initialize()

        contract = Contract(
            contract_id="contract_1",
            chunk_id="chunk_1",
            name="UserProtocol",
            contract_type=ContractType.PROTOCOL,
            file_path="/path/to/user.py",
            start_line=1,
            end_line=10,
            language="python",
            signature="class UserProtocol(Protocol):",
            dependencies=["Protocol"],
        )

        contract_id = await contract_store.store_contract(contract)
        assert contract_id == "contract_1"

        retrieved = await contract_store.get_contract(contract_id)
        assert retrieved is not None
        assert retrieved.name == "UserProtocol"
        assert retrieved.contract_type == ContractType.PROTOCOL

    @pytest.mark.asyncio
    async def test_get_contracts_by_type(self, test_config, sqlite_store):
        """Test retrieving contracts by type."""
        from icd.storage.contract_store import Contract, ContractStore, ContractType

        contract_store = ContractStore(test_config, sqlite_store)
        await contract_store.initialize()

        # Store contracts of different types
        for i, ctype in enumerate([ContractType.INTERFACE, ContractType.PROTOCOL, ContractType.INTERFACE]):
            contract = Contract(
                contract_id=f"type_test_{i}",
                chunk_id=f"chunk_{i}",
                name=f"Contract{i}",
                contract_type=ctype,
                file_path=f"/path/to/file_{i}.py",
                start_line=1,
                end_line=10,
                language="python",
            )
            await contract_store.store_contract(contract)

        interfaces = await contract_store.get_contracts_by_type(ContractType.INTERFACE)
        assert len(interfaces) == 2

        protocols = await contract_store.get_contracts_by_type(ContractType.PROTOCOL)
        assert len(protocols) == 1


class TestMemoryStore:
    """Tests for MemoryStore."""

    @pytest.mark.asyncio
    async def test_pin_and_unpin(self, test_config, sqlite_store):
        """Test pinning and unpinning chunks."""
        from icd.storage.memory_store import MemoryStore

        # First store a chunk
        await sqlite_store.store_chunk(
            chunk_id="pin_memory_test",
            file_path="/path/to/file.py",
            content="test content",
            start_line=1,
            end_line=1,
            start_byte=0,
            end_byte=12,
            symbol_name=None,
            symbol_type=None,
            language="python",
            token_count=2,
        )

        memory_store = MemoryStore(test_config, sqlite_store)
        await memory_store.initialize()

        # Pin chunk
        result = await memory_store.pin_chunk(
            "pin_memory_test",
            reason="Important for testing",
        )
        assert result is True

        # Check pinned
        pinned = await memory_store.get_pinned_chunks()
        assert "pin_memory_test" in pinned

        # Unpin
        result = await memory_store.unpin_chunk("pin_memory_test")
        assert result is True

        pinned = await memory_store.get_pinned_chunks()
        assert "pin_memory_test" not in pinned

    @pytest.mark.asyncio
    async def test_ledger_entries(self, test_config, sqlite_store):
        """Test memory ledger entries."""
        from icd.storage.memory_store import LedgerCategory, MemoryStore

        memory_store = MemoryStore(test_config, sqlite_store)
        await memory_store.initialize()

        # Add ledger entry
        entry_id = await memory_store.add_ledger_entry(
            content="Important architectural decision: use microservices",
            category=LedgerCategory.DECISION,
        )
        assert entry_id is not None

        # Retrieve entry
        entry = await memory_store.get_ledger_entry(entry_id)
        assert entry is not None
        assert "microservices" in entry.content
        assert entry.category == LedgerCategory.DECISION

    @pytest.mark.asyncio
    async def test_session_management(self, test_config, sqlite_store):
        """Test session state management."""
        from icd.storage.memory_store import MemoryStore

        memory_store = MemoryStore(test_config, sqlite_store)
        await memory_store.initialize()

        # Create session
        session_id = await memory_store.create_session()
        assert session_id is not None

        # Update session
        result = await memory_store.update_session(
            session_id,
            query="how does authentication work",
            retrieved_chunks=["chunk_1", "chunk_2"],
        )
        assert result is True

        # Get session
        session = await memory_store.get_session(session_id)
        assert session is not None
        assert len(session.query_history) == 1
        assert len(session.retrieved_chunks) == 2
