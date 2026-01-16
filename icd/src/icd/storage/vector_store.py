"""
HNSW Vector Store with float16 storage and float32 compute.

Provides efficient approximate nearest neighbor search using hnswlib.
Stores vectors in float16 to reduce memory by 50%, converts to float32 for search.
"""

from __future__ import annotations

import asyncio
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hnswlib
import numpy as np
import structlog

if TYPE_CHECKING:
    from icd.config import Config

logger = structlog.get_logger(__name__)


@dataclass
class VectorSearchResult:
    """Result from vector search."""

    chunk_id: str
    distance: float
    score: float  # Converted to similarity


class VectorStore:
    """
    HNSW-based vector store with float16 storage.

    Features:
    - Float16 storage for 50% memory reduction
    - Float32 compute for accuracy
    - Incremental index updates
    - Persistent storage with metadata
    - Efficient batch operations
    """

    def __init__(self, config: "Config") -> None:
        """
        Initialize vector store.

        Args:
            config: ICD configuration.
        """
        self.config = config
        self.index_path = config.vector_index_path
        self.metadata_path = self.index_path.with_suffix(".meta")

        self.dimension = config.embedding.dimension
        self.max_elements = config.storage.max_vectors_per_repo
        self.use_float16 = config.storage.vector_dtype.value == "float16"

        # HNSW parameters
        self.m = config.storage.hnsw_m
        self.ef_construction = config.storage.hnsw_ef_construction
        self.ef_search = config.storage.hnsw_ef_search

        self._index: hnswlib.Index | None = None
        self._id_to_chunk: dict[int, str] = {}
        self._chunk_to_id: dict[str, int] = {}
        self._vectors_f16: dict[int, np.ndarray] = {}  # Float16 storage
        self._next_id = 0
        self._lock = asyncio.Lock()
        self._dirty = False

    async def initialize(self) -> None:
        """Initialize or load the vector index."""
        logger.info(
            "Initializing vector store",
            index_path=str(self.index_path),
            dimension=self.dimension,
            max_elements=self.max_elements,
        )

        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        # Try to load existing index
        if self.index_path.exists() and self.metadata_path.exists():
            await self._load_index()
        else:
            await self._create_index()

        logger.info(
            "Vector store initialized",
            num_vectors=len(self._id_to_chunk),
        )

    async def _create_index(self) -> None:
        """Create a new HNSW index."""
        self._index = hnswlib.Index(space="cosine", dim=self.dimension)
        self._index.init_index(
            max_elements=self.max_elements,
            ef_construction=self.ef_construction,
            M=self.m,
        )
        self._index.set_ef(self.ef_search)

        self._id_to_chunk = {}
        self._chunk_to_id = {}
        self._vectors_f16 = {}
        self._next_id = 0

        logger.info("Created new HNSW index")

    async def _load_index(self) -> None:
        """Load existing index from disk."""
        try:
            # Load HNSW index
            self._index = hnswlib.Index(space="cosine", dim=self.dimension)
            self._index.load_index(str(self.index_path), max_elements=self.max_elements)
            self._index.set_ef(self.ef_search)

            # Load metadata
            with open(self.metadata_path, "rb") as f:
                metadata = pickle.load(f)
                self._id_to_chunk = metadata["id_to_chunk"]
                self._chunk_to_id = metadata["chunk_to_id"]
                self._vectors_f16 = metadata.get("vectors_f16", {})
                self._next_id = metadata["next_id"]

            logger.info(
                "Loaded existing index",
                num_vectors=len(self._id_to_chunk),
            )
        except Exception as e:
            logger.warning("Failed to load index, creating new", error=str(e))
            await self._create_index()

    async def save(self) -> None:
        """Save index and metadata to disk."""
        if not self._dirty or not self._index:
            return

        async with self._lock:
            logger.info("Saving vector index")

            # Save HNSW index
            self._index.save_index(str(self.index_path))

            # Save metadata
            metadata = {
                "id_to_chunk": self._id_to_chunk,
                "chunk_to_id": self._chunk_to_id,
                "vectors_f16": self._vectors_f16,
                "next_id": self._next_id,
            }
            with open(self.metadata_path, "wb") as f:
                pickle.dump(metadata, f)

            self._dirty = False
            logger.info("Vector index saved")

    async def close(self) -> None:
        """Close the store and save to disk."""
        await self.save()
        self._index = None

    async def add_vector(
        self,
        chunk_id: str,
        vector: np.ndarray,
    ) -> int:
        """
        Add a vector to the index.

        Args:
            chunk_id: Unique chunk identifier.
            vector: Embedding vector (will be normalized if needed).

        Returns:
            Internal vector ID.
        """
        if not self._index:
            raise RuntimeError("Index not initialized")

        async with self._lock:
            # Check if chunk already exists
            if chunk_id in self._chunk_to_id:
                # Update existing vector
                internal_id = self._chunk_to_id[chunk_id]
                return await self._update_vector(internal_id, vector)

            # Ensure float32 for indexing
            vector_f32 = vector.astype(np.float32)

            # Normalize if needed
            norm = np.linalg.norm(vector_f32)
            if norm > 0:
                vector_f32 = vector_f32 / norm

            # Assign new ID
            internal_id = self._next_id
            self._next_id += 1

            # Add to HNSW index
            self._index.add_items(
                vector_f32.reshape(1, -1),
                np.array([internal_id]),
            )

            # Store float16 copy for later retrieval
            if self.use_float16:
                self._vectors_f16[internal_id] = vector_f32.astype(np.float16)
            else:
                self._vectors_f16[internal_id] = vector_f32

            # Update mappings
            self._id_to_chunk[internal_id] = chunk_id
            self._chunk_to_id[chunk_id] = internal_id

            self._dirty = True

            return internal_id

    async def _update_vector(
        self,
        internal_id: int,
        vector: np.ndarray,
    ) -> int:
        """Update an existing vector."""
        # Ensure float32 for indexing
        vector_f32 = vector.astype(np.float32)

        # Normalize if needed
        norm = np.linalg.norm(vector_f32)
        if norm > 0:
            vector_f32 = vector_f32 / norm

        # hnswlib doesn't support direct updates, so we mark and re-add
        # The old vector will be overwritten
        self._index.mark_deleted(internal_id)
        self._index.add_items(
            vector_f32.reshape(1, -1),
            np.array([internal_id]),
        )

        # Update stored float16 vector
        if self.use_float16:
            self._vectors_f16[internal_id] = vector_f32.astype(np.float16)
        else:
            self._vectors_f16[internal_id] = vector_f32

        self._dirty = True

        return internal_id

    async def add_vectors_batch(
        self,
        chunk_ids: list[str],
        vectors: np.ndarray,
    ) -> list[int]:
        """
        Add multiple vectors in batch.

        Args:
            chunk_ids: List of chunk identifiers.
            vectors: Array of vectors (n x dimension).

        Returns:
            List of internal vector IDs.
        """
        if not self._index:
            raise RuntimeError("Index not initialized")

        if len(chunk_ids) != vectors.shape[0]:
            raise ValueError("Number of chunk_ids must match number of vectors")

        async with self._lock:
            # Ensure float32
            vectors_f32 = vectors.astype(np.float32)

            # Normalize
            norms = np.linalg.norm(vectors_f32, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            vectors_f32 = vectors_f32 / norms

            internal_ids = []
            new_vectors = []
            new_ids = []

            for i, chunk_id in enumerate(chunk_ids):
                if chunk_id in self._chunk_to_id:
                    # Update existing
                    internal_id = self._chunk_to_id[chunk_id]
                    self._index.mark_deleted(internal_id)
                else:
                    internal_id = self._next_id
                    self._next_id += 1
                    self._id_to_chunk[internal_id] = chunk_id
                    self._chunk_to_id[chunk_id] = internal_id

                internal_ids.append(internal_id)
                new_vectors.append(vectors_f32[i])
                new_ids.append(internal_id)

                # Store float16
                if self.use_float16:
                    self._vectors_f16[internal_id] = vectors_f32[i].astype(np.float16)
                else:
                    self._vectors_f16[internal_id] = vectors_f32[i]

            # Batch add to index
            if new_vectors:
                self._index.add_items(
                    np.array(new_vectors),
                    np.array(new_ids),
                )

            self._dirty = True

            return internal_ids

    async def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filter_ids: set[str] | None = None,
    ) -> list[VectorSearchResult]:
        """
        Search for nearest neighbors.

        Args:
            query_vector: Query embedding vector.
            k: Number of results to return.
            filter_ids: Optional set of chunk IDs to include.

        Returns:
            List of VectorSearchResult sorted by similarity.
        """
        if not self._index:
            raise RuntimeError("Index not initialized")

        if len(self._id_to_chunk) == 0:
            return []

        # Ensure float32 for query
        query_f32 = query_vector.astype(np.float32)

        # Normalize
        norm = np.linalg.norm(query_f32)
        if norm > 0:
            query_f32 = query_f32 / norm

        # Search with extra candidates if filtering
        search_k = min(k * 3 if filter_ids else k, len(self._id_to_chunk))

        try:
            labels, distances = self._index.knn_query(
                query_f32.reshape(1, -1),
                k=search_k,
            )
        except Exception as e:
            logger.warning("Vector search failed", error=str(e))
            return []

        results = []
        for label, distance in zip(labels[0], distances[0]):
            if label not in self._id_to_chunk:
                continue

            chunk_id = self._id_to_chunk[label]

            if filter_ids and chunk_id not in filter_ids:
                continue

            # Convert cosine distance to similarity score
            # hnswlib returns distance, not similarity
            similarity = 1.0 - distance

            results.append(
                VectorSearchResult(
                    chunk_id=chunk_id,
                    distance=float(distance),
                    score=float(similarity),
                )
            )

            if len(results) >= k:
                break

        return results

    async def get_vector(self, chunk_id: str) -> np.ndarray | None:
        """
        Get a stored vector by chunk ID.

        Converts from float16 storage to float32 for computation.

        Args:
            chunk_id: Chunk identifier.

        Returns:
            Vector as float32 numpy array, or None if not found.
        """
        if chunk_id not in self._chunk_to_id:
            return None

        internal_id = self._chunk_to_id[chunk_id]
        if internal_id not in self._vectors_f16:
            return None

        # Convert float16 to float32 for computation
        return self._vectors_f16[internal_id].astype(np.float32)

    async def get_vectors_batch(
        self,
        chunk_ids: list[str],
    ) -> dict[str, np.ndarray]:
        """
        Get multiple vectors by chunk IDs.

        Args:
            chunk_ids: List of chunk identifiers.

        Returns:
            Dictionary mapping chunk_id to vector (float32).
        """
        results = {}
        for chunk_id in chunk_ids:
            vector = await self.get_vector(chunk_id)
            if vector is not None:
                results[chunk_id] = vector
        return results

    async def delete_vector(self, chunk_id: str) -> bool:
        """
        Delete a vector from the index.

        Args:
            chunk_id: Chunk identifier.

        Returns:
            True if deleted, False if not found.
        """
        if not self._index:
            raise RuntimeError("Index not initialized")

        if chunk_id not in self._chunk_to_id:
            return False

        async with self._lock:
            internal_id = self._chunk_to_id[chunk_id]

            # Mark as deleted in HNSW index
            self._index.mark_deleted(internal_id)

            # Remove from mappings
            del self._chunk_to_id[chunk_id]
            del self._id_to_chunk[internal_id]
            if internal_id in self._vectors_f16:
                del self._vectors_f16[internal_id]

            self._dirty = True

            return True

    async def delete_vectors_batch(self, chunk_ids: list[str]) -> int:
        """
        Delete multiple vectors.

        Args:
            chunk_ids: List of chunk identifiers.

        Returns:
            Number of deleted vectors.
        """
        count = 0
        for chunk_id in chunk_ids:
            if await self.delete_vector(chunk_id):
                count += 1
        return count

    async def contains(self, chunk_id: str) -> bool:
        """Check if a chunk ID exists in the index."""
        return chunk_id in self._chunk_to_id

    async def get_stats(self) -> dict[str, Any]:
        """Get index statistics."""
        stats = {
            "num_vectors": len(self._id_to_chunk),
            "max_elements": self.max_elements,
            "dimension": self.dimension,
            "use_float16": self.use_float16,
            "hnsw_m": self.m,
            "ef_search": self.ef_search,
        }

        if self._index:
            stats["index_size_bytes"] = self._index.element_count * self.dimension * (
                2 if self.use_float16 else 4
            )

        return stats

    async def compute_similarity(
        self,
        chunk_id_a: str,
        chunk_id_b: str,
    ) -> float | None:
        """
        Compute cosine similarity between two stored vectors.

        Args:
            chunk_id_a: First chunk ID.
            chunk_id_b: Second chunk ID.

        Returns:
            Cosine similarity or None if either vector not found.
        """
        vec_a = await self.get_vector(chunk_id_a)
        vec_b = await self.get_vector(chunk_id_b)

        if vec_a is None or vec_b is None:
            return None

        # Compute cosine similarity
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    async def compute_similarities_to_set(
        self,
        query_chunk_id: str,
        target_chunk_ids: list[str],
    ) -> dict[str, float]:
        """
        Compute similarities between one vector and multiple targets.

        Args:
            query_chunk_id: Query chunk ID.
            target_chunk_ids: Target chunk IDs.

        Returns:
            Dictionary mapping target chunk_id to similarity.
        """
        query_vec = await self.get_vector(query_chunk_id)
        if query_vec is None:
            return {}

        results = {}
        for target_id in target_chunk_ids:
            target_vec = await self.get_vector(target_id)
            if target_vec is not None:
                dot_product = np.dot(query_vec, target_vec)
                norm_q = np.linalg.norm(query_vec)
                norm_t = np.linalg.norm(target_vec)
                if norm_q > 0 and norm_t > 0:
                    results[target_id] = float(dot_product / (norm_q * norm_t))

        return results
