"""
ICD Main Entry Point.

Provides the main ICDService orchestration class and CLI interface.
"""

from __future__ import annotations

import asyncio
import signal
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator

import click
import structlog

from icd.config import Config, load_config

if TYPE_CHECKING:
    from icd.indexing.chunker import Chunk
    from icd.indexing.embedder import EmbeddingBackend
    from icd.indexing.watcher import FileWatcher
    from icd.pack.compiler import PackCompiler
    from icd.retrieval.hybrid import HybridRetriever
    from icd.storage.contract_store import ContractStore
    from icd.storage.memory_store import MemoryStore
    from icd.storage.sqlite_store import SQLiteStore
    from icd.storage.vector_store import VectorStore

logger = structlog.get_logger(__name__)


@dataclass
class RetrievalResult:
    """Result from a retrieval query."""

    chunks: list["Chunk"]
    scores: list[float]
    entropy: float
    query: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PackResult:
    """Result from pack compilation."""

    content: str
    token_count: int
    chunk_ids: list[str]
    citations: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)


class ICDService:
    """
    Main ICD service orchestrating all components.

    This is the primary interface for interacting with the ICD daemon.
    It manages:
    - File watching and indexing
    - Embedding generation
    - Vector and SQLite storage
    - Hybrid retrieval
    - Pack compilation
    """

    def __init__(self, config: Config | None = None) -> None:
        """
        Initialize the ICD service.

        Args:
            config: Configuration instance. Uses default if not provided.
        """
        self.config = config or Config()
        self.config.ensure_directories()

        self._sqlite_store: SQLiteStore | None = None
        self._vector_store: VectorStore | None = None
        self._contract_store: ContractStore | None = None
        self._memory_store: MemoryStore | None = None
        self._embedder: EmbeddingBackend | None = None
        self._watcher: FileWatcher | None = None
        self._retriever: HybridRetriever | None = None
        self._pack_compiler: PackCompiler | None = None

        self._initialized = False
        self._shutdown_event = asyncio.Event()

        logger.info(
            "ICD service created",
            project_root=str(self.config.project_root),
            data_dir=str(self.config.absolute_data_dir),
        )

    async def initialize(self) -> None:
        """Initialize all components."""
        if self._initialized:
            return

        logger.info("Initializing ICD service")

        # Import here to avoid circular imports
        from icd.indexing.embedder import create_embedder
        from icd.indexing.watcher import FileWatcher
        from icd.pack.compiler import PackCompiler
        from icd.retrieval.hybrid import HybridRetriever
        from icd.storage.contract_store import ContractStore
        from icd.storage.memory_store import MemoryStore
        from icd.storage.sqlite_store import SQLiteStore
        from icd.storage.vector_store import VectorStore

        # Initialize storage
        self._sqlite_store = SQLiteStore(self.config)
        await self._sqlite_store.initialize()

        self._vector_store = VectorStore(self.config)
        await self._vector_store.initialize()

        self._contract_store = ContractStore(self.config, self._sqlite_store)
        await self._contract_store.initialize()

        self._memory_store = MemoryStore(self.config, self._sqlite_store)
        await self._memory_store.initialize()

        # Initialize embedder
        self._embedder = create_embedder(self.config)
        await self._embedder.initialize()

        # Initialize retriever
        self._retriever = HybridRetriever(
            config=self.config,
            sqlite_store=self._sqlite_store,
            vector_store=self._vector_store,
            embedder=self._embedder,
            contract_store=self._contract_store,
            memory_store=self._memory_store,
        )

        # Initialize pack compiler
        self._pack_compiler = PackCompiler(self.config)

        # Initialize file watcher
        self._watcher = FileWatcher(
            config=self.config,
            sqlite_store=self._sqlite_store,
            vector_store=self._vector_store,
            embedder=self._embedder,
            contract_store=self._contract_store,
        )

        self._initialized = True
        logger.info("ICD service initialized")

    async def shutdown(self) -> None:
        """Gracefully shutdown all components."""
        logger.info("Shutting down ICD service")
        self._shutdown_event.set()

        if self._watcher:
            await self._watcher.stop()

        if self._vector_store:
            await self._vector_store.close()

        if self._sqlite_store:
            await self._sqlite_store.close()

        self._initialized = False
        logger.info("ICD service shutdown complete")

    @asynccontextmanager
    async def session(self) -> AsyncIterator["ICDService"]:
        """Context manager for service lifecycle."""
        try:
            await self.initialize()
            yield self
        finally:
            await self.shutdown()

    async def start_watching(self) -> None:
        """Start file system watching."""
        if not self._initialized:
            await self.initialize()

        if self._watcher:
            await self._watcher.start()
            logger.info("File watching started")

    async def stop_watching(self) -> None:
        """Stop file system watching."""
        if self._watcher:
            await self._watcher.stop()
            logger.info("File watching stopped")

    async def index_directory(
        self,
        path: Path | None = None,
        force: bool = False,
    ) -> dict[str, int]:
        """
        Index a directory.

        Args:
            path: Directory to index. Uses project_root if not provided.
            force: Force re-indexing of all files.

        Returns:
            Statistics about indexed files.
        """
        if not self._initialized:
            await self.initialize()

        target = path or self.config.project_root
        logger.info("Starting directory indexing", path=str(target), force=force)

        if self._watcher:
            stats = await self._watcher.index_directory(target, force=force)
            logger.info("Directory indexing complete", stats=stats)
            return stats

        return {"files": 0, "chunks": 0, "errors": 0}

    async def retrieve(
        self,
        query: str,
        limit: int | None = None,
        focus_paths: list[Path] | None = None,
        include_contracts: bool = True,
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: Natural language query.
            limit: Maximum results to return.
            focus_paths: Paths to prioritize in retrieval.
            include_contracts: Include contract chunks in results.

        Returns:
            RetrievalResult with chunks and metadata.
        """
        if not self._initialized:
            await self.initialize()

        if not self._retriever:
            raise RuntimeError("Retriever not initialized")

        results = await self._retriever.retrieve(
            query=query,
            limit=limit or self.config.retrieval.final_results,
            focus_paths=focus_paths,
            include_contracts=include_contracts,
        )

        return results

    async def compile_pack(
        self,
        query: str,
        budget_tokens: int | None = None,
        focus_paths: list[Path] | None = None,
    ) -> PackResult:
        """
        Compile a context pack for a query.

        Args:
            query: Natural language query.
            budget_tokens: Token budget for the pack.
            focus_paths: Paths to prioritize.

        Returns:
            PackResult with compiled content.
        """
        if not self._initialized:
            await self.initialize()

        if not self._retriever or not self._pack_compiler:
            raise RuntimeError("Components not initialized")

        # Retrieve relevant chunks
        retrieval_result = await self.retrieve(
            query=query,
            focus_paths=focus_paths,
        )

        # Compile pack
        budget = budget_tokens or self.config.pack.default_budget_tokens
        pack_result = await self._pack_compiler.compile(
            chunks=retrieval_result.chunks,
            scores=retrieval_result.scores,
            budget_tokens=budget,
            query=query,
        )

        return pack_result

    async def pin_chunk(self, chunk_id: str, reason: str | None = None) -> bool:
        """
        Pin a chunk as an invariant.

        Args:
            chunk_id: ID of chunk to pin.
            reason: Optional reason for pinning.

        Returns:
            True if successfully pinned.
        """
        if not self._memory_store:
            raise RuntimeError("Memory store not initialized")

        return await self._memory_store.pin_chunk(chunk_id, reason=reason)

    async def unpin_chunk(self, chunk_id: str) -> bool:
        """
        Unpin a chunk.

        Args:
            chunk_id: ID of chunk to unpin.

        Returns:
            True if successfully unpinned.
        """
        if not self._memory_store:
            raise RuntimeError("Memory store not initialized")

        return await self._memory_store.unpin_chunk(chunk_id)

    async def get_pinned_chunks(self) -> list[str]:
        """Get all pinned chunk IDs."""
        if not self._memory_store:
            raise RuntimeError("Memory store not initialized")

        return await self._memory_store.get_pinned_chunks()

    async def add_ledger_entry(
        self,
        content: str,
        category: str = "general",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Add an entry to the memory ledger.

        Args:
            content: Ledger entry content.
            category: Entry category.
            metadata: Additional metadata.

        Returns:
            Entry ID.
        """
        if not self._memory_store:
            raise RuntimeError("Memory store not initialized")

        return await self._memory_store.add_ledger_entry(
            content=content,
            category=category,
            metadata=metadata,
        )

    async def get_stats(self) -> dict[str, Any]:
        """Get service statistics."""
        stats: dict[str, Any] = {
            "initialized": self._initialized,
            "project_root": str(self.config.project_root),
        }

        if self._sqlite_store:
            stats["sqlite"] = await self._sqlite_store.get_stats()

        if self._vector_store:
            stats["vectors"] = await self._vector_store.get_stats()

        if self._contract_store:
            stats["contracts"] = await self._contract_store.get_stats()

        return stats


# CLI Implementation
@click.group()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration file",
)
@click.option(
    "--project",
    "-p",
    type=click.Path(exists=True, path_type=Path),
    default=Path.cwd(),
    help="Project root directory",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
@click.pass_context
def cli(ctx: click.Context, config: Path | None, project: Path, verbose: bool) -> None:
    """ICD - Intelligent Code Retrieval Daemon."""
    ctx.ensure_object(dict)

    # Configure logging
    log_level = "DEBUG" if verbose else "INFO"
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(structlog, log_level)
        ),
    )

    # Load configuration
    ctx.obj["config"] = load_config(config_path=config, project_root=project)


@cli.command()
@click.pass_context
def index(ctx: click.Context) -> None:
    """Index the project directory."""
    config = ctx.obj["config"]

    async def run_index() -> None:
        service = ICDService(config)
        async with service.session():
            stats = await service.index_directory()
            click.echo(f"Indexed {stats.get('files', 0)} files")
            click.echo(f"Created {stats.get('chunks', 0)} chunks")
            if stats.get("errors", 0) > 0:
                click.echo(f"Errors: {stats['errors']}", err=True)

    asyncio.run(run_index())


@cli.command()
@click.argument("query")
@click.option("--limit", "-n", type=int, default=10, help="Number of results")
@click.option("--pack", "-p", is_flag=True, help="Compile results into a pack")
@click.option("--budget", "-b", type=int, help="Token budget for pack")
@click.pass_context
def search(
    ctx: click.Context,
    query: str,
    limit: int,
    pack: bool,
    budget: int | None,
) -> None:
    """Search the index."""
    config = ctx.obj["config"]

    async def run_search() -> None:
        service = ICDService(config)
        async with service.session():
            if pack:
                result = await service.compile_pack(
                    query=query,
                    budget_tokens=budget,
                )
                click.echo(result.content)
            else:
                result = await service.retrieve(query=query, limit=limit)
                for i, (chunk, score) in enumerate(
                    zip(result.chunks, result.scores), 1
                ):
                    click.echo(f"\n--- Result {i} (score: {score:.3f}) ---")
                    click.echo(f"File: {chunk.file_path}:{chunk.start_line}")
                    click.echo(chunk.content[:500] + "..." if len(chunk.content) > 500 else chunk.content)

    asyncio.run(run_search())


@cli.command()
@click.pass_context
def watch(ctx: click.Context) -> None:
    """Watch for file changes and update index."""
    config = ctx.obj["config"]

    async def run_watch() -> None:
        service = ICDService(config)

        # Handle signals
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(
                sig,
                lambda: asyncio.create_task(service.shutdown()),
            )

        async with service.session():
            # Initial index
            await service.index_directory()

            # Start watching
            await service.start_watching()
            click.echo("Watching for changes... (Ctrl+C to stop)")

            # Wait for shutdown
            await service._shutdown_event.wait()

    asyncio.run(run_watch())


@cli.command()
@click.pass_context
def stats(ctx: click.Context) -> None:
    """Show index statistics."""
    config = ctx.obj["config"]

    async def run_stats() -> None:
        service = ICDService(config)
        async with service.session():
            stats = await service.get_stats()
            click.echo("ICD Statistics")
            click.echo("=" * 40)
            for key, value in stats.items():
                if isinstance(value, dict):
                    click.echo(f"\n{key}:")
                    for k, v in value.items():
                        click.echo(f"  {k}: {v}")
                else:
                    click.echo(f"{key}: {value}")

    asyncio.run(run_stats())


@cli.command()
@click.pass_context
def init(ctx: click.Context) -> None:
    """Initialize ICD in the current directory."""
    config = ctx.obj["config"]

    config.ensure_directories()
    click.echo(f"Initialized ICD in {config.absolute_data_dir}")


def main() -> None:
    """Main entry point."""
    cli(obj={})


if __name__ == "__main__":
    main()
