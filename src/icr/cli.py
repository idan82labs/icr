"""
ICR CLI - Intelligent Code Retrieval Command Line Interface

The main entry point for the ICR unified tool, providing commands for
indexing, searching, context generation, and Claude Code integration.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.tree import Tree
from rich import box

from icr import __version__
from icr.core import (
    ICRConfig,
    RepoConfig,
    InvariantConfig,
    get_icr_root,
    get_repo_id,
    get_repo_root,
    ensure_initialized,
    load_config,
    save_config,
    initialize_icr,
    get_repo_data_path,
    format_file_size,
    format_duration,
    ICRError,
    NotInitializedError,
    ConfigurationError,
)

# Rich console for pretty output
console = Console()
error_console = Console(stderr=True)


def print_error(message: str) -> None:
    """Print an error message to stderr."""
    error_console.print(f"[bold red]Error:[/bold red] {message}")


def print_success(message: str) -> None:
    """Print a success message."""
    console.print(f"[bold green]✓[/bold green] {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[bold yellow]![/bold yellow] {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    console.print(f"[bold blue]i[/bold blue] {message}")


@click.group()
@click.version_option(version=__version__, prog_name="icr")
def main() -> None:
    """
    ICR - Intelligent Code Retrieval

    A unified tool for semantic code search, context generation,
    and Claude Code integration.
    """
    pass


@main.command()
def init() -> None:
    """Initialize ICR for the current user."""
    icr_root = get_icr_root()

    if icr_root.exists():
        config_path = icr_root / "config.yaml"
        if config_path.exists():
            print_warning(f"ICR already initialized at {icr_root}")
            if not click.confirm("Reinitialize? This will preserve existing data."):
                return

    with console.status("[bold blue]Initializing ICR..."):
        root = initialize_icr()

    console.print()
    console.print(Panel(
        f"[bold green]ICR initialized at {root}[/bold green]\n\n"
        f"Created:\n"
        f"  [dim]config.yaml[/dim]  - Configuration file\n"
        f"  [dim]repos/[/dim]       - Repository data\n"
        f"  [dim]cache/[/dim]       - Embedding cache\n"
        f"  [dim]logs/[/dim]        - Log files\n\n"
        f"Next steps:\n"
        f"  1. Index a repository: [cyan]icr index /path/to/repo[/cyan]\n"
        f"  2. Search for code: [cyan]icr search 'your query'[/cyan]\n"
        f"  3. Configure Claude Code: [cyan]icr configure claude-code[/cyan]",
        title="[bold]ICR Initialized[/bold]",
        border_style="green"
    ))


@main.command()
@click.argument("repo_path", type=click.Path(exists=True), default=".")
@click.option("--name", "-n", help="Name for the repository")
@click.option("--force", "-f", is_flag=True, help="Force re-indexing")
def index(repo_path: str, name: Optional[str], force: bool) -> None:
    """Index a repository for semantic search."""
    try:
        ensure_initialized()
        config = load_config()
    except NotInitializedError as e:
        print_error(str(e))
        sys.exit(1)

    # Resolve repository path
    repo_root = get_repo_root(repo_path)
    repo_id = get_repo_id(repo_root)
    repo_name = name or repo_root.name

    console.print(f"\n[bold]Indexing repository:[/bold] {repo_root}")
    console.print(f"[dim]Repository ID: {repo_id}[/dim]\n")

    # Check if already indexed
    if repo_id in config.repositories and not force:
        existing = config.repositories[repo_id]
        print_warning(f"Repository already indexed ({existing.chunk_count} chunks)")
        if not click.confirm("Re-index?"):
            return

    # Simulate indexing process (actual implementation would use icd)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        # Scanning files
        scan_task = progress.add_task("[cyan]Scanning files...", total=100)
        for i in range(100):
            progress.update(scan_task, advance=1)

        # Parsing code
        parse_task = progress.add_task("[cyan]Parsing code...", total=100)
        for i in range(100):
            progress.update(parse_task, advance=1)

        # Generating embeddings
        embed_task = progress.add_task("[cyan]Generating embeddings...", total=100)
        for i in range(100):
            progress.update(embed_task, advance=1)

        # Building index
        index_task = progress.add_task("[cyan]Building index...", total=100)
        for i in range(100):
            progress.update(index_task, advance=1)

    # Create repo data directory
    repo_data_path = get_repo_data_path(repo_id)
    repo_data_path.mkdir(parents=True, exist_ok=True)

    # Update configuration
    config.repositories[repo_id] = RepoConfig(
        repo_id=repo_id,
        path=str(repo_root),
        name=repo_name,
        indexed_at=datetime.now().isoformat(),
        chunk_count=0,  # Would be actual count from indexer
        file_count=0,   # Would be actual count from indexer
    )
    save_config(config)

    console.print()
    console.print(Panel(
        f"[bold green]Index complete![/bold green]\n\n"
        f"Repository: [cyan]{repo_name}[/cyan]\n"
        f"Path: [dim]{repo_root}[/dim]\n"
        f"ID: [dim]{repo_id}[/dim]\n\n"
        f"[dim]Note: This is a placeholder. Full indexing requires icd package.[/dim]",
        title="[bold]Indexing Complete[/bold]",
        border_style="green"
    ))


@main.command()
@click.argument("query")
@click.option("--limit", "-l", default=10, help="Maximum number of results")
@click.option("--threshold", "-t", default=0.5, type=float, help="Minimum similarity score")
@click.option("--repo", "-r", help="Limit search to specific repository")
def search(query: str, limit: int, threshold: float, repo: Optional[str]) -> None:
    """Search indexed content using semantic search."""
    try:
        ensure_initialized()
        config = load_config()
    except NotInitializedError as e:
        print_error(str(e))
        sys.exit(1)

    if not config.repositories:
        print_error("No repositories indexed. Run 'icr index <path>' first.")
        sys.exit(1)

    console.print(f"\n[bold]Searching for:[/bold] {query}")
    if repo:
        console.print(f"[dim]Limited to repository: {repo}[/dim]")
    console.print()

    # Placeholder search results (actual implementation would use icd)
    with console.status("[bold blue]Searching..."):
        # Simulate search delay
        pass

    # Display placeholder results
    table = Table(
        title="Search Results",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("File", style="cyan")
    table.add_column("Lines", style="green")
    table.add_column("Score", style="yellow", justify="right")

    # Placeholder results
    console.print(Panel(
        "[dim]No search results available.\n\n"
        "Search functionality requires the icd package to be installed.\n"
        "The search will return semantic matches from indexed repositories.[/dim]",
        title="[bold]Search Results[/bold]",
        border_style="blue"
    ))


@main.command()
@click.argument("prompt")
@click.option("--mode", "-m", type=click.Choice(["pack", "impact", "qa"]), default="pack")
@click.option("--max-chunks", "-c", default=20, help="Maximum chunks to include")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
def pack(prompt: str, mode: str, max_chunks: int, output: Optional[str]) -> None:
    """Generate a context pack for a prompt."""
    try:
        ensure_initialized()
        config = load_config()
    except NotInitializedError as e:
        print_error(str(e))
        sys.exit(1)

    console.print(f"\n[bold]Generating context pack[/bold]")
    console.print(f"Mode: [cyan]{mode}[/cyan]")
    console.print(f"Prompt: [dim]{prompt[:100]}{'...' if len(prompt) > 100 else ''}[/dim]\n")

    with console.status("[bold blue]Generating context pack..."):
        # Placeholder - actual implementation would use icd
        pass

    # Display placeholder pack
    pack_content = f"""## ICR Context Pack
Mode: {mode}
Generated: {datetime.now().isoformat()}

### Query
{prompt}

### Relevant Context
[Context would be retrieved from indexed repositories]

### Confidence
Score: N/A (requires icd package)

---
*Generated by ICR v{__version__}*
"""

    if output:
        Path(output).write_text(pack_content)
        print_success(f"Context pack written to {output}")
    else:
        console.print(Panel(
            pack_content,
            title="[bold]Context Pack[/bold]",
            border_style="blue"
        ))


@main.command()
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--depth", "-d", default=2, help="Analysis depth level")
def impact(paths: tuple, depth: int) -> None:
    """Analyze the impact of changes to specified files."""
    try:
        ensure_initialized()
        config = load_config()
    except NotInitializedError as e:
        print_error(str(e))
        sys.exit(1)

    console.print(f"\n[bold]Analyzing impact for {len(paths)} file(s)[/bold]\n")

    for path in paths:
        console.print(f"  [cyan]{path}[/cyan]")

    console.print()

    with console.status("[bold blue]Analyzing dependencies..."):
        # Placeholder - actual implementation would use icd
        pass

    # Display placeholder impact analysis
    console.print(Panel(
        "[dim]Impact analysis requires the icd package.\n\n"
        "This would show:\n"
        "  - Files that depend on the changed files\n"
        "  - Potential breaking changes\n"
        "  - Test files that should be run\n"
        "  - Related documentation[/dim]",
        title="[bold]Impact Analysis[/bold]",
        border_style="yellow"
    ))


@main.command()
@click.argument("invariant")
@click.option("--repo", "-r", help="Associate with specific repository")
def pin(invariant: str, repo: Optional[str]) -> None:
    """Pin an invariant to be included in context."""
    try:
        ensure_initialized()
        config = load_config()
    except NotInitializedError as e:
        print_error(str(e))
        sys.exit(1)

    import hashlib
    inv_id = hashlib.sha256(invariant.encode()).hexdigest()[:8]

    new_invariant = InvariantConfig(
        id=inv_id,
        content=invariant,
        repo_id=repo,
        created_at=datetime.now().isoformat(),
    )

    config.invariants.append(new_invariant)
    save_config(config)

    print_success(f"Pinned invariant with ID: {inv_id}")
    console.print(f"[dim]{invariant}[/dim]")


@main.command()
@click.argument("inv_id")
def unpin(inv_id: str) -> None:
    """Remove a pinned invariant by ID."""
    try:
        ensure_initialized()
        config = load_config()
    except NotInitializedError as e:
        print_error(str(e))
        sys.exit(1)

    original_count = len(config.invariants)
    config.invariants = [inv for inv in config.invariants if inv.id != inv_id]

    if len(config.invariants) == original_count:
        print_error(f"Invariant with ID '{inv_id}' not found")
        sys.exit(1)

    save_config(config)
    print_success(f"Removed invariant: {inv_id}")


@main.command("list-pins")
def list_pins() -> None:
    """List all pinned invariants."""
    try:
        ensure_initialized()
        config = load_config()
    except NotInitializedError as e:
        print_error(str(e))
        sys.exit(1)

    if not config.invariants:
        print_info("No pinned invariants")
        return

    table = Table(
        title="Pinned Invariants",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("ID", style="yellow", width=10)
    table.add_column("Content", style="white")
    table.add_column("Repo", style="dim")
    table.add_column("Created", style="dim")

    for inv in config.invariants:
        content = inv.content[:60] + "..." if len(inv.content) > 60 else inv.content
        table.add_row(
            inv.id,
            content,
            inv.repo_id or "-",
            inv.created_at[:10] if inv.created_at else "-",
        )

    console.print(table)


@main.command("mcp-serve")
@click.option("--transport", "-t", type=click.Choice(["stdio"]), default="stdio")
def mcp_serve(transport: str) -> None:
    """Start the MCP server for Claude Code integration."""
    try:
        ensure_initialized()
    except NotInitializedError as e:
        print_error(str(e))
        sys.exit(1)

    print_info(f"Starting MCP server with {transport} transport...")
    print_info("MCP server requires ic-mcp package to be installed.")

    # Placeholder - actual implementation would start ic-mcp server
    console.print(Panel(
        "[dim]MCP server functionality requires the ic-mcp package.\n\n"
        "The server would expose:\n"
        "  - search: Semantic code search\n"
        "  - pack: Context pack generation\n"
        "  - impact: Change impact analysis\n"
        "  - index: Repository indexing[/dim]",
        title="[bold]MCP Server[/bold]",
        border_style="blue"
    ))


# Hook subcommands
@main.group()
def hook() -> None:
    """Claude Code hook handlers."""
    pass


@hook.command("prompt-submit")
@click.option("--input", "-i", "input_file", type=click.Path(exists=True), help="Input JSON file")
def hook_prompt_submit(input_file: Optional[str]) -> None:
    """Handle UserPromptSubmit hook from Claude Code."""
    try:
        ensure_initialized()
    except NotInitializedError as e:
        print_error(str(e))
        sys.exit(1)

    # Read from stdin or file
    import json
    if input_file:
        with open(input_file) as f:
            data = json.load(f)
    else:
        data = json.load(sys.stdin)

    # Placeholder response
    response = {
        "continue": True,
        "context": None,
        "message": "ICR hook processed (placeholder)"
    }

    print(json.dumps(response))


@hook.command("stop")
def hook_stop() -> None:
    """Handle Stop hook from Claude Code."""
    try:
        ensure_initialized()
    except NotInitializedError as e:
        print_error(str(e))
        sys.exit(1)

    import json
    response = {"acknowledged": True}
    print(json.dumps(response))


@hook.command("precompact")
@click.option("--input", "-i", "input_file", type=click.Path(exists=True), help="Input JSON file")
def hook_precompact(input_file: Optional[str]) -> None:
    """Handle PreCompact hook from Claude Code."""
    try:
        ensure_initialized()
    except NotInitializedError as e:
        print_error(str(e))
        sys.exit(1)

    import json
    if input_file:
        with open(input_file) as f:
            data = json.load(f)
    else:
        data = json.load(sys.stdin)

    # Placeholder response
    response = {
        "summary": None,
        "preserve": []
    }

    print(json.dumps(response))


# Configure subcommands
@main.group()
def configure() -> None:
    """Configuration commands."""
    pass


@configure.command("claude-code")
@click.option("--enable-hooks/--disable-hooks", default=True, help="Enable/disable hooks")
@click.option("--enable-mcp/--disable-mcp", default=True, help="Enable/disable MCP")
def configure_claude_code(enable_hooks: bool, enable_mcp: bool) -> None:
    """Configure Claude Code integration."""
    try:
        ensure_initialized()
        config = load_config()
    except NotInitializedError as e:
        print_error(str(e))
        sys.exit(1)

    config.hooks.enabled = enable_hooks
    config.mcp.enabled = enable_mcp
    save_config(config)

    console.print("\n[bold]Claude Code Configuration Updated[/bold]\n")
    console.print(f"  Hooks: {'[green]enabled[/green]' if enable_hooks else '[red]disabled[/red]'}")
    console.print(f"  MCP:   {'[green]enabled[/green]' if enable_mcp else '[red]disabled[/red]'}")

    # Show integration instructions
    console.print(Panel(
        "To complete Claude Code integration:\n\n"
        "1. Add to your Claude Code settings.json:\n"
        "[cyan]{\n"
        '  "hooks": {\n'
        '    "UserPromptSubmit": [\n'
        '      {"command": "icr hook prompt-submit"}\n'
        '    ]\n'
        '  },\n'
        '  "mcpServers": {\n'
        '    "icr": {\n'
        '      "command": "icr",\n'
        '      "args": ["mcp-serve"]\n'
        '    }\n'
        '  }\n'
        "}[/cyan]\n\n"
        "2. Restart Claude Code to apply changes",
        title="[bold]Integration Instructions[/bold]",
        border_style="blue"
    ))


@configure.command("show")
def configure_show() -> None:
    """Show current ICR configuration."""
    try:
        ensure_initialized()
        config = load_config()
    except NotInitializedError as e:
        print_error(str(e))
        sys.exit(1)

    console.print("\n[bold]ICR Configuration[/bold]\n")

    # Embedding config
    console.print("[cyan]Embedding:[/cyan]")
    console.print(f"  Backend: {config.embedding.backend}")
    console.print(f"  Model: {config.embedding.model}")
    console.print(f"  Dimension: {config.embedding.dimension}")
    console.print()

    # Index config
    console.print("[cyan]Index:[/cyan]")
    console.print(f"  EF Construction: {config.index.ef_construction}")
    console.print(f"  M: {config.index.m}")
    console.print(f"  EF Search: {config.index.ef_search}")
    console.print()

    # Hooks config
    console.print("[cyan]Hooks:[/cyan]")
    status = "[green]enabled[/green]" if config.hooks.enabled else "[red]disabled[/red]"
    console.print(f"  Status: {status}")
    console.print()

    # MCP config
    console.print("[cyan]MCP:[/cyan]")
    status = "[green]enabled[/green]" if config.mcp.enabled else "[red]disabled[/red]"
    console.print(f"  Status: {status}")
    console.print(f"  Transport: {config.mcp.transport}")
    console.print()

    # Repositories
    console.print("[cyan]Repositories:[/cyan]")
    if config.repositories:
        for repo_id, repo in config.repositories.items():
            console.print(f"  {repo.name} ({repo_id})")
            console.print(f"    Path: [dim]{repo.path}[/dim]")
            console.print(f"    Chunks: {repo.chunk_count}")
    else:
        console.print("  [dim]No repositories indexed[/dim]")


@main.command()
def doctor() -> None:
    """Run health checks on ICR installation."""
    console.print("\n[bold]ICR Health Check[/bold]")
    console.print("=" * 40 + "\n")

    all_healthy = True

    # Check 1: ICR initialized
    icr_root = get_icr_root()
    if icr_root.exists():
        print_success("ICR directory exists")
    else:
        print_error(f"ICR not initialized at {icr_root}")
        console.print("  Run: [cyan]icr init[/cyan]")
        all_healthy = False
        console.print(f"\n[bold red]Status: UNHEALTHY[/bold red]")
        return

    # Check 2: Configuration file
    config_path = icr_root / "config.yaml"
    if config_path.exists():
        print_success("Configuration file found")
        try:
            config = load_config()
            print_success("Configuration is valid")
        except ConfigurationError as e:
            print_error(f"Configuration invalid: {e}")
            all_healthy = False
    else:
        print_error("Configuration file missing")
        all_healthy = False

    # Check 3: Database
    db_path = icr_root / "icr.db"
    if db_path.exists():
        print_success("Database file exists")
    else:
        print_warning("Database file not found (will be created on first index)")

    # Check 4: Required directories
    for dir_name in ["repos", "cache", "logs"]:
        dir_path = icr_root / dir_name
        if dir_path.exists():
            print_success(f"{dir_name}/ directory exists")
        else:
            print_warning(f"{dir_name}/ directory missing")

    # Check 5: Embedding backend (placeholder)
    config = load_config()
    print_success(f"Embedding backend: {config.embedding.backend}")

    # Check 6: Hooks configuration
    if config.hooks.enabled:
        print_success("Claude Code hooks enabled")
    else:
        print_warning("Claude Code hooks disabled")

    # Check 7: MCP configuration
    if config.mcp.enabled:
        print_success("MCP server enabled")
    else:
        print_warning("MCP server disabled")

    # Check 8: Indexed repositories
    if config.repositories:
        print_success(f"{len(config.repositories)} repository(ies) indexed")
    else:
        print_info("No repositories indexed yet")

    # Final status
    console.print()
    if all_healthy:
        console.print("[bold green]Status: HEALTHY[/bold green]")
    else:
        console.print("[bold red]Status: UNHEALTHY[/bold red]")


@main.command()
@click.argument("repo_path", type=click.Path(exists=True), default=".", required=False)
def status(repo_path: str) -> None:
    """Show ICR status for a repository."""
    try:
        ensure_initialized()
        config = load_config()
    except NotInitializedError as e:
        print_error(str(e))
        sys.exit(1)

    repo_root = get_repo_root(repo_path)
    repo_id = get_repo_id(repo_root)

    console.print(f"\n[bold]ICR Status[/bold]")
    console.print(f"Repository: [cyan]{repo_root}[/cyan]")
    console.print(f"ID: [dim]{repo_id}[/dim]\n")

    if repo_id in config.repositories:
        repo = config.repositories[repo_id]

        table = Table(box=box.ROUNDED, show_header=False)
        table.add_column("Property", style="cyan")
        table.add_column("Value")

        table.add_row("Status", "[green]Indexed[/green]")
        table.add_row("Name", repo.name)
        table.add_row("Path", repo.path)
        table.add_row("Files", str(repo.file_count))
        table.add_row("Chunks", str(repo.chunk_count))
        table.add_row("Indexed At", repo.indexed_at or "Unknown")

        console.print(table)
    else:
        console.print(Panel(
            "[yellow]Repository not indexed[/yellow]\n\n"
            f"Run: [cyan]icr index {repo_path}[/cyan]",
            border_style="yellow"
        ))


@main.command()
def version() -> None:
    """Show ICR version information."""
    console.print(f"\n[bold]ICR[/bold] - Intelligent Code Retrieval")
    console.print(f"Version: [cyan]{__version__}[/cyan]")
    console.print()

    # Show component versions (placeholders)
    table = Table(box=box.SIMPLE, show_header=False)
    table.add_column("Component", style="dim")
    table.add_column("Version")

    table.add_row("icr", __version__)
    table.add_row("icd", "[dim]not installed[/dim]")
    table.add_row("ic-mcp", "[dim]not installed[/dim]")
    table.add_row("ic-claude", "[dim]not installed[/dim]")

    console.print(table)


if __name__ == "__main__":
    main()
