"""
Tests for IC-MCP tools.
"""

from pathlib import Path
from uuid import UUID

import pytest

from ic_mcp.schemas.inputs import (
    AdminPingInput,
    EnvAggregateInput,
    EnvPeekInput,
    EnvSearchInput,
    EnvSliceInput,
    MemoryGetInput,
    MemoryListInput,
    MemoryPackInput,
    MemoryPinInput,
    MemoryStatsInput,
    MemoryUnpinInput,
    ProjectCommandsInput,
    ProjectImpactInput,
    ProjectMapInput,
    ProjectSymbolSearchInput,
    RlmPlanInput,
)
from ic_mcp.tools.admin import AdminTools
from ic_mcp.tools.env import EnvTools
from ic_mcp.tools.memory import MemoryTools
from ic_mcp.tools.project import ProjectTools
from ic_mcp.tools.rlm import RlmTools


class TestMemoryTools:
    """Tests for memory tools."""

    @pytest.fixture
    def memory_tools(self) -> MemoryTools:
        """Create memory tools instance."""
        return MemoryTools()

    @pytest.mark.asyncio
    async def test_memory_pack_basic(
        self,
        memory_tools: MemoryTools,
        temp_repo: Path,
        request_id: UUID,
    ):
        """Test basic memory_pack functionality."""
        input_data = MemoryPackInput(
            prompt="Find the main function",
            repo_root=str(temp_repo),
            budget_tokens=2000,
            k=10,
        )

        result = await memory_tools.memory_pack(input_data, request_id)

        assert result.ok is True
        assert result.request_id == request_id
        assert result.mode_resolved in ("pack", "rlm")
        assert result.pack_markdown is not None
        assert result.budget_used_tokens > 0
        assert result.budget_used_tokens <= input_data.budget_tokens
        assert 0.0 <= result.confidence <= 1.0
        assert 0.0 <= result.entropy <= 1.0

    @pytest.mark.asyncio
    async def test_memory_pack_with_focus_paths(
        self,
        memory_tools: MemoryTools,
        temp_repo: Path,
        request_id: UUID,
    ):
        """Test memory_pack with focus_paths filter."""
        input_data = MemoryPackInput(
            prompt="Find utility functions",
            repo_root=str(temp_repo),
            budget_tokens=2000,
            focus_paths=["src/"],
        )

        result = await memory_tools.memory_pack(input_data, request_id)

        assert result.ok is True
        # All top sources should be in src/
        for source in result.top_sources:
            assert source.path.startswith("src/") or source.path == ""

    @pytest.mark.asyncio
    async def test_memory_pin_unpin(
        self,
        memory_tools: MemoryTools,
        request_id: UUID,
    ):
        """Test memory_pin and memory_unpin."""
        # Pin a source
        pin_input = MemoryPinInput(
            source_id="test_source_1",
            path="src/main.py",
            label="Important file",
            ttl_seconds=3600,
        )

        pin_result = await memory_tools.memory_pin(pin_input, request_id)

        assert pin_result.ok is True
        assert pin_result.source_id == "test_source_1"
        assert pin_result.path == "src/main.py"
        assert pin_result.label == "Important file"
        assert pin_result.pinned_at is not None
        assert pin_result.expires_at is not None

        # Unpin the source
        unpin_input = MemoryUnpinInput(source_id="test_source_1")
        unpin_result = await memory_tools.memory_unpin(unpin_input, request_id)

        assert unpin_result.ok is True
        assert unpin_result.source_id == "test_source_1"
        assert unpin_result.was_pinned is True

        # Unpin again should show was_pinned=False
        unpin_result2 = await memory_tools.memory_unpin(unpin_input, request_id)
        assert unpin_result2.was_pinned is False

    @pytest.mark.asyncio
    async def test_memory_list(
        self,
        memory_tools: MemoryTools,
        temp_repo: Path,
        request_id: UUID,
    ):
        """Test memory_list functionality."""
        # First, pack some content to populate the store
        pack_input = MemoryPackInput(
            prompt="Find functions",
            repo_root=str(temp_repo),
            budget_tokens=1000,
            k=5,
        )
        await memory_tools.memory_pack(pack_input, request_id)

        # Now list items
        list_input = MemoryListInput(filter_type="all", limit=50)
        result = await memory_tools.memory_list(list_input, request_id)

        assert result.ok is True
        assert result.pagination is not None
        assert isinstance(result.items, list)

    @pytest.mark.asyncio
    async def test_memory_stats(
        self,
        memory_tools: MemoryTools,
        temp_repo: Path,
        request_id: UUID,
    ):
        """Test memory_stats functionality."""
        # Pack some content first
        pack_input = MemoryPackInput(
            prompt="Find functions",
            repo_root=str(temp_repo),
            budget_tokens=1000,
        )
        await memory_tools.memory_pack(pack_input, request_id)

        # Get stats
        stats_input = MemoryStatsInput(include_breakdown=True)
        result = await memory_tools.memory_stats(stats_input, request_id)

        assert result.ok is True
        assert result.total_items >= 0
        assert result.pinned_count >= 0
        assert result.total_size_bytes >= 0
        assert result.breakdown is not None


class TestEnvTools:
    """Tests for environment tools."""

    @pytest.fixture
    def env_tools(self, temp_repo: Path) -> EnvTools:
        """Create env tools instance."""
        return EnvTools(str(temp_repo))

    @pytest.mark.asyncio
    async def test_env_search_repo(
        self,
        env_tools: EnvTools,
        temp_repo: Path,
        request_id: UUID,
    ):
        """Test env_search in repo scope."""
        input_data = EnvSearchInput(
            query="Calculator",
            scope="repo",
            limit=10,
        )

        result = await env_tools.env_search(input_data, request_id)

        assert result.ok is True
        assert len(result.results) > 0
        assert any("Calculator" in r.snippet for r in result.results)

    @pytest.mark.asyncio
    async def test_env_search_with_language(
        self,
        env_tools: EnvTools,
        temp_repo: Path,
        request_id: UUID,
    ):
        """Test env_search with language filter."""
        input_data = EnvSearchInput(
            query="function",
            scope="repo",
            language="python",
            limit=10,
        )

        result = await env_tools.env_search(input_data, request_id)

        assert result.ok is True
        # All results should be Python files
        for r in result.results:
            assert r.path.endswith(".py")

    @pytest.mark.asyncio
    async def test_env_peek(
        self,
        env_tools: EnvTools,
        temp_repo: Path,
        request_id: UUID,
    ):
        """Test env_peek functionality."""
        input_data = EnvPeekInput(
            path=str(temp_repo / "src" / "main.py"),
            start_line=1,
            end_line=10,
        )

        result = await env_tools.env_peek(input_data, request_id)

        assert result.ok is True
        assert result.path == str(temp_repo / "src" / "main.py")
        assert result.start_line == 1
        assert result.end_line <= 10
        assert result.total_lines > 0
        assert "Main module" in result.content or "def main" in result.content

    @pytest.mark.asyncio
    async def test_env_peek_with_max_lines(
        self,
        env_tools: EnvTools,
        temp_repo: Path,
        request_id: UUID,
    ):
        """Test env_peek with max_lines limit."""
        input_data = EnvPeekInput(
            path=str(temp_repo / "src" / "main.py"),
            start_line=1,
            end_line=100,
            max_lines=5,
        )

        result = await env_tools.env_peek(input_data, request_id)

        assert result.ok is True
        assert result.truncated is True
        assert result.end_line - result.start_line + 1 <= 5

    @pytest.mark.asyncio
    async def test_env_slice_by_symbol(
        self,
        env_tools: EnvTools,
        temp_repo: Path,
        request_id: UUID,
    ):
        """Test env_slice by symbol name."""
        input_data = EnvSliceInput(
            path=str(temp_repo / "src" / "main.py"),
            symbol="Calculator",
            context_lines=2,
        )

        result = await env_tools.env_slice(input_data, request_id)

        assert result.ok is True
        assert result.symbol == "Calculator"
        assert "class Calculator" in result.content

    @pytest.mark.asyncio
    async def test_env_slice_by_lines(
        self,
        env_tools: EnvTools,
        temp_repo: Path,
        request_id: UUID,
    ):
        """Test env_slice by line range."""
        input_data = EnvSliceInput(
            path=str(temp_repo / "src" / "main.py"),
            start_line=5,
            end_line=10,
            context_lines=1,
        )

        result = await env_tools.env_slice(input_data, request_id)

        assert result.ok is True
        assert result.span.start_line == 5
        assert result.span.end_line == 10

    @pytest.mark.asyncio
    async def test_env_aggregate_unique(
        self,
        env_tools: EnvTools,
        request_id: UUID,
    ):
        """Test env_aggregate with unique operation."""
        input_data = EnvAggregateInput(
            op="unique",
            inputs=["apple", "banana", "apple", "cherry", "banana"],
            limit=10,
        )

        result = await env_tools.env_aggregate(input_data, request_id)

        assert result.ok is True
        assert result.op == "unique"
        values = [r.value for r in result.results]
        assert len(values) == 3
        assert set(values) == {"apple", "banana", "cherry"}

    @pytest.mark.asyncio
    async def test_env_aggregate_count(
        self,
        env_tools: EnvTools,
        request_id: UUID,
    ):
        """Test env_aggregate with count operation."""
        input_data = EnvAggregateInput(
            op="count",
            inputs=["apple", "banana", "apple", "cherry", "banana", "apple"],
            limit=10,
        )

        result = await env_tools.env_aggregate(input_data, request_id)

        assert result.ok is True
        assert result.op == "count"
        # apple should be most common
        assert result.results[0].value == "apple"
        assert result.results[0].count == 3

    @pytest.mark.asyncio
    async def test_env_aggregate_extract_regex(
        self,
        env_tools: EnvTools,
        request_id: UUID,
    ):
        """Test env_aggregate with extract_regex operation."""
        input_data = EnvAggregateInput(
            op="extract_regex",
            inputs=[
                "def hello_world():",
                "def another_function():",
                "class MyClass:",
            ],
            params={"pattern": r"def (\w+)"},
            limit=10,
        )

        result = await env_tools.env_aggregate(input_data, request_id)

        assert result.ok is True
        values = [r.value for r in result.results]
        assert "hello_world" in values
        assert "another_function" in values


class TestProjectTools:
    """Tests for project tools."""

    @pytest.fixture
    def project_tools(self) -> ProjectTools:
        """Create project tools instance."""
        return ProjectTools()

    @pytest.mark.asyncio
    async def test_project_map(
        self,
        project_tools: ProjectTools,
        temp_repo: Path,
        request_id: UUID,
    ):
        """Test project_map functionality."""
        input_data = ProjectMapInput(
            repo_root=str(temp_repo),
            depth=3,
            include_stats=True,
        )

        result = await project_tools.project_map(input_data, request_id)

        assert result.ok is True
        assert result.repo_root == str(temp_repo)
        assert result.total_files > 0
        assert result.total_directories > 0
        assert result.tree is not None
        assert "python" in result.languages or "typescript" in result.languages

    @pytest.mark.asyncio
    async def test_project_map_with_patterns(
        self,
        project_tools: ProjectTools,
        temp_repo: Path,
        request_id: UUID,
    ):
        """Test project_map with include/exclude patterns."""
        input_data = ProjectMapInput(
            repo_root=str(temp_repo),
            depth=3,
            include_patterns=["*.py"],
            exclude_patterns=["*test*"],
        )

        result = await project_tools.project_map(input_data, request_id)

        assert result.ok is True
        # Should have Python files but no test files counted
        # (Note: the tree structure may still show directories)

    @pytest.mark.asyncio
    async def test_project_symbol_search(
        self,
        project_tools: ProjectTools,
        temp_repo: Path,
        request_id: UUID,
    ):
        """Test project_symbol_search functionality."""
        input_data = ProjectSymbolSearchInput(
            query="Calculator",
            repo_root=str(temp_repo),
            limit=20,
        )

        result = await project_tools.project_symbol_search(input_data, request_id)

        assert result.ok is True
        assert len(result.results) > 0
        # Should find the Calculator class
        class_results = [r for r in result.results if r.symbol_type == "class"]
        assert len(class_results) > 0

    @pytest.mark.asyncio
    async def test_project_symbol_search_with_type_filter(
        self,
        project_tools: ProjectTools,
        temp_repo: Path,
        request_id: UUID,
    ):
        """Test project_symbol_search with symbol type filter."""
        input_data = ProjectSymbolSearchInput(
            query="add",
            repo_root=str(temp_repo),
            symbol_types=["function", "method"],
            limit=20,
        )

        result = await project_tools.project_symbol_search(input_data, request_id)

        assert result.ok is True
        # All results should be functions or methods
        for r in result.results:
            assert r.symbol_type in ("function", "method")

    @pytest.mark.asyncio
    async def test_project_impact(
        self,
        project_tools: ProjectTools,
        temp_repo: Path,
        request_id: UUID,
    ):
        """Test project_impact functionality."""
        input_data = ProjectImpactInput(
            changed_paths=["src/main.py", "src/utils.py"],
            max_nodes=100,
            max_edges=500,
        )

        result = await project_tools.project_impact(input_data, request_id)

        assert result.ok is True
        assert len(result.nodes) > 0
        assert result.root_paths == ["src/main.py", "src/utils.py"]
        assert result.depth_reached >= 0

    @pytest.mark.asyncio
    async def test_project_commands(
        self,
        project_tools: ProjectTools,
        temp_repo: Path,
        request_id: UUID,
    ):
        """Test project_commands functionality."""
        input_data = ProjectCommandsInput(
            repo_root=str(temp_repo),
            command_type="all",
        )

        result = await project_tools.project_commands(input_data, request_id)

        assert result.ok is True
        assert len(result.commands) > 0
        assert "package.json" in result.config_files or "pyproject.toml" in result.config_files

        # Should find npm scripts from package.json
        npm_commands = [c for c in result.commands if c.source == "package.json"]
        assert len(npm_commands) > 0

    @pytest.mark.asyncio
    async def test_project_commands_filtered(
        self,
        project_tools: ProjectTools,
        temp_repo: Path,
        request_id: UUID,
    ):
        """Test project_commands with type filter."""
        input_data = ProjectCommandsInput(
            repo_root=str(temp_repo),
            command_type="test",
        )

        result = await project_tools.project_commands(input_data, request_id)

        assert result.ok is True
        # All returned commands should be test commands
        for cmd in result.commands:
            assert cmd.type == "test"


class TestRlmTools:
    """Tests for RLM tools."""

    @pytest.fixture
    def rlm_tools(self) -> RlmTools:
        """Create RLM tools instance."""
        return RlmTools()

    @pytest.mark.asyncio
    async def test_rlm_plan_exploration(
        self,
        rlm_tools: RlmTools,
        request_id: UUID,
    ):
        """Test rlm_plan for exploration task."""
        input_data = RlmPlanInput(
            task="Find all functions that handle user authentication",
            scope="repo",
            budget={"max_steps": 10, "max_peek_lines": 500, "max_candidates": 50},
        )

        result = await rlm_tools.rlm_plan(input_data, request_id)

        assert result.ok is True
        assert len(result.steps) > 0
        assert len(result.stop_conditions) > 0
        assert len(result.artifacts) > 0
        assert result.estimated_tokens > 0
        assert result.estimated_steps > 0

        # Check step structure
        for step in result.steps:
            assert step.id is not None
            assert step.type in ("search", "peek", "slice", "aggregate", "impact", "pack", "stop")
            assert step.tool is not None
            assert step.rationale is not None

    @pytest.mark.asyncio
    async def test_rlm_plan_impact(
        self,
        rlm_tools: RlmTools,
        request_id: UUID,
    ):
        """Test rlm_plan for impact analysis task."""
        input_data = RlmPlanInput(
            task="Analyze the impact of changes to the User model",
            scope="repo",
        )

        result = await rlm_tools.rlm_plan(input_data, request_id)

        assert result.ok is True
        # Should include an impact step
        impact_steps = [s for s in result.steps if s.type == "impact"]
        assert len(impact_steps) > 0


class TestAdminTools:
    """Tests for admin tools."""

    @pytest.fixture
    def admin_tools(self) -> AdminTools:
        """Create admin tools instance."""
        return AdminTools()

    @pytest.mark.asyncio
    async def test_admin_ping_basic(
        self,
        admin_tools: AdminTools,
        request_id: UUID,
    ):
        """Test basic admin_ping."""
        input_data = AdminPingInput()

        result = await admin_tools.admin_ping(input_data, request_id)

        assert result.ok is True
        assert result.server_name == "icr"
        assert result.server_version is not None
        assert result.uptime_seconds >= 0.0

    @pytest.mark.asyncio
    async def test_admin_ping_with_echo(
        self,
        admin_tools: AdminTools,
        request_id: UUID,
    ):
        """Test admin_ping with echo."""
        input_data = AdminPingInput(echo="hello world")

        result = await admin_tools.admin_ping(input_data, request_id)

        assert result.ok is True
        assert result.echo == "hello world"

    @pytest.mark.asyncio
    async def test_admin_ping_with_diagnostics(
        self,
        admin_tools: AdminTools,
        request_id: UUID,
    ):
        """Test admin_ping with diagnostics."""
        input_data = AdminPingInput(include_diagnostics=True)

        result = await admin_tools.admin_ping(input_data, request_id)

        assert result.ok is True
        assert result.diagnostics is not None
        assert "python" in result.diagnostics
        assert "platform" in result.diagnostics
        assert "timestamp" in result.diagnostics
