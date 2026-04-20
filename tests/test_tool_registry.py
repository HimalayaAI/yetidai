"""Tests for core/tool_registry.py — ToolRegistry."""
import asyncio
import unittest

from core.tool_contracts import (
    ToolCategory,
    ToolParam,
    ToolSpec,
    ToolResult,
    ToolContext,
)
from core.tool_registry import ToolRegistry


def _dummy_spec(name: str = "test_tool", enabled: bool = True) -> ToolSpec:
    return ToolSpec(
        tool_id=f"test.{name}",
        name=name,
        description=f"A test tool called {name}.",
        category=ToolCategory.UTILITY,
        parameters=[
            ToolParam(name="input", type="string", description="test input"),
        ],
        enabled=enabled,
    )


async def _echo_handler(ctx: ToolContext, arguments: dict) -> ToolResult:
    """Simple handler that echoes arguments back."""
    return ToolResult(
        tool_id="test.test_tool",
        success=True,
        content=f"Echo: {arguments.get('input', '')}",
    )


async def _boom_handler(ctx: ToolContext, arguments: dict) -> ToolResult:
    """Handler that always raises."""
    raise RuntimeError("Intentional explosion")


class ToolRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ToolRegistry()
        self.ctx = ToolContext(query="test query")

    def test_register_and_list(self) -> None:
        spec = _dummy_spec()
        self.registry.register(spec, _echo_handler)

        tools = self.registry.list_tools()
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].name, "test_tool")

    def test_has(self) -> None:
        self.assertFalse(self.registry.has("test_tool"))
        self.registry.register(_dummy_spec(), _echo_handler)
        self.assertTrue(self.registry.has("test_tool"))

    def test_get_spec(self) -> None:
        spec = _dummy_spec()
        self.registry.register(spec, _echo_handler)
        found = self.registry.get_spec("test_tool")
        self.assertIsNotNone(found)
        self.assertEqual(found.tool_id, "test.test_tool")
        self.assertIsNone(self.registry.get_spec("nonexistent"))

    def test_unregister(self) -> None:
        self.registry.register(_dummy_spec(), _echo_handler)
        self.assertTrue(self.registry.unregister("test_tool"))
        self.assertFalse(self.registry.has("test_tool"))
        self.assertEqual(len(self.registry.list_tools()), 0)
        # Unregistering again returns False
        self.assertFalse(self.registry.unregister("test_tool"))

    def test_openai_tools_array(self) -> None:
        self.registry.register(_dummy_spec("tool_a"), _echo_handler)
        self.registry.register(_dummy_spec("tool_b"), _echo_handler)

        tools = self.registry.openai_tools()
        self.assertEqual(len(tools), 2)
        names = {t["function"]["name"] for t in tools}
        self.assertEqual(names, {"tool_a", "tool_b"})
        # Check Sarvam-compatible shape
        for t in tools:
            self.assertEqual(t["type"], "function")
            self.assertIn("parameters", t["function"])

    def test_openai_tools_excludes_disabled(self) -> None:
        self.registry.register(_dummy_spec("enabled_tool", enabled=True), _echo_handler)
        self.registry.register(_dummy_spec("disabled_tool", enabled=False), _echo_handler)

        enabled = self.registry.openai_tools(enabled_only=True)
        self.assertEqual(len(enabled), 1)
        self.assertEqual(enabled[0]["function"]["name"], "enabled_tool")

        all_tools = self.registry.openai_tools(enabled_only=False)
        self.assertEqual(len(all_tools), 2)

    def test_execute_known_tool(self) -> None:
        self.registry.register(_dummy_spec(), _echo_handler)
        result = asyncio.run(
            self.registry.execute("test_tool", self.ctx, {"input": "hello"})
        )
        self.assertTrue(result.success)
        self.assertEqual(result.content, "Echo: hello")

    def test_execute_unknown_tool(self) -> None:
        result = asyncio.run(
            self.registry.execute("nonexistent", self.ctx, {})
        )
        self.assertFalse(result.success)
        self.assertIn("Unknown tool", result.error)

    def test_execute_disabled_tool(self) -> None:
        self.registry.register(_dummy_spec(enabled=False), _echo_handler)
        result = asyncio.run(
            self.registry.execute("test_tool", self.ctx, {"input": "hello"})
        )
        self.assertFalse(result.success)
        self.assertIn("disabled", result.error)

    def test_handler_exception_caught(self) -> None:
        """Handler that raises should produce error ToolResult, not crash."""
        spec = _dummy_spec("boom")
        self.registry.register(spec, _boom_handler)
        result = asyncio.run(
            self.registry.execute("boom", self.ctx, {})
        )
        self.assertFalse(result.success)
        self.assertIn("RuntimeError", result.error)
        self.assertIn("Intentional explosion", result.error)

    def test_re_register_overwrites(self) -> None:
        async def handler_v1(ctx, args):
            return ToolResult(tool_id="test.test_tool", success=True, content="v1")

        async def handler_v2(ctx, args):
            return ToolResult(tool_id="test.test_tool", success=True, content="v2")

        spec = _dummy_spec()
        self.registry.register(spec, handler_v1)
        self.registry.register(spec, handler_v2)

        result = asyncio.run(
            self.registry.execute("test_tool", self.ctx, {})
        )
        self.assertEqual(result.content, "v2")
        self.assertEqual(len(self.registry.list_tools()), 1)


if __name__ == "__main__":
    unittest.main()
