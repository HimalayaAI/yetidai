"""Tests for tools/osint/plugin.py — OSINT plugin registration and handler."""
import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from core.tool_contracts import ToolContext, ToolResult
from core.tool_registry import ToolRegistry

from tools.osint.plugin import OSINT_SPEC, handle_osint, register


class OsintPluginSpecTests(unittest.TestCase):
    """Verify the OSINT tool spec is well-formed."""

    def test_spec_has_correct_id(self) -> None:
        self.assertEqual(OSINT_SPEC.tool_id, "osint.nepal.live_context")

    def test_spec_has_correct_name(self) -> None:
        self.assertEqual(OSINT_SPEC.name, "get_nepal_live_context")

    def test_spec_generates_valid_openai_tool(self) -> None:
        tool = OSINT_SPEC.to_openai_tool()
        self.assertEqual(tool["type"], "function")
        func = tool["function"]
        self.assertEqual(func["name"], "get_nepal_live_context")
        self.assertIn("focus", func["parameters"]["properties"])
        self.assertEqual(func["parameters"]["required"], ["focus"])
        self.assertFalse(func["parameters"]["additionalProperties"])

    def test_spec_is_enabled(self) -> None:
        self.assertTrue(OSINT_SPEC.enabled)


class OsintPluginRegistrationTests(unittest.TestCase):
    """Verify the plugin registers correctly with a ToolRegistry."""

    def test_register_adds_to_registry(self) -> None:
        registry = ToolRegistry()

        # Patch get_registry to return our test registry
        with patch("tools.osint.plugin.get_registry", return_value=registry):
            register()

        self.assertTrue(registry.has("get_nepal_live_context"))
        spec = registry.get_spec("get_nepal_live_context")
        self.assertEqual(spec.tool_id, "osint.nepal.live_context")

    def test_openai_tools_includes_osint(self) -> None:
        registry = ToolRegistry()

        with patch("tools.osint.plugin.get_registry", return_value=registry):
            register()

        tools = registry.openai_tools()
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["function"]["name"], "get_nepal_live_context")


class OsintPluginHandlerTests(unittest.TestCase):
    """Test handle_osint with mocked OSINT internals."""

    def test_handler_returns_context_on_success(self) -> None:
        ctx = ToolContext(query="inflation update", llm_client=None)

        mock_plan = AsyncMock()
        mock_plan.return_value.use_nepalosint = True

        mock_bundle = AsyncMock(return_value={
            "payloads": {"economy_snapshot": {"sections": {}}},
            "errors": {},
        })

        mock_brief = "Macro snapshot\n- Inflation: 5.20%"

        with (
            patch("tools.osint.plugin.resolve_route_plan", mock_plan),
            patch("tools.osint.plugin.fetch_context_bundle", mock_bundle),
            patch("tools.osint.plugin.build_context_brief", return_value=mock_brief),
        ):
            result = asyncio.run(handle_osint(ctx, {"focus": "inflation"}))

        self.assertIsInstance(result, ToolResult)
        self.assertTrue(result.success)
        self.assertEqual(result.content, mock_brief)
        self.assertEqual(result.tool_id, "osint.nepal.live_context")

    def test_handler_returns_none_content_when_not_needed(self) -> None:
        ctx = ToolContext(query="hello namaste", llm_client=None)

        mock_plan = AsyncMock()
        mock_plan.return_value.use_nepalosint = False

        with patch("tools.osint.plugin.resolve_route_plan", mock_plan):
            result = asyncio.run(handle_osint(ctx, {"focus": "greeting"}))

        self.assertTrue(result.success)
        self.assertIsNone(result.content)

    def test_handler_returns_error_on_exception(self) -> None:
        ctx = ToolContext(query="test", llm_client=None)

        with patch(
            "tools.osint.plugin.resolve_route_plan",
            side_effect=ConnectionError("Cannot reach API"),
        ):
            result = asyncio.run(handle_osint(ctx, {"focus": "test"}))

        self.assertFalse(result.success)
        self.assertIn("ConnectionError", result.error)
        self.assertIn("Cannot reach API", result.error)

    def test_handler_via_registry_execute(self) -> None:
        """End-to-end: register plugin, execute via registry."""
        registry = ToolRegistry()

        with patch("tools.osint.plugin.get_registry", return_value=registry):
            register()

        ctx = ToolContext(query="debt update", llm_client=None)

        mock_plan = AsyncMock()
        mock_plan.return_value.use_nepalosint = True

        mock_bundle = AsyncMock(return_value={
            "payloads": {"debt_clock": {"debt_now_npr": 2500000000000}},
            "errors": {},
        })

        with (
            patch("tools.osint.plugin.resolve_route_plan", mock_plan),
            patch("tools.osint.plugin.fetch_context_bundle", mock_bundle),
            patch("tools.osint.plugin.build_context_brief", return_value="Debt: NPR 2.5T"),
        ):
            result = asyncio.run(
                registry.execute("get_nepal_live_context", ctx, {"focus": "debt"})
            )

        self.assertTrue(result.success)
        self.assertEqual(result.content, "Debt: NPR 2.5T")


if __name__ == "__main__":
    unittest.main()
