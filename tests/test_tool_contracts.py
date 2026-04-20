"""Tests for core/tool_contracts.py — Pydantic models."""
import json
import unittest

from core.tool_contracts import (
    ToolCategory,
    ToolParam,
    ToolSpec,
    ToolResult,
    ToolContext,
)


class ToolParamTests(unittest.TestCase):
    def test_defaults(self) -> None:
        p = ToolParam(name="query", description="search query")
        self.assertEqual(p.type, "string")
        self.assertTrue(p.required)
        self.assertIsNone(p.enum)

    def test_enum_values(self) -> None:
        p = ToolParam(
            name="category",
            type="string",
            description="content category",
            enum=["news", "macro", "debt"],
        )
        self.assertEqual(p.enum, ["news", "macro", "debt"])


class ToolSpecTests(unittest.TestCase):
    def _make_spec(self) -> ToolSpec:
        return ToolSpec(
            tool_id="osint.nepal.live_context",
            name="get_nepal_live_context",
            description="Fetch live Nepal context.",
            category=ToolCategory.OSINT,
            parameters=[
                ToolParam(name="focus", type="string", description="Topic focus.", required=True),
                ToolParam(name="limit", type="number", description="Max items.", required=False),
            ],
        )

    def test_to_openai_tool_shape(self) -> None:
        """Must match the schema Sarvam expects (see test_sarvam_tool_calling_live.py)."""
        tool = self._make_spec().to_openai_tool()

        self.assertEqual(tool["type"], "function")
        func = tool["function"]
        self.assertEqual(func["name"], "get_nepal_live_context")
        self.assertIn("parameters", func)

        params = func["parameters"]
        self.assertEqual(params["type"], "object")
        self.assertFalse(params["additionalProperties"])
        self.assertIn("focus", params["properties"])
        self.assertIn("limit", params["properties"])
        # Only "focus" is required
        self.assertEqual(params["required"], ["focus"])

    def test_to_openai_tool_enum_serialized(self) -> None:
        spec = ToolSpec(
            tool_id="test.enum",
            name="test_enum",
            description="test",
            category=ToolCategory.UTILITY,
            parameters=[
                ToolParam(name="mode", type="string", description="mode", enum=["fast", "slow"]),
            ],
        )
        tool = spec.to_openai_tool()
        self.assertEqual(tool["function"]["parameters"]["properties"]["mode"]["enum"], ["fast", "slow"])

    def test_empty_parameters(self) -> None:
        spec = ToolSpec(
            tool_id="test.noparam",
            name="no_params",
            description="tool with no params",
            category=ToolCategory.DATA,
        )
        tool = spec.to_openai_tool()
        self.assertEqual(tool["function"]["parameters"]["properties"], {})
        self.assertEqual(tool["function"]["parameters"]["required"], [])


class ToolResultTests(unittest.TestCase):
    def test_to_tool_message_success(self) -> None:
        result = ToolResult(
            tool_id="osint.nepal.live_context",
            success=True,
            content="Inflation: 5.20%",
        )
        msg = result.to_tool_message("call_123")
        self.assertEqual(msg["role"], "tool")
        self.assertEqual(msg["tool_call_id"], "call_123")
        self.assertEqual(msg["content"], "Inflation: 5.20%")

    def test_to_tool_message_error(self) -> None:
        result = ToolResult(
            tool_id="osint.nepal.live_context",
            success=False,
            error="Timeout connecting to NepalOSINT",
        )
        msg = result.to_tool_message("call_456")
        self.assertEqual(msg["role"], "tool")
        parsed = json.loads(msg["content"])
        self.assertIn("error", parsed)

    def test_to_tool_message_no_content(self) -> None:
        result = ToolResult(tool_id="test", success=True)
        msg = result.to_tool_message("call_789")
        parsed = json.loads(msg["content"])
        self.assertIn("message", parsed)


class ToolContextTests(unittest.TestCase):
    def test_accepts_arbitrary_llm_client(self) -> None:
        ctx = ToolContext(query="test", llm_client=object())
        self.assertEqual(ctx.query, "test")
        self.assertIsNotNone(ctx.llm_client)

    def test_optional_fields_default_none(self) -> None:
        ctx = ToolContext(query="hello")
        self.assertIsNone(ctx.history)
        self.assertIsNone(ctx.channel_id)
        self.assertIsNone(ctx.user_id)


if __name__ == "__main__":
    unittest.main()
