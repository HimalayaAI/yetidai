"""
core/tool_contracts.py — Pydantic models for the YetiDai tool-calling framework.

These models mirror the OpenAI / Sarvam function-calling schema so the
registry can auto-generate the `tools` array for chat.completions().

Reference: tests/test_sarvam_tool_calling_live.py for the wire format.
"""
from __future__ import annotations

import json
from enum import Enum
from typing import Any, Callable, Awaitable

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────


class ToolCategory(str, Enum):
    """Logical grouping for tools."""
    OSINT = "osint"
    AUTOMATION = "automation"       # n8n, webhooks, cron triggers
    DATA = "data"                   # databases, live feeds
    UTILITY = "utility"             # calculators, converters


# ── Tool definition models ───────────────────────────────────────


class ToolParam(BaseModel):
    """One parameter inside a tool's JSON-Schema `properties` block."""
    name: str
    type: str = "string"            # JSON-Schema type
    description: str = ""
    required: bool = True
    enum: list[str] | None = None   # constrained values
    examples: list[str] | None = None  # illustrative values (shown to LLM)


class ToolSpec(BaseModel):
    """
    Declarative spec for a single tool.

    Generates the exact dict that Sarvam/OpenAI expect in the `tools`
    array:

        {
            "type": "function",
            "function": {
                "name": "...",
                "description": "...",
                "parameters": {
                    "type": "object",
                    "properties": {...},
                    "required": [...],
                    "additionalProperties": false
                }
            }
        }
    """
    tool_id: str = Field(
        ...,
        description="Namespaced identifier, e.g. 'osint.nepal.live_context'.",
    )
    name: str = Field(
        ...,
        description="Function name the LLM will use in tool_calls.",
    )
    description: str = Field(
        ...,
        description="Natural-language description shown to the LLM.",
    )
    category: ToolCategory
    parameters: list[ToolParam] = Field(default_factory=list)
    enabled: bool = True
    # Per-tool wall-clock timeout. None → use the global default in bot.py.
    # Slow fan-out tools (aggregators, scrapers) should set this higher than
    # the default so a few slow endpoints don't cascade into a loop-level
    # failure. Fast local lookups can pin it lower for faster UX.
    timeout_seconds: float | None = None

    def to_openai_tool(self) -> dict[str, Any]:
        """
        Convert to the dict Sarvam expects in the `tools` array.

        Matches the schema used in test_sarvam_tool_calling_live.py:
            {"type": "function", "function": {"name": ..., "parameters": {...}}}
        """
        properties: dict[str, Any] = {}
        required: list[str] = []

        for p in self.parameters:
            prop: dict[str, Any] = {
                "type": p.type,
                "description": p.description,
            }
            if p.enum is not None:
                prop["enum"] = p.enum
            if p.examples is not None:
                prop["examples"] = p.examples
            properties[p.name] = prop
            if p.required:
                required.append(p.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }


# ── Execution models ─────────────────────────────────────────────


class ToolResult(BaseModel):
    """Uniform envelope returned by every tool handler."""
    tool_id: str
    success: bool
    content: str | None = None          # formatted text for the LLM
    raw_data: dict[str, Any] | None = None
    error: str | None = None
    # When True, bot.py should automatically execute the named fallback tool
    # with `fallback_args` as an extra tool call in the same turn, without
    # asking the LLM. Used for "OSINT returned nothing useful → try web".
    trigger_fallback: bool = False
    fallback_tool: str | None = None
    fallback_args: dict[str, Any] | None = None
    # Structured metadata for logging (endpoints hit, partial flags, etc.)
    meta: dict[str, Any] | None = None

    def to_tool_message(self, tool_call_id: str) -> dict[str, str]:
        """
        Format as a `role: tool` message for the Sarvam chat API.

        Wire format (from test_sarvam_tool_calling_live.py):
            {
                "role": "tool",
                "tool_call_id": "<id>",
                "content": "<json or text>"
            }
        """
        if self.success and self.content:
            body = self.content
        elif self.error:
            body = json.dumps({"error": self.error}, ensure_ascii=False)
        else:
            body = json.dumps({"message": "No data returned."})
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": body,
        }


class ToolContext(BaseModel):
    """
    Runtime context injected into every tool handler.

    Carries the user query, conversation history, and references
    the tool might need (like the LLM client for sub-calls).
    """
    query: str
    history: list[Any] | None = None
    llm_client: Any = None
    channel_id: int | None = None
    user_id: int | None = None

    model_config = {"arbitrary_types_allowed": True}


# ── Handler type alias ───────────────────────────────────────────

ToolHandler = Callable[[ToolContext, dict[str, Any]], Awaitable[ToolResult]]
