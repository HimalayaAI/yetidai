"""
tools/osint/plugin.py — Nepal OSINT plugin for YetiDai.

This is the ONLY module that bot.py imports from the OSINT package.
It registers itself with the global ToolRegistry so the LLM can
invoke it via Sarvam's native tool_choice mechanism.

Sarvam tool-call contract (see tests/test_sarvam_tool_calling_live.py):
    1. bot sends  tools=[{type: "function", function: {name, description, parameters}}]
    2. LLM returns finish_reason="tool_calls" with tool_calls[].function.{name, arguments}
    3. bot executes handler, sends back {role: "tool", tool_call_id, content}
    4. LLM produces final text answer
"""
from __future__ import annotations

from typing import Any

from core.tool_contracts import (
    ToolSpec,
    ToolParam,
    ToolCategory,
    ToolResult,
    ToolContext,
)
from core.tool_registry import get_registry

from tools.osint.nepalosint_client import NepalOSINTClient
from tools.osint.context_router import fetch_context_bundle
from tools.osint.context_formatter import build_context_brief
from tools.osint.retrieval_planner import resolve_route_plan


# ── Tool specification ────────────────────────────────────────────

OSINT_SPEC = ToolSpec(
    tool_id="osint.nepal.live_context",
    name="get_nepal_live_context",
    description=(
        "Fetch live Nepal public-information context from NepalOSINT: "
        "current news, NRB macroeconomic data (inflation, remittance, reserves, trade), "
        "government decisions, public debt, Federal Parliament updates, "
        "and NEPSE/stock-market data. "
        "Call this when the user asks about current events, economic indicators, "
        "government actions, or market conditions in Nepal."
    ),
    category=ToolCategory.OSINT,
    parameters=[
        ToolParam(
            name="focus",
            type="string",
            description="Brief description of the macro area or topic the user is asking about.",
            required=True,
        ),
    ],
)


# ── Singleton OSINT client ───────────────────────────────────────

_client: NepalOSINTClient | None = None


def _get_client() -> NepalOSINTClient:
    global _client
    if _client is None:
        _client = NepalOSINTClient()
    return _client


# ── Handler ───────────────────────────────────────────────────────

async def handle_osint(ctx: ToolContext, arguments: dict[str, Any]) -> ToolResult:
    """
    Async handler invoked by the ToolRegistry when the LLM calls
    `get_nepal_live_context`.

    Parameters
    ----------
    ctx : ToolContext
        Runtime context (query, history, llm_client, etc.).
    arguments : dict
        Arguments from the LLM's tool_call (e.g. {"focus": "inflation"}).
    """
    osint = _get_client()

    try:
        route_plan = await resolve_route_plan(
            ctx.llm_client, ctx.query, ctx.history,
        )

        if not route_plan.use_nepalosint:
            return ToolResult(
                tool_id=OSINT_SPEC.tool_id,
                success=True,
                content=None,
                raw_data=None,
            )

        context_bundle = await fetch_context_bundle(osint, ctx.query, route_plan)
        context_message = build_context_brief(context_bundle, max_chars=1800)

        return ToolResult(
            tool_id=OSINT_SPEC.tool_id,
            success=True,
            content=context_message,
            raw_data=context_bundle.get("payloads"),
        )

    except Exception as exc:
        return ToolResult(
            tool_id=OSINT_SPEC.tool_id,
            success=False,
            error=f"{type(exc).__name__}: {exc}",
        )


# ── Registration ──────────────────────────────────────────────────

def register() -> None:
    """Register the OSINT tool with the global ToolRegistry."""
    get_registry().register(OSINT_SPEC, handle_osint)
