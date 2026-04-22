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
        "PRIMARY tool for ANY Nepal-related question. Fetches live, authoritative "
        "Nepal public-information context from NepalOSINT across multiple "
        "endpoints, auto-routed from the user's query.\n\n"
        "CAPABILITIES (all composed from one call):\n"
        "  • Macro / NRB: inflation, remittance, FX reserves, trade balance, "
        "imports/exports, tourism arrivals, migrant permits, money supply, "
        "T-bill yields, interbank, bank rate.\n"
        "  • NEPSE / markets: index, stock news, IPOs, dividends, bonus/right "
        "shares, sector-specific mentions (hydro, banking, insurance).\n"
        "  • Public debt: real-time debt clock (total / external / domestic, "
        "debt/GDP), IMF/WB/ADB financing news.\n"
        "  • Government: latest cabinet decisions, ministry announcements, "
        "official notices, PM/minister actions.\n"
        "  • Federal Parliament: session verbatim summaries, bill tracker with "
        "status, House of Representatives & National Assembly activity.\n"
        "  • General news: consolidated recent stories (last 24h), unified "
        "cross-source search, semantic/embedding similarity search, social "
        "signals from tracked handles.\n"
        "  • HISTORICAL date-range queries: include any date or relative phrase "
        "in the user query (routed automatically, not in `focus`) — the tool "
        "detects ISO dates ('2026-02-28'), English months ('feb 28', "
        "'28 february 2026'), Devanagari ('फेब्रुअरी २८'), relative Nepali "
        "('हिजो', 'गत हप्ता', 'अघिल्लो महिना'), and romanized ('hijo', "
        "'gaeko hapta').\n\n"
        "USE FOR: every question about Nepal — current facts, current people "
        "(ministers, PM, officials), economic data, markets, government, "
        "parliament, news (today or past dates). Your training data is STALE; "
        "always try this tool first for Nepal topics.\n\n"
        "DO NOT USE FOR: purely non-Nepal world topics (world cup, foreign "
        "CEOs, celebrities from other countries). For those, use "
        "`internet_search`."
    ),
    category=ToolCategory.OSINT,
    parameters=[
        ToolParam(
            name="focus",
            type="string",
            description=(
                "Topic keyword that seeds the auto-router. Use one of the "
                "canonical values below when it fits, else a concise phrase in "
                "English or Nepali. Do NOT put dates here — put them in the "
                "user's original message or leave them implicit.\n"
                "Canonical values:\n"
                "  inflation, remittance, reserves, trade, tourism — macro\n"
                "  nepse, ipo, dividend — markets\n"
                "  debt, external_debt — debt\n"
                "  government_decisions, announcements — govt\n"
                "  parliament, bills, verbatim — parliament\n"
                "  general_news — catch-all for 'what happened' questions\n"
                "  who_is:<role> — identity lookups (e.g. 'who_is:finance_minister')"
            ),
            required=True,
            examples=[
                "inflation",
                "remittance",
                "reserves",
                "trade",
                "nepse",
                "debt",
                "parliament",
                "bills",
                "government_decisions",
                "announcements",
                "general_news",
                "who_is:prime_minister",
                "who_is:finance_minister",
            ],
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

        payloads = context_bundle.get("payloads") or {}
        errors = context_bundle.get("errors") or {}
        meta = {
            "endpoints_ok": sorted(payloads.keys()),
            "endpoints_failed": sorted(errors.keys()),
            "intents": list(route_plan.intents),
            "is_who_is": bool(route_plan.is_who_is),
            "wants_history": bool(route_plan.wants_history),
            "history_range": [route_plan.history_start_date, route_plan.history_end_date],
            "cache_hits": getattr(osint, "cache_hits", 0),
            "cache_misses": getattr(osint, "cache_misses", 0),
        }

        # Auto-fallback to internet_search when OSINT yielded no useful
        # context. The bot layer will execute the fallback tool in the same
        # turn without asking the LLM.
        fetch_failed = (
            context_message is None
            or (isinstance(context_message, str) and context_message.startswith("[NEPALOSINT_FETCH_FAILED]"))
        )
        if fetch_failed:
            return ToolResult(
                tool_id=OSINT_SPEC.tool_id,
                success=True,
                content=context_message
                or "[NEPALOSINT_NO_MATCH] NepalOSINT had no matching context; falling back to web search.",
                raw_data=payloads or None,
                trigger_fallback=True,
                fallback_tool="internet_search",
                fallback_args={"query": ctx.query},
                meta=meta,
            )

        return ToolResult(
            tool_id=OSINT_SPEC.tool_id,
            success=True,
            content=context_message,
            raw_data=payloads or None,
            meta=meta,
        )

    except Exception as exc:
        # Network / auth / parse error → surface, but also hint fallback so
        # bot.py can still give the user a useful answer.
        return ToolResult(
            tool_id=OSINT_SPEC.tool_id,
            success=False,
            error=f"{type(exc).__name__}: {exc}",
            trigger_fallback=True,
            fallback_tool="internet_search",
            fallback_args={"query": ctx.query},
        )


# ── Registration ──────────────────────────────────────────────────

def register() -> None:
    """Register the OSINT tool with the global ToolRegistry."""
    get_registry().register(OSINT_SPEC, handle_osint)
