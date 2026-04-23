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
from tools.osint.freshness import assess_freshness


# Emitted in the tool message when the user asked for "today" but the
# newest OSINT payload is older than the freshness threshold. The bot
# loop interprets this as non-real content so the model cannot hide
# behind a stale citation.
STALE_DATA_MARKER = "[STALE_DATA]"


def _nepal_scoped_query(raw: str) -> str:
    """Build a Nepal-scoped query for the internet_search fallback.

    Without this, a bare "aja ko samachar" / "आजको news" fallback query
    on DuckDuckGo returns Hindi-language Indian news portals (aajtak.in,
    indiatv.in, amarujala.com — observed in production). Prepending
    "Nepal" biases the SERP toward Nepal-domain sources.
    """
    q = (raw or "").strip()
    if not q:
        return "Nepal news today"
    lowered = q.lower()
    if "nepal" in lowered or "नेपाल" in q:
        return q
    return f"Nepal {q}"


# ── Tool specification ────────────────────────────────────────────

OSINT_SPEC = ToolSpec(
    tool_id="osint.nepal.live_context",
    name="get_nepal_live_context",
    description=(
        "PRIMARY tool for ANY Nepal-related question. Fetches live, "
        "authoritative Nepal public-information context from NepalOSINT "
        "across multiple endpoints.\n\n"
        "HOW TO USE — pick one `intent` value so the tool targets the "
        "right endpoints directly (faster, cheaper, no internal router):\n"
        "  • macro         — NRB / economy: inflation, remittance, FX "
        "reserves, trade balance, imports/exports, tourism, migrant permits, "
        "money supply, T-bill yields, interbank, bank rate.\n"
        "  • trading       — NEPSE / markets: index, stocks, IPOs, "
        "dividends, bonus/right shares, sector mentions (hydro, banking, "
        "insurance). Pass a ticker (e.g. `NABIL`, `UPPER`) or sector in "
        "`focus`.\n"
        "  • debt          — Public debt clock, external/domestic debt, "
        "IMF/WB/ADB financing.\n"
        "  • government    — Cabinet decisions, ministry announcements, "
        "official notices, PM/minister actions.\n"
        "  • parliament    — Session verbatim summaries, bill tracker, "
        "HoR & National Assembly.\n"
        "  • general_news  — Consolidated recent stories (last 24 h), "
        "cross-source search. Use when the user asks 'what happened' "
        "without naming a domain.\n"
        "  • who_is        — Identity lookup for Nepal officials (finance "
        "minister, PM, secretary, …). Put the role in `focus`.\n"
        "  • history       — Date-range query. Include the date phrase in "
        "`focus` OR leave it in the user's original message — the tool "
        "parses ISO ('2026-02-28'), English ('feb 28'), Devanagari "
        "('फेब्रुअरी २८'), relative Nepali ('हिजो', 'गत हप्ता'), and "
        "romanized ('hijo', 'gaeko hapta').\n\n"
        "PARALLEL FAN-OUT: if the user asks about two unrelated Nepal "
        "domains (e.g. 'inflation and NEPSE today'), call this tool TWICE "
        "in parallel with different `intent` values rather than relying on "
        "the catch-all router.\n\n"
        "DEFAULT: your training data is stale. Always try this tool first "
        "for Nepal topics before answering from memory.\n\n"
        "DO NOT USE FOR: non-Nepal world topics (foreign heads of state, "
        "international sports, global celebrities). Use `internet_search` "
        "for those. For reading a specific URL cited in OSINT output, use "
        "`fetch_url`."
    ),
    category=ToolCategory.OSINT,
    parameters=[
        ToolParam(
            name="intent",
            type="string",
            description=(
                "Pick ONE of the canonical intents. When you set this, the "
                "tool skips its internal router LLM call and goes straight "
                "to the matching NepalOSINT endpoints — faster and cheaper. "
                "Omit ONLY when you genuinely cannot classify the query."
            ),
            required=False,
            enum=[
                "macro",
                "trading",
                "debt",
                "government",
                "parliament",
                "general_news",
                "who_is",
                "history",
            ],
            examples=[
                "macro",
                "trading",
                "government",
                "who_is",
                "history",
            ],
        ),
        ToolParam(
            name="focus",
            type="string",
            description=(
                "Short hint for the endpoints — a keyword, ticker, or role. "
                "Examples: 'inflation', 'NABIL', 'finance_minister', "
                "'cabinet'. Optional when `intent` is specific enough by "
                "itself. Do NOT dump the full user message here."
            ),
            required=False,
            examples=[
                "inflation",
                "remittance",
                "NABIL",
                "UPPER",
                "finance_minister",
                "prime_minister",
                "cabinet_decisions",
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
    hinted_intent = None
    raw_intent = arguments.get("intent")
    if isinstance(raw_intent, str) and raw_intent.strip():
        hinted_intent = raw_intent.strip().lower()

    try:
        route_plan = await resolve_route_plan(
            ctx.llm_client, ctx.query, ctx.history,
            hinted_intent=hinted_intent,
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

        # Freshness check: if the user asked about "today"/"aja" but the
        # newest story in the bundle is days old, annotate the tool result
        # and trigger a web-search fallback in the same turn. This is the
        # fix for the transcript where Yeti served 2022 COVID stories as
        # "April 22 news".
        freshness = assess_freshness(ctx.query, payloads)
        meta = {
            "endpoints_ok": sorted(payloads.keys()),
            "endpoints_failed": sorted(errors.keys()),
            "intents": list(route_plan.intents),
            "is_who_is": bool(route_plan.is_who_is),
            "wants_history": bool(route_plan.wants_history),
            "history_range": [route_plan.history_start_date, route_plan.history_end_date],
            "cache_hits": getattr(osint, "cache_hits", 0),
            "cache_misses": getattr(osint, "cache_misses", 0),
            "freshness": freshness,
        }

        if freshness.get("stale"):
            age = freshness.get("age_days")
            newest = freshness.get("newest")
            stale_header = (
                f"{STALE_DATA_MARKER} NepalOSINT's freshest story is "
                f"{age} days old ({newest}), but the user asked about a "
                f"current event. DO NOT present this as today's news — "
                f"say the data is stale and rely on the web-search "
                f"fallback that runs next.\n\n"
            )
            # Return a stale-flagged result and chain to internet_search.
            # The loop will surface the web result and the model can
            # decide which to cite.
            return ToolResult(
                tool_id=OSINT_SPEC.tool_id,
                success=True,
                content=stale_header + (context_message or ""),
                raw_data=payloads or None,
                trigger_fallback=True,
                fallback_tool="internet_search",
                fallback_args={"query": _nepal_scoped_query(ctx.query)},
                meta=meta,
            )

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
                fallback_args={"query": _nepal_scoped_query(ctx.query)},
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
            fallback_args={"query": _nepal_scoped_query(ctx.query)},
        )


# ── Registration ──────────────────────────────────────────────────

def register() -> None:
    """Register the OSINT tool with the global ToolRegistry."""
    get_registry().register(OSINT_SPEC, handle_osint)
