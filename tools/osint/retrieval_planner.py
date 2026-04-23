from __future__ import annotations

import json
import os
import re
import asyncio
from typing import Any

from tools.osint.context_router import (
    ALLOWED_INTENTS as _ROUTER_ALLOWED_INTENTS,
    RoutePlan,
    plan_from_intent,
    route_query,
)


# Accept both the canonical six plus the two shortcuts the OSINT tool
# exposes directly to the LLM (`who_is`, `history`). `plan_from_intent`
# validates and materialises those into a RoutePlan.
ALLOWED_INTENTS = set(_ROUTER_ALLOWED_INTENTS)
HINTED_INTENT_VALUES = ALLOWED_INTENTS | {"who_is", "history"}

FOLLOW_UP_MARKERS = (
    "यसले",
    "यसमा",
    "यो",
    "त्यो",
    "esle",
    "yesle",
    "tesle",
    "yo",
    "tyo",
    "this",
    "that",
    "it",
    "they",
    "those",
    "effect",
    "impact",
)


# Small-talk / greeting markers. When a query matches and doesn't name a
# Nepal entity, we short-circuit to "no OSINT needed" so simple chats
# like "k xa" / "halkhabar" don't waste an endpoint fan-out.
#
# The observed failure mode: user says "tapai ko halkhabar" (what's up).
# The word "khabar" literally means "news", so the keyword router would
# happily classify it as general_news and burn 1-2 OSINT calls returning
# data nobody wanted. This filter fixes that.
_SMALLTALK_MARKERS: tuple[str, ...] = (
    # Devanagari greetings / small-talk
    "नमस्ते", "नमस्कार", "के छ", "के छौ", "के गर्दै",
    # Romanized Nepali
    "k xa", "k cha", "ke xa", "ke cha", "halkhabar", "halchal",
    "ke garne", "ke garnu", "ke garera", "sanchai",
    "kasto xa", "kasto chha", "kasto cha",
    # English greetings
    "hi", "hey", "hello", "yo", "sup", "what's up", "whats up", "wassup",
    "howdy", "good morning", "good evening",
    # Identity probes
    "who are you", "तिमी को", "तपाईं को", "tapai ko ho", "tapai k ho",
)

# Tokens that clearly indicate a Nepal-specific request even inside a
# short message — if any of these appear, we do NOT short-circuit.
_NEPAL_ENTITY_HINTS: tuple[str, ...] = (
    "nepal", "नेपाल", "nrb", "nepse", "pdmo", "kathmandu", "काठमाडौं",
    "parliament", "संसद", "cabinet", "मन्त्रिपरिषद्", "minister", "मन्त्री",
    "inflation", "मुद्रास्फीति", "remittance", "रेमिट्यान्स",
    "ipo", "share", "शेयर", "सेयर", "pm", "प्रधानमन्त्री",
)


_SHORT_MARKER_LEN = 4
_SMALLTALK_WORD_CACHE: dict[str, "re.Pattern[str]"] = {}


def _marker_matches(text: str, marker: str) -> bool:
    """Word-boundary match for short ASCII markers (so "hi" doesn't match
    inside "this"), substring for longer strings and Devanagari."""
    if marker.isascii() and len(marker) <= _SHORT_MARKER_LEN:
        pat = _SMALLTALK_WORD_CACHE.get(marker)
        if pat is None:
            pat = re.compile(rf"\b{re.escape(marker)}\b", flags=re.IGNORECASE)
            _SMALLTALK_WORD_CACHE[marker] = pat
        return pat.search(text) is not None
    return marker in text


def _is_smalltalk(query: str) -> bool:
    """True when the query reads like greeting / chit-chat and does NOT
    name a Nepal entity that needs OSINT."""
    if not query:
        return False
    normalized = query.lower().strip()
    # Trivially short — almost certainly small talk.
    if len(normalized) <= 2:
        return True
    # Long messages are unlikely to be pure small talk.
    if len(normalized.split()) > 6:
        return False
    if any(_marker_matches(normalized, hint) for hint in _NEPAL_ENTITY_HINTS):
        return False
    return any(_marker_matches(normalized, marker) for marker in _SMALLTALK_MARKERS)


def _is_ambiguous(query: str, base_plan: RoutePlan) -> bool:
    normalized = query.lower().strip()
    specific_intents = [intent for intent in base_plan.intents if intent != "general_news"]
    has_followup_marker = any(marker in normalized for marker in FOLLOW_UP_MARKERS)
    is_short = len(normalized.split()) <= 5
    multiple_domains = len(specific_intents) > 1
    no_specific_domain = not specific_intents

    return has_followup_marker or multiple_domains or (no_specific_domain and is_short)


def _normalize_history_date(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return value
    return None


def _extract_json_blob(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _plan_from_payload(payload: dict[str, Any], fallback: RoutePlan) -> RoutePlan:
    intents = payload.get("intents") or fallback.intents
    if not isinstance(intents, list):
        intents = fallback.intents

    cleaned_intents: list[str] = []
    for intent in intents:
        if isinstance(intent, str) and intent in ALLOWED_INTENTS and intent not in cleaned_intents:
            cleaned_intents.append(intent)

    if not cleaned_intents:
        cleaned_intents = fallback.intents

    wants_history = bool(payload.get("wants_history", fallback.wants_history))
    history_start_date = _normalize_history_date(payload.get("history_start_date")) or fallback.history_start_date
    history_end_date = _normalize_history_date(payload.get("history_end_date")) or fallback.history_end_date
    history_category = payload.get("history_category")
    if history_category not in {"economic", "political", "security", "social", None}:
        history_category = fallback.history_category

    use_nepalosint = bool(payload.get("use_nepalosint", fallback.use_nepalosint))
    if cleaned_intents:
        use_nepalosint = True

    return RoutePlan(
        use_nepalosint=use_nepalosint,
        intents=cleaned_intents,
        wants_history=wants_history,
        history_start_date=history_start_date,
        history_end_date=history_end_date,
        history_category=history_category or fallback.history_category,
    )


async def resolve_route_plan(
    llm_client: Any,
    query: str,
    previous_messages: list[Any] | None = None,
    *,
    hinted_intent: str | None = None,
) -> RoutePlan:
    """Decide which NepalOSINT endpoints to hit for `query`.

    Decision order:
      1. `hinted_intent` — if the LLM passed an explicit enum value via the
         tool arguments, honour it and skip the planner round-trip entirely.
         This is the fast path: 0 extra LLM calls.
      2. `route_query` keyword rules — deterministic, local, 0 LLM calls.
         Used when the intent is unambiguous from keywords alone.
      3. LLM planner (Sarvam) — only for ambiguous short queries or
         multi-domain queries. One extra completion call, ~1.5 s budget,
         with fall-through to the keyword base plan on any failure.

    Parameters
    ----------
    hinted_intent : str, optional
        One of `ALLOWED_INTENTS` or the shortcuts `who_is` / `history`.
        Typically supplied by the OSINT tool from the `intent` tool
        parameter. Invalid values are silently ignored (fall through to
        keyword path) so a hallucinated enum value doesn't break routing.
    """
    # Fast path — LLM already told us what it wants.
    if hinted_intent:
        hinted_plan = plan_from_intent(hinted_intent, query)
        if hinted_plan is not None:
            return hinted_plan

    # Small-talk short-circuit: for greetings / chit-chat with no Nepal
    # entity in sight, return a plan that skips OSINT entirely. Saves the
    # planner round-trip and the endpoint fan-out.
    if _is_smalltalk(query):
        return RoutePlan(use_nepalosint=False)

    base_plan = route_query(query)
    if not _is_ambiguous(query, base_plan):
        return base_plan

    history_lines: list[str] = []
    if previous_messages:
        for message in previous_messages[-3:]:
            content = getattr(message, "content", "") or ""
            author = getattr(getattr(message, "author", None), "name", "user")
            if content.strip():
                history_lines.append(f"{author}: {content.strip()}")

    planner_prompt = (
        "तिमी retrieval planner हौ। प्रयोगकर्ताको सन्देश र सानो वार्तालाप सन्दर्भ हेरेर "
        "NepalOSINT बाट live context चाहिन्छ कि चाहिँदैन भनेर निर्णय गर। "
        "केवल JSON फर्काऊ। कुनै व्याख्या नलेख।\n\n"
        "Rule:\n"
        "- current Nepal news, macro, government decisions, debt, parliament, market/ticker/company, official announcements => use_nepalosint=true\n"
        "- chit-chat, timeless general knowledge, greeting => use_nepalosint=false\n"
        "- intents values must be from: general_news, macro, government, debt, parliament, trading\n"
        "- wants_history=true only if the user explicitly asks about yesterday/last week/previous/history/date-range\n"
        "- if uncertain but the user seems to ask about what is happening now in Nepal, prefer use_nepalosint=true\n\n"
        "Return JSON with keys:\n"
        "{"
        "\"use_nepalosint\": true|false, "
        "\"intents\": [\"general_news\"], "
        "\"wants_history\": true|false, "
        "\"history_start_date\": \"YYYY-MM-DD or null\", "
        "\"history_end_date\": \"YYYY-MM-DD or null\", "
        "\"history_category\": \"economic|political|security|social|null\""
        "}"
    )

    planner_messages = [
        {"role": "system", "content": planner_prompt},
    ]
    if history_lines:
        planner_messages.append(
            {
                "role": "user",
                "content": "Recent conversation:\n" + "\n".join(history_lines),
            }
        )
    planner_messages.append({"role": "user", "content": f"Current user query:\n{query}"})

    try:
        timeout_seconds = float(os.getenv("SARVAM_ROUTER_TIMEOUT_SECONDS", "1.5"))
        response = await asyncio.wait_for(
            llm_client.chat.completions(
                model=os.getenv("SARVAM_ROUTER_MODEL", "sarvam-30b"),
                messages=planner_messages,
            ),
            timeout=timeout_seconds,
        )
        planner_text = ""
        if hasattr(response, "choices") and response.choices:
            planner_text = response.choices[0].message.content or ""
        payload = _extract_json_blob(planner_text)
        if not payload:
            return base_plan
        return _plan_from_payload(payload, base_plan)
    except Exception:
        return base_plan
