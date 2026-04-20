from __future__ import annotations

import json
import os
import re
import asyncio
from typing import Any

from tools.osint.context_router import RoutePlan, route_query


ALLOWED_INTENTS = {"general_news", "macro", "government", "debt", "parliament", "trading"}

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
) -> RoutePlan:
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
