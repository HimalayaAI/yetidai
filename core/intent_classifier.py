"""
core/intent_classifier.py — LLM-driven pre-pass routing classifier.

Why this exists: the 10-prompt live test showed Sarvam-30b skipping tool
calls for canonical Bucket 2/3 questions (Modi's Nepal visits, Nepal's
PM). The systemPrompt's 6-bucket decision rubric is pure documentation;
bot.py relied on deterministic keyword gates (`needs_tool_use`) and an
empty-promise retry safety net that only fires AFTER the promise has
already been emitted to the user.

This module adds a proactive layer: one fast Sarvam call, strict-JSON
output, classifying the user message into (bucket, tools, osint_intent,
recency, confidence). The result drives:

  • Small-talk short-circuit — skip the main tool loop entirely.
  • High-confidence tool buckets — run the first round of the main loop
    with `tool_choice="required"` instead of "auto", forcing Sarvam to
    emit a tool call.
  • Routing hint — injected as a system message so the main turn can
    see the classifier's recommendation without being bound by it.

Fails open: any classifier error (timeout, bad JSON, network) returns
None and the caller falls back to the existing keyword path. The
classifier cannot BREAK routing — only improve it when it works.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

logger = logging.getLogger("yetidai.intent_classifier")


CLASSIFIER_TIMEOUT_SECONDS = 5.0
CLASSIFIER_CONFIDENCE_THRESHOLD = 0.7


# Prompt shape: minimal, strict-JSON, no prose. Kept separate from the
# main systemPrompt because this is a routing decision, not a generation
# decision. Changing the main prompt doesn't require touching this.
CLASSIFIER_PROMPT = """You are a strict-JSON routing classifier for the YetiDai Nepali Discord bot.

Read the user's latest message (in English, Nepali, Devanagari, or Romanized Nepali) and emit EXACTLY one JSON object on one line, no prose, no markdown fences. Schema:

{"bucket": 1-6, "tools_needed": [...], "osint_intent": "...|null", "recency_required": true|false, "confidence": 0.0-1.0}

Bucket legend:
  1 = Small talk / greeting / "who are you" ("k xa", "hello", "namaste"). No tools.
  2 = Nepal factual question (NRB, NEPSE, PM/minister, inflation, parliament, Nepal news). Tools: ["get_nepal_live_context"].
  3 = Non-Nepal world question (foreign leaders, global sports, world tech, non-Nepal trivia). Tools: ["internet_search"].
  4 = Follow a specific URL the user pasted (not a GitHub repo URL). Tools: ["fetch_url"].
  5 = GitHub repo analysis OR listing an account's repos. Tools: ["analyze_github_repo"] or ["list_github_repos"].
  6 = Parallel / composed (Nepal-vs-world comparison, README + commit, multiple Nepal intents). Tools: 2+ entries.

osint_intent (only when tools_needed includes get_nepal_live_context, else null):
  macro | trading | government | who_is | parliament | debt | general_news | history

recency_required: true if the user's message implies "now / today / aja / आज / latest / current / ahile / हालको" OR the topic is a changing current state (current PM, today's NEPSE, current inflation). False for historical or timeless topics.

confidence: your self-assessment of how certain you are. 1.0 for obvious cases, 0.5 for genuinely ambiguous.

Examples:
user: "k xa" → {"bucket":1,"tools_needed":[],"osint_intent":null,"recency_required":false,"confidence":0.95}
user: "nepalko pm ko ho ahile" → {"bucket":2,"tools_needed":["get_nepal_live_context"],"osint_intent":"who_is","recency_required":true,"confidence":0.9}
user: "modi le kahile nepal bhraman gareko" → {"bucket":3,"tools_needed":["internet_search"],"osint_intent":null,"recency_required":false,"confidence":0.85}
user: "https://github.com/HimalayaAI/yetidai ke ho" → {"bucket":5,"tools_needed":["analyze_github_repo"],"osint_intent":null,"recency_required":false,"confidence":0.95}
user: "nepal r india ko inflation tulana gar" → {"bucket":6,"tools_needed":["get_nepal_live_context","internet_search"],"osint_intent":"macro","recency_required":true,"confidence":0.9}

OUTPUT: ONE JSON object. Nothing else."""


_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

_VALID_BUCKETS = {1, 2, 3, 4, 5, 6}
_VALID_TOOL_NAMES = {
    "get_nepal_live_context",
    "internet_search",
    "fetch_url",
    "analyze_github_repo",
    "list_github_repos",
}
_VALID_OSINT_INTENTS = {
    "macro", "trading", "government", "who_is",
    "parliament", "debt", "general_news", "history",
}


def _parse_classification(raw: str) -> dict[str, Any] | None:
    """Parse the classifier's JSON output. None on any malformation.

    Strict — rejects anything that doesn't match the schema. The caller
    falls back to the keyword path on None.
    """
    if not raw:
        return None
    stripped = _CODE_FENCE_RE.sub("", raw).strip()
    # Occasionally Sarvam wraps the JSON in a sentence. Try to extract
    # the outermost {...} block if the whole string doesn't parse.
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None

    if not isinstance(data, dict):
        return None

    try:
        bucket = int(data.get("bucket", 0))
    except (TypeError, ValueError):
        return None
    if bucket not in _VALID_BUCKETS:
        return None

    raw_tools = data.get("tools_needed") or []
    if not isinstance(raw_tools, list):
        return None
    tools_needed = [
        t for t in raw_tools if isinstance(t, str) and t in _VALID_TOOL_NAMES
    ]

    osint_intent_raw = data.get("osint_intent")
    osint_intent: str | None
    if isinstance(osint_intent_raw, str) and osint_intent_raw in _VALID_OSINT_INTENTS:
        osint_intent = osint_intent_raw
    else:
        osint_intent = None

    recency = bool(data.get("recency_required", False))

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    return {
        "bucket": bucket,
        "tools_needed": tools_needed,
        "osint_intent": osint_intent,
        "recency_required": recency,
        "confidence": confidence,
    }


async def classify_intent(
    llm_client: Any,
    query: str,
    *,
    model: str,
    timeout_s: float = CLASSIFIER_TIMEOUT_SECONDS,
) -> dict[str, Any] | None:
    """One-shot classification call. Fails open — returns None on any error.

    Does NOT consume history — only the current user message. History
    context would add tokens for little gain on short classifier prompts.
    If follow-up context becomes important, we can thread `history` later.
    """
    query = (query or "").strip()
    if not query:
        return None

    messages = [
        {"role": "system", "content": CLASSIFIER_PROMPT},
        {"role": "user", "content": query},
    ]

    try:
        resp = await asyncio.wait_for(
            llm_client.chat.completions(
                model=model,
                messages=messages,
                tools=None,
                tool_choice=None,
            ),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        logger.info("intent_classifier timed out after %.1fs", timeout_s)
        return None
    except Exception as exc:
        logger.info("intent_classifier error: %s", exc)
        return None

    if not resp or not getattr(resp, "choices", None):
        return None

    raw = getattr(resp.choices[0].message, "content", "") or ""
    result = _parse_classification(raw)
    if result is None:
        logger.info("intent_classifier produced unparseable output: %r", raw[:200])
    else:
        logger.info(
            "intent_classifier: bucket=%d tools=%s intent=%s recency=%s conf=%.2f",
            result["bucket"], result["tools_needed"],
            result["osint_intent"], result["recency_required"],
            result["confidence"],
        )
    return result


def routing_hint_message(classification: dict[str, Any]) -> dict[str, str]:
    """Build a system-message nudge carrying the classifier's recommendation.

    Advisory, not prescriptive — the main turn's tool_choice enforcement
    does the actual forcing. The message gives Sarvam visibility into
    what the router decided so it can follow, or deviate with reason.
    """
    bits = [f"bucket={classification['bucket']}"]
    if classification["tools_needed"]:
        bits.append(f"tools={','.join(classification['tools_needed'])}")
    if classification["osint_intent"]:
        bits.append(f"osint_intent={classification['osint_intent']}")
    if classification["recency_required"]:
        bits.append("recency=true")
    bits.append(f"confidence={classification['confidence']:.2f}")
    text = (
        "ROUTING HINT (internal pre-classifier): "
        + " | ".join(bits)
        + ". Use this to pick the right tool on your first tool_call. "
        "Do not describe this hint to the user."
    )
    return {"role": "system", "content": text}


def should_force_tool_call(
    classification: dict[str, Any],
    *,
    threshold: float = CLASSIFIER_CONFIDENCE_THRESHOLD,
) -> bool:
    """True when the classifier is confident that a tool call is required.

    Callers use this to upgrade the first round's `tool_choice` from
    "auto" to "required", preventing Sarvam from skipping tools and
    drifting into training-data answers.
    """
    return (
        classification.get("confidence", 0.0) >= threshold
        and classification.get("bucket") != 1
        and bool(classification.get("tools_needed"))
    )
