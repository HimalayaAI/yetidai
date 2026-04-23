"""
core/bot_helpers.py — side-effect-free helpers used by bot.py.

Kept deliberately free of the `discord` and `sarvamai` imports so tests can
exercise them directly without stubbing the whole Discord/SDK surface.

Groups:
  - Discord text shaping: extract_urls, split_body_and_sources,
    chunk_for_discord, safe_field_value.
  - Failure guards:       is_bot_apology, is_transient_llm_error,
                          classify_llm_error.
  - Output shaping:       normalize_digits, ensure_sources_line.
  - Tool-loop plumbing:   hash_tool_call, tool_calls_signature,
                          is_real_tool_content, structured error markers.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
from typing import Any, Iterable

# ── Discord limits we must respect ────────────────────────────────
DISCORD_MSG_LIMIT = 2000
DISCORD_EMBED_FIELD_VALUE_LIMIT = 1024
DISCORD_EMBED_FOOTER_LIMIT = 2048

# ── Bot-generated apology strings (do not replay these as history) ─
GENERIC_TECH_ERROR = (
    "माफ गर्नुहोस्, एउटा प्राविधिक समस्या देखियो। कृपया फेरि प्रयास गर्नुहोस्।"
)
BOT_APOLOGY_PREFIXES: tuple[str, ...] = (
    "माफ गर्नुहोस्, एउटा प्राविधिक समस्या",
    "माफ गर्नुहोस्, उत्तर तयार गर्न सकिएन",
    "माफ गर्नुहोस्, Sarvam",
    "माफ गर्नुहोस्, नेटवर्क",
)

URL_RE = re.compile(r"https?://[^\s)\]\"<>]+")

# Translation table for ASCII → Devanagari digits.
_ASCII_TO_DEVANAGARI = str.maketrans("0123456789", "०१२३४५६७८९")


def is_bot_apology(content: str) -> bool:
    """True if a message looks like one of our own generic failure strings.

    Filtered out of replayed history so Sarvam never learns to parrot it.
    """
    if not content:
        return False
    stripped = content.lstrip()
    return any(stripped.startswith(p) for p in BOT_APOLOGY_PREFIXES)


def extract_urls(text: str | None) -> list[str]:
    """Pull http(s) URLs out of tool output for citation embeds. Deduped, ordered."""
    if not text:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for url in URL_RE.findall(text):
        url = url.rstrip(".,;:")
        if url not in seen:
            seen.add(url)
            out.append(url)
    return out


def split_body_and_sources(answer: str) -> tuple[str, str]:
    """Return (body_without_sources, sources_line_or_empty)."""
    idx = answer.rfind("स्रोत:")
    if idx < 0:
        return answer, ""
    return answer[:idx].rstrip(), answer[idx:].strip()


def chunk_for_discord(text: str, limit: int = DISCORD_MSG_LIMIT) -> list[str]:
    """Break text at newline/whitespace boundaries to fit Discord's per-message cap.

    Falls back to a hard slice only when no boundary exists inside the window
    (e.g. an unbroken URL or single long token).
    """
    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        window = remaining[:limit]
        cut = window.rfind("\n")
        if cut < limit // 2:
            ws = window.rfind(" ")
            if ws > cut:
                cut = ws
        if cut <= 0:
            cut = limit
        chunks.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


def safe_field_value(url: str) -> str:
    """Fit a URL into Discord's 1024-char embed field value limit."""
    if len(url) <= DISCORD_EMBED_FIELD_VALUE_LIMIT:
        return url
    return url[: DISCORD_EMBED_FIELD_VALUE_LIMIT - 1] + "…"


# ── LLM error classification ──────────────────────────────────────

def _error_status_code(exc: BaseException) -> int | None:
    """Best-effort extraction of an HTTP status code from a client-library exception."""
    direct = getattr(exc, "status_code", None)
    if isinstance(direct, int):
        return direct
    response = getattr(exc, "response", None)
    indirect = getattr(response, "status_code", None)
    return indirect if isinstance(indirect, int) else None


def is_transient_llm_error(exc: BaseException) -> bool:
    """True if the LLM call deserves one retry (timeout, 5xx, 429, net error)."""
    if isinstance(exc, asyncio.TimeoutError):
        return True
    status = _error_status_code(exc)
    if isinstance(status, int) and (status == 429 or 500 <= status < 600):
        return True
    name = type(exc).__name__.lower()
    return any(k in name for k in ("timeout", "connect", "network", "remoteproto"))


def classify_llm_error(exc: BaseException | None) -> str:
    """Return a user-facing Nepali string for the given exception.

    Preserves the BOT_APOLOGY_PREFIXES contract so the history filter keeps
    these out of replayed context.
    """
    if exc is None:
        return GENERIC_TECH_ERROR
    if isinstance(exc, asyncio.TimeoutError):
        return (
            "माफ गर्नुहोस्, Sarvam जवाफ दिन ढिला भयो। एकछिन पछि पुनः प्रयास गर्नुहोस्।"
        )
    status = _error_status_code(exc)
    name = type(exc).__name__.lower()
    if status == 429 or "ratelimit" in name:
        return (
            "माफ गर्नुहोस्, Sarvam अहिले व्यस्त छ। केही सेकेन्डपछि पुनः प्रयास गर्नुहोस्।"
        )
    if isinstance(status, int) and 500 <= status < 600:
        return (
            "माफ गर्नुहोस्, Sarvam सेवामा समस्या छ। एकछिन पछि पुनः प्रयास गर्नुहोस्।"
        )
    if "connect" in name or "network" in name:
        return "माफ गर्नुहोस्, नेटवर्क समस्या देखियो। पुनः प्रयास गर्नुहोस्।"
    return GENERIC_TECH_ERROR


def with_turn_id(message: str, turn_id: str | None) -> str:
    """Append a small `(त्रुटि कोड: …)` footer so a screenshot maps to a log line."""
    if not turn_id:
        return message
    return f"{message}\n(त्रुटि कोड: {turn_id})"


# ── Deterministic output fixes ────────────────────────────────────

def normalize_digits(text: str) -> str:
    """Convert ASCII digits to Devanagari, but leave URL substrings untouched.

    The output validator flags ASCII digits as a hard error; most such cases
    are trivially fixable without asking the LLM to regenerate.
    """
    if not text:
        return text
    out: list[str] = []
    last = 0
    for m in URL_RE.finditer(text):
        out.append(text[last:m.start()].translate(_ASCII_TO_DEVANAGARI))
        out.append(m.group(0))
        last = m.end()
    out.append(text[last:].translate(_ASCII_TO_DEVANAGARI))
    return "".join(out)


# ── Tool-loop plumbing ────────────────────────────────────────────
#
# Structured markers the bot injects into tool-result `content` when the
# registry call itself failed. Putting these in the content (instead of
# silently substituting a bland error string) makes the failure visible to
# the LLM so it can decide between "retry with different args", "switch to
# a fallback tool", or "apologize in Nepali with a partial answer". This
# mirrors Anthropic's `is_error` convention — we just encode it inline
# because the OpenAI/Sarvam tool message shape has no dedicated flag.

TOOL_ERROR_MARKER = "[TOOL_ERROR]"
TOOL_TIMEOUT_MARKER = "[TOOL_TIMEOUT]"
TOOL_DEDUP_MARKER = "[TOOL_DEDUP_HIT]"
# Emitted by the OSINT plugin when it detects stale data for a recency
# query. Treated like the error/timeout markers so `is_real_tool_content`
# returns False — the validator would otherwise demand a citation and
# the model could cite a 2022 story as "today's news".
TOOL_STALE_MARKER = "[STALE_DATA]"

_TOOL_STATUS_MARKERS: tuple[str, ...] = (
    TOOL_ERROR_MARKER,
    TOOL_TIMEOUT_MARKER,
    TOOL_DEDUP_MARKER,
    TOOL_STALE_MARKER,
)


# Keys stripped from tool_call args before hashing. The model often varies
# these trivially between rounds (limit=8 vs limit=10, hours=24 vs hours=48,
# read_pages=2 vs read_pages=3) — treating those as distinct calls defeats
# the dedup cache. Tunable per-tool if a plugin genuinely distinguishes
# on these values (none currently do).
_DEDUP_NOISE_KEYS: frozenset[str] = frozenset({
    "limit", "max_items", "top_k", "page", "offset", "hours",
    "read_pages", "min_similarity", "dedupe",
})


def canonical_tool_args(args: dict[str, Any] | None) -> dict[str, Any]:
    """Drop pagination/noise keys + lowercase string values so near-
    identical calls collapse to the same dedup signature.

    Example:
        {"query": "NEPSE", "limit": 8}  →  {"query": "nepse"}
        {"query": "nepse", "limit": 10} →  {"query": "nepse"}
    Both hash to the same key and the second call replays the cached
    result instead of hitting the plugin again.
    """
    if not args:
        return {}
    cleaned: dict[str, Any] = {}
    for key, value in args.items():
        if not isinstance(key, str) or key in _DEDUP_NOISE_KEYS:
            continue
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                continue
            cleaned[key] = stripped.lower()
        else:
            cleaned[key] = value
    return cleaned


def hash_tool_call(name: str, args: dict[str, Any]) -> str:
    """Return a stable signature for a tool call.

    Used for in-turn dedup: if the model emits the same (name, args) pair
    across rounds we reuse the cached result and annotate the tool message
    so the model notices it's looping. `canonical_tool_args` normalises
    pagination-y keys so trivial variations still collide.
    """
    canonical_args = canonical_tool_args(args)
    try:
        canonical = json.dumps(canonical_args, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        canonical = repr(canonical_args)
    digest = hashlib.sha256(f"{name}|{canonical}".encode("utf-8")).hexdigest()
    return digest[:16]


def tool_calls_signature(
    tool_calls: Iterable[Any],
) -> tuple[tuple[str, str], ...]:
    """Aggregate signature for a whole round's tool_calls.

    Two consecutive rounds with the same signature means the model is
    spinning on identical calls and won't converge — bot.py uses this to
    break the loop early and fall through to "force text answer".
    """
    out: list[tuple[str, str]] = []
    for tc in tool_calls:
        name = getattr(getattr(tc, "function", None), "name", None) or ""
        raw = getattr(getattr(tc, "function", None), "arguments", None) or ""
        try:
            args = json.loads(raw) if raw else {}
        except (TypeError, ValueError, json.JSONDecodeError):
            args = {"_raw": raw}
        out.append((name, hash_tool_call(name, args)))
    return tuple(sorted(out))


def is_tool_status_marker(content: str | None) -> bool:
    """True when the content starts with one of our structured error markers."""
    if not content:
        return False
    stripped = content.lstrip()
    return any(stripped.startswith(m) for m in _TOOL_STATUS_MARKERS)


def is_real_tool_content(result: Any) -> bool:
    """True iff the tool actually produced usable content.

    Gates `tool_was_used` so the validator doesn't demand a `स्रोत:` line
    when every tool call failed or was deduped — we'd just be asking the
    model to invent a citation.
    """
    if result is None:
        return False
    if not getattr(result, "success", False):
        return False
    content = getattr(result, "content", None)
    if not content or not content.strip():
        return False
    return not is_tool_status_marker(content)


# ── Correction / count-intent detection ──────────────────────────
#
# Observed in production: user says "aja ko 30 khabar" → bot lists 30
# news *portals*. User replies "haina, samachar portal bahenyko haina"
# (not portals) → bot repeats the same wrong list. These helpers let
# bot.py inject a corrective system message before the LLM turn.

_CORRECTION_MARKERS: tuple[str, ...] = (
    # Devanagari
    "होइन", "गलत", "त्यो होइन", "फरक", "फेरि पढ",
    # Romanised Nepali
    "haina", "hoina", "bahenyko haina", "bhaneko haina",
    "purano", "maile bhaneko", "galat", "milaunu",
    # English
    "not that", "wrong answer", "that's wrong", "that is not",
    "you misunderstood", "re-read", "i asked", "i said",
    "different from", "you didn't understand",
)

# Rough "N items requested" detector. Handles Devanagari + ASCII digits
# and common English number words. Returns None if nothing obvious.
_COUNT_RE = re.compile(
    r"(?:^|\s)(\d{1,3}|[०-९]{1,3})\s*(?:वटा|ota|wata|items?|news|stories|headlines?)",
    re.IGNORECASE,
)
_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def looks_like_correction(user_text: str | None) -> bool:
    """True when the user's latest message reads like a correction of our
    previous reply ("haina", "you misunderstood", "purano khabar", …)."""
    if not user_text:
        return False
    lower = user_text.lower()
    return any(marker in lower for marker in _CORRECTION_MARKERS)


def detect_requested_count(user_text: str | None) -> int | None:
    """Extract an explicit 'N items' count from the user message.

    Used to surface intent to the LLM: if the user asked for 30 and we
    only have 5 real stories, it's better to say so than to pad with
    duplicates (the transcript's "NepalKhabar, NepalKhabar, …" failure).
    """
    if not user_text:
        return None
    m = _COUNT_RE.search(user_text.translate(_DEVANAGARI_DIGITS))
    if not m:
        return None
    try:
        n = int(m.group(1))
    except ValueError:
        return None
    return n if 1 <= n <= 200 else None


def build_correction_nudge(
    user_text: str,
    *,
    requested_count: int | None = None,
) -> str:
    """System-message text to inject when we detect a user correction.

    The goal is to break the "parrot the previous bad answer" loop. The
    nudge is in Nepali to stay in-register and names the specific anti-
    patterns we've seen (duplicate entries, wrong type of answer).
    """
    extra = ""
    if requested_count:
        extra = (
            f"\nप्रयोगकर्ताले ठीक {requested_count} वटा माग्नुभएको छ। "
            "यदि tool output मा {requested_count} वटा पर्याप्त छैनन् भने, "
            "दोहोरो प्रविष्टि (duplicate entries) कहिल्यै नलेख्नुहोस् — "
            "इमानदार भएर कति मात्र उपलब्ध छन् भन्नुहोस्।"
        )
    return (
        "प्रयोगकर्ताले अघिल्लो जवाफ अस्वीकार गर्नुभएको छ। पहिले भन्दा फरक, "
        "प्रयोगकर्ताको वास्तविक प्रश्न के हो ध्यान दिएर पुनः लेख्नुहोस्। "
        "अघिल्लो जवाफ दोहोऱ्याउनुहुन्न। कुनै पनि entry दुई पटक नलेख्नुहोस् — "
        "tool output मा जति unique items छन् त्यति नै देखाउनुहोस्। "
        "यदि tool ले पर्याप्त डेटा दिएन भने, माफी मागेर खुलस्त भन्नुहोस्।"
        f"{extra}"
    )


def ensure_sources_line(
    answer: str,
    citation_urls: Iterable[str],
    *,
    max_urls: int = 3,
) -> str:
    """Append a `स्रोत:` section when the model forgot to cite tool output.

    If the answer already contains `स्रोत:`, it is returned unchanged. We only
    inject when there are citation URLs; otherwise the LLM retry can do
    better (it may know non-URL source names).
    """
    if not answer or "स्रोत:" in answer:
        return answer
    urls = [u for u in citation_urls if u][:max_urls]
    if not urls:
        return answer
    body = answer.rstrip()
    lines = "\n".join(f"- {u}" for u in urls)
    return f"{body}\n\nस्रोत:\n{lines}"
