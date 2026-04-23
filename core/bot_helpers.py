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
        # DEBUG: caller reached the error branch without an exception —
        # usually means ai_response was empty / all retries produced nothing.
        return f"{GENERIC_TECH_ERROR}\n[debug] no-exception; empty answer"
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
    # DEBUG: surface exception class + short message so the unknown-error
    # failure mode can be diagnosed without server log access. Revert this
    # once the root cause is identified.
    detail = f"{type(exc).__name__}: {str(exc)[:160]}"
    return f"{GENERIC_TECH_ERROR}\n[debug] {detail}"


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


# ── Empty-promise / tool-use enforcement ─────────────────────────
#
# Production failure: user asks "aja ko samachar sunau", Sarvam replies
# "म नेपालको आजको ताजा समाचार बताउँछु।" and stops — no tool_call, no
# actual news. The bot sends the promise straight to Discord because
# nothing detects "you vowed but didn't deliver". These helpers catch
# that shape so bot.py can inject a forcing nudge and retry once.

_EMPTY_PROMISE_PATTERNS: tuple["re.Pattern[str]", ...] = (
    re.compile(r"म .{0,40}बताउँछु"),               # "म समाचार बताउँछु"
    re.compile(r"म .{0,40}सुनाउँछु"),                # "म खबर सुनाउँछु"
    re.compile(r"म .{0,40}भन्छु"),                  # "म भन्छु"
    re.compile(r"म .{0,40}दिन्छु"),                  # "म जानकारी दिन्छु"
    re.compile(r"म .{0,40}प्रदान गर्छु"),             # "म समाचार प्रदान गर्छु"
    re.compile(r"म .{0,40}पठाउँछु"),                # "म पठाउँछु"
    # "ल्याउँछु / ल्याउनेछु" — "I will bring/fetch". This is the
    # specific verb Yeti reached for when saying "NepalOSINT बाट ताजा
    # समाचार ल्याउँछु" without ever calling the tool.
    re.compile(r"म .{0,60}ल्याउँछु"),
    re.compile(r"म .{0,60}ल्याउनेछु"),
    re.compile(r".{0,40}ल्याउँछु।?$"),
    # Future-tense "will do / will help / will look" — the GitHub-commit
    # trace landed on "help गर्नेछु" which the earlier pattern set missed.
    re.compile(r"म .{0,60}(?:गर्नेछु|गर्छु|हेर्नेछु|हेर्छु|खोज्नेछु|खोज्छु)"),
    re.compile(r"तपाईं(?:ँ|ले|लाई).{0,60}(?:बताउँछु|सुनाउँछु|दिन्छु|पठाउँछु|"
               r"गर्नेछु|गर्छु|हेर्नेछु|हेर्छु|ल्याउँछु|ल्याउनेछु)"),
    re.compile(r"help गर्(?:छु|नेछु)", re.IGNORECASE),
    re.compile(r"let me (fetch|search|check|look|see|analyze|bring|find)", re.IGNORECASE),
    re.compile(
        r"i ?(will|'ll|'d) (fetch|search|check|look|provide|get|tell|help|analyze|see|bring|find)",
        re.IGNORECASE,
    ),
    # "मलाई ... खोज्न दिनुहोस्" — permissive/imperative phrasing that
    # functions as an empty promise. Observed: PM query returned only
    # "तपाईँले नेपालको वर्तमान प्रधानमन्त्री को हो भनेर सोध्दै हुनुहुन्छ।
    # मलाई त्यो जानकारी खोज्न दिनुहोस्।" — classic "let me go find out"
    # shape that the म-X-छु patterns miss entirely.
    re.compile(r"मलाई\s+.{0,80}(?:खोज्न|पत्ता\s+लगाउन|हेर्न|बताउन)\s+दिनु"),
    # Ditto "म ... खोज्न चाहन्छु / हेर्न चाहन्छु" (I want to search).
    re.compile(r"म\s+.{0,80}(?:खोज्न|हेर्न|पत्ता\s+लगाउन)\s+चाहन्छु"),
)

# Short reply + any promise pattern is the smoking-gun shape. Long
# replies often contain "म बताउँछु" as a natural phrase inside a real
# answer, so we gate on length to keep false-positives low.
_EMPTY_PROMISE_MAX_CHARS = 240


def is_empty_promise(text: str | None, *, tool_was_used: bool = False) -> bool:
    """Heuristic: did the model promise to do something without doing it?

    Criteria (all must hold):
      1. No tool was successfully used this turn.
      2. The reply is short — promises are almost always one-liners.
      3. At least one empty-promise pattern matches.
    """
    if tool_was_used or not text:
        return False
    stripped = text.strip()
    if not stripped or len(stripped) > _EMPTY_PROMISE_MAX_CHARS:
        return False
    return any(pat.search(stripped) for pat in _EMPTY_PROMISE_PATTERNS)


# Patterns that strongly suggest the turn needs a tool call. Used to
# decide whether an empty-promise retry is worthwhile — if the user
# was chatting, let it go; if they asked for data, force Sarvam to
# actually call a tool.
_TOOL_NEEDED_PATTERNS: tuple["re.Pattern[str]", ...] = (
    # URLs of any kind
    re.compile(r"https?://", re.IGNORECASE),
    # Nepal info keywords (English / romanized / Devanagari)
    re.compile(
        r"\b(news|samachar|khabar|halkhabar|update|inflation|nepse|"
        r"parliament|cabinet|minister|pradanmantri|arthamantri|"
        r"pm|ipo|dividend|rin|ऋण|कर्जा)\b",
        re.IGNORECASE,
    ),
    re.compile(r"(समाचार|खबर|मुद्रास्फीति|रेमिट्यान्स|मन्त्री|प्रधानमन्त्री|संसद|विधेयक)"),
    # Explicit date phrases ("aja", "आज", "today", ISO date)
    re.compile(r"\b(aja|aaja|today|hijo|yesterday|bhako|bhayo)\b", re.IGNORECASE),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    # "who is X / X ko ho" identity probes
    re.compile(r"\b(who is|ko ho|को हो|को हुन|को हुनुहुन्छ)\b", re.IGNORECASE),
)


def needs_tool_use(user_text: str | None) -> bool:
    """True if the user's message likely requires a tool call to answer.

    Conservative: only returns True on strong positive signals. Small-talk
    and freeform chat returns False so we don't force tools on "नमस्ते"
    or "k xa".
    """
    if not user_text:
        return False
    return any(pat.search(user_text) for pat in _TOOL_NEEDED_PATTERNS)


# ── News-shape validator ──────────────────────────────────────────
#
# Production failure: user asked for "aja ko taja samachar". Yeti
# replied with a list of VEGETABLES (interpreted "ताजा समाचार" as
# "ताजा तरकारी"). Neither the empty-promise detector nor the
# fabricated-URL detector catches an off-topic but confidently-wrong
# answer. This pair of helpers does.

_NEWS_REQUEST_PATTERNS: tuple["re.Pattern[str]", ...] = (
    re.compile(r"\b(samachar|khabar|headline|news|samachaar)\b", re.IGNORECASE),
    re.compile(r"(समाचार|खबर)"),
)


def user_asked_for_news(user_text: str | None) -> bool:
    """True if the user's message explicitly asks for news."""
    if not user_text:
        return False
    return any(pat.search(user_text) for pat in _NEWS_REQUEST_PATTERNS)


def looks_like_news_answer(answer: str | None) -> bool:
    """Heuristic: does the answer actually look like a news reply?

    Signals (any one suffices):
      * Contains an http(s) URL.
      * Has a `स्रोत:` citation line.
      * Contains at least two dated-headline-like markers (Devanagari
        year 2026/2083 near a verb, or ISO date).
    This is deliberately loose — we only want to catch the cases where
    the answer is obviously NOT news (vegetables, weather, random
    chat).
    """
    if not answer:
        return False
    if URL_RE.search(answer):
        return True
    if "स्रोत:" in answer or "स्रोत :" in answer:
        return True
    dated_markers = len(
        re.findall(r"(?:20[0-9]{2}|२०[०-९]{2})", answer)
    )
    return dated_markers >= 2


def news_answer_off_topic(
    user_text: str | None,
    answer: str | None,
    *,
    tool_was_used: bool,
) -> bool:
    """True when the user asked for news but the answer clearly isn't.

    Only fires when a tool was NOT used — if the tool returned real
    content and the model still went off-topic, the fabricated-URL /
    fabricated-filename detectors are the right guards (the tool output
    is the ground truth there). This is the catch for cases where no
    tool ran at all.
    """
    if tool_was_used:
        return False
    if not user_asked_for_news(user_text):
        return False
    return not looks_like_news_answer(answer)


def build_force_tool_nudge(user_text: str) -> str:
    """System message used to retry after an empty-promise answer.

    Kept short so it doesn't push the already-long system prompt off the
    attention window. The model sees this AFTER its promise text, so it
    effectively reads: "I said I'd fetch. System reminder: actually fetch."
    """
    return (
        "तपाईंले अघिल्लो turn मा कुनै tool call emit नगरी 'म बताउँछु' जस्तो "
        "वाचा मात्र लेख्नुभयो। यो bug हो — प्रयोगकर्ताले actual data माग्दै "
        "हुनुहुन्छ। अहिले नै उपयुक्त tool call emit गर्नुहोस् "
        "(get_nepal_live_context / internet_search / fetch_url / "
        "analyze_github_repo मध्ये एक)। वाचा नगरी सिधै tool call दिनुहोस्।"
    )


# ── Anti-hallucination for GitHub answers ────────────────────────
#
# Production failure: analyze_github_repo ran, returned the real tree,
# but Sarvam still invented "NepaliNewsAggregator.py" in its answer
# (a file that doesn't exist in the repo). This helper flags any file
# name the model cites that isn't present in the actual tool output.

_FILENAME_IN_ANSWER_RE = re.compile(
    r"\b([\w.\-]+\.(?:py|md|ts|tsx|js|jsx|json|yaml|yml|toml|txt|rs|go|java|kt|rb|c|cpp|h|sh))\b"
)


def detect_fabricated_urls(answer: str, tool_output: str) -> list[str]:
    """Return URLs cited in `answer` whose host doesn't appear in `tool_output`.

    Catches the "Sarvam cites https://nprc.gov.np/ but the tool output
    contains no such URL and no such host" class of hallucination. We
    check by host (not full URL) because real tools often cite an
    article URL while Sarvam paraphrases to the homepage — same host
    is fine, different host is fabrication.
    """
    if not answer or not tool_output:
        return []
    answer_urls = URL_RE.findall(answer)
    if not answer_urls:
        return []
    # Collect all hostnames from the tool output, case-insensitive.
    tool_hosts: set[str] = set()
    for m in URL_RE.finditer(tool_output):
        host = _extract_host(m.group(0))
        if host:
            tool_hosts.add(host.lower())
    if not tool_hosts:
        # No URLs in the tool output — every URL in the answer is suspicious.
        # Return them so the caller can decide to nuke the citations.
        return sorted({u.rstrip(".,;:") for u in answer_urls})
    bad: list[str] = []
    for url in answer_urls:
        url = url.rstrip(".,;:")
        host = _extract_host(url)
        if host and host.lower() not in tool_hosts:
            bad.append(url)
    # Dedupe while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for u in bad:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return unique


def _extract_host(url: str) -> str | None:
    """Parse a hostname out of a URL, tolerating minor malformations."""
    try:
        import urllib.parse
        parsed = urllib.parse.urlparse(url.strip())
        host = (parsed.hostname or "").strip()
        # Strip a leading "www." so api.example.com and www.example.com
        # don't trip the detector when the real tool quoted one and
        # the model wrote the other.
        if host.startswith("www."):
            host = host[4:]
        return host or None
    except Exception:
        return None


# News-org names that frequently appear in hallucinated स्रोत: blocks.
# When the model cites one of these WITHOUT a URL to back it, flag as
# fabricated — Yeti was caught writing "स्रोत: The Associated Press,
# Reuters" with no links and no real tool output from either source.
_FAKE_SOURCE_NAMES: frozenset[str] = frozenset({
    "the associated press", "associated press", "ap ",
    "reuters", "bbc", "cnn", "al jazeera", "aljazeera",
    "new york times", "the guardian", "washington post",
    "bloomberg", "economist", "financial times",
})


def detect_fabricated_source_names(
    answer: str, tool_output: str,
) -> list[str]:
    """Return news-org names cited in the answer's स्रोत: block that
    have neither a URL next to them nor any appearance in tool_output.

    This is the AP/Reuters fake-citation pattern — the model names a
    western outlet as a source but the outlet was never hit by any
    tool this turn.
    """
    if not answer or "स्रोत:" not in answer:
        return []
    idx = answer.rfind("स्रोत:")
    sources_block = answer[idx:]
    # If the sources block has URLs, the URL-level validator already
    # covers us. We only run this when the block is URL-free (a
    # classic hallucination shape).
    if URL_RE.search(sources_block):
        return []
    lowered_block = sources_block.lower()
    lowered_tool = (tool_output or "").lower()
    bad: list[str] = []
    for name in _FAKE_SOURCE_NAMES:
        if name in lowered_block and name not in lowered_tool:
            bad.append(name.strip().rstrip())
    # Dedupe, preserve order, title-case for the user-facing message.
    seen: set[str] = set()
    unique: list[str] = []
    for n in bad:
        key = n.lower()
        if key not in seen:
            seen.add(key)
            unique.append(n.title() if n.isascii() else n)
    return unique


def detect_fabricated_filenames(answer: str, tool_output: str) -> list[str]:
    """Return file names cited in `answer` that do not appear in `tool_output`.

    Empty list = no obvious fabrication. `tool_output` is the concatenated
    content of any analyze_github_repo / fetch_url / internet_search
    tool results this turn — if a file name in the answer doesn't appear
    there, the model hallucinated it.
    """
    if not answer or not tool_output:
        return []
    answer_names = set(_FILENAME_IN_ANSWER_RE.findall(answer))
    if not answer_names:
        return []
    return sorted(n for n in answer_names if n not in tool_output)


def shorten_for_citation(url: str, *, max_chars: int = 42) -> str:
    """Return a Discord-markdown link with a short visible label.

    `[myrepublica.com/news](https://myrepublica.nagariknetwork.com/news/...)`

    Discord renders this as an inline clickable label, keeping the
    citation line readable while preserving the full URL target.
    Non-URL input is returned unchanged.
    """
    if not url or not url.startswith(("http://", "https://")):
        return url
    host = _extract_host(url) or ""
    import urllib.parse as _up
    try:
        parsed = _up.urlparse(url)
    except Exception:
        return url
    path = (parsed.path or "").rstrip("/")
    label = host
    if path and len(path) > 1 and len(host) < max_chars - 4:
        first_seg = path.lstrip("/").split("/", 1)[0]
        if first_seg:
            remain = max_chars - len(host) - 1  # '/'
            if remain > 2:
                if len(first_seg) > remain:
                    first_seg = first_seg[: remain - 1] + "…"
                label = f"{host}/{first_seg}"
    if len(label) > max_chars:
        label = label[: max_chars - 1] + "…"
    return f"[{label}]({url})"


def rewrite_sources_as_markdown(answer: str) -> str:
    """Rewrite bare URLs inside the `स्रोत:` block as Discord-markdown links.

    Only touches lines that are bullets + bare URL (`- https://…`), so
    Sarvam's own markdown links or surrounding prose stay untouched.
    """
    if not answer or "स्रोत:" not in answer:
        return answer
    idx = answer.rfind("स्रोत:")
    body = answer[:idx]
    sources = answer[idx:]
    bullet_url_re = re.compile(
        r"^(\s*[-*•]?\s*\d*\.?\s*)(https?://[^\s)\]\"<>]+)\s*$"
    )
    out_lines: list[str] = []
    for line in sources.split("\n"):
        m = bullet_url_re.match(line)
        if m:
            prefix = m.group(1)
            url = m.group(2).rstrip(".,;:")
            out_lines.append(f"{prefix}{shorten_for_citation(url)}")
        else:
            out_lines.append(line)
    return body + "\n".join(out_lines)


def ensure_sources_line(
    answer: str,
    citation_urls: Iterable[str],
    *,
    max_urls: int = 3,
) -> str:
    """Ensure the `स्रोत:` block actually cites URLs when the tool had some.

    Three cases:
      1. No `स्रोत:` header at all → append one with shortened URLs.
      2. `स्रोत:` header present AND already contains URLs → leave alone.
      3. `स्रोत:` header present but has NO URLs (e.g. the model wrote
         "स्रोत: नेपालOSINT" with no links) → rewrite the block to include
         the real tool URLs so citations are actually clickable.

    Case 3 matters because Sarvam likes to write "स्रोत: NepalOSINT" as a
    flat name which satisfies the validator but gives the user nothing
    verifiable.
    """
    if not answer:
        return answer
    urls = [u.rstrip(".,;:") for u in citation_urls if u][:max_urls]
    urls = [u for u in urls if u]
    if not urls:
        return answer

    if "स्रोत:" not in answer:
        body = answer.rstrip()
        lines = "\n".join(f"- {shorten_for_citation(u)}" for u in urls)
        return f"{body}\n\nस्रोत:\n{lines}"

    # Case 3 detection: a स्रोत: block exists but contains no URL.
    idx = answer.rfind("स्रोत:")
    body_before = answer[:idx].rstrip()
    sources_block = answer[idx:]
    if URL_RE.search(sources_block):
        return answer  # Already has URLs, leave it.
    lines = "\n".join(f"- {shorten_for_citation(u)}" for u in urls)
    return f"{body_before}\n\nस्रोत:\n{lines}"
