"""
core/preflight.py — deterministic pre-flight tool execution.

Why this exists: Sarvam-30B is unreliable at picking tools from a long
system prompt. It frequently emits "म खोज्छु" as plain text when it
should have called get_nepal_live_context(). The reactive retry in
bot.py catches some of those, but not all — and forcing a retry costs
an extra LLM round-trip.

This module runs a cheap classifier on the user's message BEFORE the
first Sarvam turn. For high-confidence query shapes, it returns a
`(tool_name, arguments)` pair that bot.py executes directly and feeds
back as a synthetic prior tool call — so Sarvam's first turn already
has the data in hand and only has to write the Nepali summary.

Deterministic preflight rules (match order matters):
  1. Bare GitHub URL in message      → analyze_github_repo
  2. Any other http(s) URL           → fetch_url
  3. Minister role detected          → get_nepal_live_context(government, who_is)
  4. Clear "news" / samachar / khabar → get_nepal_live_context(general_news)
  5. GDP / macro keyword             → get_nepal_live_context(macro) + web fallback note
  6. NEPSE ticker + trading context  → get_nepal_live_context(trading)
  7. Cabinet / decisions             → get_nepal_live_context(government)

Returns None when no rule matches — the normal tool-loop handles the
query.
"""
from __future__ import annotations

import re
import urllib.parse
from typing import Any

from tools.osint.context_router import (
    DEBT_KEYWORDS,
    GOVT_KEYWORDS,
    MACRO_KEYWORDS,
    PARLIAMENT_KEYWORDS,
    _contains_any,
    detect_minister_role,
)

# Keep these import-light so bot.py can call preflight synchronously
# during message-list build, before any plugins run.

_URL_RE = re.compile(r"https?://[^\s)\]\"<>]+", re.IGNORECASE)
_GITHUB_HOST_RE = re.compile(r"^(?:https?://)?(?:www\.)?github\.com/", re.IGNORECASE)

# Pulled from context_router.ROMANIZED_GENERAL_PATTERNS / GENERAL_KEYWORDS
# — reproduced here as a short list so we can cheap-check without
# importing the full router every call.
_NEWS_HINT_RE = re.compile(
    r"\b(samachar|samachaar|khabar|news|headline|headlines|"
    r"aja|aaja|aaj|aajko|latest|breaking)\b",
    re.IGNORECASE,
)
_NEWS_HINT_DEV_RE = re.compile(r"(समाचार|खबर|ताजा|आज)")


def _extract_first_url(text: str) -> str | None:
    m = _URL_RE.search(text or "")
    if not m:
        return None
    return m.group(0).rstrip(".,;:)")


def _is_github_url(url: str) -> bool:
    return bool(_GITHUB_HOST_RE.match(url))


def _extract_ticker(text: str) -> str | None:
    """If the user query contains a NEPSE ticker (whitelisted), return it."""
    from tools.osint.context_router import _NEPSE_TICKERS, _TICKER_CANDIDATE_RE
    for tok in _TICKER_CANDIDATE_RE.findall(text):
        if tok in _NEPSE_TICKERS:
            return tok
    return None


def plan_preflight(user_text: str | None) -> tuple[str, dict[str, Any]] | None:
    """Return (tool_name, arguments) for a high-confidence preflight.

    Only fires when the query is unambiguous. For fuzzy / chat / multi-
    intent queries, returns None so the normal Sarvam-picks-tool flow
    runs.
    """
    if not user_text:
        return None
    text = user_text.strip()
    lowered = text.lower()

    # 1. URL in message — either GitHub or generic fetch.
    url = _extract_first_url(text)
    if url:
        if _is_github_url(url):
            return ("analyze_github_repo", {"repo": url})
        return ("fetch_url", {"url": url})

    # 2. Minister role → government + who_is (OSINT handler reads these
    #    from the route plan; `focus` carries the canonical role tag).
    role = detect_minister_role(text)
    if role:
        return (
            "get_nepal_live_context",
            {"intent": "government", "focus": role},
        )

    # Domain-specific checks BEFORE the news catch-all. Otherwise a
    # query like "NEPSE aja kasto?" would match the news rule first
    # just because of "aja" — wrong: it's a markets query.
    #
    # 3. GDP / macro keyword.
    if _contains_any(lowered, MACRO_KEYWORDS):
        return (
            "get_nepal_live_context",
            {"intent": "macro", "focus": "inflation"},
        )

    # NOTE: ticker / trading queries used to preflight into OSINT's
    # trading intent, but OSINT's trading endpoints return news rows
    # about a ticker — not the rich company background (capacity,
    # developer, project location, COD) users want for "RURU share
    # ko barima information". Sarvam's own choice — internet_search
    # against ruruhydro.com / icranepal.com / doed.gov.np / wikipedia
    # / merolagani — is strictly better for these. So we deliberately
    # do NOT preflight ticker/trading queries; let Sarvam pick.
    #
    # If the reactive force-tool retry finds an empty-promise reply,
    # it'll still fire a tool call. The fast-path preflight is for
    # shapes where Sarvam was RELIABLY picking the wrong tool (news,
    # minister names, GDP), not for shapes where it was working.

    # 5. Public debt.
    if _contains_any(lowered, DEBT_KEYWORDS):
        return (
            "get_nepal_live_context",
            {"intent": "debt", "focus": "debt"},
        )

    # 6. Parliament (bills, sessions, verbatim).
    if _contains_any(lowered, PARLIAMENT_KEYWORDS):
        return (
            "get_nepal_live_context",
            {"intent": "parliament", "focus": "bills"},
        )

    # 7. Cabinet / govt decisions (the "nepal govt ko naya decision" shape).
    if _contains_any(lowered, GOVT_KEYWORDS):
        return (
            "get_nepal_live_context",
            {"intent": "government", "focus": "cabinet_decisions"},
        )

    # 6. Generic news request — runs LAST as a catch-all for "aja ko
    #    samachar" with no specific domain. Limited to <= 12 words.
    if len(lowered.split()) <= 12 and (
        _NEWS_HINT_RE.search(lowered) or _NEWS_HINT_DEV_RE.search(text)
    ):
        return (
            "get_nepal_live_context",
            {"intent": "general_news", "focus": "general_news"},
        )

    return None
