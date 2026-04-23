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

# Politics-scoped news. "political samachar" / "राजनीतिक खबर" / etc.
# deserves the consolidated-stories endpoint with category=political
# so the bundle hits actual political stories (parliament, cabinet,
# party moves) instead of the generic recent-stories stream which
# can be economic-heavy.
_POLITICS_RE = re.compile(
    r"\b(political|politics|raajnitik|rajnitik|rajniti)\b",
    re.IGNORECASE,
)
_POLITICS_DEV_RE = re.compile(r"(राजनीत|राजनीति|राजनीतिक)")

# Explicit "use web search" / "use google" commands. When the user
# literally tells us to go search, preflight routes to internet_search
# instead of whatever our usual classifier picked. Addresses the
# "web search garnus alchi nagarnus google chalaunus" trace.
_EXPLICIT_WEB_SEARCH_RE = re.compile(
    r"(web\s*search|google\s*(?:chalaunus|chalaun|garnus|gara|garau|gar)|"
    r"online\s*(?:khoju|search)|internet\s*bata|"
    r"google\s*ma|google\s*garera|google\s*garnus|"
    r"गुगल|वेब\s*सर्च|इन्टरनेटमा)",
    re.IGNORECASE,
)

# Tokens we strip when forwarding a user's "web search X" command to
# the search tool — leaves just the actual search subject.
_COMMAND_STRIP_RE = re.compile(
    r"\b(web\s*search|google\s*(?:chalaunus|chalaun|garnus|garera|gara|garau|gar)|"
    r"online\s*(?:khoju|search)|internet\s*bata|alchi\s*nagarnus|"
    r"please|kripaya|hajur|garnus|garnu)\b",
    re.IGNORECASE,
)


def _strip_command_tokens(text: str) -> str:
    """Remove imperative words so 'web search garnus Nepal PM' becomes
    'Nepal PM' — a much cleaner query for DuckDuckGo."""
    cleaned = _COMMAND_STRIP_RE.sub(" ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.!?")
    return cleaned or text


def _is_political_news_request(text: str, lowered: str) -> bool:
    """User asked for political news specifically."""
    if _POLITICS_RE.search(lowered) or _POLITICS_DEV_RE.search(text):
        return True
    return False


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

    # 1a. Explicit "web search" command — honour the user's direction.
    # "web search garnus", "google garera heru na", "online khoju".
    if _EXPLICIT_WEB_SEARCH_RE.search(lowered):
        return ("internet_search", {"query": _strip_command_tokens(text)})

    # 2. Minister role → government + who_is. For "current / अहिले"
    #    identity queries, parallel-fire a web search because Wikipedia
    #    caches lag Nepal's real political changes (observed: Yeti cited
    #    Sushila Karki as PM when Balen Shah was actually sitting PM —
    #    the wikipedia page hadn't updated). Returning internet_search
    #    as the preflight tool for the "current PM" shape gets fresh
    #    news over stale reference pages.
    role = detect_minister_role(text)
    if role:
        is_current = bool(
            re.search(r"\b(current|currently|now|ahile|right\s*now)\b", lowered)
            or re.search(r"(अहिले|हाल)", text)
        )
        if is_current:
            # Fresh news over stale reference. The search plugin's
            # Nepal-host filter keeps results on-domain; the
            # fabricated-source-name guard in bot.py catches any
            # "Reuters/AP" flim-flam.
            role_label = role.replace("_", " ")
            return ("internet_search", {"query": f"current {role_label} of Nepal 2026"})
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

    # Ticker + company-details requests — force internet_search.
    # OSINT's trading endpoints have sparse ticker coverage (typically
    # just the latest bonus-share / dividend announcement), so for
    # "RURU ko details / project information / background" the right
    # tool is a web search against the company's own site + icranepal
    # + doed.gov.np + merolagani. Without this rule Sarvam tends to
    # call OSINT trading and come back with a one-line answer.
    ticker = _extract_ticker(text)
    wants_details = bool(
        re.search(
            r"\b(details?|information|info|background|profile|barima|"
            r"overview|project|company|baare|barema|barima)\b",
            lowered,
            re.IGNORECASE,
        )
        or re.search(r"(विवरण|जानकारी|बारे|बारेमा|प्रोफाइल)", text)
    )
    if ticker and wants_details:
        # Build a rich web query: "RURU hydropower Nepal project details"
        parts = [ticker, "Nepal"]
        if re.search(r"hydro|jal", lowered):
            parts.append("hydropower")
        parts.append("project details")
        return ("internet_search", {"query": " ".join(parts)})

    # Bare ticker (no "details" qualifier) → fall through, let Sarvam
    # decide. If it picks OSINT it gets the ticker news; if it picks
    # internet_search it gets background. Either is acceptable.

    # 4.5 Political news — "political samachar" / "राजनीतिक खबर".
    # Routes to government intent which hits govt-decisions +
    # announcements + political-categorised stories. Placed before
    # the debt/parliament/govt rules so the more specific news
    # framing wins (we want *news stories*, not a cabinet dump).
    if _is_political_news_request(text, lowered):
        return (
            "get_nepal_live_context",
            {"intent": "government", "focus": "political_news"},
        )

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
