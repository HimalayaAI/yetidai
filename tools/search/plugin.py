"""
tools/search/plugin.py — general-purpose web search + page extraction.

Design:
  * SERP layer: DuckDuckGo HTML (no API key, no rate-limit token).
  * Page-read layer: fetch top N result pages in parallel (httpx async),
    extract main text with a BeautifulSoup heuristic, truncate to a per-
    page budget so the tool result fits Sarvam's context window.
  * Token budget: MAX_TOTAL_CHARS bounds the whole tool message.
  * Fail-soft: a slow/403/404 result is replaced by its DDG snippet, not
    dropped — the LLM still sees the URL and a usable summary.

This upgrade is the main thing that keeps Yeti off its training cutoff:
the old plugin returned 5 one-line snippets, which is barely more than
"here is a URL, trust yourself". Reading real page text means the final
Nepali answer can quote specific facts.
"""
from __future__ import annotations

import asyncio
import logging
import re
import urllib.parse
from typing import Any

import httpx
from bs4 import BeautifulSoup

from core.tool_contracts import (
    ToolCategory,
    ToolContext,
    ToolParam,
    ToolResult,
    ToolSpec,
)
from core.tool_registry import get_registry

logger = logging.getLogger("yetidai.search")


# ── Budgets ───────────────────────────────────────────────────────
# Tuned so a single internet_search call fits comfortably inside
# Sarvam's prompt budget even when combined with a parallel OSINT call.
MAX_RESULTS = 5          # DDG results requested
MAX_READ_PAGES = 3       # how many pages we actually fetch+extract
MAX_PAGE_CHARS = 1400    # per-page extracted-text cap
MAX_TOTAL_CHARS = 4800   # whole-tool-message cap

PAGE_TIMEOUT_SECONDS = 6.0
SERP_TIMEOUT_SECONDS = 8.0

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.1 Safari/605.1.15"
)

# Hosts that proved to pollute "Nepal news" SERPs with Hindi / Indian
# political content. When the query looks Nepal-scoped we drop matches
# on these hosts so they never reach the user as "today's Nepal news".
_NON_NEPAL_NEWS_HOSTS: frozenset[str] = frozenset({
    "aajtak.in", "indiatv.in", "amarujala.com", "news18.com",
    "navbharattimes.indiatimes.com", "jagran.com", "bhaskar.com",
    "hindustantimes.com", "ndtv.com", "zeenews.india.com",
    "timesofindia.indiatimes.com", "news24online.com",
    # Astrology / panchang pollution — unrelated to news
    "hamropatro.com",
})

# HTML elements that never contain useful page content. Dropped before
# text extraction so nav/cookie banners don't eat the char budget.
_NOISE_SELECTORS = (
    "script", "style", "nav", "footer", "aside", "form",
    "noscript", "iframe", "svg", "button",
)

# Rough priority for text containers. First match wins per page.
_MAIN_SELECTORS = (
    "article",
    "main",
    "[role=main]",
    "div#content",
    "div.content",
    "div.article",
    "div.post",
    "div.entry-content",
    "div.story",
    "div.news",
)


SEARCH_SPEC = ToolSpec(
    tool_id="search.internet",
    name="internet_search",
    description=(
        "Live web search + page reading. Use for anything that depends on "
        "real-world information, since your training data is stale.\n\n"
        "WHAT IT DOES:\n"
        "  1. Runs a DuckDuckGo query for `query`.\n"
        "  2. Fetches the top 3 result pages and extracts their main "
        "article text.\n"
        "  3. Returns a compact bundle: for each result, the URL, a DDG "
        "snippet, and up to ~1.4 KB of extracted page text.\n\n"
        "USE FOR:\n"
        "  • Non-Nepal topics (world news, foreign leaders, sports, "
        "science, product info).\n"
        "  • Fact-checking a recent claim against primary sources.\n"
        "  • Reading up-to-date documentation, release notes, changelogs.\n"
        "  • When `get_nepal_live_context` returned nothing useful.\n\n"
        "DO NOT USE FOR:\n"
        "  • Nepal public information — `get_nepal_live_context` has "
        "better coverage (NRB, NEPSE, cabinet, parliament).\n"
        "  • Following a known URL — use `fetch_url` instead, it's "
        "faster and doesn't burn a SERP query.\n"
        "  • Analysing a GitHub repo — use `analyze_github_repo`.\n\n"
        "QUERY TIPS:\n"
        "  • English works best on DDG.\n"
        "  • Include the current year for time-sensitive queries.\n"
        "  • Quote multi-word names or exact phrases.\n\n"
        "OUTPUT: English/multilingual raw text. You MUST summarise / "
        "translate into Nepali in the final answer — never paste raw "
        "English to the user. Cite at least one URL with a `स्रोत:` line."
    ),
    category=ToolCategory.UTILITY,
    parameters=[
        ToolParam(
            name="query",
            type="string",
            description=(
                "Concise English search query. Include year for recency, "
                "quote exact names/phrases."
            ),
            required=True,
            examples=[
                "UEFA Champions League 2026 winner",
                "\"Llama 4\" release date",
                "Mount Everest height meters",
                "current president of India 2026",
            ],
        ),
        ToolParam(
            name="read_pages",
            type="integer",
            description=(
                f"How many of the top DDG results to fetch and extract. "
                f"Default {MAX_READ_PAGES}. Set to 0 for snippets-only "
                f"(fast but shallow). Capped at {MAX_RESULTS}."
            ),
            required=False,
        ),
    ],
    timeout_seconds=25.0,
)


# ── SERP (DuckDuckGo HTML) ────────────────────────────────────────

def _clean_ddg_href(raw: str) -> str:
    """DDG wraps result URLs in `/l/?uddg=<encoded>`. Unwrap when we see it."""
    if raw.startswith("/l/?") or raw.startswith("//duckduckgo.com/l/?"):
        parsed = urllib.parse.urlparse(raw)
        qs = urllib.parse.parse_qs(parsed.query)
        if "uddg" in qs and qs["uddg"]:
            return urllib.parse.unquote(qs["uddg"][0])
    return raw


def _is_nepal_scoped_query(query: str) -> bool:
    """True when the query explicitly scopes the search to Nepal.

    Used to decide whether to apply the Indian-news host deny-list —
    we only drop those hosts when the user (or the OSINT fallback)
    said "Nepal". For a global search like "UEFA 2025 winner" we
    don't want to filter any hosts.
    """
    if not query:
        return False
    q = query.lower()
    return "nepal" in q or "नेपाल" in query


def _host_from_url(url: str) -> str | None:
    """Lowercase hostname without leading www. — used for host filters."""
    import urllib.parse as _up
    try:
        host = (_up.urlparse(url).hostname or "").lower()
    except Exception:
        return None
    if host.startswith("www."):
        host = host[4:]
    return host or None


def _apply_nepal_filter(results: list[dict[str, str]], query: str) -> list[dict[str, str]]:
    """Drop known non-Nepal-news hosts when the query is Nepal-scoped."""
    if not _is_nepal_scoped_query(query):
        return results
    kept: list[dict[str, str]] = []
    dropped: list[str] = []
    for r in results:
        host = _host_from_url(r.get("href", ""))
        if host and host in _NON_NEPAL_NEWS_HOSTS:
            dropped.append(host)
            continue
        kept.append(r)
    if dropped:
        logger.info(
            "Dropped %d non-Nepal hosts from Nepal-scoped query %r: %s",
            len(dropped), query, sorted(set(dropped)),
        )
    return kept


async def _ddg_search(client: httpx.AsyncClient, query: str) -> list[dict[str, str]]:
    """Return up to MAX_RESULTS DDG results as dicts {title, snippet, href}."""
    url = "https://html.duckduckgo.com/html/"
    response = await client.post(
        url,
        data={"q": query},
        headers={"User-Agent": _UA},
        timeout=SERP_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    out: list[dict[str, str]] = []
    for node in soup.find_all("div", class_="result"):
        title_node = node.find("a", class_="result__a")
        snippet_node = node.find("a", class_="result__snippet") or node.find("div", class_="result__snippet")
        href_node = node.find("a", class_="result__url") or title_node
        if not title_node or not href_node:
            continue
        href = _clean_ddg_href(href_node.get("href", "").strip())
        if not href or not href.startswith("http"):
            continue
        out.append({
            "title": title_node.get_text(" ", strip=True),
            "snippet": snippet_node.get_text(" ", strip=True) if snippet_node else "",
            "href": href,
        })
        if len(out) >= MAX_RESULTS + 5:  # extra headroom for filtering
            break
    return _apply_nepal_filter(out, query)[:MAX_RESULTS]


# ── Page extraction ───────────────────────────────────────────────

_WHITESPACE_RE = re.compile(r"\s+")


def _extract_main_text(html: str, *, cap: int = MAX_PAGE_CHARS) -> str:
    """Cheap readability-ish extractor.

    Strategy:
      1. Drop obviously non-content tags (scripts, nav, forms).
      2. Look for a main-content container via a prioritised selector list.
         Fall back to <body> if none match.
      3. Concatenate block text, collapse whitespace, truncate to `cap`.

    Not as good as python-readability, but has no extra dependency and is
    deterministic — fine for a 1-2 KB snippet that the LLM will re-read.
    """
    soup = BeautifulSoup(html, "html.parser")
    for sel in _NOISE_SELECTORS:
        for tag in soup.find_all(sel):
            tag.decompose()

    container = None
    for selector in _MAIN_SELECTORS:
        container = soup.select_one(selector)
        if container is not None:
            break
    if container is None:
        container = soup.body or soup

    text = container.get_text(" ", strip=True)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    if len(text) <= cap:
        return text
    # Truncate at last sentence boundary inside the cap window.
    window = text[:cap]
    for sep in (". ", "। ", "! ", "? "):
        idx = window.rfind(sep)
        if idx > cap // 2:
            return window[: idx + 1].strip() + " …"
    return window.rstrip() + " …"


async def _fetch_and_extract(
    client: httpx.AsyncClient,
    result: dict[str, str],
) -> dict[str, str]:
    """Return the original result dict augmented with `body` (extracted text).

    On any failure the `body` falls back to the DDG `snippet` so downstream
    still has *something* to cite. Errors are logged but not raised.
    """
    href = result["href"]
    try:
        resp = await client.get(
            href,
            headers={"User-Agent": _UA, "Accept": "text/html"},
            timeout=PAGE_TIMEOUT_SECONDS,
            follow_redirects=True,
        )
        resp.raise_for_status()
        ctype = resp.headers.get("content-type", "")
        if "html" not in ctype.lower():
            raise ValueError(f"non-html content-type: {ctype}")
        body = _extract_main_text(resp.text)
        if not body:
            body = result.get("snippet", "") or "(no extractable text)"
    except Exception as exc:
        logger.info("fetch_and_extract failed for %s: %s", href, exc)
        body = result.get("snippet", "") or "(fetch failed)"
    return {**result, "body": body}


# ── Handler ───────────────────────────────────────────────────────

def _format_results(results: list[dict[str, str]]) -> str:
    """Render fetched results into a compact, citable block."""
    lines: list[str] = []
    used = 0
    for idx, r in enumerate(results, start=1):
        title = r.get("title", "").strip() or "(untitled)"
        body = r.get("body") or r.get("snippet") or ""
        href = r.get("href", "")
        block = f"[{idx}] {title}\n{body}\nSource: {href}"
        if used + len(block) + 2 > MAX_TOTAL_CHARS:
            block = block[: max(0, MAX_TOTAL_CHARS - used - 20)].rstrip() + " …"
            lines.append(block)
            break
        lines.append(block)
        used += len(block) + 2
    return "Internet Search Results:\n\n" + "\n\n".join(lines)


async def handle_search(ctx: ToolContext, arguments: dict[str, Any]) -> ToolResult:
    query = arguments.get("query")
    if not query or not isinstance(query, str):
        return ToolResult(tool_id=SEARCH_SPEC.tool_id, success=False, error="Missing query")

    try:
        read_pages = int(arguments.get("read_pages", MAX_READ_PAGES))
    except (TypeError, ValueError):
        read_pages = MAX_READ_PAGES
    read_pages = max(0, min(read_pages, MAX_RESULTS))

    try:
        async with httpx.AsyncClient(
            http2=False,
            timeout=PAGE_TIMEOUT_SECONDS,
        ) as client:
            serp = await _ddg_search(client, query)
            if not serp:
                return ToolResult(
                    tool_id=SEARCH_SPEC.tool_id,
                    success=True,
                    content="No search results were found for this query.",
                )

            if read_pages == 0:
                enriched = [{**r, "body": r.get("snippet", "")} for r in serp]
            else:
                fetched = await asyncio.gather(
                    *[_fetch_and_extract(client, r) for r in serp[:read_pages]],
                    return_exceptions=False,
                )
                enriched = list(fetched) + [
                    {**r, "body": r.get("snippet", "")}
                    for r in serp[read_pages:]
                ]

        content = _format_results(enriched)
        meta = {
            "serp_count": len(serp),
            "pages_read": min(read_pages, len(serp)),
        }
        return ToolResult(
            tool_id=SEARCH_SPEC.tool_id,
            success=True,
            content=content,
            meta=meta,
        )
    except Exception as exc:
        logger.exception("internet_search failed")
        return ToolResult(
            tool_id=SEARCH_SPEC.tool_id,
            success=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def register() -> None:
    get_registry().register(SEARCH_SPEC, handle_search)
