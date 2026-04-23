"""
tools/fetch/plugin.py — direct URL reader.

Companion to `internet_search`. Use when Yeti already knows the URL
(from an OSINT citation, an earlier search result, a user-pasted link):
skipping the SERP round-trip is roughly a 1-2 second latency save and
avoids burning a DDG query.

Safety:
  * http/https only.
  * Timeout + per-page char cap reused from tools.search.plugin so
    context sizing stays consistent.
  * Bare IP + localhost addresses rejected to avoid SSRF from a
    hallucinated URL.
"""
from __future__ import annotations

import ipaddress
import logging
import re
import urllib.parse
from typing import Any

import httpx

from core.tool_contracts import (
    ToolCategory,
    ToolContext,
    ToolParam,
    ToolResult,
    ToolSpec,
)
from core.tool_registry import get_registry
from tools.search.plugin import _UA, _extract_main_text

logger = logging.getLogger("yetidai.fetch")


# Per-fetch budgets — a bit larger than search plugin's per-result
# budget because this is the *primary* context on a focused read.
FETCH_TIMEOUT_SECONDS = 8.0
MAX_CHARS = 4000


FETCH_SPEC = ToolSpec(
    tool_id="fetch.url",
    name="fetch_url",
    description=(
        "Fetch and extract the main readable text of a specific URL. "
        "Use when you already have a URL to read — e.g. following a "
        "citation from `get_nepal_live_context` or `internet_search`, or "
        "reading a page the user pasted.\n\n"
        "OUTPUT: up to ~4 KB of cleaned article/page text. Nav menus, "
        "ads, scripts, and cookie banners are stripped.\n\n"
        "LIMITS:\n"
        "  • http/https only.\n"
        "  • One URL per call — call in parallel if you need multiple.\n"
        "  • Truncated at a sentence boundary; longer pages are clipped."
    ),
    category=ToolCategory.UTILITY,
    parameters=[
        ToolParam(
            name="url",
            type="string",
            description="Absolute http(s) URL to fetch.",
            required=True,
            examples=[
                "https://www.nrb.org.np/contents/uploads/2026/03/Macroeconomic-Situation.pdf",
                "https://thehimalayantimes.com/nepal/…",
                "https://github.com/anthropics/claude-code",
            ],
        ),
    ],
    timeout_seconds=15.0,
)


_SCHEME_RE = re.compile(r"^https?://", re.IGNORECASE)


def _is_safe_url(url: str) -> tuple[bool, str | None]:
    """Reject non-http(s), loopback, link-local, and private-IP targets."""
    if not _SCHEME_RE.match(url):
        return False, "url must start with http:// or https://"
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception as exc:
        return False, f"malformed URL: {exc}"
    host = (parsed.hostname or "").strip().lower()
    if not host:
        return False, "missing host"
    # Block localhost shortcuts.
    if host in {"localhost", "localhost.localdomain"}:
        return False, "localhost not allowed"
    # Block bare IPs that fall into private / loopback / link-local ranges.
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip is not None and (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved):
        return False, "private/loopback IP not allowed"
    return True, None


async def handle_fetch(ctx: ToolContext, arguments: dict[str, Any]) -> ToolResult:
    url = arguments.get("url")
    if not url or not isinstance(url, str):
        return ToolResult(tool_id=FETCH_SPEC.tool_id, success=False, error="Missing url")

    ok, err = _is_safe_url(url.strip())
    if not ok:
        return ToolResult(tool_id=FETCH_SPEC.tool_id, success=False, error=err)

    try:
        async with httpx.AsyncClient(timeout=FETCH_TIMEOUT_SECONDS) as client:
            resp = await client.get(
                url.strip(),
                headers={"User-Agent": _UA, "Accept": "text/html,application/xhtml+xml"},
                follow_redirects=True,
            )
            resp.raise_for_status()
            ctype = resp.headers.get("content-type", "").lower()
            if "html" not in ctype and "text" not in ctype:
                return ToolResult(
                    tool_id=FETCH_SPEC.tool_id,
                    success=False,
                    error=f"unsupported content-type: {ctype or 'unknown'}",
                )
            text = _extract_main_text(resp.text, cap=MAX_CHARS)

        if not text:
            return ToolResult(
                tool_id=FETCH_SPEC.tool_id,
                success=True,
                content=f"(no extractable text)\nSource: {url}",
            )

        content = f"{text}\n\nSource: {url}"
        return ToolResult(
            tool_id=FETCH_SPEC.tool_id,
            success=True,
            content=content,
            meta={"bytes": len(text)},
        )
    except httpx.HTTPStatusError as exc:
        logger.info("fetch_url HTTP %s for %s", exc.response.status_code, url)
        return ToolResult(
            tool_id=FETCH_SPEC.tool_id,
            success=False,
            error=f"HTTP {exc.response.status_code}",
        )
    except Exception as exc:
        logger.exception("fetch_url failed")
        return ToolResult(
            tool_id=FETCH_SPEC.tool_id,
            success=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def register() -> None:
    get_registry().register(FETCH_SPEC, handle_fetch)
