"""Social-media feed tool.

The background Discord task posts new tweets automatically. This tool exposes
the same SQLite-backed feed state to YetiDai's normal auto tool-calling loop,
so users can ask what recently arrived without triggering a fresh scrape.
"""
from __future__ import annotations

import json
from typing import Any

from core.tool_contracts import (
    ToolCategory,
    ToolContext,
    ToolParam,
    ToolResult,
    ToolSpec,
)
from core.tool_registry import get_registry
from tools.social.twitter_feed import get_social_feed_service


SOCIAL_FEED_SPEC = ToolSpec(
    tool_id="social.twitter.recent_feed",
    name="get_social_media_feed",
    description=(
        "Return recent AI/social-media posts that YetiDai's automatic "
        "#social-media feed has already seen. Use this when the user asks "
        "what the social feed, AI handles, Twitter/X list, or recent scraped "
        "tweets are saying. This tool reads local feed state and does not "
        "perform a fresh web scrape."
    ),
    category=ToolCategory.DATA,
    parameters=[
        ToolParam(
            name="limit",
            type="integer",
            description="How many recent feed items to return. Default 10, max 50.",
            required=False,
        ),
        ToolParam(
            name="author",
            type="string",
            description="Optional X/Twitter username, with or without @.",
            required=False,
            examples=["karpathy", "sama", "HimalayaAILabs"],
        ),
    ],
    timeout_seconds=5.0,
)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


async def handle_social_feed(ctx: ToolContext, arguments: dict[str, Any]) -> ToolResult:
    limit = max(1, min(_safe_int(arguments.get("limit"), 10), 50))
    author = arguments.get("author")
    if isinstance(author, str):
        author = author.strip().lstrip("@") or None
    else:
        author = None

    rows = get_social_feed_service().recent(limit=limit, author=author)
    if not rows:
        scope = f" for @{author}" if author else ""
        return ToolResult(
            tool_id=SOCIAL_FEED_SPEC.tool_id,
            success=True,
            content=f"No social-media feed items are stored yet{scope}.",
            raw_data={"items": []},
        )

    lines = ["Recent #social-media feed items:"]
    for idx, row in enumerate(rows, start=1):
        media_urls = json.loads(row.get("media_urls_json") or "[]")
        video_urls = json.loads(row.get("video_urls_json") or "[]")
        media_note = ""
        if video_urls:
            media_note = " [video]"
        elif media_urls:
            media_note = f" [images: {len(media_urls)}]"
        timestamp = row.get("tweeted_at") or row.get("first_seen_at") or ""
        text = (row.get("text") or "").replace("\n", " ").strip()
        if len(text) > 240:
            text = f"{text[:237]}..."
        lines.append(
            f"{idx}. @{row['author_username']}{media_note} ({timestamp})\n"
            f"   {text}\n"
            f"   {row['x_url']}"
        )

    return ToolResult(
        tool_id=SOCIAL_FEED_SPEC.tool_id,
        success=True,
        content="\n".join(lines),
        raw_data={"items": rows},
    )


def register() -> None:
    get_registry().register(SOCIAL_FEED_SPEC, handle_social_feed)
