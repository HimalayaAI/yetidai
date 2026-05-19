"""Social-media feed tool.

The background Discord task posts new tweets automatically. This tool exposes
that SQLite-backed feed state to YetiDai's normal auto tool-calling loop and
can optionally do a bounded live refresh before answering.
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
        "Return recent AI/social-media posts from YetiDai's Twitter/X feed. "
        "Use this when the user asks what the social feed, AI handles, "
        "Twitter/X list, or recent scraped tweets are saying. Set refresh=true "
        "for live/latest/new/update requests; it performs a bounded Nitter "
        "refresh, stores newly seen tweets, and avoids repeating old tweet IDs."
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
        ToolParam(
            name="refresh",
            type="boolean",
            description=(
                "When true, fetch fresh posts before returning recent items. "
                "Exact-author refreshes only that author; broad refreshes scan "
                "a bounded set of configured accounts."
            ),
            required=False,
        ),
    ],
    timeout_seconds=25.0,
)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _format_rows(rows: list[dict], *, heading: str) -> str:
    lines = [heading]
    for idx, row in enumerate(rows, start=1):
        media_urls = json.loads(row.get("media_urls_json") or "[]")
        video_urls = json.loads(row.get("video_urls_json") or "[]")
        media_bits: list[str] = []
        if video_urls:
            media_bits.append(f"video: {len(video_urls)}")
        if media_urls:
            media_bits.append(f"images: {len(media_urls)}")
        media_note = f" [{', '.join(media_bits)}]" if media_bits else ""
        timestamp = row.get("tweeted_at") or row.get("first_seen_at") or ""
        text = (row.get("text") or "").replace("\n", " ").strip()
        if len(text) > 320:
            text = f"{text[:317]}..."
        lines.append(
            f"{idx}. @{row['author_username']}{media_note} ({timestamp})\n"
            f"   {text}\n"
            f"   {row['x_url']}"
        )
    return "\n".join(lines)


async def handle_social_feed(ctx: ToolContext, arguments: dict[str, Any]) -> ToolResult:
    limit = max(1, min(_safe_int(arguments.get("limit"), 10), 50))
    refresh = _safe_bool(arguments.get("refresh"), False)
    author = arguments.get("author")
    if isinstance(author, str):
        author = author.strip().lstrip("@") or None
    else:
        author = None

    service = get_social_feed_service()
    refresh_result = None
    if refresh:
        refresh_result = await service.refresh_recent(author=author)

    rows = service.recent(limit=limit, author=author)
    if not rows:
        scope = f" for @{author}" if author else ""
        if refresh_result and refresh_result.errors:
            errors = "; ".join(refresh_result.errors[:5])
            message = f"No live social-media feed items found{scope}. Scrape warnings: {errors}"
        else:
            message = f"No social-media feed items are stored yet{scope}."
        return ToolResult(
            tool_id=SOCIAL_FEED_SPEC.tool_id,
            success=True,
            content=message,
            raw_data={"items": []},
            meta={
                "refresh": refresh,
                "new_items": len(refresh_result.tweets_to_post) if refresh_result else 0,
                "scanned": refresh_result.scanned if refresh_result else 0,
                "errors": refresh_result.errors if refresh_result else [],
            },
        )

    heading = "Recent #social-media feed items:"
    if refresh_result:
        if refresh_result.tweets_to_post:
            heading = (
                f"Live #social-media refresh found {len(refresh_result.tweets_to_post)} "
                "new item(s). Recent feed items:"
            )
        else:
            heading = "Live #social-media refresh found no new items. Recent stored feed items:"

    return ToolResult(
        tool_id=SOCIAL_FEED_SPEC.tool_id,
        success=True,
        content=_format_rows(rows, heading=heading),
        raw_data={"items": rows},
        meta={
            "refresh": refresh,
            "new_items": len(refresh_result.tweets_to_post) if refresh_result else 0,
            "scanned": refresh_result.scanned if refresh_result else 0,
            "errors": refresh_result.errors if refresh_result else [],
        },
    )


def register() -> None:
    get_registry().register(SOCIAL_FEED_SPEC, handle_social_feed)
