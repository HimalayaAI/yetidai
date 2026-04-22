"""
bot.py — YetiDai Discord bot with tool-calling support.

Flow:
    1. User sends message → bot builds message list
    2. Sends to Sarvam with tools array from the ToolRegistry
    3. If Sarvam returns tool_calls → execute via registry → send results back
    4. Sarvam produces final text answer → bot sends to Discord

Backend: Sarvam (`sarvam-30b`) via the `sarvamai` async client. Configured
through `SARVAM_API_KEY` in the environment.
"""
import discord
import json
import logging
import os
import re
import time
import uuid
from dotenv import load_dotenv
from sarvamai import AsyncSarvamAI
from functionality import functional

# ── Core framework ────────────────────────────────────────────────
from core.tool_registry import get_registry
from core.tool_contracts import ToolContext
from core.output_validator import validate_answer, build_fix_message
from core.request_log import log_turn

# ── Register plugins ──────────────────────────────────────────────
import tools.osint.plugin as osint_plugin
import tools.search.plugin as search_plugin

osint_plugin.register()
search_plugin.register()
# ── Initialization ────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("yetidai")

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

llm_client = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)
LLM_MODEL = os.getenv("SARVAM_ROUTER_MODEL", "sarvam-30b")
YETI_BACKEND = "sarvam"
logger.info("Using Sarvam backend (model=%s).", LLM_MODEL)

with open("systemPrompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

intents = discord.Intents.default()
intents.message_content = True
bot = discord.Client(intents=intents)

chad = functional(bot=bot)

registry = get_registry()

# Safety cap: max tool-call round-trips before forcing a text answer
MAX_TOOL_ROUNDS = 5

# Discord hard limits we need to respect when sending content.
_DISCORD_MSG_LIMIT = 2000
_DISCORD_EMBED_FIELD_VALUE_LIMIT = 1024
_DISCORD_EMBED_FOOTER_LIMIT = 2048

_URL_RE = re.compile(r"https?://[^\s)\]\"<>]+")

# Generic-apology strings the bot itself emits on failure. We never want
# these replayed back to Sarvam as assistant history — otherwise Sarvam
# learns to parrot the apology on healthy turns.
_BOT_APOLOGY_PREFIXES = (
    "माफ गर्नुहोस्, एउटा प्राविधिक समस्या",
    "माफ गर्नुहोस्, उत्तर तयार गर्न सकिएन",
)
_GENERIC_TECH_ERROR = (
    "माफ गर्नुहोस्, एउटा प्राविधिक समस्या देखियो। कृपया फेरि प्रयास गर्नुहोस्।"
)


def _is_bot_apology(content: str) -> bool:
    """True if a message looks like one of our own generic failure messages."""
    if not content:
        return False
    stripped = content.lstrip()
    return any(stripped.startswith(p) for p in _BOT_APOLOGY_PREFIXES)


def _extract_urls(text: str | None) -> list[str]:
    """Pull http(s) URLs out of tool output for citation embeds."""
    if not text:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for url in _URL_RE.findall(text):
        url = url.rstrip(".,;:")
        if url not in seen:
            seen.add(url)
            out.append(url)
    return out


def _split_body_and_sources(answer: str) -> tuple[str, str]:
    """Separate the main Nepali body from the trailing `स्रोत:` citation line."""
    idx = answer.rfind("स्रोत:")
    if idx < 0:
        return answer, ""
    return answer[:idx].rstrip(), answer[idx:].strip()


def _chunk_for_discord(text: str, limit: int = _DISCORD_MSG_LIMIT) -> list[str]:
    """Break text into Discord-sized chunks at a newline or whitespace boundary.

    Prefers the last newline inside the window, then the last whitespace; only
    falls back to a hard slice when no boundary exists in the window (e.g. an
    unbroken URL or single long token).
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


def _safe_field_value(url: str) -> str:
    """Fit a URL into Discord's 1024-char embed field value limit."""
    if len(url) <= _DISCORD_EMBED_FIELD_VALUE_LIMIT:
        return url
    # Truncate with an ellipsis marker so users see the link was clipped.
    return url[: _DISCORD_EMBED_FIELD_VALUE_LIMIT - 1] + "…"


async def _send_discord(channel, answer: str, citation_urls: list[str]) -> None:
    """Send answer; attach a citations embed when we have URLs.

    Body chunks and the citations embed are sent independently: a failure to
    build or send the embed must not prevent the body from being delivered,
    and vice versa.
    """
    body, sources_line = _split_body_and_sources(answer)
    text = body if (citation_urls and body) else answer

    for chunk in _chunk_for_discord(text, _DISCORD_MSG_LIMIT):
        await channel.send(chunk)

    if not citation_urls:
        return

    try:
        embed = discord.Embed(title="स्रोत / Sources", color=0x2D72D2)
        for idx, url in enumerate(citation_urls[:5], start=1):
            embed.add_field(name=f"{idx}.", value=_safe_field_value(url), inline=False)
        if sources_line:
            embed.set_footer(text=sources_line[:_DISCORD_EMBED_FOOTER_LIMIT])
        await channel.send(embed=embed)
    except Exception:
        # Never let a bad citation embed clobber the answer the user already got.
        logger.exception("Failed to send citations embed (body already delivered)")


@bot.event
async def on_ready():
    tool_names = [t.name for t in registry.list_tools()]
    logger.info("Logged in as %s (ID: %s)", bot.user, bot.user.id)
    logger.info("Registered tools: %s", tool_names)


async def _run_llm_turn(messages, tools_array):
    """One Sarvam round-trip. Raised exceptions propagate to the caller."""
    return await llm_client.chat.completions(
        model=LLM_MODEL,
        messages=messages,
        tools=tools_array if tools_array else None,
        tool_choice="auto" if tools_array else None,
    )


@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    await chad.call(message)

    if not chad.user_input:
        return

    async with message.channel.typing():
        turn_id = uuid.uuid4().hex[:8]
        t0 = time.time()
        tool_calls_log: list[dict] = []
        fallback_used = False
        osint_endpoints_ok: list[str] = []
        osint_endpoints_failed: list[str] = []
        cache_stats: dict = {}
        tool_was_used = False
        validator_retries = 0
        ai_response = ""
        citation_urls: list[str] = []
        llm_failed = False

        # ── Build message list ────────────────────────────────────
        # No LLM traffic yet — these failures are local and trivial, but if
        # they raise we fall back to the generic error (see outermost guard).
        try:
            previous_messages = await chad.get_message_history(
                message.channel, limit=5,
            )

            import datetime
            today_str = datetime.date.today().strftime("%Y-%m-%d")
            dynamic_system_prompt = (
                f"{SYSTEM_PROMPT}\n\n# CURRENT DATE:\nToday's Date is: {today_str}"
            )
            messages = [{"role": "system", "content": dynamic_system_prompt}]

            for prev_msg in previous_messages:
                if prev_msg.id == message.id or not prev_msg.content.strip():
                    continue
                if prev_msg.author == bot.user:
                    # Skip our own prior apologies so Sarvam doesn't learn to
                    # parrot them. Anything else we sent is fair context.
                    if _is_bot_apology(prev_msg.content):
                        continue
                    messages.append({
                        "role": "assistant",
                        "content": prev_msg.content,
                    })
                else:
                    messages.append({
                        "role": "user",
                        "content": f"{prev_msg.author.name}: {prev_msg.content}",
                    })

            messages.append({"role": "user", "content": chad.user_input})
            tools_array = registry.openai_tools()
        except Exception:
            logger.exception("Failed building request context")
            await message.channel.send(_GENERIC_TECH_ERROR)
            log_turn(
                turn_id=turn_id,
                user_id=getattr(message.author, "id", None),
                channel_id=getattr(message.channel, "id", None),
                query=chad.user_input,
                tool_calls=tool_calls_log,
                fallback_used=fallback_used,
                osint_endpoints_ok=osint_endpoints_ok,
                osint_endpoints_failed=osint_endpoints_failed,
                cache=cache_stats,
                validator_retries=validator_retries,
                latency_ms=int((time.time() - t0) * 1000),
                backend=YETI_BACKEND,
                model=LLM_MODEL,
            )
            return

        # ── Tool-call loop ────────────────────────────────────────
        # The LLM decides whether to call tools via tool_choice="auto".
        # If it returns tool_calls, we execute them and loop. If it returns
        # a text answer, we break.
        response = None
        try:
            for _round in range(MAX_TOOL_ROUNDS):
                response = await _run_llm_turn(messages, tools_array)

                if not response or not getattr(response, "choices", None):
                    break
                choice = response.choices[0]

                tool_calls = getattr(choice.message, "tool_calls", None) or []
                if choice.finish_reason != "tool_calls" or not tool_calls:
                    break

                # Append the assistant message (with tool_calls) to history
                messages.append({
                    "role": "assistant",
                    "content": choice.message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                })

                ctx = ToolContext(
                    query=chad.user_input,
                    history=previous_messages,
                    llm_client=llm_client,
                    channel_id=message.channel.id,
                    user_id=message.author.id,
                )

                for tc in tool_calls:
                    tool_was_used = True
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        logger.warning(
                            "Bad tool_call arguments JSON for %s: %r",
                            tc.function.name, tc.function.arguments,
                        )
                        args = {}

                    try:
                        result = await registry.execute(tc.function.name, ctx, args)
                    except Exception:
                        # A single tool failing shouldn't poison the whole turn —
                        # feed an error result back so the LLM can recover or
                        # choose a different tool on the next round.
                        logger.exception(
                            "Tool %s raised; returning error to LLM", tc.function.name,
                        )
                        from core.tool_contracts import ToolResult
                        result = ToolResult(
                            success=False,
                            content=f"[TOOL_ERROR] {tc.function.name} failed internally.",
                        )

                    messages.append(result.to_tool_message(tc.id))
                    tool_calls_log.append({
                        "name": tc.function.name,
                        "args": args,
                        "success": result.success,
                    })
                    citation_urls.extend(_extract_urls(result.content))

                    if result.meta:
                        osint_endpoints_ok = result.meta.get(
                            "endpoints_ok", osint_endpoints_ok,
                        )
                        osint_endpoints_failed = result.meta.get(
                            "endpoints_failed", osint_endpoints_failed,
                        )
                        cache_stats = {
                            "hits": result.meta.get(
                                "cache_hits", cache_stats.get("hits", 0),
                            ),
                            "misses": result.meta.get(
                                "cache_misses", cache_stats.get("misses", 0),
                            ),
                        }

                    logger.info(
                        "Tool call: %s(args=%s) → success=%s",
                        tc.function.name, args, result.success,
                    )

                    # Auto-fallback: execute a second tool call in the same
                    # turn when the primary tool asked for it (e.g. OSINT
                    # returned no match → fall back to internet_search).
                    if result.trigger_fallback and result.fallback_tool:
                        fallback_call_id = f"autofb_{uuid.uuid4().hex[:8]}"
                        fb_args = result.fallback_args or {}
                        messages.append({
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [{
                                "id": fallback_call_id,
                                "type": "function",
                                "function": {
                                    "name": result.fallback_tool,
                                    "arguments": json.dumps(fb_args, ensure_ascii=False),
                                },
                            }],
                        })
                        try:
                            fb_result = await registry.execute(
                                result.fallback_tool, ctx, fb_args,
                            )
                        except Exception:
                            logger.exception(
                                "Auto-fallback tool %s raised", result.fallback_tool,
                            )
                            from core.tool_contracts import ToolResult
                            fb_result = ToolResult(
                                success=False,
                                content=f"[TOOL_ERROR] {result.fallback_tool} failed internally.",
                            )
                        messages.append(fb_result.to_tool_message(fallback_call_id))
                        tool_calls_log.append({
                            "name": result.fallback_tool,
                            "args": fb_args,
                            "success": fb_result.success,
                            "auto_fallback_from": tc.function.name,
                        })
                        citation_urls.extend(_extract_urls(fb_result.content))
                        fallback_used = True
                        logger.info(
                            "Auto-fallback %s → %s(args=%s) success=%s",
                            tc.function.name, result.fallback_tool,
                            fb_args, fb_result.success,
                        )

            # Extract final answer
            if response and getattr(response, "choices", None):
                ai_response = response.choices[0].message.content or ""
        except Exception:
            logger.exception("Sarvam call / tool loop failed")
            llm_failed = True

        # ── Validator + one-shot retry (never fatal) ──────────────
        if ai_response and not llm_failed:
            try:
                issues = validate_answer(ai_response, tool_was_used=tool_was_used)
                if issues:
                    logger.info("Validator issues: %s — retrying once.", issues)
                    messages.append({"role": "assistant", "content": ai_response})
                    messages.append({
                        "role": "system",
                        "content": build_fix_message(issues),
                    })
                    retry_resp = await llm_client.chat.completions(
                        model=LLM_MODEL,
                        messages=messages,
                        tools=None,
                        tool_choice=None,
                    )
                    retry_content = (
                        retry_resp.choices[0].message.content or ""
                    ) if retry_resp and getattr(retry_resp, "choices", None) else ""
                    if retry_content:
                        ai_response = retry_content
                        validator_retries = 1
            except Exception:
                # Retry failing is not a user-visible error: we already have
                # a valid-enough answer in ai_response.
                logger.exception("Validator retry failed; keeping original answer")

        # ── Send to Discord ──────────────────────────────────────
        try:
            if ai_response:
                await _send_discord(message.channel, ai_response, citation_urls)
            elif llm_failed:
                await message.channel.send(_GENERIC_TECH_ERROR)
            else:
                await message.channel.send("माफ गर्नुहोस्, उत्तर तयार गर्न सकिएन।")
        except Exception:
            logger.exception("Discord send failed")

        log_turn(
            turn_id=turn_id,
            user_id=getattr(message.author, "id", None),
            channel_id=getattr(message.channel, "id", None),
            query=chad.user_input,
            tool_calls=tool_calls_log,
            fallback_used=fallback_used,
            osint_endpoints_ok=osint_endpoints_ok,
            osint_endpoints_failed=osint_endpoints_failed,
            cache=cache_stats,
            validator_retries=validator_retries,
            latency_ms=int((time.time() - t0) * 1000),
            backend=YETI_BACKEND,
            model=LLM_MODEL,
        )


if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
