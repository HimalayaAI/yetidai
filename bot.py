"""
bot.py — YetiDai Discord bot with tool-calling support.

Flow:
    1. User sends message → bot builds message list
    2. Sends to Sarvam with tools array from the ToolRegistry
    3. If Sarvam returns tool_calls → execute via registry → send results back
    4. Sarvam produces final text answer → bot sends to Discord

Backend: Sarvam (`sarvam-30b`) via the `sarvamai` async client. Configured
through `SARVAM_API_KEY` in the environment.

Resilience:
  - Per-phase try/except so one failure can't masquerade as another.
  - Sarvam calls wrapped in asyncio.wait_for with one transient retry.
  - Last tool-round forces tools=None so the LLM must emit text.
  - Deterministic fixups (ASCII→Devanagari digits, सरस्रोत line injection)
    run before invoking a second LLM turn for validator nudges.
  - Error messages classified into distinct Nepali strings and tagged with
    the turn_id for log correlation.
"""
import asyncio
import datetime
import json
import logging
import os
import random
import re
import time
import uuid

import discord
from dotenv import load_dotenv
from sarvamai import AsyncSarvamAI

from functionality import functional

# ── Core framework ────────────────────────────────────────────────
from core.tool_registry import get_registry
from core.tool_contracts import ToolContext, ToolResult
from core.output_validator import validate_answer, build_fix_message
from core.request_log import log_turn
from core.bot_helpers import (
    DISCORD_EMBED_FOOTER_LIMIT,
    DISCORD_MSG_LIMIT,
    GENERIC_TECH_ERROR,
    chunk_for_discord,
    classify_llm_error,
    ensure_sources_line,
    extract_urls,
    is_bot_apology,
    is_transient_llm_error,
    normalize_digits,
    safe_field_value,
    split_body_and_sources,
    with_turn_id,
)

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
SARVAM_TIMEOUT_SECONDS = float(os.getenv("SARVAM_TIMEOUT_SECONDS", "25"))
YETI_BACKEND = "sarvam"
logger.info(
    "Using Sarvam backend (model=%s, timeout=%.1fs).",
    LLM_MODEL, SARVAM_TIMEOUT_SECONDS,
)

with open("systemPrompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

intents = discord.Intents.default()
intents.message_content = True
bot = discord.Client(intents=intents)

chad = functional(bot=bot)

registry = get_registry()

# Safety cap: max tool-call round-trips before forcing a text answer.
MAX_TOOL_ROUNDS = 5


async def _send_discord(channel, answer: str, citation_urls: list[str]) -> None:
    """Send answer; attach a citations embed when we have URLs.

    Body chunks and the citations embed are sent independently: a failure to
    build or send the embed must not prevent the body from being delivered.
    """
    body, sources_line = split_body_and_sources(answer)
    text = body if (citation_urls and body) else answer

    for chunk in chunk_for_discord(text, DISCORD_MSG_LIMIT):
        await channel.send(chunk)

    if not citation_urls:
        return

    try:
        embed = discord.Embed(title="स्रोत / Sources", color=0x2D72D2)
        for idx, url in enumerate(citation_urls[:5], start=1):
            embed.add_field(name=f"{idx}.", value=safe_field_value(url), inline=False)
        if sources_line:
            embed.set_footer(text=sources_line[:DISCORD_EMBED_FOOTER_LIMIT])
        await channel.send(embed=embed)
    except Exception:
        logger.exception("Failed to send citations embed (body already delivered)")


async def _run_llm_turn(messages, tools_array, *, tool_choice: str | None):
    """One Sarvam round-trip with timeout and one transient retry.

    Raises the last exception if both attempts fail. Non-transient errors
    (auth, schema, ...) raise on the first attempt with no retry.
    """
    last_exc: BaseException | None = None
    for attempt in range(2):
        try:
            return await asyncio.wait_for(
                llm_client.chat.completions(
                    model=LLM_MODEL,
                    messages=messages,
                    tools=tools_array if tools_array else None,
                    tool_choice=tool_choice if tools_array else None,
                ),
                timeout=SARVAM_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as exc:
            last_exc = exc
            logger.warning("Sarvam timeout (attempt %d/2)", attempt + 1)
        except Exception as exc:
            if not is_transient_llm_error(exc):
                raise
            last_exc = exc
            logger.warning(
                "Sarvam transient error (attempt %d/2): %s", attempt + 1, exc,
            )
        if attempt == 0:
            await asyncio.sleep(0.5 + random.random() * 0.5)
    assert last_exc is not None
    raise last_exc


@bot.event
async def on_ready():
    tool_names = [t.name for t in registry.list_tools()]
    logger.info("Logged in as %s (ID: %s)", bot.user, bot.user.id)
    logger.info("Registered tools: %s", tool_names)


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
        llm_exc: BaseException | None = None

        # ── Build message list ────────────────────────────────────
        try:
            previous_messages = await chad.get_message_history(
                message.channel, limit=5,
            )

            today_str = datetime.date.today().strftime("%Y-%m-%d")
            dynamic_system_prompt = (
                f"{SYSTEM_PROMPT}\n\n# CURRENT DATE:\nToday's Date is: {today_str}"
            )
            messages = [{"role": "system", "content": dynamic_system_prompt}]

            for prev_msg in previous_messages:
                if prev_msg.id == message.id or not prev_msg.content.strip():
                    continue
                if prev_msg.author == bot.user:
                    if is_bot_apology(prev_msg.content):
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
            await message.channel.send(with_turn_id(GENERIC_TECH_ERROR, turn_id))
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
        # On the final round we strip tools to force Sarvam to emit text,
        # eliminating the "ran out of rounds with empty ai_response" failure
        # mode.
        response = None
        try:
            for _round in range(MAX_TOOL_ROUNDS):
                is_last_round = (_round == MAX_TOOL_ROUNDS - 1)
                response = await _run_llm_turn(
                    messages,
                    tools_array if (tools_array and not is_last_round) else None,
                    tool_choice=("auto" if (tools_array and not is_last_round) else None),
                )

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
                        logger.exception(
                            "Tool %s raised; returning error to LLM", tc.function.name,
                        )
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
                    citation_urls.extend(extract_urls(result.content))

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
                                    "arguments": json.dumps(
                                        fb_args, ensure_ascii=False,
                                    ),
                                },
                            }],
                        })
                        try:
                            fb_result = await registry.execute(
                                result.fallback_tool, ctx, fb_args,
                            )
                        except Exception:
                            logger.exception(
                                "Auto-fallback tool %s raised",
                                result.fallback_tool,
                            )
                            fb_result = ToolResult(
                                success=False,
                                content=(
                                    f"[TOOL_ERROR] {result.fallback_tool} "
                                    "failed internally."
                                ),
                            )
                        messages.append(fb_result.to_tool_message(fallback_call_id))
                        tool_calls_log.append({
                            "name": result.fallback_tool,
                            "args": fb_args,
                            "success": fb_result.success,
                            "auto_fallback_from": tc.function.name,
                        })
                        citation_urls.extend(extract_urls(fb_result.content))
                        fallback_used = True
                        logger.info(
                            "Auto-fallback %s → %s(args=%s) success=%s",
                            tc.function.name, result.fallback_tool,
                            fb_args, fb_result.success,
                        )

            # Extract final answer
            if response and getattr(response, "choices", None):
                ai_response = response.choices[0].message.content or ""
        except Exception as exc:
            logger.exception("Sarvam call / tool loop failed")
            llm_exc = exc

        # ── Deterministic fixups + validator retry (non-fatal) ────
        if ai_response and llm_exc is None:
            # Mechanical fixes first — cheap, don't need the LLM.
            ai_response = normalize_digits(ai_response)
            if tool_was_used:
                ai_response = ensure_sources_line(ai_response, citation_urls)

            try:
                issues = validate_answer(ai_response, tool_was_used=tool_was_used)
                if issues:
                    logger.info(
                        "Validator issues after deterministic fixes: %s — retrying once.",
                        issues,
                    )
                    messages.append({"role": "assistant", "content": ai_response})
                    messages.append({
                        "role": "system",
                        "content": build_fix_message(issues),
                    })
                    retry_resp = await _run_llm_turn(
                        messages, tools_array=None, tool_choice=None,
                    )
                    retry_content = (
                        retry_resp.choices[0].message.content or ""
                    ) if retry_resp and getattr(retry_resp, "choices", None) else ""
                    if retry_content:
                        # Apply the same deterministic fixes to the retry output.
                        retry_content = normalize_digits(retry_content)
                        if tool_was_used:
                            retry_content = ensure_sources_line(
                                retry_content, citation_urls,
                            )
                        ai_response = retry_content
                        validator_retries = 1
            except Exception:
                logger.exception("Validator retry failed; keeping original answer")

        # ── Send to Discord ──────────────────────────────────────
        try:
            if ai_response:
                await _send_discord(message.channel, ai_response, citation_urls)
            elif llm_exc is not None:
                await message.channel.send(
                    with_turn_id(classify_llm_error(llm_exc), turn_id),
                )
            else:
                await message.channel.send(
                    with_turn_id("माफ गर्नुहोस्, उत्तर तयार गर्न सकिएन।", turn_id),
                )
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
