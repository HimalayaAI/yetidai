"""
bot.py — YetiDai Discord bot with tool-calling support.

Flow:
    1. User sends message → bot builds message list
    2. Sends to Sarvam with tools array from the ToolRegistry
    3. If Sarvam returns tool_calls → execute via registry → send results back
    4. Sarvam produces final text answer → bot sends to Discord

Backend: Sarvam (`sarvam-30b`) via the `sarvamai` async client. Configured
through `SARVAM_API_KEY` in the environment.

Tool-loop design (mirrors Anthropic's "let the model decide" model):
  - Parallel tool execution per round via asyncio.gather. Anthropic's SDK
    treats parallel tool use as the default; we match that so the model
    can fan out several OSINT calls in one round without serialising them.
  - Per-tool timeout (ToolSpec.timeout_seconds, else YETI_TOOL_TIMEOUT).
    A single hung endpoint cannot starve the loop any more.
  - Structured error markers ([TOOL_ERROR], [TOOL_TIMEOUT], [TOOL_DEDUP_HIT])
    encoded in the tool result content — the analogue of Anthropic's
    `is_error` flag in a system that has no dedicated field for it.
  - Cross-round dedup cache keyed by (name, args_hash): the same call made
    twice returns the cached result wrapped in a dedup marker so the model
    sees it's looping.
  - Progress check: two consecutive rounds with the same tool_calls
    signature → break and force a text round, instead of burning the
    MAX_TOOL_ROUNDS budget.
  - tool_was_used gates on actually-useful content, so the validator
    doesn't demand a citation line when every tool call failed.

Resilience:
  - Per-phase try/except so one failure can't masquerade as another.
  - Sarvam calls wrapped in asyncio.wait_for with one transient retry.
  - Last tool-round forces tools=None so the LLM must emit text.
  - Deterministic fixups (ASCII→Devanagari digits, स्रोत line injection)
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
from core.nepali_date import format_bs_ne, format_bs_iso
from core.date_context import build_date_block
from core.preflight import plan_preflight
from core.bot_helpers import (
    DISCORD_EMBED_FOOTER_LIMIT,
    DISCORD_MSG_LIMIT,
    GENERIC_TECH_ERROR,
    TOOL_DEDUP_MARKER,
    TOOL_ERROR_MARKER,
    TOOL_STALE_MARKER,
    TOOL_TIMEOUT_MARKER,
    build_correction_nudge,
    build_force_tool_nudge,
    build_tool_narration_nudge,
    build_tool_output_ignored_nudge,
    chunk_for_discord,
    classify_llm_error,
    detect_fabricated_filenames,
    detect_fabricated_source_names,
    detect_fabricated_urls,
    detect_requested_count,
    ensure_sources_line,
    extract_urls,
    hash_tool_call,
    is_bot_apology,
    is_empty_promise,
    is_real_tool_content,
    is_tool_narration,
    is_tool_output_ignored,
    is_transient_llm_error,
    looks_like_correction,
    needs_tool_use,
    news_answer_off_topic,
    normalize_digits,
    rewrite_sources_as_markdown,
    safe_field_value,
    split_body_and_sources,
    tool_calls_signature,
    with_turn_id,
)

# ── Register plugins ──────────────────────────────────────────────
import tools.osint.plugin as osint_plugin
import tools.search.plugin as search_plugin
import tools.fetch.plugin as fetch_plugin
import tools.github.plugin as github_plugin

osint_plugin.register()
search_plugin.register()
fetch_plugin.register()
github_plugin.register()
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
# Per-tool wall-clock limit. Tools can override via ToolSpec.timeout_seconds
# (slow aggregators) or be pinned lower for fast local lookups.
YETI_TOOL_TIMEOUT = float(os.getenv("YETI_TOOL_TIMEOUT", "15"))


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


async def _execute_tool_call(
    tc,
    ctx: ToolContext,
    *,
    dedup_cache: dict,
    default_timeout: float,
) -> tuple[str, dict, ToolResult, dict]:
    """Run a single tool call with dedup + per-tool timeout + error capture.

    Returns (tool_call_id, parsed_args, result, log_extra). The caller owns
    appending the tool message and extending tool_calls_log — keeping those
    out of this helper makes it trivial to unit-test in isolation.

    Semantics:
        * Bad JSON in arguments → args becomes {}, error_class="bad_args_json".
          We still call the tool because plugins tolerate missing keys better
          than they tolerate being skipped.
        * Same (name, args_hash) seen earlier this turn → return the cached
          ToolResult wrapped in a [TOOL_DEDUP_HIT] marker so the model sees
          it looped. This is cross-round within one user message.
        * Timeout → ToolResult with [TOOL_TIMEOUT] marker; the model can
          choose to retry or switch tools.
        * Other exceptions → ToolResult with [TOOL_ERROR] marker; content
          deliberately short so the tool message doesn't dominate context.

    Never raises. A well-formed ToolResult is always returned.
    """
    name = tc.function.name
    raw_args = tc.function.arguments or ""
    t_start = time.time()
    error_class: str | None = None
    dedup_hit = False

    try:
        args = json.loads(raw_args) if raw_args else {}
    except json.JSONDecodeError:
        logger.warning(
            "Bad tool_call arguments JSON for %s: %r", name, raw_args,
        )
        args = {}
        error_class = "bad_args_json"

    sig = hash_tool_call(name, args)
    cached: ToolResult | None = dedup_cache.get(sig)
    if cached is not None:
        dedup_hit = True
        original = (cached.content or "").strip()
        dedup_content = (
            f"{TOOL_DEDUP_MARKER} {name} already executed earlier this turn "
            f"with the same arguments. Reusing prior result:\n{original}"
        )
        result = ToolResult(
            tool_id=cached.tool_id,
            success=cached.success,
            content=dedup_content,
            raw_data=cached.raw_data,
            meta=cached.meta,
            trigger_fallback=False,  # suppress chained fallbacks on replay
        )
        error_class = error_class or "dedup"
        logger.info("Tool call dedup hit: %s(args=%s)", name, args)
        log_extra = {
            "latency_ms": int((time.time() - t_start) * 1000),
            "error_class": error_class,
            "dedup": True,
        }
        return (tc.id, args, result, log_extra)

    spec = registry.get_spec(name)
    timeout_s = (
        spec.timeout_seconds
        if (spec is not None and spec.timeout_seconds is not None)
        else default_timeout
    )

    try:
        result = await asyncio.wait_for(
            registry.execute(name, ctx, args),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        logger.warning("Tool %s timed out after %.1fs", name, timeout_s)
        error_class = "timeout"
        result = ToolResult(
            tool_id=name,
            success=False,
            content=(
                f"{TOOL_TIMEOUT_MARKER} {name} exceeded {timeout_s:.1f}s."
                " Consider a different tool or narrower query."
            ),
            error="timeout",
        )
    except Exception as exc:  # noqa: BLE001 — we deliberately catch-all
        logger.exception("Tool %s raised", name)
        error_class = type(exc).__name__
        result = ToolResult(
            tool_id=name,
            success=False,
            content=(
                f"{TOOL_ERROR_MARKER} {name} failed internally: "
                f"{type(exc).__name__}."
            ),
            error=f"{type(exc).__name__}: {exc}",
        )

    dedup_cache[sig] = result
    log_extra = {
        "latency_ms": int((time.time() - t_start) * 1000),
        "error_class": error_class,
        "dedup": dedup_hit,
    }
    return (tc.id, args, result, log_extra)


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
        # Accumulated tool content for the turn — used by the hallucination
        # check AND populated by the preflight step below. Must be defined
        # before the try-block so preflight and in-loop paths share it.
        tool_output_accum: list[str] = []

        # ── Build message list ────────────────────────────────────
        try:
            previous_messages = await chad.get_message_history(
                message.channel, limit=5,
            )

            today = datetime.date.today()
            date_block = build_date_block(today)
            dynamic_system_prompt = f"{SYSTEM_PROMPT}\n\n{date_block}"
            messages = [{"role": "system", "content": dynamic_system_prompt}]

            # Per-message try/except: one weird Discord message (None
            # content, missing author, system notification) shouldn't
            # nuke the whole turn — skip it and carry on.
            for prev_msg in previous_messages:
                try:
                    if prev_msg.id == message.id:
                        continue
                    content = getattr(prev_msg, "content", None) or ""
                    if not content.strip():
                        continue
                    if prev_msg.author == bot.user:
                        if is_bot_apology(content):
                            continue
                        messages.append({
                            "role": "assistant",
                            "content": content,
                        })
                    else:
                        author_name = getattr(
                            getattr(prev_msg, "author", None), "name", "user",
                        )
                        messages.append({
                            "role": "user",
                            "content": f"{author_name}: {content}",
                        })
                except Exception:
                    logger.exception(
                        "Skipping bad history message (turn=%s)", turn_id,
                    )
                    continue

            # Correction / count-intent nudges. These are cheap signals that
            # materially change the model's next turn — we inject them as
            # a system message RIGHT BEFORE the current user turn so Sarvam
            # reads them fresh without paying attention-decay on a long
            # history.
            if looks_like_correction(chad.user_input):
                requested_count = detect_requested_count(chad.user_input)
                messages.append({
                    "role": "system",
                    "content": build_correction_nudge(
                        chad.user_input,
                        requested_count=requested_count,
                    ),
                })
                logger.info(
                    "Correction detected in user_input; injected nudge "
                    "(requested_count=%s).",
                    requested_count,
                )

            messages.append({"role": "user", "content": chad.user_input})
            tools_array = registry.openai_tools()

            # ── Pre-flight tool execution ─────────────────────────
            # Deterministic rule-based classifier decides if the query
            # needs a specific tool. If yes, we execute it NOW and feed
            # the result into messages as a synthetic prior tool call.
            # Sarvam's first turn then has the data in context and only
            # needs to write the Nepali summary — it literally cannot
            # emit "म खोज्छु" any more, because the work is done.
            preflight = plan_preflight(chad.user_input)
            if preflight is not None:
                pf_name, pf_args = preflight
                pf_tc_id = f"preflight_{uuid.uuid4().hex[:8]}"
                pf_ctx = ToolContext(
                    query=chad.user_input,
                    history=previous_messages,
                    llm_client=llm_client,
                    channel_id=message.channel.id,
                    user_id=message.author.id,
                )
                logger.info(
                    "Preflight (turn=%s): %s(%s)", turn_id, pf_name, pf_args,
                )
                try:
                    pf_result = await asyncio.wait_for(
                        registry.execute(pf_name, pf_ctx, pf_args),
                        timeout=YETI_TOOL_TIMEOUT,
                    )
                except Exception as exc:
                    logger.exception("Preflight failed: %s", exc)
                    pf_result = None

                if pf_result is not None:
                    # Append the synthetic tool_call + tool result as if
                    # Sarvam had already chosen this tool. Sarvam's next
                    # turn sees a completed interaction and continues.
                    messages.append({
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": pf_tc_id,
                            "type": "function",
                            "function": {
                                "name": pf_name,
                                "arguments": json.dumps(
                                    pf_args, ensure_ascii=False,
                                ),
                            },
                        }],
                    })
                    messages.append(pf_result.to_tool_message(pf_tc_id))
                    tool_calls_log.append({
                        "name": pf_name,
                        "args": pf_args,
                        "success": pf_result.success,
                        "preflight": True,
                    })
                    if is_real_tool_content(pf_result):
                        tool_was_used = True
                        if pf_result.content:
                            tool_output_accum.append(pf_result.content)
                    citation_urls.extend(extract_urls(pf_result.content))
                    if pf_result.meta:
                        osint_endpoints_ok = pf_result.meta.get(
                            "endpoints_ok", osint_endpoints_ok,
                        )
                        osint_endpoints_failed = pf_result.meta.get(
                            "endpoints_failed", osint_endpoints_failed,
                        )

                    # If the preflight triggered a fallback (e.g. OSINT
                    # returned empty → internet_search), execute the
                    # fallback too so Sarvam has that data as well.
                    if pf_result.trigger_fallback and pf_result.fallback_tool:
                        fb_tc_id = f"preflight_fb_{uuid.uuid4().hex[:8]}"
                        fb_args = pf_result.fallback_args or {}
                        try:
                            fb_result = await asyncio.wait_for(
                                registry.execute(
                                    pf_result.fallback_tool, pf_ctx, fb_args,
                                ),
                                timeout=YETI_TOOL_TIMEOUT,
                            )
                        except Exception:
                            fb_result = None
                        if fb_result is not None:
                            messages.append({
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [{
                                    "id": fb_tc_id,
                                    "type": "function",
                                    "function": {
                                        "name": pf_result.fallback_tool,
                                        "arguments": json.dumps(
                                            fb_args, ensure_ascii=False,
                                        ),
                                    },
                                }],
                            })
                            messages.append(fb_result.to_tool_message(fb_tc_id))
                            tool_calls_log.append({
                                "name": pf_result.fallback_tool,
                                "args": fb_args,
                                "success": fb_result.success,
                                "preflight_fallback": True,
                            })
                            if is_real_tool_content(fb_result):
                                tool_was_used = True
                                if fb_result.content:
                                    tool_output_accum.append(fb_result.content)
                            citation_urls.extend(extract_urls(fb_result.content))
                            fallback_used = True
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
        # mode. If the model emits the same tool_calls signature two rounds
        # in a row we also break early — no point burning more budget on a
        # loop that won't converge.
        response = None
        dedup_cache: dict[str, ToolResult] = {}
        last_round_signature: tuple | None = None
        final_nudge_injected = False
        try:
            for _round in range(MAX_TOOL_ROUNDS):
                is_last_round = (_round == MAX_TOOL_ROUNDS - 1)

                # On the forced-text round, tell the model explicitly that
                # no more tools are available. Without this nudge Sarvam
                # sometimes narrates "I would call X but..." in the final
                # text. Injected once, just before the last LLM turn.
                if is_last_round and not final_nudge_injected and tools_array:
                    messages.append({
                        "role": "system",
                        "content": (
                            "NO MORE TOOL CALLS. तपाईंसँग अब कुनै tool उपलब्ध छैन। "
                            "अहिलेसम्म collect भएको tool data प्रयोग गरेर अन्तिम "
                            "नेपाली जवाफ लेख्नुहोस्। यदि डेटा पर्याप्त छैन भने, "
                            "छोटो माफी माग्दै प्रयोगकर्तालाई के छैन र के गर्न "
                            "सकिन्छ भनी बताउनुहोस् — कुनै काल्पनिक तथ्य नलेख्नुहोस्। "
                            "स्रोत दिँदा केवल tool output मा देखिएका URL मात्र "
                            "उद्धरण गर्नुहोस्।"
                        ),
                    })
                    final_nudge_injected = True

                response = await _run_llm_turn(
                    messages,
                    tools_array if (tools_array and not is_last_round) else None,
                    tool_choice=("auto" if (tools_array and not is_last_round) else None),
                )

                if not response or not getattr(response, "choices", None):
                    break
                choice = response.choices[0]
                finish_reason = getattr(choice, "finish_reason", None)

                tool_calls = getattr(choice.message, "tool_calls", None) or []
                if finish_reason != "tool_calls" or not tool_calls:
                    logger.info(
                        "LLM stop_reason=%s (round=%d, tool_calls=%d) → breaking loop",
                        finish_reason, _round, len(tool_calls),
                    )
                    break

                # Anthropic-style visibility: when the model narrates its plan
                # alongside tool_calls, surface that reasoning in logs so we
                # can audit *why* it chose those calls.
                assistant_text = (choice.message.content or "").strip()
                if assistant_text:
                    logger.info(
                        "LLM inter-round narration (round=%d): %s",
                        _round, assistant_text[:300],
                    )

                # Progress check: identical tool_calls signature twice in a
                # row = loop. Cut to the forced-text round.
                this_signature = tool_calls_signature(tool_calls)
                if this_signature == last_round_signature:
                    logger.info(
                        "No-progress detected at round %d (same signature) → "
                        "breaking to force text answer.",
                        _round,
                    )
                    break
                last_round_signature = this_signature

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

                # Parallel fan-out: run every primary tool in this round
                # concurrently. Anthropic's tool-use loop treats parallel
                # calls as the default; we match that so two independent
                # OSINT lookups don't serialise on the wire.
                exec_results = await asyncio.gather(*[
                    _execute_tool_call(
                        tc, ctx,
                        dedup_cache=dedup_cache,
                        default_timeout=YETI_TOOL_TIMEOUT,
                    )
                    for tc in tool_calls
                ])

                # Materialise round results in original tool_calls order so
                # the tool_call_id → tool_message pairing stays correct, then
                # chain any auto-fallbacks sequentially (they depend on the
                # primary's trigger flag).
                for (tc_id, args, result, log_extra) in exec_results:
                    messages.append(result.to_tool_message(tc_id))
                    log_entry = {
                        "name": next(
                            (t.function.name for t in tool_calls if t.id == tc_id),
                            tc_id,
                        ),
                        "args": args,
                        "success": result.success,
                        **log_extra,
                    }
                    tool_calls_log.append(log_entry)
                    if is_real_tool_content(result):
                        tool_was_used = True
                        if result.content:
                            tool_output_accum.append(result.content)
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
                        "Tool call: %s(args=%s) → success=%s latency=%dms class=%s",
                        log_entry["name"], args, result.success,
                        log_extra["latency_ms"], log_extra["error_class"],
                    )

                    # Auto-fallback: execute a second tool call in the same
                    # turn when the primary tool asked for it (e.g. OSINT
                    # returned no match → fall back to internet_search).
                    # Dedup replays suppress this via trigger_fallback=False.
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
                        fb_tc = type(
                            "FallbackTC", (), {
                                "id": fallback_call_id,
                                "function": type(
                                    "F", (), {
                                        "name": result.fallback_tool,
                                        "arguments": json.dumps(
                                            fb_args, ensure_ascii=False,
                                        ),
                                    },
                                )(),
                            },
                        )()
                        _, _, fb_result, fb_log_extra = await _execute_tool_call(
                            fb_tc, ctx,
                            dedup_cache=dedup_cache,
                            default_timeout=YETI_TOOL_TIMEOUT,
                        )
                        messages.append(fb_result.to_tool_message(fallback_call_id))
                        tool_calls_log.append({
                            "name": result.fallback_tool,
                            "args": fb_args,
                            "success": fb_result.success,
                            "auto_fallback_from": log_entry["name"],
                            **fb_log_extra,
                        })
                        if is_real_tool_content(fb_result):
                            tool_was_used = True
                            if fb_result.content:
                                tool_output_accum.append(fb_result.content)
                        citation_urls.extend(extract_urls(fb_result.content))
                        fallback_used = True
                        logger.info(
                            "Auto-fallback %s → %s(args=%s) success=%s latency=%dms",
                            log_entry["name"], result.fallback_tool,
                            fb_args, fb_result.success,
                            fb_log_extra["latency_ms"],
                        )

            # Extract final answer
            if response and getattr(response, "choices", None):
                ai_response = response.choices[0].message.content or ""

            # Tool-narration rescue (same-turn):
            # Model used the tool but its final reply narrates the mechanics
            # ("tool call सफल भयो / query process गरिँदैछ") instead of using
            # the returned data. The tool output is already in `messages` —
            # just ask the model to rewrite WITHOUT another tool call.
            if tool_was_used and is_tool_narration(ai_response):
                logger.info(
                    "Tool-narration detected (turn=%s): %r — forcing rewrite.",
                    turn_id, ai_response[:120],
                )
                messages.append({"role": "assistant", "content": ai_response})
                messages.append({
                    "role": "system",
                    "content": build_tool_narration_nudge(),
                })
                try:
                    rewrite_resp = await _run_llm_turn(
                        messages, tools_array=None, tool_choice=None,
                    )
                except Exception:
                    rewrite_resp = None
                if rewrite_resp and getattr(rewrite_resp, "choices", None):
                    rewrite_text = (
                        rewrite_resp.choices[0].message.content or ""
                    ).strip()
                    if rewrite_text and not is_tool_narration(rewrite_text):
                        ai_response = rewrite_text

            # Tool-output-ignored rescue (same-turn):
            # Tool returned real content + citation URLs, but the model's
            # answer denies having any data ("भेटिएन / डेटामा छैन"). Same
            # mechanism — inject a corrective system msg + rewrite without
            # another tool call, since the data is already in the history.
            if is_tool_output_ignored(
                ai_response,
                tool_was_used=tool_was_used,
                citation_urls=citation_urls,
            ):
                logger.info(
                    "Tool-output-ignored detected (turn=%s): %r — forcing rewrite.",
                    turn_id, ai_response[:120],
                )
                messages.append({"role": "assistant", "content": ai_response})
                messages.append({
                    "role": "system",
                    "content": build_tool_output_ignored_nudge(),
                })
                try:
                    rewrite_resp = await _run_llm_turn(
                        messages, tools_array=None, tool_choice=None,
                    )
                except Exception:
                    rewrite_resp = None
                if rewrite_resp and getattr(rewrite_resp, "choices", None):
                    rewrite_text = (
                        rewrite_resp.choices[0].message.content or ""
                    ).strip()
                    if rewrite_text and not is_tool_output_ignored(
                        rewrite_text,
                        tool_was_used=tool_was_used,
                        citation_urls=citation_urls,
                    ):
                        ai_response = rewrite_text

            # Empty-promise rescue (same-turn):
            # 1. Pure empty promise ("म बताउँछु / I'll fetch") with no tool used
            #    and the query clearly needed a tool.
            # 2. User asked for news but the answer looks nothing like news
            #    (the "tarkari instead of samachar" failure mode). Same
            #    mechanism — force a tool-call retry with tools=auto.
            needs_retry = (
                tools_array
                and (
                    (
                        is_empty_promise(ai_response, tool_was_used=tool_was_used)
                        and needs_tool_use(chad.user_input)
                    )
                    or news_answer_off_topic(
                        chad.user_input, ai_response, tool_was_used=tool_was_used,
                    )
                )
            )
            if needs_retry:
                logger.info(
                    "Empty-promise detected (turn=%s): %r — forcing tool retry.",
                    turn_id, ai_response[:100],
                )
                messages.append({"role": "assistant", "content": ai_response})
                messages.append({
                    "role": "system",
                    "content": build_force_tool_nudge(chad.user_input),
                })
                # First attempt: tool_choice="required" — strongest hint
                # we can give Sarvam that it MUST emit a tool_call this
                # round. Some SDK versions reject "required"; fall back
                # to "auto" on any error from the SDK side.
                try:
                    force_resp = await _run_llm_turn(
                        messages, tools_array=tools_array, tool_choice="required",
                    )
                except Exception:
                    try:
                        force_resp = await _run_llm_turn(
                            messages, tools_array=tools_array, tool_choice="auto",
                        )
                    except Exception:
                        force_resp = None
                if force_resp and getattr(force_resp, "choices", None):
                    force_choice = force_resp.choices[0]
                    force_calls = getattr(force_choice.message, "tool_calls", None) or []
                    if force_calls:
                        # Execute the forced calls and feed results back.
                        messages.append({
                            "role": "assistant",
                            "content": force_choice.message.content or "",
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                }
                                for tc in force_calls
                            ],
                        })
                        ctx_force = ToolContext(
                            query=chad.user_input,
                            history=previous_messages,
                            llm_client=llm_client,
                            channel_id=message.channel.id,
                            user_id=message.author.id,
                        )
                        force_exec = await asyncio.gather(*[
                            _execute_tool_call(
                                tc, ctx_force,
                                dedup_cache=dedup_cache,
                                default_timeout=YETI_TOOL_TIMEOUT,
                            )
                            for tc in force_calls
                        ])
                        for (tc_id, args, result, log_extra) in force_exec:
                            messages.append(result.to_tool_message(tc_id))
                            if is_real_tool_content(result):
                                tool_was_used = True
                                if result.content:
                                    tool_output_accum.append(result.content)
                            citation_urls.extend(extract_urls(result.content))
                            tool_calls_log.append({
                                "name": next(
                                    (t.function.name for t in force_calls if t.id == tc_id),
                                    tc_id,
                                ),
                                "args": args,
                                "success": result.success,
                                "forced": True,
                                **log_extra,
                            })
                        # Ask Sarvam to compose the real answer now (no more tools).
                        final_resp = await _run_llm_turn(
                            messages, tools_array=None, tool_choice=None,
                        )
                        if final_resp and getattr(final_resp, "choices", None):
                            forced_answer = final_resp.choices[0].message.content or ""
                            if forced_answer.strip():
                                ai_response = forced_answer
                    else:
                        # Model emitted text again on forced round — keep it
                        # only if it's no longer an empty promise.
                        retry_text = force_choice.message.content or ""
                        if retry_text and not is_empty_promise(retry_text):
                            ai_response = retry_text

                # Final safety net: if the first pass AND the forced
                # retry both produced empty-promise text, replace the
                # user-visible answer with an honest apology so the
                # bot never ships a bare "म खोज्छु" that goes nowhere.
                if is_empty_promise(ai_response, tool_was_used=tool_was_used):
                    ai_response = (
                        "माफ गर्नुहोस् हजुर — अहिले यो प्रश्नको लागि "
                        "live data ल्याउन सकिएन। केही सेकेन्डपछि पुनः "
                        "सोध्नुहोस्, वा अलि विस्तृत प्रश्न दिनुहोस्।"
                    )
                    logger.info(
                        "Empty promise persisted through forced retry "
                        "(turn=%s) — replacing with honest apology.",
                        turn_id,
                    )
        except Exception as exc:
            logger.exception("Sarvam call / tool loop failed")
            llm_exc = exc

        # ── Unconditional empty-promise safety net ───────────────
        # The in-loop safety net above only fires inside `if needs_retry`.
        # If the retry path is short-circuited for any reason (Sarvam
        # rejecting tool_choice="required" on both passes, an exception
        # bubbling through both LLM calls, tools_array being empty on a
        # weird turn, etc.), an empty promise like "म यसलाई खोज्छु।"
        # would ship straight to Discord. This unconditional pass
        # catches that final case — if the answer is still just a
        # promise and the user clearly asked for data, replace it with
        # an honest apology rather than the bare promise.
        if (
            ai_response
            and is_empty_promise(ai_response, tool_was_used=tool_was_used)
            and needs_tool_use(chad.user_input)
        ):
            logger.warning(
                "Empty promise reached post-loop unconditionally (turn=%s): %r — "
                "replacing with apology.",
                turn_id, ai_response[:120],
            )
            ai_response = (
                "माफ गर्नुहोस् हजुर — अहिले यो प्रश्नको लागि live data "
                "ल्याउन सकिएन। केही सेकेन्डपछि पुनः सोध्नुहोस्, वा अलि "
                "विस्तृत प्रश्न दिनुहोस्।"
            )

        # ── Anti-hallucination: fabricated filenames ─────────────
        #
        # When analyze_github_repo / fetch_url / internet_search returned
        # real content, the final answer must not cite file names that
        # aren't in that content. If we find one, inject a correction
        # system message and retry once with no tools (the right data is
        # already in context).
        if (
            ai_response
            and llm_exc is None
            and tool_output_accum
        ):
            joined_output = "\n".join(tool_output_accum)
            fabricated_files = detect_fabricated_filenames(ai_response, joined_output)
            fabricated_urls = detect_fabricated_urls(ai_response, joined_output)
            fabricated_srcs = detect_fabricated_source_names(ai_response, joined_output)
            if fabricated_files or fabricated_urls or fabricated_srcs:
                logger.info(
                    "Fabrication in answer (turn=%s): files=%s urls=%s names=%s — retrying.",
                    turn_id, fabricated_files, fabricated_urls, fabricated_srcs,
                )
                parts = []
                if fabricated_files:
                    parts.append(
                        f"यी फाइल नामहरू tool output मा छैनन्: "
                        f"{', '.join(fabricated_files)}"
                    )
                if fabricated_urls:
                    parts.append(
                        f"यी URL tool output मा छैनन् (hallucinated): "
                        f"{', '.join(fabricated_urls)}"
                    )
                if fabricated_srcs:
                    parts.append(
                        f"यी news-org नामहरू (source block मा) tool output "
                        f"मा कहीँ देखिँदैनन्: {', '.join(fabricated_srcs)}"
                    )
                nudge = (
                    " | ".join(parts)
                    + "। यी काल्पनिक हुन्। पुनः लेख्नुहोस्, केवल tool output "
                    "मा देखिएका real file / real URL मात्र उद्धरण गर्नुहोस्। "
                    "यदि tool output मा citable URL छैन भने स्रोत: रेखामा "
                    "endpoint नाम मात्र राख्नुहोस् — काल्पनिक URL नलेख्नुहोस्।"
                )
                messages.append({"role": "assistant", "content": ai_response})
                messages.append({"role": "system", "content": nudge})
                try:
                    anti_resp = await _run_llm_turn(
                        messages, tools_array=None, tool_choice=None,
                    )
                    if anti_resp and getattr(anti_resp, "choices", None):
                        corrected = anti_resp.choices[0].message.content or ""
                        if corrected.strip():
                            ai_response = corrected
                except Exception:
                    logger.exception("Anti-hallucination retry failed; keeping answer")

                # Final line of defence: if the retry STILL has
                # fabricated source names AND no real URLs in tool
                # output, replace with an honest "I don't have info"
                # instead of shipping an invented citation again.
                # User quote: "if it doesnt find straightup say I
                # dont have info".
                still_bad = detect_fabricated_source_names(
                    ai_response, joined_output,
                )
                if still_bad:
                    logger.info(
                        "Hallucinated sources persisted after retry "
                        "(turn=%s): %s — replacing with honest apology.",
                        turn_id, still_bad,
                    )
                    ai_response = (
                        "माफ गर्नुहोस् हजुर — यो प्रश्नको लागि मलाई "
                        "भरपर्दो source भेटिएन। NepalOSINT मा यो विषय "
                        "अहिले indexed छैन, र web search ले पनि "
                        "पुष्टि गर्न सकिने लिङ्क दिएन। काल्पनिक "
                        "स्रोत लेख्ननभन्दा खुलस्त भन्दै छु: मलाई "
                        "थाहा भएन।"
                    )

        # ── Deterministic fixups + validator retry (non-fatal) ────
        if ai_response and llm_exc is None:
            try:
                # Track whether a github tool fired this turn — the
                # validator uses this to catch fabricated
                # `github.com/HimalayaAI/<repo>` URLs in the final answer.
                github_tool_was_used = any(
                    entry.get("name") in ("analyze_github_repo", "list_github_repos")
                    for entry in tool_calls_log
                )

                # Check pre-fix state so we can distinguish "model was fine"
                # from "fixups rescued it" in the log.
                pre_issues = validate_answer(
                    ai_response,
                    tool_was_used=tool_was_used,
                    github_tool_was_used=github_tool_was_used,
                )

                # Mechanical fixes first — cheap, don't need the LLM.
                ai_response = normalize_digits(ai_response)
                if tool_was_used:
                    ai_response = ensure_sources_line(ai_response, citation_urls)
                # Shorten any bare URLs in the स्रोत: block to Discord-markdown
                # links regardless of who wrote the block (model or helper).
                ai_response = rewrite_sources_as_markdown(ai_response)

                # Re-validate *after* fixups: if the only problems were
                # ASCII digits and a missing स्रोत line, we've just solved
                # them without burning a Sarvam call.
                post_issues = validate_answer(
                    ai_response,
                    tool_was_used=tool_was_used,
                    github_tool_was_used=github_tool_was_used,
                )

                if post_issues:
                    logger.info(
                        "Validator issues remain after deterministic fixes "
                        "(pre=%s post=%s) — retrying with LLM once.",
                        pre_issues, post_issues,
                    )
                    messages.append({"role": "assistant", "content": ai_response})
                    messages.append({
                        "role": "system",
                        "content": build_fix_message(post_issues),
                    })
                    retry_resp = await _run_llm_turn(
                        messages, tools_array=None, tool_choice=None,
                    )
                    retry_content = (
                        retry_resp.choices[0].message.content or ""
                    ) if retry_resp and getattr(retry_resp, "choices", None) else ""
                    if retry_content:
                        retry_content = normalize_digits(retry_content)
                        if tool_was_used:
                            retry_content = ensure_sources_line(
                                retry_content, citation_urls,
                            )
                        ai_response = retry_content
                        validator_retries = 1
                elif pre_issues:
                    logger.info(
                        "Validator issues %s resolved by deterministic "
                        "fixes alone — skipped LLM retry.",
                        pre_issues,
                    )
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
