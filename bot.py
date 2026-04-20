"""
bot.py — YetiDai Discord bot with tool-calling support.

Flow:
    1. User sends message → bot builds message list
    2. Sends to the LLM with tools array from the ToolRegistry
    3. If LLM returns tool_calls → execute via registry → send results back
    4. LLM produces final text answer → bot sends to Discord

Default backend: local Claude Haiku 4.5 via the `claude` CLI
(see core/llm_haiku.py). Set YETI_BACKEND=sarvam to fall back to the
remote Sarvam client.
"""
import discord
import json
import logging
import os
import re
import time
import uuid
from dotenv import load_dotenv
from functionality import functional
from core.llm_haiku import HaikuClient

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
YETI_BACKEND = os.getenv("YETI_BACKEND", "haiku").lower()

if YETI_BACKEND == "sarvam":
    from sarvamai import AsyncSarvamAI
    SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
    llm_client = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)
    LLM_MODEL = "sarvam-30b"
    logger.info("Using Sarvam backend (model=%s).", LLM_MODEL)
else:
    # Local Haiku via `claude` CLI, low thinking effort.
    llm_client = HaikuClient(
        model=os.getenv("YETI_HAIKU_MODEL", "haiku"),
        effort=os.getenv("YETI_HAIKU_EFFORT", "low"),
        timeout=float(os.getenv("YETI_HAIKU_TIMEOUT", "90")),
    )
    LLM_MODEL = "haiku"
    logger.info(
        "Using local Haiku backend (model=%s, effort=%s).",
        llm_client.model, llm_client.effort,
    )

with open("systemPrompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

intents = discord.Intents.default()
intents.message_content = True
bot = discord.Client(intents=intents)

chad = functional(bot=bot)

registry = get_registry()

# Safety cap: max tool-call round-trips before forcing a text answer
MAX_TOOL_ROUNDS = 5

_URL_RE = re.compile(r"https?://[^\s)\]\"<>]+")


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


async def _send_discord(channel, answer: str, citation_urls: list[str]) -> None:
    """Send answer; attach a citations embed when we have URLs."""
    body, sources_line = _split_body_and_sources(answer)
    text = body if (citation_urls and body) else answer

    # Chunk body text at 2000 chars (Discord hard cap).
    for i in range(0, len(text), 2000):
        await channel.send(text[i : i + 2000])

    if citation_urls:
        embed = discord.Embed(
            title="स्रोत / Sources",
            color=0x2D72D2,
        )
        for idx, url in enumerate(citation_urls[:5], start=1):
            embed.add_field(name=f"{idx}.", value=url, inline=False)
        if sources_line:
            embed.set_footer(text=sources_line[:2048])
        await channel.send(embed=embed)


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

        try:
            # Get message history for context
            previous_messages = await chad.get_message_history(
                message.channel, limit=5,
            )

            import datetime
            today_str = datetime.date.today().strftime("%Y-%m-%d")
            dynamic_system_prompt = f"{SYSTEM_PROMPT}\n\n# CURRENT DATE:\nToday's Date is: {today_str}"

            # Build the message list
            messages = [{"role": "system", "content": dynamic_system_prompt}]

            # Format previous messages
            for prev_msg in previous_messages:
                if prev_msg.id != message.id and prev_msg.content.strip():
                    if prev_msg.author == bot.user:
                        messages.append({
                            "role": "assistant",
                            "content": prev_msg.content,
                        })
                    else:
                        messages.append({
                            "role": "user",
                            "content": f"{prev_msg.author.name}: {prev_msg.content}",
                        })

            # Add current user message
            messages.append({"role": "user", "content": chad.user_input})

            # Get the tools array from the registry
            tools_array = registry.openai_tools()

            # ── Tool-call loop ────────────────────────────────────
            # The LLM decides whether to call tools via tool_choice="auto".
            # If it returns tool_calls, we execute them and loop.
            # If it returns a text answer, we break.

            response = None
            citation_urls: list[str] = []
            for _round in range(MAX_TOOL_ROUNDS):
                response = await llm_client.chat.completions(
                    model=LLM_MODEL,
                    messages=messages,
                    tools=tools_array if tools_array else None,
                    tool_choice="auto" if tools_array else None,
                )

                choice = response.choices[0]

                # If the LLM produced a regular text answer, we're done
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

                # Execute each tool call via the registry
                ctx = ToolContext(
                    query=chad.user_input,
                    history=previous_messages,
                    llm_client=llm_client,
                    channel_id=message.channel.id,
                    user_id=message.author.id,
                )

                for tc in tool_calls:
                    tool_was_used = True
                    args = json.loads(tc.function.arguments)
                    result = await registry.execute(tc.function.name, ctx, args)
                    messages.append(result.to_tool_message(tc.id))
                    tool_calls_log.append({
                        "name": tc.function.name,
                        "args": args,
                        "success": result.success,
                    })
                    citation_urls.extend(_extract_urls(result.content))

                    if result.meta:
                        osint_endpoints_ok = result.meta.get("endpoints_ok", osint_endpoints_ok)
                        osint_endpoints_failed = result.meta.get("endpoints_failed", osint_endpoints_failed)
                        cache_stats = {
                            "hits": result.meta.get("cache_hits", cache_stats.get("hits", 0)),
                            "misses": result.meta.get("cache_misses", cache_stats.get("misses", 0)),
                        }

                    logger.info(
                        "Tool call: %s(args=%s) → success=%s",
                        tc.function.name, args, result.success,
                    )

                    # ── Auto-fallback: execute a second tool call in the same
                    #    turn when the primary tool asked for it (e.g. OSINT
                    #    returned no match → fall back to internet_search).
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
                        fb_result = await registry.execute(result.fallback_tool, ctx, fb_args)
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

            # ── Extract final answer ──────────────────────────────
            if response and hasattr(response, "choices") and len(response.choices) > 0:
                ai_response = response.choices[0].message.content or ""
            else:
                ai_response = ""

            # ── Post-generation validator + one targeted retry ──
            if ai_response:
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
                    ) if retry_resp and retry_resp.choices else ""
                    if retry_content:
                        ai_response = retry_content
                        validator_retries = 1

            # ── Send to Discord (embed for citations, if any) ───
            if ai_response:
                await _send_discord(message.channel, ai_response, citation_urls)
            else:
                await message.channel.send(
                    "माफ गर्नुहोस्, उत्तर तयार गर्न सकिएन।"
                )

        except Exception:
            logger.exception("Failed calling API")
            await message.channel.send(
                "माफ गर्नुहोस्, एउटा प्राविधिक समस्या देखियो। कृपया फेरि प्रयास गर्नुहोस्।"
            )

        finally:
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
