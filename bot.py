"""
bot.py — YetiDai Discord bot with tool-calling support.

Flow:
    1. User sends message → bot builds message list
    2. Sends to Sarvam with tools array from the ToolRegistry
    3. If LLM returns tool_calls → execute via registry → send results back
    4. LLM produces final text answer → bot sends to Discord

Sarvam tool-call contract (see tests/test_sarvam_tool_calling_live.py):
    - First call:  tool_choice="auto", tools=[...]
    - If finish_reason="tool_calls": execute handlers, send role="tool" messages
    - Second call: tool_choice="auto", LLM gets tool results, produces answer
"""
import discord
import json
import logging
import os
from dotenv import load_dotenv
from sarvamai import AsyncSarvamAI
from functionality import functional

# ── Core framework ────────────────────────────────────────────────
from core.tool_registry import get_registry
from core.tool_contracts import ToolContext

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

with open("systemPrompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

intents = discord.Intents.default()
intents.message_content = True
bot = discord.Client(intents=intents)

chad = functional(bot=bot)

registry = get_registry()

# Safety cap: max tool-call round-trips before forcing a text answer
MAX_TOOL_ROUNDS = 5


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
            for _round in range(MAX_TOOL_ROUNDS):
                response = await llm_client.chat.completions(
                    model="sarvam-30b",
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
                # Wire format matches test_sarvam_tool_calling_live.py
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
                    args = json.loads(tc.function.arguments)
                    result = await registry.execute(tc.function.name, ctx, args)
                    messages.append(result.to_tool_message(tc.id))

                    logger.info(
                        "Tool call: %s(args=%s) → success=%s",
                        tc.function.name, args, result.success,
                    )

            # ── Extract final answer ──────────────────────────────
            if response and hasattr(response, "choices") and len(response.choices) > 0:
                ai_response = response.choices[0].message.content
            else:
                ai_response = "I couldn't generate a response."

            if ai_response:
                for i in range(0, len(ai_response), 2000):
                    await message.channel.send(ai_response[i : i + 2000])

        except Exception as e:
            logger.exception("Failed calling API")
            await message.channel.send(
                "Sorry, I encountered an error while processing your request."
            )


if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
