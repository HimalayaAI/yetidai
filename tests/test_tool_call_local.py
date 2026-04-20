"""
Local integration test — simulates the bot.py tool-calling flow
without Discord, using the real Sarvam API + real NepalOSINT.

Usage:
    python tests/test_tool_call_local.py

Requires SARVAM_API_KEY in .env.
"""
import asyncio
import json
import os
import sys

from dotenv import load_dotenv
from sarvamai import AsyncSarvamAI

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.tool_registry import get_registry
from core.tool_contracts import ToolContext
import tools.osint.plugin as osint_plugin
import tools.search.plugin as search_plugin

# ── Setup ─────────────────────────────────────────────────────────

load_dotenv()

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
if not SARVAM_API_KEY:
    print("❌ SARVAM_API_KEY not found in .env — cannot run local test.")
    sys.exit(1)

llm_client = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)

with open("systemPrompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

# Register plugins
osint_plugin.register()
search_plugin.register()

registry = get_registry()

MAX_TOOL_ROUNDS = 5


async def simulate_bot_flow(user_query: str) -> None:
    """Simulate the exact flow from bot.py on_message handler."""

    print("=" * 70)
    print(f"🧪 LOCAL TEST — simulating bot.py tool-call flow")
    print(f"📝 Query: {user_query}")
    print("=" * 70)

    # 1. Build message list (same as bot.py)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": user_query})

    # 2. Get tools array from registry
    tools_array = registry.openai_tools()
    print(f"\n🔧 Registered tools: {[t['function']['name'] for t in tools_array]}")
    print(f"   Tool schema sent to Sarvam:")
    for t in tools_array:
        print(f"   - {t['function']['name']}: {t['function']['description'][:80]}...")

    # 3. Tool-call loop (same as bot.py)
    response = None
    for round_num in range(MAX_TOOL_ROUNDS):
        print(f"\n{'─' * 50}")
        print(f"🔄 Round {round_num + 1}: Calling Sarvam (tool_choice='auto')...")

        response = await llm_client.chat.completions(
            model="sarvam-30b",
            messages=messages,
            tools=tools_array if tools_array else None,
            tool_choice="auto" if tools_array else None,
        )

        choice = response.choices[0]
        tool_calls = getattr(choice.message, "tool_calls", None) or []

        print(f"   finish_reason: {choice.finish_reason}")
        print(f"   tool_calls: {len(tool_calls)}")

        if choice.finish_reason != "tool_calls" or not tool_calls:
            print("   ✅ LLM returned a text answer (no more tool calls).")
            break

        # Append assistant message with tool_calls
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

        # Execute each tool call via registry
        ctx = ToolContext(
            query=user_query,
            history=None,
            llm_client=llm_client,
        )

        for tc in tool_calls:
            args = json.loads(tc.function.arguments)
            print(f"\n   🛠️  Tool call: {tc.function.name}")
            print(f"      Arguments: {json.dumps(args, ensure_ascii=False)}")
            print(f"      tool_call_id: {tc.id}")

            result = await registry.execute(tc.function.name, ctx, args)

            print(f"      success: {result.success}")
            if result.error:
                print(f"      error: {result.error}")
            if result.content:
                preview = result.content[:300]
                print(f"      content preview ({len(result.content)} chars):")
                for line in preview.split("\n")[:8]:
                    print(f"        {line}")
                if len(result.content) > 300:
                    print(f"        ... ({len(result.content) - 300} more chars)")

            # Append tool result message
            messages.append(result.to_tool_message(tc.id))

    # 4. Final answer
    print(f"\n{'═' * 70}")
    if response and hasattr(response, "choices") and len(response.choices) > 0:
        ai_response = response.choices[0].message.content
        print(f"🤖 YetiDai final answer:\n")
        print(ai_response)
    else:
        print("❌ No response generated.")

    print(f"\n{'═' * 70}")
    print(f"📊 Total messages in conversation: {len(messages)}")
    roles = [m.get('role', '?') for m in messages]
    print(f"   Message roles: {roles}")
    print("✅ Test complete.")


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "नेपालमा आज के भइरहेको छ?"
    asyncio.run(simulate_bot_flow(query))
