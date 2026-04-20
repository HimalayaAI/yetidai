"""Quick routing test — verifies Haiku picks the right tool (or no tool)
for a battery of representative queries. Does NOT execute tools.

Usage:
    python tests/test_haiku_routing.py
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.llm_haiku import HaikuClient
from core.tool_registry import get_registry
import tools.osint.plugin as osint_plugin
import tools.search.plugin as search_plugin

with open("systemPrompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

osint_plugin.register()
search_plugin.register()
tools_array = get_registry().openai_tools()

client = HaikuClient(model="haiku", effort="low", timeout=90)


CASES = [
    ("नमस्ते Yeti!", "expect: no tool, greeting reply"),
    ("hey", "expect: no tool, greeting reply"),
    ("नेपालको प्रधानमन्त्री को हुनुहुन्छ?", "expect: get_nepal_live_context(who_is:prime_minister)"),
    ("feb 28 ma k bhako thiyo nepal ma?", "expect: get_nepal_live_context(general_news)"),
    ("गत हप्ता संसदमा के भयो?", "expect: get_nepal_live_context(parliament|bills)"),
    ("2025 ko UEFA Champions League kasle jityo?", "expect: internet_search"),
    ("भारतको प्रधानमन्त्री को हुनुहुन्छ?", "expect: internet_search"),
    ("नेपाल र भारतको मुद्रास्फीति तुलना गर्नुहोस्।",
     "expect: BOTH get_nepal_live_context + internet_search in one turn"),
]


async def main() -> None:
    for user_msg, expect in CASES:
        print("=" * 70)
        print(f"USER: {user_msg}")
        print(f"EXPECT: {expect}")
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        resp = await client.chat.completions(
            model="haiku", messages=msgs, tools=tools_array, tool_choice="auto"
        )
        choice = resp.choices[0]
        tcs = getattr(choice.message, "tool_calls", None) or []
        print(f"FINISH: {choice.finish_reason}")
        if tcs:
            for tc in tcs:
                print(f"TOOL: {tc.function.name}({tc.function.arguments})")
        else:
            print(f"CONTENT: {choice.message.content[:200]}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
