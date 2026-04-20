import json
import os
import unittest

from dotenv import load_dotenv
from sarvamai import AsyncSarvamAI


def _live_tests_enabled() -> bool:
    return os.getenv("RUN_LIVE_SARVAM_TESTS", "0") == "1"


@unittest.skipUnless(_live_tests_enabled(), "Set RUN_LIVE_SARVAM_TESTS=1 to run live Sarvam tests.")
class SarvamToolCallingLiveTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        load_dotenv(dotenv_path=".env")
        api_key = os.getenv("SARVAM_API_KEY")
        if not api_key:
            self.skipTest("SARVAM_API_KEY not set in environment/.env")
        self.client = AsyncSarvamAI(api_subscription_key=api_key)
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_nepalosint_macro_snapshot",
                    "description": "Fetch latest Nepal macroeconomic snapshot from NepalOSINT.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "focus": {
                                "type": "string",
                                "description": "What macro area the user asked about.",
                            }
                        },
                        "required": ["focus"],
                        "additionalProperties": False,
                    },
                },
            }
        ]

    async def test_required_tool_choice_returns_tool_call(self) -> None:
        response = await self.client.chat.completions(
            model="sarvam-30b",
            messages=[
                {
                    "role": "system",
                    "content": "If the user asks for current Nepal macro updates, call the function.",
                },
                {"role": "user", "content": "नेपालको मुद्रास्फीति र रेमिट्यान्सको अपडेट देऊ।"},
            ],
            tools=self.tools,
            tool_choice="required",
            temperature=0,
            max_tokens=200,
        )

        choice = response.choices[0]
        self.assertEqual(choice.finish_reason, "tool_calls")
        tool_calls = getattr(choice.message, "tool_calls", None) or []
        self.assertGreaterEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].function.name, "get_nepalosint_macro_snapshot")
        parsed_args = json.loads(tool_calls[0].function.arguments)
        self.assertIn("focus", parsed_args)

    async def test_tool_call_round_trip_produces_final_answer(self) -> None:
        first = await self.client.chat.completions(
            model="sarvam-30b",
            messages=[
                {"role": "system", "content": "Use tools for live Nepal macro questions."},
                {"role": "user", "content": "नेपालको मुद्रास्फीति अपडेट देऊ"},
            ],
            tools=self.tools,
            tool_choice="required",
            temperature=0,
            max_tokens=200,
        )
        first_choice = first.choices[0]
        tool_calls = getattr(first_choice.message, "tool_calls", None) or []
        self.assertTrue(tool_calls, "Expected tool call in first turn.")
        tool_call = tool_calls[0]

        tool_payload = {
            "as_of": "Falgun 2082",
            "inflation": "5.20%",
            "remittance_yoy": "17.0%",
            "source": "NRB via NepalOSINT",
        }

        second = await self.client.chat.completions(
            model="sarvam-30b",
            messages=[
                {"role": "system", "content": "Answer in concise Nepali and include source mention."},
                {"role": "user", "content": "नेपालको मुद्रास्फीति अपडेट देऊ"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_payload, ensure_ascii=False),
                },
            ],
            tools=self.tools,
            tool_choice="auto",
            temperature=0,
            max_tokens=250,
        )

        second_choice = second.choices[0]
        self.assertEqual(second_choice.finish_reason, "stop")
        self.assertTrue((second_choice.message.content or "").strip())


if __name__ == "__main__":
    unittest.main()
