"""Test that the retrieval planner short-circuits on small talk."""
from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from tools.osint.retrieval_planner import _is_smalltalk, resolve_route_plan


class SmalltalkDetectorTests(unittest.TestCase):
    def test_k_xa(self) -> None:
        self.assertTrue(_is_smalltalk("k xa"))

    def test_halkhabar(self) -> None:
        self.assertTrue(_is_smalltalk("tapai ko halkhabar k cha"))

    def test_namaste(self) -> None:
        self.assertTrue(_is_smalltalk("नमस्ते हजुर"))

    def test_nepal_word_in_question_is_not_smalltalk(self) -> None:
        self.assertFalse(_is_smalltalk("nepal ko pm ko ho"))

    def test_long_message_not_smalltalk(self) -> None:
        self.assertFalse(_is_smalltalk(
            "I want a detailed summary of the latest events and parliamentary votes"
        ))


class ResolveRoutePlanSmalltalkTests(unittest.TestCase):
    def test_smalltalk_skips_llm_and_osint(self) -> None:
        class PoisonedClient:
            """Any call to the LLM planner is a bug — the short-circuit
            must fire *before* we build the planner request."""
            class chat:
                @staticmethod
                async def completions(**_):
                    raise AssertionError("LLM planner should not run for small talk")

        plan = asyncio.run(
            resolve_route_plan(PoisonedClient(), "k xa", previous_messages=[])
        )
        self.assertFalse(plan.use_nepalosint)

    def test_hinted_intent_still_wins_over_smalltalk(self) -> None:
        # Even a small-talk-looking query falls through when the LLM
        # explicitly passed `intent=macro` via the tool call.
        llm_client = SimpleNamespace()  # never called
        plan = asyncio.run(
            resolve_route_plan(
                llm_client, "k xa", previous_messages=[],
                hinted_intent="macro",
            )
        )
        self.assertTrue(plan.use_nepalosint)
        self.assertIn("macro", plan.intents)


if __name__ == "__main__":
    unittest.main()
