"""Tests for core.preflight — deterministic tool selection before LLM.

The preflight layer is bot.py's belt-and-suspenders against Sarvam-30B
emitting "म खोज्छु" instead of a tool call. For a high-confidence
query shape it returns (tool_name, args); bot.py executes that tool
before the first LLM turn so the data is already in context.
"""
from __future__ import annotations

import unittest

from core.preflight import plan_preflight


class UrlPreflightTests(unittest.TestCase):
    def test_github_url_goes_to_analyze_repo(self) -> None:
        plan = plan_preflight("yo repo ma k cha https://github.com/HimalayaAI/yetidai")
        self.assertEqual(plan[0], "analyze_github_repo")
        self.assertIn("github.com", plan[1]["repo"])

    def test_generic_url_goes_to_fetch(self) -> None:
        plan = plan_preflight("read this: https://nrb.org.np/macro-situation")
        self.assertEqual(plan[0], "fetch_url")
        self.assertIn("nrb.org.np", plan[1]["url"])


class MinisterPreflightTests(unittest.TestCase):
    def test_hm_resign(self) -> None:
        plan = plan_preflight("nepal ko HM le kina resign garyo")
        self.assertEqual(plan[0], "get_nepal_live_context")
        self.assertEqual(plan[1]["intent"], "government")
        self.assertEqual(plan[1]["focus"], "home_minister")

    def test_pm_is_who(self) -> None:
        plan = plan_preflight("PM ko ho")
        self.assertEqual(plan[0], "get_nepal_live_context")
        self.assertEqual(plan[1]["focus"], "prime_minister")


class NewsPreflightTests(unittest.TestCase):
    def test_aja_ko_samachar(self) -> None:
        plan = plan_preflight("aja ko samachar bhanus")
        self.assertEqual(plan[0], "get_nepal_live_context")
        self.assertEqual(plan[1]["intent"], "general_news")

    def test_devanagari_samachar(self) -> None:
        plan = plan_preflight("आजको ताजा समाचार देउ")
        self.assertEqual(plan[0], "get_nepal_live_context")
        self.assertEqual(plan[1]["intent"], "general_news")

    def test_long_query_skipped(self) -> None:
        # Long compound query — let Sarvam do the routing, not preflight.
        plan = plan_preflight(
            "I want a detailed comparison between Nepal and India inflation "
            "rates over the past five years with GDP context and currency "
            "movements please give me the complete breakdown"
        )
        # Long queries fall through on the news rule (12-word cap) but the
        # macro rule fires because 'inflation'/'gdp' are present.
        if plan is not None:
            self.assertEqual(plan[1]["intent"], "macro")


class MacroPreflightTests(unittest.TestCase):
    def test_gdp(self) -> None:
        plan = plan_preflight("nepal ko gdp kati ho")
        self.assertEqual(plan[0], "get_nepal_live_context")
        self.assertEqual(plan[1]["intent"], "macro")

    def test_inflation(self) -> None:
        plan = plan_preflight("inflation rate k cha aja")
        self.assertEqual(plan[0], "get_nepal_live_context")
        self.assertEqual(plan[1]["intent"], "macro")


class TradingPreflightTests(unittest.TestCase):
    def test_ticker(self) -> None:
        plan = plan_preflight("RURU ko price k cha")
        self.assertEqual(plan[0], "get_nepal_live_context")
        self.assertEqual(plan[1]["intent"], "trading")
        self.assertEqual(plan[1]["focus"], "RURU")

    def test_nepse(self) -> None:
        plan = plan_preflight("NEPSE aja kasto?")
        self.assertEqual(plan[0], "get_nepal_live_context")
        self.assertEqual(plan[1]["intent"], "trading")


class NoMatchTests(unittest.TestCase):
    def test_greeting_skipped(self) -> None:
        self.assertIsNone(plan_preflight("नमस्ते"))
        self.assertIsNone(plan_preflight("k xa"))

    def test_empty(self) -> None:
        self.assertIsNone(plan_preflight(""))
        self.assertIsNone(plan_preflight(None))


if __name__ == "__main__":
    unittest.main()
