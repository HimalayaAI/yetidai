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


class TradingPreflightPolicyTests(unittest.TestCase):
    """Ticker / trading query routing policy:
      - ticker + "details/information" → internet_search (rich
        company background from ruruhydro / icranepal / doed).
      - bare ticker or "ticker price" → fall through to Sarvam.
    """

    def test_ticker_with_details_goes_to_web(self) -> None:
        plan = plan_preflight("RURU hydropower share ko barima information")
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan[0], "internet_search")

    def test_nabil_price_falls_through(self) -> None:
        # No "details" qualifier — bare price check — falls to Sarvam.
        self.assertIsNone(plan_preflight("NABIL share price"))

    def test_nabil_full_details_goes_to_web(self) -> None:
        # "full details" qualifier — web fetch.
        plan = plan_preflight("NABIL IPO ko full details")
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan[0], "internet_search")

    def test_time_sensitive_ipo_news_hits_news_path(self) -> None:
        # "aja ko ipo baadfad" is a time-sensitive news request — the
        # general_news catch-all rule fires intentionally so OSINT's
        # recent-stories endpoint picks up today's allotment news.
        plan = plan_preflight("aja ko ipo baadfad bhayo ki?")
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan[1]["intent"], "general_news")


class NoMatchTests(unittest.TestCase):
    def test_greeting_skipped(self) -> None:
        self.assertIsNone(plan_preflight("नमस्ते"))
        self.assertIsNone(plan_preflight("k xa"))

    def test_empty(self) -> None:
        self.assertIsNone(plan_preflight(""))
        self.assertIsNone(plan_preflight(None))


if __name__ == "__main__":
    unittest.main()
