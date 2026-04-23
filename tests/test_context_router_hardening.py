"""Regression tests for the post-v2 router hardening.

Covers:
  * NEPSE ticker whitelist replaces the \\b[A-Z]{2,6}\\b false positives.
  * `plan_from_intent` short-circuits the keyword router.
  * `_apply_endpoint_cap` trims tasks by priority.
"""
from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from tools.osint.context_router import (
    ALLOWED_INTENTS,
    MAX_ENDPOINTS_PER_TURN,
    RoutePlan,
    _apply_endpoint_cap,
    fetch_context_bundle,
    plan_from_intent,
    route_query,
)


class TickerWhitelistTests(unittest.TestCase):
    def test_usa_does_not_trigger_trading(self) -> None:
        plan = route_query("USA ma aja k bhayo?")
        self.assertNotIn("trading", plan.intents)

    def test_hiv_does_not_trigger_trading(self) -> None:
        plan = route_query("HIV infection rate in Nepal")
        self.assertNotIn("trading", plan.intents)

    def test_ceo_alone_does_not_trigger_trading(self) -> None:
        plan = route_query("What does CEO mean?")
        self.assertNotIn("trading", plan.intents)

    def test_known_ticker_triggers_trading(self) -> None:
        plan = route_query("NABIL ko price k cha?")
        self.assertIn("trading", plan.intents)

    def test_unknown_uppercase_with_nepse_context_triggers_trading(self) -> None:
        # A not-yet-whitelisted ticker still routes when the query names NEPSE.
        plan = route_query("BRAND NEPSE ma kasto chha?")
        self.assertIn("trading", plan.intents)

    def test_unknown_uppercase_without_context_does_not(self) -> None:
        plan = route_query("CRM हुँदा तपाईंलाई के लाग्छ?")
        self.assertNotIn("trading", plan.intents)


class PlanFromIntentTests(unittest.TestCase):
    def test_explicit_macro_intent(self) -> None:
        plan = plan_from_intent("macro", "inflation k cha")
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.intents, ["macro"])
        self.assertTrue(plan.use_nepalosint)

    def test_who_is_shortcut(self) -> None:
        plan = plan_from_intent("who_is", "nepalko arthamantri")
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertTrue(plan.is_who_is)
        self.assertTrue(plan.use_nepalosint)

    def test_history_requires_date(self) -> None:
        # No extractable date → None (caller falls back to keyword routing).
        self.assertIsNone(plan_from_intent("history", "what happened"))

    def test_history_with_iso_date(self) -> None:
        plan = plan_from_intent("history", "between 2026-04-01 to 2026-04-05")
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertTrue(plan.wants_history)
        self.assertEqual(plan.history_start_date, "2026-04-01")
        self.assertEqual(plan.history_end_date, "2026-04-05")

    def test_invalid_intent_returns_none(self) -> None:
        self.assertIsNone(plan_from_intent("blarg", "nepal ko news"))

    def test_allowed_intents_exposed(self) -> None:
        # Must match the set referenced by retrieval_planner.
        self.assertEqual(
            set(ALLOWED_INTENTS),
            {"general_news", "macro", "government", "debt", "parliament", "trading"},
        )


class EndpointCapTests(unittest.TestCase):
    def test_noop_when_under_cap(self) -> None:
        tasks = {"debt_clock": 1, "economy_snapshot": 2}
        kept = _apply_endpoint_cap(tasks)
        self.assertEqual(kept, tasks)

    def test_drops_lowest_priority_when_over_cap(self) -> None:
        # Construct MAX_ENDPOINTS_PER_TURN + 2 tasks with mixed priorities.
        tasks = {
            "embedding_search": 1,     # priority 50
            "general_search": 1,       # 55
            "trading_search": 1,       # 55
            "recent_news": 1,          # 60
            "announcements": 1,        # 70
            "govt_decisions": 1,       # 75
            "debt_clock": 1,           # 100
        }
        self.assertGreater(len(tasks), MAX_ENDPOINTS_PER_TURN)
        kept = _apply_endpoint_cap(tasks)
        self.assertEqual(len(kept), MAX_ENDPOINTS_PER_TURN)
        # Top-priority ones survive.
        self.assertIn("debt_clock", kept)
        self.assertIn("govt_decisions", kept)
        # Lowest-priority one is dropped first.
        self.assertNotIn("embedding_search", kept)


class FetchBundleCapTests(unittest.TestCase):
    def test_fetch_bundle_respects_cap(self) -> None:
        class DummyClient:
            def __init__(self) -> None:
                self.max_context_items = 8

            # Stubs for every possible endpoint the router might request.
            async def get_economy_snapshot(self): return {"ok": 1}
            async def get_govt_decisions_latest(self, **_: object): return {"ok": 1}
            async def get_announcements_summary(self, **_: object): return {"ok": 1}
            async def get_debt_clock(self): return {"ok": 1}
            async def get_verbatim_summary(self): return {"ok": 1}
            async def get_parliament_bills(self, **_: object): return {"ok": 1}
            async def search_unified(self, *_a, **_kw): return {"ok": 1}
            async def search_embeddings(self, *_a, **_kw): return {"ok": 1}
            async def get_consolidated_recent(self, **_: object): return {"ok": 1}
            async def get_consolidated_history(self, **_: object): return {"ok": 1}

        plan = RoutePlan(
            use_nepalosint=True,
            intents=["macro", "government", "debt", "parliament", "trading"],
            wants_history=False,
            is_who_is=True,  # forces embedding_search too
        )
        bundle = asyncio.run(fetch_context_bundle(DummyClient(), "test", plan))
        self.assertLessEqual(
            len(bundle["payloads"]) + len(bundle["errors"]),
            MAX_ENDPOINTS_PER_TURN,
        )


if __name__ == "__main__":
    unittest.main()
