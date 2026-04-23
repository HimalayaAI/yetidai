"""Tests for ticker-details / current-role preflight fallbacks.

Both motivated by live traces where OSINT gave a one-line answer /
stale Wikipedia data when the user wanted rich project background
or the actually-current minister.
"""
from __future__ import annotations

import unittest

from core.preflight import plan_preflight


class TickerDetailsRoutingTests(unittest.TestCase):
    def test_ticker_plus_details_forces_web_search(self) -> None:
        plan = plan_preflight("RURU hydropower ko stock ra project ko details chahiyo")
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan[0], "internet_search")
        q = plan[1]["query"]
        self.assertIn("RURU", q)
        self.assertIn("Nepal", q)
        self.assertIn("hydropower", q)

    def test_ticker_plus_information(self) -> None:
        plan = plan_preflight("NABIL share ko information chahiyo")
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan[0], "internet_search")
        self.assertIn("NABIL", plan[1]["query"])

    def test_ticker_plus_devanagari_barima(self) -> None:
        plan = plan_preflight("NABIL बारेमा जानकारी देउ")
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan[0], "internet_search")

    def test_bare_ticker_falls_through(self) -> None:
        # No "details / information" → let Sarvam decide.
        self.assertIsNone(plan_preflight("RURU"))

    def test_ticker_price_falls_through(self) -> None:
        # "RURU price" is an OSINT trading query, not a background one.
        self.assertIsNone(plan_preflight("RURU ko price"))


class CurrentRoleIdentityTests(unittest.TestCase):
    def test_current_pm_goes_to_web(self) -> None:
        plan = plan_preflight("nepal ko current PM k ho")
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan[0], "internet_search")
        self.assertIn("prime minister", plan[1]["query"].lower())
        self.assertIn("Nepal", plan[1]["query"])

    def test_ahile_ko_pm(self) -> None:
        plan = plan_preflight("ahile ko PM k ho")
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan[0], "internet_search")

    def test_non_current_pm_query_still_goes_to_osint(self) -> None:
        # A historical PM-name query should NOT flip to internet_search.
        plan = plan_preflight("PM ko barema bhanus")
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan[0], "get_nepal_live_context")

    def test_current_home_minister(self) -> None:
        plan = plan_preflight("current home minister of Nepal")
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan[0], "internet_search")
        self.assertIn("home minister", plan[1]["query"].lower())


if __name__ == "__main__":
    unittest.main()
