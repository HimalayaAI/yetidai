"""Regression tests for minister-role detection.

Motivated by:
  User: "nepal ko HM le kina resign garnu vako"
  Yeti: "तपाईँको प्रश्न नेपालको प्रधानमन्त्रीको राजीनामासँग सम्बन्धित छ…"
        (confused HM with PM)

detect_minister_role() resolves the ambiguity; route_query() uses the
tag to force `government` intent + `is_who_is=True`.
"""
from __future__ import annotations

import unittest

from tools.osint.context_router import detect_minister_role, route_query


class DetectMinisterRoleTests(unittest.TestCase):
    def test_hm_abbreviation(self) -> None:
        self.assertEqual(
            detect_minister_role("nepal ko HM le kina resign garnu vako"),
            "home_minister",
        )

    def test_pm_abbreviation(self) -> None:
        self.assertEqual(
            detect_minister_role("PM ko ho?"),
            "prime_minister",
        )

    def test_devanagari_home_minister(self) -> None:
        self.assertEqual(
            detect_minister_role("गृहमन्त्री ले के भने?"),
            "home_minister",
        )

    def test_romanized_arthamantri(self) -> None:
        self.assertEqual(
            detect_minister_role("arthamantri ko kaam k cha"),
            "finance_minister",
        )

    def test_english_finance_minister(self) -> None:
        self.assertEqual(
            detect_minister_role("current finance minister of Nepal"),
            "finance_minister",
        )

    def test_not_a_minister_query(self) -> None:
        self.assertIsNone(detect_minister_role("nepse ko price"))
        self.assertIsNone(detect_minister_role("aja ko samachar"))

    def test_hm_does_not_match_inside_word(self) -> None:
        # `hm` is 2 chars ASCII so word-boundary matching applies.
        self.assertIsNone(detect_minister_role("ahem, something"))


class RouterUsesMinisterDetectionTests(unittest.TestCase):
    def test_hm_query_routes_to_government(self) -> None:
        plan = route_query("nepal ko HM le kina resign garnu vako")
        self.assertIn("government", plan.intents)
        self.assertTrue(plan.is_who_is)

    def test_pm_query_routes_to_government(self) -> None:
        plan = route_query("nepal ko PM ko ho")
        self.assertIn("government", plan.intents)
        self.assertTrue(plan.is_who_is)

    def test_gdp_routes_to_macro(self) -> None:
        plan = route_query("nepal ko gdp kati ho")
        self.assertIn("macro", plan.intents)


if __name__ == "__main__":
    unittest.main()
