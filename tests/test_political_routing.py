"""Routing regression tests for political / resignation queries.

Motivated by production traces:
  - "nepalosint bata political samachar haru" → web fallback polluted
    with the phrase "nepalosint" (got my skeleton GitHub repo).
  - "sudan gurung le kina resign gareyko" → Yeti hallucinated Sudan
    (the country) instead of searching for the Nepali home minister's
    resignation.
"""
from __future__ import annotations

import unittest

from core.preflight import plan_preflight
from tools.osint.context_router import route_query


class PoliticalNewsPreflightTests(unittest.TestCase):
    def test_political_samachar_routes_to_government(self) -> None:
        plan = plan_preflight("nepalosint bata political samachar haru bhanus")
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan[1]["intent"], "government")
        self.assertEqual(plan[1]["focus"], "political_news")

    def test_devanagari_rajnaitik_samachar(self) -> None:
        plan = plan_preflight("राजनीतिक समाचार देउ")
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan[1]["intent"], "government")

    def test_rajnitik_romanized(self) -> None:
        plan = plan_preflight("rajnitik samachar haru")
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan[1]["intent"], "government")


class ResignRoutingTests(unittest.TestCase):
    def test_resign_verb_routes_to_government(self) -> None:
        plan = plan_preflight("sudan gurung le kina resign gareyko")
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan[1]["intent"], "government")

    def test_rajinama_romanized(self) -> None:
        plan = plan_preflight("Sudhan Gurung ko rajinama kina")
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan[1]["intent"], "government")

    def test_devanagari_rajinama(self) -> None:
        plan = plan_preflight("गृहमन्त्रीले किन राजीनामा दिए?")
        self.assertIsNotNone(plan)
        assert plan is not None
        # Minister role (गृहमन्त्री) takes priority over the resign
        # keyword — both route to government which is the correct
        # destination.
        self.assertEqual(plan[1]["intent"], "government")

    def test_router_picks_up_resign(self) -> None:
        plan = route_query("sudan gurung le resign gareyko")
        self.assertIn("government", plan.intents)


if __name__ == "__main__":
    unittest.main()
