"""Regression tests for the 'aja ko samachar' router gap.

Motivated by a production trace:
  User: "aja ko samachar bhanus ta"
  → route_query returned use_nepalosint=False
  → OSINT tool ran but returned empty
  → Sarvam fabricated gov.np URLs (HTTP 000 when checked).

The fix lands `samachar`, `khabar`, `aja`, `aaja`, `aaj` in
GENERAL_KEYWORDS and adds romanized news phrases to
ROMANIZED_GENERAL_PATTERNS.
"""
from __future__ import annotations

import unittest

from tools.osint.context_router import route_query


class RomanizedNewsTests(unittest.TestCase):
    def test_aja_ko_samachar_triggers_osint(self) -> None:
        plan = route_query("aja ko samachar bhanus ta")
        self.assertTrue(plan.use_nepalosint)
        self.assertIn("general_news", plan.intents)

    def test_aaja_ko_khabar(self) -> None:
        plan = route_query("aaja ko khabar sunau ta")
        self.assertTrue(plan.use_nepalosint)
        self.assertIn("general_news", plan.intents)

    def test_bare_samachar(self) -> None:
        plan = route_query("malai samachar deu")
        self.assertTrue(plan.use_nepalosint)
        self.assertIn("general_news", plan.intents)

    def test_khabar_bhana(self) -> None:
        plan = route_query("khabar bhana hajur")
        self.assertTrue(plan.use_nepalosint)

    def test_headline_query(self) -> None:
        plan = route_query("aja ko headlines suna")
        self.assertTrue(plan.use_nepalosint)

    def test_greeting_still_skipped(self) -> None:
        # Sanity — small talk still doesn't trigger OSINT.
        plan = route_query("k xa")
        # route_query is Tree-1 (keyword); smalltalk filter is Tree-2.
        # The bare word "k xa" shouldn't produce any intents either way.
        self.assertEqual(plan.intents, [])


if __name__ == "__main__":
    unittest.main()
