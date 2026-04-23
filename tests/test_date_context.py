"""Tests for core.date_context.build_date_block."""
from __future__ import annotations

import unittest
from datetime import date

from core.date_context import build_date_block


class DateBlockTests(unittest.TestCase):
    def test_contains_all_three_anchors(self) -> None:
        block = build_date_block(date(2026, 4, 23))
        self.assertIn("आज (Today)", block)
        self.assertIn("हिजो (Yesterday)", block)
        self.assertIn("गत हप्ता (1 week ago)", block)

    def test_ad_dates_correct(self) -> None:
        block = build_date_block(date(2026, 4, 23))
        self.assertIn("2026-04-23 AD", block)
        self.assertIn("2026-04-22 AD", block)
        self.assertIn("2026-04-16 AD", block)

    def test_includes_bs_when_available(self) -> None:
        block = build_date_block(date(2026, 4, 23))
        # Either BS data is in the block, or the fallback line is.
        has_bs = "BS 2083-01-10" in block
        has_fallback = "BS conversion unavailable" in block
        self.assertTrue(has_bs or has_fallback)

    def test_rules_present(self) -> None:
        block = build_date_block(date(2026, 4, 23))
        self.assertIn("ground truth", block)
        self.assertIn("Never hallucinate a date", block)
        self.assertIn("BOTH calendars", block)


if __name__ == "__main__":
    unittest.main()
