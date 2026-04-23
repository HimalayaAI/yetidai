"""Tests for tools.osint.freshness."""
from __future__ import annotations

import unittest
from datetime import date, timedelta

from tools.osint.freshness import (
    RECENCY_THRESHOLD_DAYS,
    assess_freshness,
    is_recency_query,
    newest_date,
)


class RecencyDetectorTests(unittest.TestCase):
    def test_aja_triggers(self) -> None:
        self.assertTrue(is_recency_query("aja ko khabar"))

    def test_devanagari_aja(self) -> None:
        self.assertTrue(is_recency_query("आज के भयो?"))

    def test_plain_historical_does_not(self) -> None:
        self.assertFalse(is_recency_query("last year's budget"))


class NewestDateTests(unittest.TestCase):
    def test_walks_nested_payload(self) -> None:
        payloads = {
            "stories": [
                {"title": "a", "published_at": "2026-04-10"},
                {"title": "b", "published_at": "2026-04-20T12:00:00Z"},
                {"title": "c", "created_at": "2024-03-01"},
            ],
            "meta": {"as_of": "2025-12-31"},
        }
        self.assertEqual(newest_date(payloads), date(2026, 4, 20))

    def test_ignores_future_dates(self) -> None:
        far_future = (date.today() + timedelta(days=30)).isoformat()
        payloads = {"a": {"date": far_future}, "b": {"date": "2026-04-01"}}
        self.assertEqual(newest_date(payloads), date(2026, 4, 1))

    def test_handles_empty(self) -> None:
        self.assertIsNone(newest_date({}))

    def test_epoch_seconds(self) -> None:
        from datetime import datetime, timezone
        expected = date(2026, 4, 15)
        ts = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc).timestamp()
        payloads = {"x": {"timestamp": int(ts)}}
        self.assertEqual(newest_date(payloads), expected)


class AssessFreshnessTests(unittest.TestCase):
    # These tests pass coverage_until=None to isolate payload-age logic
    # from the global OSINT_COVERAGE_UNTIL anchor. Separate tests below
    # exercise coverage-gap behavior explicitly.

    def test_fresh_data_ok(self) -> None:
        fresh = (date.today() - timedelta(days=1)).isoformat()
        info = assess_freshness(
            "aja ko news", {"x": {"date": fresh}}, coverage_until=None,
        )
        self.assertFalse(info["stale"])
        self.assertTrue(info["required"])

    def test_stale_for_recency_query(self) -> None:
        stale = (date.today() - timedelta(days=RECENCY_THRESHOLD_DAYS + 2)).isoformat()
        info = assess_freshness(
            "aja ko news", {"x": {"date": stale}}, coverage_until=None,
        )
        self.assertTrue(info["stale"])

    def test_stale_regardless_of_recency_gate(self) -> None:
        # Behavior change: staleness is now decoupled from recency keywords.
        # A year-old payload is stale even for non-recency queries, so the
        # fallback fires and the user doesn't get 2024 numbers served as
        # current.
        stale = (date.today() - timedelta(days=365)).isoformat()
        info = assess_freshness(
            "last year budget", {"x": {"date": stale}}, coverage_until=None,
        )
        self.assertTrue(info["stale"])
        self.assertFalse(info["required"])


class CoverageGapTests(unittest.TestCase):
    def test_coverage_gap_fires_on_fresh_payload(self) -> None:
        # Payload is 1 day old (fresh), but upstream coverage anchor was
        # 30 days ago — the whole dataset is behind, so stale + gap fire.
        fresh = (date.today() - timedelta(days=1)).isoformat()
        anchor = date.today() - timedelta(days=30)
        info = assess_freshness(
            "nepalko pm ko ho",
            {"x": {"date": fresh}},
            coverage_until=anchor,
        )
        self.assertTrue(info["coverage_gap"])
        self.assertTrue(info["stale"])
        self.assertEqual(info["gap_days"], 30)

    def test_coverage_anchor_inside_threshold_no_gap(self) -> None:
        # Anchor within threshold (2 days ago, threshold=3) → no gap.
        anchor = date.today() - timedelta(days=2)
        info = assess_freshness("anything", {}, coverage_until=anchor)
        self.assertFalse(info["coverage_gap"])
        self.assertFalse(info["stale"])

    def test_coverage_disabled_with_none(self) -> None:
        info = assess_freshness("aja", {}, coverage_until=None)
        self.assertFalse(info["coverage_gap"])
        self.assertFalse(info["stale"])
        self.assertIsNone(info["gap_days"])


if __name__ == "__main__":
    unittest.main()
