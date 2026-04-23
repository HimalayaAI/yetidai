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
    def test_fresh_data_ok(self) -> None:
        fresh = (date.today() - timedelta(days=1)).isoformat()
        info = assess_freshness("aja ko news", {"x": {"date": fresh}})
        self.assertFalse(info["stale"])
        self.assertTrue(info["required"])

    def test_stale_for_recency_query(self) -> None:
        stale = (date.today() - timedelta(days=RECENCY_THRESHOLD_DAYS + 2)).isoformat()
        info = assess_freshness("aja ko news", {"x": {"date": stale}})
        self.assertTrue(info["stale"])

    def test_not_stale_when_recency_not_requested(self) -> None:
        # Same old data, but the user didn't ask about today — not stale.
        stale = (date.today() - timedelta(days=365)).isoformat()
        info = assess_freshness("last year budget", {"x": {"date": stale}})
        self.assertFalse(info["stale"])
        self.assertFalse(info["required"])


if __name__ == "__main__":
    unittest.main()
