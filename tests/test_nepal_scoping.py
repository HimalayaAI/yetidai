"""Tests for the Nepal-news scoping guardrails.

Motivated by a production trace where the user asked "आजको Nepali news
मा के-के छ त?", OSINT returned NO_MATCH, the internet_search fallback
fired with the bare Devanagari query, and DuckDuckGo returned Hindi
news portals (aajtak.in, indiatv.in, amarujala.com). Yeti summarised
Indian news as "आजको Nepal news".
"""
from __future__ import annotations

import unittest

from tools.osint.plugin import _nepal_scoped_query
from tools.search.plugin import (
    _apply_nepal_filter,
    _is_nepal_scoped_query,
)


class NepalScopedQueryTests(unittest.TestCase):
    def test_prepends_nepal_when_missing(self) -> None:
        self.assertEqual(_nepal_scoped_query("aja ko samachar"), "Nepal aja ko samachar")

    def test_does_not_double_prepend(self) -> None:
        self.assertEqual(_nepal_scoped_query("Nepal debt today"), "Nepal debt today")

    def test_respects_devanagari_nepal(self) -> None:
        self.assertEqual(
            _nepal_scoped_query("नेपालको ऋण कति छ"),
            "नेपालको ऋण कति छ",
        )

    def test_empty_fallbacks_to_default(self) -> None:
        self.assertEqual(_nepal_scoped_query(""), "Nepal news today")

    def test_none_fallbacks_to_default(self) -> None:
        self.assertEqual(_nepal_scoped_query(None), "Nepal news today")

    def test_strips_nepalosint_meta_token(self) -> None:
        """'nepalosint bata X' should search for X (Nepal-scoped),
        not search for the phrase 'nepalosint'."""
        out = _nepal_scoped_query("nepalosint bata political samachar haru")
        # "nepalosint bata" is gone.
        self.assertNotIn("nepalosint", out.lower())
        # The actual intent survives.
        self.assertIn("samachar", out)
        # Still Nepal-scoped.
        self.assertIn("Nepal", out)

    def test_strips_osint_ma(self) -> None:
        out = _nepal_scoped_query("osint ma aja ko khabar")
        self.assertNotIn("osint", out.lower())
        self.assertIn("khabar", out)


class NepalScopedDetectionTests(unittest.TestCase):
    def test_english_nepal(self) -> None:
        self.assertTrue(_is_nepal_scoped_query("Nepal news today"))

    def test_devanagari(self) -> None:
        self.assertTrue(_is_nepal_scoped_query("नेपाल समाचार"))

    def test_non_nepal(self) -> None:
        self.assertFalse(_is_nepal_scoped_query("UEFA 2025 winner"))


class NepalFilterTests(unittest.TestCase):
    """The actual production failure shape — bad URLs from the
    'Indian news' pollution ring."""

    def test_drops_indian_hosts_on_nepal_query(self) -> None:
        results = [
            {"title": "India news", "snippet": "…", "href": "https://www.aajtak.in/"},
            {"title": "India TV", "snippet": "…", "href": "https://www.indiatv.in/hindi-samachar"},
            {"title": "Nepal news", "snippet": "…", "href": "https://merolagani.com/x"},
            {"title": "Kathmandu post", "snippet": "…", "href": "https://kathmandupost.com/y"},
        ]
        filtered = _apply_nepal_filter(results, "Nepal news today")
        hosts = [r["href"] for r in filtered]
        self.assertNotIn("https://www.aajtak.in/", hosts)
        self.assertNotIn("https://www.indiatv.in/hindi-samachar", hosts)
        self.assertIn("https://merolagani.com/x", hosts)
        self.assertIn("https://kathmandupost.com/y", hosts)

    def test_does_not_filter_non_nepal_queries(self) -> None:
        # A world query (UEFA) should NOT have hosts filtered even if
        # aajtak happened to hit.
        results = [
            {"title": "UEFA stat", "snippet": "…", "href": "https://www.aajtak.in/x"},
            {"title": "BBC", "snippet": "…", "href": "https://www.bbc.com/sport"},
        ]
        filtered = _apply_nepal_filter(results, "UEFA Champions League 2025 winner")
        # Both still present.
        self.assertEqual(len(filtered), 2)

    def test_drops_hamropatro_panchang_pollution(self) -> None:
        # hamropatro.com is a panchang/astrology site that DDG kept
        # injecting into Nepal news SERPs.
        results = [
            {"title": "panchang", "snippet": "…", "href": "https://www.hamropatro.com/rashifal"},
            {"title": "real nepal news", "snippet": "…", "href": "https://setopati.com/x"},
        ]
        filtered = _apply_nepal_filter(results, "Nepal aja ko samachar")
        hosts = [r["href"] for r in filtered]
        self.assertNotIn("https://www.hamropatro.com/rashifal", hosts)
        self.assertIn("https://setopati.com/x", hosts)


if __name__ == "__main__":
    unittest.main()
