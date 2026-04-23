"""Tests for the v3 additions to core.bot_helpers.

Covers:
  * canonical_tool_args strips pagination keys / normalises strings.
  * hash_tool_call collapses trivially-different args to the same key.
  * looks_like_correction + detect_requested_count.
  * TOOL_STALE_MARKER is treated as a status marker.
"""
from __future__ import annotations

import types
import unittest

from core.bot_helpers import (
    TOOL_STALE_MARKER,
    build_correction_nudge,
    canonical_tool_args,
    detect_requested_count,
    hash_tool_call,
    is_real_tool_content,
    is_tool_status_marker,
    looks_like_correction,
)


class CanonicalArgsTests(unittest.TestCase):
    def test_drops_pagination_keys(self) -> None:
        args = {"query": "NEPSE", "limit": 10, "offset": 20, "top_k": 5}
        self.assertEqual(canonical_tool_args(args), {"query": "nepse"})

    def test_lowercases_strings(self) -> None:
        self.assertEqual(
            canonical_tool_args({"focus": "  Inflation  "}),
            {"focus": "inflation"},
        )

    def test_empty(self) -> None:
        self.assertEqual(canonical_tool_args(None), {})
        self.assertEqual(canonical_tool_args({}), {})


class HashCollapsesNoiseTests(unittest.TestCase):
    def test_different_limits_hash_same(self) -> None:
        a = hash_tool_call("x", {"query": "nepse", "limit": 8})
        b = hash_tool_call("x", {"query": "nepse", "limit": 20})
        self.assertEqual(a, b)

    def test_case_differences_hash_same(self) -> None:
        a = hash_tool_call("x", {"query": "Nepal OSINT"})
        b = hash_tool_call("x", {"query": "nepal osint"})
        self.assertEqual(a, b)

    def test_distinct_queries_hash_different(self) -> None:
        a = hash_tool_call("x", {"query": "inflation"})
        b = hash_tool_call("x", {"query": "nepse"})
        self.assertNotEqual(a, b)


class CorrectionTests(unittest.TestCase):
    def test_romanized_correction(self) -> None:
        self.assertTrue(looks_like_correction("haina, tyo ho hoina"))

    def test_english_correction(self) -> None:
        self.assertTrue(looks_like_correction("that is not what I asked"))

    def test_benign_message_not_correction(self) -> None:
        self.assertFalse(looks_like_correction("kasto chha"))

    def test_count_detection_devanagari(self) -> None:
        self.assertEqual(detect_requested_count("मलाई ३० वटा खबर देउ"), 30)

    def test_count_detection_english(self) -> None:
        self.assertEqual(detect_requested_count("give me 5 news stories"), 5)

    def test_count_detection_none(self) -> None:
        self.assertIsNone(detect_requested_count("aja ko khabar"))

    def test_correction_nudge_mentions_count(self) -> None:
        text = build_correction_nudge("haina, 30 chahinchha", requested_count=30)
        self.assertIn("30", text)
        self.assertIn("पुनः", text)


class StaleMarkerTests(unittest.TestCase):
    def test_stale_marker_is_status(self) -> None:
        self.assertTrue(is_tool_status_marker(f"{TOOL_STALE_MARKER} 12 days old"))

    def test_is_real_tool_content_rejects_stale(self) -> None:
        result = types.SimpleNamespace(
            success=True, content=f"{TOOL_STALE_MARKER} stale data",
        )
        self.assertFalse(is_real_tool_content(result))


if __name__ == "__main__":
    unittest.main()
