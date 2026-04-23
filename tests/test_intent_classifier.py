"""Tests for core/intent_classifier.py.

Focused on the parse path and derived helpers — the async Sarvam call
itself is not exercised here (would need a live fixture). `classify_intent`
is tested by mocking the client to return a canned response.
"""
from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from core.intent_classifier import (
    CLASSIFIER_CONFIDENCE_THRESHOLD,
    _parse_classification,
    classify_intent,
    routing_hint_message,
    should_force_tool_call,
)


class ParseClassificationTests(unittest.TestCase):
    def test_clean_json(self) -> None:
        r = _parse_classification(
            '{"bucket":2,"tools_needed":["get_nepal_live_context"],'
            '"osint_intent":"who_is","recency_required":true,"confidence":0.9}'
        )
        self.assertEqual(r["bucket"], 2)
        self.assertEqual(r["tools_needed"], ["get_nepal_live_context"])
        self.assertEqual(r["osint_intent"], "who_is")
        self.assertTrue(r["recency_required"])
        self.assertAlmostEqual(r["confidence"], 0.9)

    def test_code_fenced_json(self) -> None:
        r = _parse_classification(
            '```json\n{"bucket":1,"tools_needed":[],'
            '"osint_intent":null,"recency_required":false,"confidence":0.95}\n```'
        )
        self.assertEqual(r["bucket"], 1)
        self.assertEqual(r["tools_needed"], [])
        self.assertIsNone(r["osint_intent"])

    def test_json_embedded_in_prose(self) -> None:
        r = _parse_classification(
            'Classification: {"bucket":3,"tools_needed":["internet_search"],'
            '"osint_intent":null,"recency_required":false,"confidence":0.8}.'
        )
        self.assertEqual(r["bucket"], 3)
        self.assertEqual(r["tools_needed"], ["internet_search"])

    def test_malformed_returns_none(self) -> None:
        self.assertIsNone(_parse_classification("not json"))
        self.assertIsNone(_parse_classification(""))
        self.assertIsNone(_parse_classification('{"bucket": 99}'))  # out of range
        self.assertIsNone(_parse_classification('{"bucket": "two"}'))  # wrong type

    def test_unknown_tool_names_filtered(self) -> None:
        r = _parse_classification(
            '{"bucket":2,"tools_needed":["get_nepal_live_context","fake_tool"],'
            '"osint_intent":null,"recency_required":false,"confidence":0.9}'
        )
        self.assertEqual(r["tools_needed"], ["get_nepal_live_context"])

    def test_unknown_osint_intent_normalised_to_none(self) -> None:
        r = _parse_classification(
            '{"bucket":2,"tools_needed":["get_nepal_live_context"],'
            '"osint_intent":"not_a_real_intent","recency_required":false,"confidence":0.9}'
        )
        self.assertIsNone(r["osint_intent"])

    def test_confidence_clamped(self) -> None:
        r_high = _parse_classification(
            '{"bucket":2,"tools_needed":[],"osint_intent":null,'
            '"recency_required":false,"confidence":2.5}'
        )
        self.assertEqual(r_high["confidence"], 1.0)
        r_low = _parse_classification(
            '{"bucket":2,"tools_needed":[],"osint_intent":null,'
            '"recency_required":false,"confidence":-1.0}'
        )
        self.assertEqual(r_low["confidence"], 0.0)


class ShouldForceToolCallTests(unittest.TestCase):
    def _cls(self, *, bucket, tools, confidence):
        return {
            "bucket": bucket,
            "tools_needed": tools,
            "osint_intent": None,
            "recency_required": False,
            "confidence": confidence,
        }

    def test_forces_when_confident_and_tools_needed(self) -> None:
        self.assertTrue(should_force_tool_call(
            self._cls(bucket=2, tools=["get_nepal_live_context"], confidence=0.9)
        ))

    def test_does_not_force_smalltalk(self) -> None:
        self.assertFalse(should_force_tool_call(
            self._cls(bucket=1, tools=[], confidence=0.99)
        ))

    def test_does_not_force_when_low_confidence(self) -> None:
        # One notch below the threshold.
        self.assertFalse(should_force_tool_call(
            self._cls(
                bucket=2,
                tools=["get_nepal_live_context"],
                confidence=CLASSIFIER_CONFIDENCE_THRESHOLD - 0.01,
            )
        ))

    def test_does_not_force_when_tools_empty(self) -> None:
        # Even a confident non-smalltalk bucket with no tools shouldn't
        # force — nothing to force to.
        self.assertFalse(should_force_tool_call(
            self._cls(bucket=2, tools=[], confidence=0.99)
        ))


class RoutingHintMessageTests(unittest.TestCase):
    def test_system_role(self) -> None:
        hint = routing_hint_message({
            "bucket": 2,
            "tools_needed": ["get_nepal_live_context"],
            "osint_intent": "who_is",
            "recency_required": True,
            "confidence": 0.9,
        })
        self.assertEqual(hint["role"], "system")
        self.assertIn("ROUTING HINT", hint["content"])
        self.assertIn("bucket=2", hint["content"])
        self.assertIn("who_is", hint["content"])
        self.assertIn("recency=true", hint["content"])

    def test_minimal_hint(self) -> None:
        hint = routing_hint_message({
            "bucket": 1,
            "tools_needed": [],
            "osint_intent": None,
            "recency_required": False,
            "confidence": 0.95,
        })
        self.assertIn("bucket=1", hint["content"])
        self.assertNotIn("tools=", hint["content"])
        self.assertNotIn("osint_intent", hint["content"])


class ClassifyIntentTests(unittest.TestCase):
    """End-to-end with a mocked Sarvam client — no network."""

    def _make_client(self, content: str) -> MagicMock:
        client = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = content
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        client.chat.completions = AsyncMock(return_value=mock_resp)
        return client

    def test_parses_clean_response(self) -> None:
        client = self._make_client(
            '{"bucket":2,"tools_needed":["get_nepal_live_context"],'
            '"osint_intent":"macro","recency_required":true,"confidence":0.88}'
        )
        result = asyncio.run(
            classify_intent(client, "nepalko inflation kati cha", model="sarvam-30b")
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["bucket"], 2)
        self.assertEqual(result["osint_intent"], "macro")

    def test_empty_query_returns_none(self) -> None:
        client = self._make_client("{}")
        result = asyncio.run(
            classify_intent(client, "", model="sarvam-30b")
        )
        self.assertIsNone(result)

    def test_client_error_returns_none(self) -> None:
        client = MagicMock()
        client.chat.completions = AsyncMock(side_effect=RuntimeError("boom"))
        result = asyncio.run(
            classify_intent(client, "something", model="sarvam-30b")
        )
        self.assertIsNone(result)

    def test_timeout_returns_none(self) -> None:
        async def _slow(*args, **kwargs):
            await asyncio.sleep(10)
        client = MagicMock()
        client.chat.completions = _slow
        result = asyncio.run(
            classify_intent(client, "something", model="sarvam-30b", timeout_s=0.05)
        )
        self.assertIsNone(result)

    def test_bad_json_returns_none(self) -> None:
        client = self._make_client("definitely not json")
        result = asyncio.run(
            classify_intent(client, "something", model="sarvam-30b")
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
