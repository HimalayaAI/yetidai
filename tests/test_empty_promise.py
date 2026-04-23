"""Tests for the empty-promise / tool-use enforcement helpers.

Motivated by a production transcript where Sarvam replied
  "म नेपालको आजको ताजा समाचार बताउँछु।"
without ever calling a tool. The bot passed that text straight to
Discord, leaving the user with a promise and no data.
"""
from __future__ import annotations

import unittest

from core.bot_helpers import (
    build_force_tool_nudge,
    detect_fabricated_filenames,
    is_empty_promise,
    needs_tool_use,
)


class EmptyPromiseTests(unittest.TestCase):
    def test_nepali_batauchhu(self) -> None:
        self.assertTrue(is_empty_promise(
            "म नेपालको आजको ताजा समाचार बताउँछु।",
        ))

    def test_sunaauchhu(self) -> None:
        self.assertTrue(is_empty_promise("म तपाईंलाई सुनाउँछु।"))

    def test_provide(self) -> None:
        self.assertTrue(is_empty_promise("म समाचार प्रदान गर्छु।"))

    def test_english_let_me(self) -> None:
        self.assertTrue(is_empty_promise("Let me fetch that for you"))

    def test_ill_provide(self) -> None:
        self.assertTrue(is_empty_promise("I'll provide the news."))

    def test_real_answer_not_promise(self) -> None:
        long_answer = (
            "नेपालको मुद्रास्फीति ४.८२% रहेको छ। यो गत वर्षको तुलनामा "
            "केही कम हो। NRB को तथ्याङ्कअनुसार अप्रिलमा यो थप कम हुने "
            "अपेक्षा गरिएको छ। विस्तृत तथ्याङ्कका लागि तल हेर्नुहोस्।"
            "\n\nस्रोत: https://www.nrb.org.np/"
        )
        self.assertFalse(is_empty_promise(long_answer))

    def test_false_when_tool_already_used(self) -> None:
        self.assertFalse(is_empty_promise(
            "म नेपालको आजको ताजा समाचार बताउँछु।",
            tool_was_used=True,
        ))

    def test_empty_text(self) -> None:
        self.assertFalse(is_empty_promise(""))
        self.assertFalse(is_empty_promise(None))

    def test_lyaaunchhu(self) -> None:
        # The exact text from the "will it ping me" transcript.
        self.assertTrue(is_empty_promise(
            "लौ, नेपालOSINT बाट अघिल्लो २४ घण्टाको ताजा समाचारहरू ल्याउँछु।"
        ))

    def test_khojnechhu(self) -> None:
        self.assertTrue(is_empty_promise("म यसको बारेमा खोज्नेछु।"))

    def test_ill_bring(self) -> None:
        self.assertTrue(is_empty_promise("I'll bring you the latest."))

    def test_let_me_find(self) -> None:
        self.assertTrue(is_empty_promise("Let me find that."))


class NeedsToolUseTests(unittest.TestCase):
    def test_urls_need_tool(self) -> None:
        self.assertTrue(needs_tool_use("यो repo हेर https://github.com/foo/bar"))

    def test_romanized_news_needs_tool(self) -> None:
        self.assertTrue(needs_tool_use("aja ko news bhana"))

    def test_devanagari_news_needs_tool(self) -> None:
        self.assertTrue(needs_tool_use("आजको समाचार चाहियो"))

    def test_pm_identity_needs_tool(self) -> None:
        self.assertTrue(needs_tool_use("nepal ko pradanmantri ko ho"))

    def test_who_is_devanagari(self) -> None:
        self.assertTrue(needs_tool_use("अर्थमन्त्री को हुनुहुन्छ?"))

    def test_iso_date_needs_tool(self) -> None:
        self.assertTrue(needs_tool_use("2026-04-15 ma k bhako thiyo?"))

    def test_greeting_does_not_need_tool(self) -> None:
        self.assertFalse(needs_tool_use("नमस्ते!"))
        self.assertFalse(needs_tool_use("k xa"))

    def test_empty(self) -> None:
        self.assertFalse(needs_tool_use(""))
        self.assertFalse(needs_tool_use(None))


class FabricatedFilenamesTests(unittest.TestCase):
    def test_detects_fake_file(self) -> None:
        tool_output = (
            "Top-level tree (main):\n"
            "  📄 bot.py (7211 B)\n"
            "  📄 README.md (4893 B)\n"
        )
        answer = (
            "यो repo मा NepaliNewsAggregator.py फाइल छ जसले समाचार ल्याउँछ।"
        )
        self.assertEqual(
            detect_fabricated_filenames(answer, tool_output),
            ["NepaliNewsAggregator.py"],
        )

    def test_real_file_passes(self) -> None:
        tool_output = "  📄 bot.py (7211 B)"
        answer = "यो repo मा bot.py मुख्य entrypoint हो।"
        self.assertEqual(detect_fabricated_filenames(answer, tool_output), [])

    def test_no_filenames(self) -> None:
        self.assertEqual(
            detect_fabricated_filenames("यो बढिया छ।", "whatever"),
            [],
        )

    def test_empty_inputs(self) -> None:
        self.assertEqual(detect_fabricated_filenames("", "x"), [])
        self.assertEqual(detect_fabricated_filenames("x", ""), [])


class ForceToolNudgeShapeTests(unittest.TestCase):
    def test_nudge_mentions_tools(self) -> None:
        msg = build_force_tool_nudge("aja ko khabar")
        for name in ("get_nepal_live_context", "internet_search"):
            self.assertIn(name, msg)


if __name__ == "__main__":
    unittest.main()
