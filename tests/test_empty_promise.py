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
    looks_like_news_answer,
    needs_tool_use,
    news_answer_off_topic,
    user_asked_for_news,
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


class NewsShapeTests(unittest.TestCase):
    """Guards the 'samachar → tarkari' (vegetables) failure mode."""

    def test_user_asked_for_news_devanagari(self) -> None:
        self.assertTrue(user_asked_for_news("आजको ताजा समाचार देउ"))

    def test_user_asked_for_news_romanized(self) -> None:
        self.assertTrue(user_asked_for_news("aja ko taja samachar dinus"))

    def test_small_talk_not_news(self) -> None:
        self.assertFalse(user_asked_for_news("नमस्ते"))
        self.assertFalse(user_asked_for_news("k xa"))

    def test_answer_with_url_looks_like_news(self) -> None:
        self.assertTrue(looks_like_news_answer(
            "नेप्से २८०० भन्दा तल बन्द।\nस्रोत: https://merolagani.com/x"
        ))

    def test_vegetable_list_does_not_look_like_news(self) -> None:
        answer = (
            "आजका लागि ताजा तरकारीहरू:\nकाउली\nफर्सी\nगाँजर\nगाँड्यौलो"
        )
        self.assertFalse(looks_like_news_answer(answer))

    def test_off_topic_fires(self) -> None:
        # User wanted news, got vegetables, no tool ran → fire retry.
        self.assertTrue(news_answer_off_topic(
            "aja ko taja samachar dinus",
            "आजका लागि ताजा तरकारीहरू:\nकाउली\nफर्सी",
            tool_was_used=False,
        ))

    def test_no_fire_when_tool_was_used(self) -> None:
        # If a tool ran, the fabricated-URL/filename detectors are the
        # right guards. We don't fire the news-shape check.
        self.assertFalse(news_answer_off_topic(
            "aja ko taja samachar",
            "आजका लागि ताजा तरकारीहरू:\nकाउली",
            tool_was_used=True,
        ))

    def test_no_fire_when_user_didnt_ask_for_news(self) -> None:
        self.assertFalse(news_answer_off_topic(
            "नमस्ते, kasto xa",
            "ए हजुर, ठिकै छु।",
            tool_was_used=False,
        ))

    def test_news_answer_with_citations_passes(self) -> None:
        # Real news reply — we must NOT false-positive.
        answer = (
            "आज नेपालका मुख्य खबरहरू:\n"
            "१. नेप्से २८०० भन्दा तल बन्द भयो।\n"
            "२. सुदूरपश्चिममा विद्युत उपयोग कम।\n\n"
            "स्रोत:\n- https://merolagani.com/NewsDetail.aspx?newsID=125524"
        )
        self.assertFalse(news_answer_off_topic(
            "aja ko samachar", answer, tool_was_used=True,
        ))


class ForceToolNudgeShapeTests(unittest.TestCase):
    def test_nudge_mentions_tools(self) -> None:
        msg = build_force_tool_nudge("aja ko khabar")
        for name in ("get_nepal_live_context", "internet_search"):
            self.assertIn(name, msg)


if __name__ == "__main__":
    unittest.main()
