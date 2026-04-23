"""Tests for the fake-source-name guard and the explicit web-search
command preflight.

Traces motivating each:
  - User: "nepal ko pradhanmantri ra cabinet mantri ko ko cha ahile"
    Yeti: (stale confident answer) स्रोत: The Associated Press, Reuters
    — neither source was actually hit by any tool this turn. Pure
    hallucinated citation.
  - User: "web search garnus alchi nagarnus google chalaunus"
    Yeti: same generic answer as before — ignored the explicit order
    to search the web.
"""
from __future__ import annotations

import unittest

from core.bot_helpers import detect_fabricated_source_names
from core.preflight import _strip_command_tokens, plan_preflight


class FabricatedSourceNameTests(unittest.TestCase):
    def test_flags_ap_when_tool_output_empty(self) -> None:
        answer = (
            "केही details।\n\nस्रोत:\nThe Associated Press\nReuters"
        )
        bad = detect_fabricated_source_names(answer, tool_output="")
        self.assertIn("The Associated Press", " ".join(bad))
        self.assertIn("Reuters", " ".join(bad))

    def test_skips_when_url_present_in_sources(self) -> None:
        # If the block already has URLs the URL-level validator
        # handles it — we don't double-flag.
        answer = (
            "details\n\nस्रोत:\n- https://merolagani.com/x\n"
            "The Associated Press"
        )
        bad = detect_fabricated_source_names(answer, tool_output="")
        self.assertEqual(bad, [])

    def test_passes_when_name_appears_in_tool_output(self) -> None:
        # If a tool actually fetched from reuters.com, mentioning
        # "Reuters" is legitimate.
        answer = "details\n\nस्रोत:\nReuters"
        tool_output = "… reuters.com/article/xyz …"
        bad = detect_fabricated_source_names(answer, tool_output)
        # "reuters" appears in tool_output → not fabricated.
        self.assertEqual(bad, [])

    def test_empty_answer(self) -> None:
        self.assertEqual(
            detect_fabricated_source_names("", "tool output"),
            [],
        )


class WebSearchCommandPreflightTests(unittest.TestCase):
    def test_web_search_garnus(self) -> None:
        plan = plan_preflight("web search garnus alchi nagarnus")
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan[0], "internet_search")

    def test_google_chalaunus(self) -> None:
        plan = plan_preflight("google chalaunus yo barima")
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan[0], "internet_search")

    def test_online_khoju(self) -> None:
        plan = plan_preflight("online khoju hajur nepal inflation")
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan[0], "internet_search")

    def test_command_tokens_stripped_from_query(self) -> None:
        # The subject ("Nepal PM") survives; the "web search garnus"
        # / "please" / "hajur" noise is removed.
        stripped = _strip_command_tokens(
            "web search garnus hajur Nepal PM ko ho"
        )
        self.assertIn("Nepal", stripped)
        self.assertIn("PM", stripped)
        self.assertNotIn("web search", stripped.lower())
        self.assertNotIn("garnus", stripped.lower())

    def test_no_command_returns_none_fallthrough(self) -> None:
        # A regular query without "web search"/"google" falls to the
        # other preflight rules.
        plan = plan_preflight("aja ko news bhanus")
        self.assertIsNotNone(plan)
        assert plan is not None
        # Not internet_search — should be OSINT general_news.
        self.assertNotEqual(plan[0], "internet_search")


if __name__ == "__main__":
    unittest.main()
