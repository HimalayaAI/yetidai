"""Regression tests for the GitHub commit-URL handling.

Motivated by:
  User: "https://github.com/HimalayaAI/yetidai/commit/cb3a55b... yo ma k k changes cha"
  → Yeti: "म तपाईँलाई help गर्नेछु" (empty promise, no tool call)

Two bugs addressed:
  1. _parse_repo() rejected /commit/<sha> URLs entirely.
  2. is_empty_promise() missed the "गर्नेछु" future-tense shape.
"""
from __future__ import annotations

import unittest

from core.bot_helpers import is_empty_promise
from tools.github.plugin import _parse_repo


class ParseRepoTests(unittest.TestCase):
    def test_bare_owner_repo(self) -> None:
        self.assertEqual(
            _parse_repo("HimalayaAI/yetidai"),
            ("HimalayaAI", "yetidai", None, None, None),
        )

    def test_repo_url(self) -> None:
        self.assertEqual(
            _parse_repo("https://github.com/HimalayaAI/yetidai"),
            ("HimalayaAI", "yetidai", None, None, None),
        )

    def test_tree_url(self) -> None:
        self.assertEqual(
            _parse_repo("https://github.com/HimalayaAI/yetidai/tree/main"),
            ("HimalayaAI", "yetidai", "main", None, None),
        )

    def test_blob_url_with_path(self) -> None:
        self.assertEqual(
            _parse_repo("https://github.com/HimalayaAI/yetidai/blob/main/bot.py"),
            ("HimalayaAI", "yetidai", "main", None, "bot.py"),
        )

    def test_commit_url(self) -> None:
        self.assertEqual(
            _parse_repo(
                "https://github.com/HimalayaAI/yetidai/commit/cb3a55be60ee0c9f"
            ),
            ("HimalayaAI", "yetidai", None, "cb3a55be60ee0c9f", None),
        )

    def test_pull_url_keeps_repo_only(self) -> None:
        # PRs aren't wired to a dedicated endpoint; fall back to repo view.
        owner, name, branch, sha, path = _parse_repo(
            "https://github.com/HimalayaAI/yetidai/pull/12"
        )
        self.assertEqual((owner, name), ("HimalayaAI", "yetidai"))
        self.assertIsNone(branch)
        self.assertIsNone(sha)
        self.assertIsNone(path)

    def test_trailing_slash_and_query(self) -> None:
        self.assertEqual(
            _parse_repo("https://github.com/HimalayaAI/yetidai/?tab=readme"),
            ("HimalayaAI", "yetidai", None, None, None),
        )

    def test_dot_git_suffix(self) -> None:
        self.assertEqual(
            _parse_repo("https://github.com/HimalayaAI/yetidai.git"),
            ("HimalayaAI", "yetidai", None, None, None),
        )

    def test_invalid_input(self) -> None:
        with self.assertRaises(ValueError):
            _parse_repo("not a repo")


class EmptyPromiseFutureTenseTests(unittest.TestCase):
    def test_help_garnechu(self) -> None:
        # The exact text from the trace.
        self.assertTrue(is_empty_promise(
            "तपाईँले दिनुभएको GitHub commit को changes हेर्न म तपाईँलाई help गर्नेछु।"
        ))

    def test_garnechu_generic(self) -> None:
        self.assertTrue(is_empty_promise("म तपाईंलाई जानकारी प्रदान गर्नेछु।"))

    def test_hernechu(self) -> None:
        self.assertTrue(is_empty_promise("म यो हेर्नेछु है।"))

    def test_will_analyze_english(self) -> None:
        self.assertTrue(is_empty_promise("I'll analyze that for you."))


if __name__ == "__main__":
    unittest.main()
