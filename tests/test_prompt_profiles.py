"""Unit tests for prompt-profile helpers."""

from __future__ import annotations

import unittest

from core.prompt_profiles import (
    DEFAULT_SMALL_MODEL_NAME,
    LARGE_PROFILE_NAME,
    SMALL_PROFILE_NAME,
    build_runtime_system_prompt,
    build_small_runtime_prompt,
    parse_stop_tokens,
    resolve_prompt_profile,
    resolve_runtime_model,
)


class ResolvePromptProfileTests(unittest.TestCase):
    def test_auto_resolves_small_for_tiny_model(self) -> None:
        profile = resolve_prompt_profile("auto", "himalayagpt-0.5b-it")
        self.assertEqual(profile, SMALL_PROFILE_NAME)

    def test_auto_resolves_large_for_non_tiny_model(self) -> None:
        profile = resolve_prompt_profile("auto", "gpt-4.1-mini")
        self.assertEqual(profile, LARGE_PROFILE_NAME)

    def test_explicit_small_alias_wins(self) -> None:
        profile = resolve_prompt_profile("small", "gpt-4.1-mini")
        self.assertEqual(profile, SMALL_PROFILE_NAME)


class PromptBodyTests(unittest.TestCase):
    def test_small_prompt_stays_compact_and_tool_bounded(self) -> None:
        prompt = build_small_runtime_prompt()
        self.assertIn("get_nepal_live_context", prompt)
        self.assertIn("internet_search", prompt)
        self.assertIn("fetch_url", prompt)
        self.assertIn("बढीमा ४ बुँदा", prompt)
        self.assertIn("स्रोत:", prompt)
        self.assertLess(len(prompt), 450)

    def test_profile_dispatch_works(self) -> None:
        small = build_runtime_system_prompt(SMALL_PROFILE_NAME)
        large = build_runtime_system_prompt(LARGE_PROFILE_NAME)
        self.assertNotEqual(small, large)


class StopTokenParserTests(unittest.TestCase):
    def test_parse_stop_tokens(self) -> None:
        self.assertEqual(parse_stop_tokens("###,  END ,"), ["###", "END"])

    def test_parse_stop_tokens_none(self) -> None:
        self.assertIsNone(parse_stop_tokens(""))
        self.assertIsNone(parse_stop_tokens(None))


class RuntimeModelResolutionTests(unittest.TestCase):
    def test_small_profile_forces_small_model_default(self) -> None:
        model = resolve_runtime_model(SMALL_PROFILE_NAME, "himalaya-q8")
        self.assertEqual(model, DEFAULT_SMALL_MODEL_NAME)

    def test_small_profile_allows_explicit_small_override(self) -> None:
        model = resolve_runtime_model(SMALL_PROFILE_NAME, "himalaya-q8", "himalaya-custom")
        self.assertEqual(model, "himalaya-custom")

    def test_large_profile_keeps_configured_model(self) -> None:
        model = resolve_runtime_model(LARGE_PROFILE_NAME, "gpt-4.1-mini")
        self.assertEqual(model, "gpt-4.1-mini")


if __name__ == "__main__":
    unittest.main()
