"""Tests for final-answer validation."""
from __future__ import annotations

import unittest

from core.output_validator import validate_answer


class OutputValidatorTests(unittest.TestCase):
    def test_flags_non_nepali_body(self) -> None:
        answer = "I can help you with that right now."
        issues = validate_answer(answer, tool_was_used=False)
        self.assertTrue(
            any("मुख्य जवाफ देवनागरी" in issue for issue in issues),
            msg=f"Expected Nepali-body issue, got: {issues}",
        )

    def test_allows_ascii_digits_by_default(self) -> None:
        answer = "हाल मुद्रास्फीति 5% वरिपरि छ।"
        issues = validate_answer(answer, tool_was_used=False)
        self.assertFalse(
            any("ASCII अङ्क" in issue for issue in issues),
            msg=f"Did not expect ASCII-digit issue in default mode: {issues}",
        )

    def test_flags_ascii_digits_in_strict_mode(self) -> None:
        answer = "हाल मुद्रास्फीति 5% वरिपरि छ।"
        issues = validate_answer(
            answer,
            tool_was_used=False,
            enforce_devanagari_digits=True,
        )
        self.assertTrue(
            any("ASCII अङ्क" in issue for issue in issues),
            msg=f"Expected ASCII-digit issue in strict mode, got: {issues}",
        )


if __name__ == "__main__":
    unittest.main()
