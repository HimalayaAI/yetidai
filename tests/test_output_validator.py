"""Tests for final-answer validation."""
from __future__ import annotations

import unittest

from core.output_validator import validate_answer


class OutputValidatorTests(unittest.TestCase):
    def test_flags_obvious_hindi_prose(self) -> None:
        answer = "मैं आपकी मदद कर सकता हूँ। यह अभी उपलब्ध है।"

        issues = validate_answer(answer, tool_was_used=False)

        self.assertTrue(
            any("हिन्दी" in issue for issue in issues),
            msg=f"Expected Hindi-drift issue, got: {issues}",
        )


if __name__ == "__main__":
    unittest.main()
