"""Regression test for the citation-plumbing fix.

The bug: core/output_validator.py rule 3 gated `स्रोत:` enforcement on
`tool_was_used`, which was itself gated on is_real_tool_content — meaning
error-marker results with real URLs produced no inline स्रोत: despite
the citation-URL list being populated. 8/8 tool turns in the live test
showed this failure.

Fix: rule 3 also fires when `citation_urls_len > 0`, independent of
tool_was_used. This test locks in that behavior.
"""
from __future__ import annotations

import unittest

from core.output_validator import validate_answer


class CitationPlumbingTests(unittest.TestCase):
    def test_flags_missing_stroot_when_urls_harvested(self) -> None:
        # Simulates the P4 (Modi) scenario: tool returned partial content
        # with an error marker, is_real_tool_content is False so
        # tool_was_used stayed False, but URLs were still extracted.
        answer = (
            "नेपाल र भारत बीच सम्बन्ध सुमधुर छ। "
            "मोदी नरेन्द्रले नेपाल भ्रमण गरेका थिए।"
        )
        issues = validate_answer(
            answer, tool_was_used=False, citation_urls_len=2,
        )
        self.assertTrue(
            any("स्रोत" in i for i in issues),
            f"expected स्रोत: rule to fire; got {issues}",
        )

    def test_no_flag_when_stroot_already_present(self) -> None:
        answer = (
            "नेपाल र भारत बीच सम्बन्ध सुमधुर छ।\n"
            "स्रोत: https://example.com"
        )
        issues = validate_answer(
            answer, tool_was_used=False, citation_urls_len=2,
        )
        self.assertFalse(
            any("स्रोत" in i for i in issues),
            f"should not flag when स्रोत: present; got {issues}",
        )

    def test_legacy_callers_preserve_old_behavior(self) -> None:
        # Callers that don't pass citation_urls_len must behave exactly
        # as before: no stale स्रोत: nag when tool_was_used=False.
        answer = "नमस्ते हजुर! म यती हुँ।"
        issues = validate_answer(answer, tool_was_used=False)
        self.assertFalse(any("स्रोत" in i for i in issues))

    def test_original_contract_still_works(self) -> None:
        # tool_was_used=True with no स्रोत: → still flagged.
        answer = "नेपालको PM बालेन्द्र शाह हुनुहुन्छ।"
        issues = validate_answer(
            answer, tool_was_used=True, citation_urls_len=0,
        )
        self.assertTrue(any("स्रोत" in i for i in issues))


if __name__ == "__main__":
    unittest.main()
