import unittest

from tools.osint.context_formatter import build_context_brief
from tools.osint.context_router import RoutePlan


class ContextFormatterTests(unittest.TestCase):
    def test_returns_none_when_no_payloads_or_errors(self) -> None:
        brief = build_context_brief({"query": "hello", "payloads": {}, "errors": {}, "plan": None})
        self.assertIsNone(brief)

    def test_returns_fallback_when_payload_fetch_failed(self) -> None:
        brief = build_context_brief(
            {
                "query": "नेपालमा आज के भइरहेको छ?",
                "payloads": {},
                "errors": {"general_search": "ConnectError: timeout"},
                "plan": RoutePlan(use_nepalosint=True, intents=["general_news"]),
            }
        )
        self.assertIsNotNone(brief)
        # When everything fails, the formatter returns the machine-readable
        # marker; bot.py detects it and auto-falls-back to internet_search.
        self.assertIn("[NEPALOSINT_FETCH_FAILED]", brief)

    def test_builds_history_and_macro_blocks_with_footer(self) -> None:
        brief = build_context_brief(
            {
                "query": "2026-04-01 देखि 2026-04-05 सम्मको मुद्रास्फीति",
                "plan": RoutePlan(
                    use_nepalosint=True,
                    intents=["macro", "general_news"],
                    wants_history=True,
                    history_start_date="2026-04-01",
                    history_end_date="2026-04-05",
                    history_category="economic",
                ),
                "payloads": {
                    "history": {
                        "items": [
                            {"canonical_headline": "Headline A", "source_name": "Source A"},
                        ]
                    },
                    "economy_snapshot": {
                        "as_of_label": "Falgun 2082",
                        "sections": {
                            "prices": {
                                "metrics": [
                                    {
                                        "label": "Inflation",
                                        "display_value": "5.20%",
                                        "change_pct": 0.4,
                                    }
                                ]
                            }
                        },
                    },
                },
                "errors": {},
            }
        )

        self.assertIsNotNone(brief)
        self.assertIn("Historical news", brief)
        self.assertIn("Macro snapshot", brief)
        # Footer is a single SOURCES: line (Nepali footer and "उत्तर दिने
        # नियम:" were removed in the Claude-native rewrite).
        self.assertIn("SOURCES:", brief)


if __name__ == "__main__":
    unittest.main()
