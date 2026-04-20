import asyncio
import unittest
from types import SimpleNamespace

from context_router import RoutePlan, route_query
from retrieval_planner import _extract_json_blob, _plan_from_payload, resolve_route_plan


class RetrievalPlannerTests(unittest.TestCase):
    def test_extract_json_blob_from_wrapped_text(self) -> None:
        payload = _extract_json_blob(
            "planner result:\n{\"use_nepalosint\": true, \"intents\": [\"macro\"]}\nthanks"
        )
        self.assertEqual(payload, {"use_nepalosint": True, "intents": ["macro"]})

    def test_plan_from_payload_filters_unknown_intents(self) -> None:
        fallback = RoutePlan(
            use_nepalosint=True,
            intents=["general_news"],
            wants_history=False,
            history_start_date=None,
            history_end_date=None,
            history_category=None,
        )
        plan = _plan_from_payload(
            {
                "intents": ["macro", "unknown", "macro"],
                "wants_history": True,
                "history_start_date": "2026-04-01",
                "history_end_date": "2026-04-03",
                "history_category": "economic",
            },
            fallback,
        )

        self.assertEqual(plan.intents, ["macro"])
        self.assertTrue(plan.use_nepalosint)
        self.assertTrue(plan.wants_history)
        self.assertEqual(plan.history_start_date, "2026-04-01")
        self.assertEqual(plan.history_end_date, "2026-04-03")
        self.assertEqual(plan.history_category, "economic")

    def test_resolve_route_plan_falls_back_when_llm_errors(self) -> None:
        class BrokenChat:
            async def completions(self, **_: object):
                raise RuntimeError("planner unavailable")

        llm_client = SimpleNamespace(chat=BrokenChat())
        query = "what about this?"
        resolved = asyncio.run(resolve_route_plan(llm_client, query, previous_messages=[]))

        self.assertEqual(resolved, route_query(query))

    def test_resolve_route_plan_uses_llm_json_for_ambiguous_query(self) -> None:
        planner_json = (
            "{\"use_nepalosint\": true, \"intents\": [\"debt\", \"general_news\"], "
            "\"wants_history\": false, \"history_start_date\": null, "
            "\"history_end_date\": null, \"history_category\": null}"
        )

        class FakeChat:
            async def completions(self, **_: object):
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=planner_json))]
                )

        llm_client = SimpleNamespace(chat=FakeChat())
        resolved = asyncio.run(resolve_route_plan(llm_client, "what about this?", previous_messages=[]))

        self.assertEqual(resolved.intents, ["debt", "general_news"])
        self.assertTrue(resolved.use_nepalosint)
        self.assertFalse(resolved.wants_history)


if __name__ == "__main__":
    unittest.main()
