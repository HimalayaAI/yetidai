import asyncio
import unittest

from tools.osint.context_router import RoutePlan, fetch_context_bundle, route_query


class ContextRouterTests(unittest.TestCase):
    def test_explicit_iso_date_range_triggers_history(self) -> None:
        plan = route_query("Give me updates from 2026-04-01 to 2026-04-05")

        self.assertTrue(plan.wants_history)
        self.assertEqual(plan.history_start_date, "2026-04-01")
        self.assertEqual(plan.history_end_date, "2026-04-05")
        self.assertIn("general_news", plan.intents)

    def test_macro_fetch_does_not_request_dashboard_bootstrap(self) -> None:
        class DummyClient:
            def __init__(self) -> None:
                self.max_context_items = 8

            async def get_economy_snapshot(self):
                return {"sections": {}}

            async def get_dashboard_bootstrap(self, preset: str):
                raise AssertionError("get_dashboard_bootstrap should not be called")

        bundle = asyncio.run(
            fetch_context_bundle(
                DummyClient(),
                "inflation update",
                RoutePlan(use_nepalosint=True, intents=["macro"]),
            )
        )

        self.assertIn("economy_snapshot", bundle["payloads"])
        self.assertNotIn("economy_bootstrap", bundle["payloads"])
        self.assertNotIn("economy_bootstrap", bundle["errors"])


if __name__ == "__main__":
    unittest.main()
