import asyncio
import unittest
from typing import Any

from nepalosint_client import NepalOSINTClient


class FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeAsyncClient:
    def __init__(self, request_responses: list[FakeResponse]) -> None:
        self._request_responses = list(request_responses)
        self.post_calls: list[str] = []
        self.request_calls: list[dict[str, Any]] = []

    async def post(self, path: str) -> FakeResponse:
        self.post_calls.append(path)
        return FakeResponse(200, {"access_token": "token-1", "expires_in": 3600})

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> FakeResponse:
        self.request_calls.append(
            {
                "method": method,
                "path": path,
                "params": params,
                "json": json,
                "headers": headers or {},
            }
        )
        return self._request_responses.pop(0)

    async def aclose(self) -> None:
        return None


class NepalOSINTClientTests(unittest.TestCase):
    def _new_client(self, *, public_auth_enabled: bool = True) -> NepalOSINTClient:
        client = NepalOSINTClient(public_auth_enabled=public_auth_enabled)
        asyncio.run(client._client.aclose())
        return client

    def test_request_retries_with_cached_token_after_401(self) -> None:
        client = self._new_client(public_auth_enabled=True)
        fake_http = FakeAsyncClient(
            [
                FakeResponse(401, {"detail": "Not authenticated"}),
                FakeResponse(200, {"ok": True}),
            ]
        )
        client._client = fake_http

        payload = asyncio.run(
            client._request("GET", "/search/unified", params={"q": "nepal"}, auth_preferred=True)
        )

        self.assertEqual(payload, {"ok": True})
        self.assertEqual(fake_http.post_calls, ["/auth/public"])
        self.assertEqual(len(fake_http.request_calls), 2)
        self.assertEqual(
            fake_http.request_calls[0]["headers"].get("Authorization"),
            "Bearer token-1",
        )
        self.assertEqual(
            fake_http.request_calls[1]["headers"].get("Authorization"),
            "Bearer token-1",
        )

    def test_bootstrap_returns_none_when_public_auth_disabled(self) -> None:
        client = self._new_client(public_auth_enabled=False)
        token = asyncio.run(client._bootstrap_public_token())
        self.assertIsNone(token)

    def test_history_endpoint_uses_expected_request_contract(self) -> None:
        client = self._new_client(public_auth_enabled=True)
        captured: dict[str, Any] = {}

        async def fake_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
            captured["method"] = method
            captured["path"] = path
            captured["kwargs"] = kwargs
            return {"ok": True}

        client._request = fake_request  # type: ignore[method-assign]

        result = asyncio.run(
            client.get_consolidated_history(
                start_date="2026-04-01",
                end_date="2026-04-05",
                limit=5,
                offset=10,
                category="economic",
                source="merolagani",
            )
        )

        self.assertEqual(result, {"ok": True})
        self.assertEqual(captured["method"], "GET")
        self.assertEqual(captured["path"], "/analytics/consolidated-stories/history")
        self.assertEqual(
            captured["kwargs"]["params"],
            {
                "start_date": "2026-04-01",
                "end_date": "2026-04-05",
                "limit": 5,
                "offset": 10,
                "category": "economic",
                "source": "merolagani",
            },
        )
        self.assertTrue(captured["kwargs"]["auth_preferred"])


if __name__ == "__main__":
    unittest.main()
