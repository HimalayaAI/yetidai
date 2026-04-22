import os
import time
from typing import Any

import httpx


class NepalOSINTClient:
    def __init__(
        self,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        public_auth_enabled: bool | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("NEPALOSINT_BASE_URL") or "https://nepalosint.com/api/v1").rstrip("/")
        self.timeout_seconds = float(timeout_seconds or os.getenv("NEPALOSINT_TIMEOUT_SECONDS") or 8)
        auth_flag = os.getenv("NEPALOSINT_PUBLIC_AUTH_ENABLED", "true").strip().lower()
        self.public_auth_enabled = public_auth_enabled if public_auth_enabled is not None else auth_flag in {"1", "true", "yes", "on"}
        self.max_context_items = int(os.getenv("NEPALOSINT_MAX_CONTEXT_ITEMS", "8"))
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout_seconds, follow_redirects=True)
        self._access_token: str | None = None
        self._token_expiry_epoch = 0.0

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _bootstrap_public_token(self) -> str | None:
        if not self.public_auth_enabled:
            return None

        now = time.time()
        if self._access_token and now < self._token_expiry_epoch - 60:
            return self._access_token

        response = await self._client.post("/auth/public")
        response.raise_for_status()
        payload = response.json()

        token = payload.get("access_token")
        expires_in = int(payload.get("expires_in") or 0)
        if not token:
            return None

        self._access_token = token
        self._token_expiry_epoch = now + expires_in if expires_in else now + 3600
        return self._access_token

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        auth_preferred: bool = False,
        retry_on_auth: bool = True,
    ) -> Any:
        headers: dict[str, str] = {}

        if auth_preferred:
            token = await self._bootstrap_public_token()
            if token:
                headers["Authorization"] = f"Bearer {token}"

        response = await self._client.request(method, path, params=params, json=json_body, headers=headers)
        if response.status_code == 401 and retry_on_auth and self.public_auth_enabled:
            token = await self._bootstrap_public_token()
            if token:
                response = await self._client.request(
                    method,
                    path,
                    params=params,
                    json=json_body,
                    headers={"Authorization": f"Bearer {token}"},
                )

        response.raise_for_status()
        return response.json()

    async def get_economy_snapshot(self) -> Any:
        return await self._request("GET", "/economy/nrb-snapshot")

    async def get_dashboard_bootstrap(self, preset: str) -> Any:
        return await self._request("GET", "/dashboard/bootstrap", params={"preset": preset})

    async def get_govt_decisions_latest(self, limit: int = 10, dedupe: bool = True) -> Any:
        return await self._request(
            "GET",
            "/govt-decisions/latest",
            params={"limit": limit, "dedupe": str(dedupe).lower()},
        )

    async def get_announcements_summary(self, limit: int = 10) -> Any:
        return await self._request("GET", "/announcements/summary", params={"limit": limit}, auth_preferred=True)

    async def get_debt_clock(self) -> Any:
        return await self._request("GET", "/debt-clock/nepal", auth_preferred=True)

    async def get_verbatim_summary(self) -> Any:
        return await self._request("GET", "/verbatim/summary")

    async def get_parliament_bills(self, limit: int = 6) -> Any:
        return await self._request("GET", "/parliament/bills", params={"limit": limit}, auth_preferred=True)

    async def search_unified(self, query: str, limit: int = 8) -> Any:
        return await self._request(
            "GET",
            "/search/unified",
            params={"q": query, "limit": limit},
            auth_preferred=True,
        )

    async def search_embeddings(
        self,
        query: str,
        *,
        hours: int = 720,
        top_k: int = 8,
        min_similarity: float = 0.45,
    ) -> Any:
        payload = {
            "query": query,
            "hours": hours,
            "top_k": top_k,
            "min_similarity": min_similarity,
        }
        return await self._request(
            "POST",
            "/embeddings/search",
            json_body=payload,
            auth_preferred=True,
        )

    async def get_consolidated_recent(
        self,
        *,
        hours: int = 24,
        limit: int = 12,
        category: str | None = None,
    ) -> Any:
        params: dict[str, Any] = {"hours": hours, "limit": limit}
        if category:
            params["category"] = category
        return await self._request("GET", "/analytics/consolidated-stories", params=params)

    async def get_consolidated_history(
        self,
        *,
        start_date: str,
        end_date: str,
        limit: int = 8,
        offset: int = 0,
        category: str | None = None,
        source: str | None = None,
    ) -> Any:
        params: dict[str, Any] = {
            "start_date": start_date,
            "end_date": end_date,
            "limit": limit,
            "offset": offset,
        }
        if category:
            params["category"] = category
        if source:
            params["source"] = source
        return await self._request(
            "GET",
            "/analytics/consolidated-stories/history",
            params=params,
            auth_preferred=True,
        )
