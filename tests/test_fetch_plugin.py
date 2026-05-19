from __future__ import annotations

import asyncio

import tools.fetch.plugin as fetch_plugin
from tools.fetch.plugin import handle_fetch


class _FakeStreamResponse:
    def __init__(self, content_type: str):
        self.headers = {"content-type": content_type}
        self.read_attempted = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    def raise_for_status(self) -> None:
        return None

    async def aread(self) -> bytes:
        self.read_attempted = True
        return b""


class _FakeClient:
    def __init__(self, response: _FakeStreamResponse):
        self.response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    def stream(self, *args, **kwargs):
        return self.response


def test_fetch_url_rejects_video_without_reading_body(monkeypatch):
    response = _FakeStreamResponse("video/mp4")
    monkeypatch.setattr(
        fetch_plugin.httpx,
        "AsyncClient",
        lambda *args, **kwargs: _FakeClient(response),
    )

    result = asyncio.run(
        handle_fetch(
            ctx=None,
            arguments={
                "url": "https://video.twimg.com/ext_tw_video/123/vid/avc1/test.mp4",
            },
        )
    )

    assert not response.read_attempted
    assert not result.success
    assert result.error == "unsupported content-type: video/mp4"
