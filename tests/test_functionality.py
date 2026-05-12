"""Tests for the message dispatch helper."""
from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from functionality import functional


class FakeChannel:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, content: str) -> None:
        self.sent.append(content)


class FakeBotUser:
    def __init__(self, user_id: int = 123) -> None:
        self.id = user_id

    def mentioned_in(self, message) -> bool:
        mention = f"<@{self.id}>"
        mention_nick = f"<@!{self.id}>"
        return mention in (message.content or "") or mention_nick in (message.content or "")


class FakeBot:
    def __init__(self) -> None:
        self.user = FakeBotUser()


class FunctionalCallTests(unittest.TestCase):
    def test_plain_english_message_sets_user_input(self) -> None:
        bot = FakeBot()
        helper = functional(bot)
        message = SimpleNamespace(
            author=SimpleNamespace(bot=False),
            content="Hello, are you there?",
            mentions=[],
            channel=FakeChannel(),
        )

        asyncio.run(helper.call(message))

        self.assertEqual(helper.user_input, "Hello, are you there?")

    def test_bot_messages_are_ignored(self) -> None:
        bot = FakeBot()
        helper = functional(bot)
        message = SimpleNamespace(
            author=bot.user,
            content="Hello from the bot",
            mentions=[],
            channel=FakeChannel(),
        )

        asyncio.run(helper.call(message))

        self.assertIsNone(helper.user_input)


if __name__ == "__main__":
    unittest.main()
