from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from tools.social.plugin import handle_social_feed
from tools.social.twitter_feed import (
    DEFAULT_AI_ACCOUNTS,
    FeedPollResult,
    NitterScraper,
    SocialFeedConfig,
    SocialFeedService,
    SocialFeedStore,
    SocialTweet,
)
from core.preflight import plan_preflight


def test_nitter_parser_extracts_images_and_video_sources():
    html = """
    <div class="timeline-item">
      <div class="tweet-body">
        <a class="username" href="/karpathy">@karpathy</a>
        <a class="fullname" href="/karpathy">Andrej Karpathy</a>
        <span class="tweet-date"><a href="/karpathy/status/123#m" title="May 18, 2026 · 1:00 PM UTC">now</a></span>
        <div class="tweet-content">image and video post</div>
        <span class="tweet-stat"><div class="icon-container">1</div></span>
        <span class="tweet-stat"><div class="icon-container">2</div></span>
        <span class="tweet-stat"><div class="icon-container">3</div></span>
        <span class="tweet-stat"><div class="icon-container">4</div></span>
        <div class="attachments">
          <a class="still-image" href="/pic/orig/media%2Fimage.jpg"><img src="/pic/media%2Fimage.jpg"/></a>
          <video poster="/pic/thumb.jpg">
            <source src="https://video.twimg.com/ext_tw_video/123/vid/avc1/test.mp4" type="video/mp4"/>
            <source src="https://video.twimg.com/ext_tw_video/123/vid/avc1/test.mp4" type="video/mp4"/>
          </video>
        </div>
      </div>
    </div>
    """
    scraper = NitterScraper(SocialFeedConfig())

    tweets = scraper._parse_tweets(html, "https://nitter.example")

    assert len(tweets) == 1
    tweet = tweets[0]
    assert tweet.tweet_id == "123"
    assert tweet.author_username == "karpathy"
    assert tweet.media_urls == ["https://nitter.example/pic/orig/media%2Fimage.jpg"]
    assert tweet.video_urls == ["https://video.twimg.com/ext_tw_video/123/vid/avc1/test.mp4"]
    assert tweet.video_thumb_urls == ["https://nitter.example/pic/thumb.jpg"]
    assert tweet.reply_count == 1
    assert tweet.like_count == 4


def test_social_feed_defaults_are_small_prioritized_and_non_spammy():
    assert DEFAULT_AI_ACCOUNTS == [
        "OpenAI",
        "AnthropicAI",
        "GoogleDeepMind",
        "xai",
        "MistralAI",
        "huggingface",
        "sama",
        "karpathy",
        "simonw",
        "goodside",
        "nlethetech",
        "HimalayaAILabs",
    ]

    config = SocialFeedConfig()

    assert config.accounts == DEFAULT_AI_ACCOUNTS
    assert config.poll_seconds == 180
    assert config.max_posts_per_poll == 10
    assert config.nitter_instances == ["https://nitter.poast.org"]
    assert not config.post_existing_on_first_run
    assert not config.include_retweets
    assert not config.include_replies


def test_social_feed_store_dedupes_and_returns_recent(tmp_path):
    store = SocialFeedStore(tmp_path / "social.sqlite3")
    tweet = SocialTweet(
        tweet_id="1",
        author_username="sama",
        author_name="Sam Altman",
        text="hello",
        tweeted_at=datetime(2026, 5, 18, 13, 0, tzinfo=timezone.utc),
        media_urls=["https://example.com/a.jpg"],
    )

    assert not store.is_seen("1")
    store.mark_seen(tweet, "@sama", posted=False)
    store.mark_seen(tweet, "@sama", posted=True)

    assert store.is_seen("1")
    rows = store.recent(limit=5)
    assert len(rows) == 1
    assert rows[0]["tweet_id"] == "1"
    assert rows[0]["posted_at"] is not None
    assert store.recent(limit=5, author="karpathy") == []
    store.close()


def test_social_feed_first_poll_marks_existing_without_posting(tmp_path):
    class FakeScraper:
        def __init__(self, config):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def scrape_user_timeline(self, account, *, is_known_tweet=None):
            return type("Result", (), {
                "error": None,
                "tweets": [
                    SocialTweet(
                        tweet_id="2",
                        author_username=account,
                        author_name=account,
                        text="existing",
                        tweeted_at=datetime.now(timezone.utc),
                    ),
                ],
            })()

    from tools.social import twitter_feed

    original = twitter_feed.NitterScraper
    twitter_feed.NitterScraper = FakeScraper
    try:
        config = SocialFeedConfig(
            state_db_path=tmp_path / "social.sqlite3",
            accounts=["karpathy"],
            post_existing_on_first_run=False,
            delay_between_requests=0,
        )
        service = SocialFeedService(config=config)
        result = asyncio.run(service.poll_once())
        assert result.tweets_to_post == []
        assert result.marked_seen == 1
        assert service.store.is_seen("2")
    finally:
        twitter_feed.NitterScraper = original


def test_social_feed_live_refresh_marks_new_items_and_skips_repeats(tmp_path):
    class FakeScraper:
        def __init__(self, config):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def scrape_user_timeline(self, account, *, is_known_tweet=None):
            tweet = SocialTweet(
                tweet_id=f"{account}-new",
                author_username=account,
                author_name=account,
                text=f"fresh from {account}",
                tweeted_at=datetime.now(timezone.utc),
            )
            tweets = [] if is_known_tweet and is_known_tweet(tweet.tweet_id) else [tweet]
            return type("Result", (), {"error": None, "tweets": tweets})()

    from tools.social import twitter_feed

    original = twitter_feed.NitterScraper
    twitter_feed.NitterScraper = FakeScraper
    try:
        config = SocialFeedConfig(
            state_db_path=tmp_path / "social.sqlite3",
            accounts=["karpathy", "sama"],
            live_refresh_accounts=2,
            live_refresh_concurrency=2,
            delay_between_requests=0,
        )
        service = SocialFeedService(config=config)

        first = asyncio.run(service.refresh_recent())
        second = asyncio.run(service.refresh_recent())

        assert [tweet.tweet_id for tweet in first.tweets_to_post] == [
            "karpathy-new",
            "sama-new",
        ]
        assert first.marked_seen == 2
        assert second.tweets_to_post == []
        assert service.store.is_seen("karpathy-new")
        assert service.store.is_seen("sama-new")
    finally:
        twitter_feed.NitterScraper = original


def test_social_feed_poll_posts_items_seen_by_live_refresh(tmp_path):
    class FakeScraper:
        def __init__(self, config):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def scrape_user_timeline(self, account, *, is_known_tweet=None):
            tweet = SocialTweet(
                tweet_id="refresh-new",
                author_username=account,
                author_name=account,
                text="fresh item should still auto-post",
                tweeted_at=datetime.now(timezone.utc),
            )
            tweets = [] if is_known_tweet and is_known_tweet(tweet.tweet_id) else [tweet]
            return type("Result", (), {"error": None, "tweets": tweets})()

    from tools.social import twitter_feed

    original = twitter_feed.NitterScraper
    twitter_feed.NitterScraper = FakeScraper
    try:
        config = SocialFeedConfig(
            state_db_path=tmp_path / "social.sqlite3",
            accounts=["karpathy"],
            delay_between_requests=0,
        )
        service = SocialFeedService(config=config)

        refresh = asyncio.run(service.refresh_recent(author="karpathy"))
        poll = asyncio.run(service.poll_once())

        assert [tweet.tweet_id for tweet in refresh.tweets_to_post] == ["refresh-new"]
        assert [tweet.tweet_id for tweet in poll.tweets_to_post] == ["refresh-new"]
        assert not service.store.is_known_for_autopost("refresh-new")
        service.mark_posted(poll.tweets_to_post[0])

        after_post = asyncio.run(service.poll_once())

        assert after_post.tweets_to_post == []
        assert service.store.is_known_for_autopost("refresh-new")
    finally:
        twitter_feed.NitterScraper = original


def test_social_feed_tool_returns_stored_items(tmp_path, monkeypatch):
    store = SocialFeedStore(tmp_path / "social.sqlite3")
    tweet = SocialTweet(
        tweet_id="3",
        author_username="karpathy",
        author_name="Andrej Karpathy",
        text="new model notes",
        tweeted_at=datetime(2026, 5, 18, 13, 5, tzinfo=timezone.utc),
        video_urls=["https://video.twimg.com/test.mp4"],
    )
    store.mark_seen(tweet, "@karpathy", posted=True)
    service = SocialFeedService(
        config=SocialFeedConfig(state_db_path=tmp_path / "social.sqlite3"),
        store=store,
    )

    monkeypatch.setattr("tools.social.plugin.get_social_feed_service", lambda: service)

    result = asyncio.run(handle_social_feed(ctx=None, arguments={"limit": 5}))

    assert result.success
    assert "@karpathy" in result.content
    assert "[video: 1]" in result.content
    assert "https://x.com/karpathy/status/3" in result.content
    store.close()


def test_social_feed_tool_refreshes_before_returning_rows(tmp_path, monkeypatch):
    class FakeService:
        async def refresh_recent(self, *, author=None):
            return FeedPollResult(
                tweets_to_post=[
                    SocialTweet(
                        tweet_id="4",
                        author_username="sama",
                        author_name="Sam Altman",
                        text="fresh model note",
                    )
                ],
                scanned=1,
                marked_seen=1,
                errors=[],
            )

        def recent(self, *, limit=10, author=None):
            return [
                {
                    "tweet_id": "4",
                    "author_username": "sama",
                    "author_name": "Sam Altman",
                    "text": "fresh model note",
                    "x_url": "https://x.com/sama/status/4",
                    "tweeted_at": "2026-05-18T13:05:00+00:00",
                    "first_seen_at": "2026-05-18T13:06:00+00:00",
                    "media_urls_json": "[]",
                    "video_urls_json": "[]",
                }
            ]

    monkeypatch.setattr("tools.social.plugin.get_social_feed_service", lambda: FakeService())

    result = asyncio.run(handle_social_feed(ctx=None, arguments={"limit": 5, "refresh": True}))

    assert result.success
    assert "Live #social-media refresh found 1 new item" in result.content
    assert "@sama" in result.content
    assert result.meta["new_items"] == 1


def test_social_feed_preflight_routes_social_questions():
    plan = plan_preflight("what are the AI handles saying in social media?")

    assert plan == ("get_social_media_feed", {"limit": 10, "refresh": True})


def test_automatic_social_poster_does_not_call_llm():
    source = Path("bot.py").read_text()
    start = source.index("async def _run_social_feed")
    end = source.index("async def _send_discord", start)
    social_poster_source = source[start:end]

    assert "_run_llm_turn" not in social_poster_source
    assert "llm_client" not in social_poster_source
    assert "chat.completions" not in social_poster_source
    assert "channel.send" in social_poster_source
    assert "poll_once" in social_poster_source
