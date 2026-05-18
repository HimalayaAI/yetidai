"""Nitter-backed social feed for YetiDai.

This module is deliberately self-contained: YetiDai does not need a main
database to run the social-media feed. A tiny SQLite state file tracks seen
tweet IDs and recent tweet details so the background poster and the LLM tool
share the same source of truth.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import aiohttp
from bs4 import BeautifulSoup

logger = logging.getLogger("yetidai.social")

DEFAULT_AI_ACCOUNTS = [
    "karpathy",
    "fchollet",
    "ylecun",
    "AndrewYNg",
    "rasbt",
    "dair_ai",
    "lilianweng",
    "jeremyphoward",
    "simonw",
    "_akhaliq",
    "ID_AA_Carmack",
    "gwern",
    "goodside",
    "drfeifei",
    "demishassabis",
    "sama",
    "nlethetech",
    "HimalayaAILabs",
]


@dataclass(frozen=True)
class SocialFeedConfig:
    enabled: bool = True
    channel_id: int | None = None
    channel_name: str = "social-media"
    state_db_path: Path = Path("logs/social_feed.sqlite3")
    accounts: list[str] = field(default_factory=lambda: list(DEFAULT_AI_ACCOUNTS))
    nitter_instances: list[str] = field(default_factory=lambda: [
        "https://nitter.poast.org",
        "https://nitter.privacydev.net",
    ])
    poll_seconds: int = 300
    request_timeout_seconds: int = 30
    delay_between_requests: float = 2.0
    max_timeline_pages: int = 1
    max_posts_per_poll: int = 25
    post_existing_on_first_run: bool = False
    include_retweets: bool = False
    include_replies: bool = False


@dataclass
class SocialTweet:
    tweet_id: str
    author_username: str
    author_name: str
    text: str
    tweeted_at: datetime | None = None
    is_retweet: bool = False
    is_reply: bool = False
    reply_count: int = 0
    retweet_count: int = 0
    quote_count: int = 0
    like_count: int = 0
    media_urls: list[str] = field(default_factory=list)
    video_urls: list[str] = field(default_factory=list)
    video_thumb_urls: list[str] = field(default_factory=list)
    instance_url: str = ""

    @property
    def x_url(self) -> str:
        return f"https://x.com/{self.author_username}/status/{self.tweet_id}"


@dataclass
class ScrapeResult:
    success: bool
    tweets: list[SocialTweet] = field(default_factory=list)
    error: str | None = None
    instance_used: str = ""


@dataclass
class FeedPollResult:
    tweets_to_post: list[SocialTweet]
    scanned: int
    marked_seen: int
    errors: list[str]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid integer for %s=%r; using %s", name, raw, default)
        return default


def _csv_env(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if not raw:
        return list(default)
    values = [item.strip().lstrip("@") for item in raw.split(",") if item.strip()]
    return values or list(default)


def load_social_feed_config() -> SocialFeedConfig:
    channel_id_raw = os.getenv("YETI_SOCIAL_CHANNEL_ID", "").strip()
    channel_id = int(channel_id_raw) if channel_id_raw.isdigit() else None
    return SocialFeedConfig(
        enabled=_env_bool("YETI_SOCIAL_FEED_ENABLED", True),
        channel_id=channel_id,
        channel_name=os.getenv("YETI_SOCIAL_CHANNEL_NAME", "social-media"),
        state_db_path=Path(os.getenv("YETI_SOCIAL_STATE_DB", "logs/social_feed.sqlite3")),
        accounts=_csv_env("YETI_SOCIAL_ACCOUNTS", DEFAULT_AI_ACCOUNTS),
        nitter_instances=_csv_env(
            "YETI_SOCIAL_NITTER_INSTANCES",
            ["https://nitter.poast.org", "https://nitter.privacydev.net"],
        ),
        poll_seconds=_env_int("YETI_SOCIAL_POLL_SECONDS", 300),
        request_timeout_seconds=_env_int("YETI_SOCIAL_REQUEST_TIMEOUT_SECONDS", 30),
        max_timeline_pages=_env_int("YETI_SOCIAL_MAX_TIMELINE_PAGES", 1),
        max_posts_per_poll=_env_int("YETI_SOCIAL_MAX_POSTS_PER_POLL", 25),
        post_existing_on_first_run=_env_bool("YETI_SOCIAL_POST_EXISTING_ON_FIRST_RUN", False),
        include_retweets=_env_bool("YETI_SOCIAL_INCLUDE_RETWEETS", False),
        include_replies=_env_bool("YETI_SOCIAL_INCLUDE_REPLIES", False),
    )


class SocialFeedStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS social_tweets (
                tweet_id TEXT PRIMARY KEY,
                author_username TEXT NOT NULL,
                author_name TEXT,
                text TEXT NOT NULL,
                x_url TEXT NOT NULL,
                tweeted_at TEXT,
                source_label TEXT,
                media_urls_json TEXT NOT NULL DEFAULT '[]',
                video_urls_json TEXT NOT NULL DEFAULT '[]',
                video_thumb_urls_json TEXT NOT NULL DEFAULT '[]',
                first_seen_at TEXT NOT NULL,
                posted_at TEXT
            )
            """
        )
        self._conn.commit()

    def is_seen(self, tweet_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM social_tweets WHERE tweet_id = ?",
            (tweet_id,),
        ).fetchone()
        return row is not None

    def mark_seen(self, tweet: SocialTweet, source_label: str, *, posted: bool) -> None:
        now = datetime.now(timezone.utc).isoformat()
        posted_at = now if posted else None
        tweeted_at = tweet.tweeted_at.isoformat() if tweet.tweeted_at else None
        self._conn.execute(
            """
            INSERT INTO social_tweets (
                tweet_id, author_username, author_name, text, x_url, tweeted_at,
                source_label, media_urls_json, video_urls_json,
                video_thumb_urls_json, first_seen_at, posted_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(tweet_id) DO UPDATE SET
                author_username = excluded.author_username,
                author_name = excluded.author_name,
                text = excluded.text,
                x_url = excluded.x_url,
                tweeted_at = COALESCE(excluded.tweeted_at, social_tweets.tweeted_at),
                source_label = COALESCE(excluded.source_label, social_tweets.source_label),
                media_urls_json = excluded.media_urls_json,
                video_urls_json = excluded.video_urls_json,
                video_thumb_urls_json = excluded.video_thumb_urls_json,
                posted_at = COALESCE(social_tweets.posted_at, excluded.posted_at)
            """,
            (
                tweet.tweet_id,
                tweet.author_username,
                tweet.author_name,
                tweet.text,
                tweet.x_url,
                tweeted_at,
                source_label,
                json.dumps(tweet.media_urls),
                json.dumps(tweet.video_urls),
                json.dumps(tweet.video_thumb_urls),
                now,
                posted_at,
            ),
        )
        self._conn.commit()

    def recent(self, *, limit: int = 10, author: str | None = None) -> list[dict]:
        limit = max(1, min(int(limit), 50))
        params: list[object] = []
        where = ""
        if author:
            where = "WHERE lower(author_username) = lower(?)"
            params.append(author.lstrip("@"))
        rows = self._conn.execute(
            f"""
            SELECT * FROM social_tweets
            {where}
            ORDER BY COALESCE(tweeted_at, first_seen_at) DESC
            LIMIT ?
            """,
            (*params, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        self._conn.close()


@dataclass
class NitterInstance:
    url: str
    consecutive_failures: int = 0
    last_failure_at: datetime | None = None

    @property
    def is_healthy(self) -> bool:
        if self.consecutive_failures == 0 or self.last_failure_at is None:
            return True
        backoff_minutes = min(60, 5 * (3 ** (self.consecutive_failures - 1)))
        return datetime.now(timezone.utc) > self.last_failure_at + timedelta(minutes=backoff_minutes)

    def record_success(self) -> None:
        self.consecutive_failures = 0

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        self.last_failure_at = datetime.now(timezone.utc)


class NitterScraper:
    POW_CHALLENGE_RE = re.compile(r"'([A-Fa-f0-9]{40})'")
    TWEET_ID_RE = re.compile(r"/status/(\d+)")
    CURSOR_RE = re.compile(r'[?&]cursor=([^"&]+)')
    NOT_FOUND = "__NOT_FOUND__"

    def __init__(self, config: SocialFeedConfig):
        self.config = config
        self.instances = [NitterInstance(url.rstrip("/")) for url in config.nitter_instances]
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "NitterScraper":
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.request_timeout_seconds),
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            },
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    @staticmethod
    def _solve_pow(challenge: str) -> str:
        n1 = int(challenge[0], 16)
        challenge_bytes = challenge.encode("ascii")
        for i in range(10_000_000):
            digest = hashlib.sha1(challenge_bytes + str(i).encode("ascii")).digest()
            if digest[n1] == 0xB0 and digest[n1 + 1] == 0x0B:
                return str(i)
        raise RuntimeError("failed to solve Nitter proof-of-work challenge")

    async def _fetch(self, instance: NitterInstance, path: str) -> str | None:
        if not self._session:
            raise RuntimeError("NitterScraper must be used as an async context manager")
        url = f"{instance.url}{path}"
        try:
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    instance.record_success()
                    return await resp.text()
                if resp.status == 404:
                    instance.record_success()
                    return self.NOT_FOUND
                if resp.status == 429:
                    logger.warning("Nitter rate limited by %s", instance.url)
                    await asyncio.sleep(30)
                    instance.record_failure()
                    return None
                if resp.status == 502:
                    logger.warning("Temporary Nitter 502 from %s", url)
                    await asyncio.sleep(5)
                    return None
                if resp.status not in (403, 503):
                    logger.warning("Unexpected Nitter status %s from %s", resp.status, url)
                    instance.record_failure()
                    return None
                html = await resp.text()
                match = self.POW_CHALLENGE_RE.search(html)
                if not match:
                    instance.record_failure()
                    return None
                solution = self._solve_pow(match.group(1))
                self._session.cookie_jar.update_cookies(
                    {"res": f"{match.group(1)}{solution}"},
                    response_url=resp.url,
                )
            async with self._session.get(url) as retry_resp:
                if retry_resp.status == 200:
                    instance.record_success()
                    return await retry_resp.text()
                instance.record_failure()
                return None
        except (asyncio.TimeoutError, aiohttp.ClientError) as exc:
            logger.warning("Nitter fetch failed for %s: %s", url, exc)
            instance.record_failure()
            return None

    async def _fetch_with_failover(self, path: str) -> tuple[str | None, str]:
        instances = [item for item in self.instances if item.is_healthy] or sorted(
            self.instances, key=lambda item: item.consecutive_failures,
        )[:1]
        for instance in instances:
            html = await self._fetch(instance, path)
            if html == self.NOT_FOUND:
                return self.NOT_FOUND, instance.url
            if html:
                return html, instance.url
            await asyncio.sleep(0.5)
        return None, ""

    async def scrape_user_timeline(
        self,
        username: str,
        *,
        is_known_tweet: Callable[[str], bool] | None = None,
    ) -> ScrapeResult:
        tweets: list[SocialTweet] = []
        seen_ids: set[str] = set()
        path = f"/{username}"
        instance_used = ""

        for page in range(max(1, self.config.max_timeline_pages)):
            html, instance_used = await self._fetch_with_failover(path)
            if html == self.NOT_FOUND:
                return ScrapeResult(success=True, error=f"@{username} not found")
            if not html:
                if page == 0:
                    return ScrapeResult(success=False, error=f"all Nitter instances failed for @{username}")
                break

            hit_known = False
            for tweet in self._parse_tweets(html, instance_used):
                if tweet.tweet_id in seen_ids:
                    continue
                seen_ids.add(tweet.tweet_id)
                if is_known_tweet and is_known_tweet(tweet.tweet_id):
                    hit_known = True
                    continue
                tweets.append(tweet)
            if hit_known:
                break

            cursor = self._extract_cursor(html)
            if not cursor:
                break
            path = f"/{username}?cursor={cursor}"
            await asyncio.sleep(self.config.delay_between_requests)

        return ScrapeResult(success=True, tweets=tweets, instance_used=instance_used)

    def _parse_tweets(self, html: str, instance_url: str) -> list[SocialTweet]:
        soup = BeautifulSoup(html, "html.parser")
        parsed: list[SocialTweet] = []
        for item in soup.select("div.timeline-item"):
            tweet = self._parse_single_tweet(item, instance_url)
            if tweet:
                parsed.append(tweet)
        return parsed

    def _parse_single_tweet(self, item, instance_url: str) -> SocialTweet | None:
        body = item.select_one("div.tweet-body")
        if not body:
            return None
        username_el = body.select_one("a.username")
        fullname_el = body.select_one("a.fullname")
        date_link = body.select_one("span.tweet-date a")
        content_el = body.select_one("div.tweet-content")
        if not username_el or not date_link or not content_el:
            return None
        id_match = self.TWEET_ID_RE.search(date_link.get("href", ""))
        if not id_match:
            return None
        text = content_el.get_text(" ", strip=True)
        if not text:
            return None

        stats = [
            self._parse_stat((stat.select_one("div.icon-container") or stat).get_text(strip=True))
            for stat in body.select("span.tweet-stat")
        ]
        media_urls: list[str] = []
        video_urls: list[str] = []
        video_thumb_urls: list[str] = []
        attachments = body.select_one("div.attachments")
        if attachments:
            for media in attachments.select("a.still-image"):
                raw_url = media.get("href") or media.get("src") or ""
                if raw_url:
                    media_urls.append(raw_url if raw_url.startswith("http") else f"{instance_url}{raw_url}")
            for video in attachments.select("video"):
                poster = video.get("poster", "")
                if poster:
                    video_thumb_urls.append(poster if poster.startswith("http") else f"{instance_url}{poster}")
            for source in attachments.select("video source"):
                raw_url = source.get("src", "")
                if raw_url:
                    video_urls.append(raw_url if raw_url.startswith("http") else f"{instance_url}{raw_url}")

        return SocialTweet(
            tweet_id=id_match.group(1),
            author_username=username_el.get_text(strip=True).lstrip("@"),
            author_name=fullname_el.get_text(" ", strip=True) if fullname_el else "",
            text=text,
            tweeted_at=self._parse_date(date_link.get("title", "")),
            is_retweet=bool(item.select_one("div.retweet-header")),
            is_reply=bool(body.select_one("div.replying-to")),
            reply_count=stats[0] if len(stats) > 0 else 0,
            retweet_count=stats[1] if len(stats) > 1 else 0,
            quote_count=stats[2] if len(stats) > 2 else 0,
            like_count=stats[3] if len(stats) > 3 else 0,
            media_urls=media_urls,
            video_urls=video_urls,
            video_thumb_urls=video_thumb_urls,
            instance_url=instance_url,
        )

    @staticmethod
    def _parse_stat(stat_text: str) -> int:
        value = (stat_text or "").strip().replace(",", "")
        if not value:
            return 0
        try:
            if value.endswith("K"):
                return int(float(value[:-1]) * 1_000)
            if value.endswith("M"):
                return int(float(value[:-1]) * 1_000_000)
            return int(value)
        except ValueError:
            return 0

    @staticmethod
    def _parse_date(title: str) -> datetime | None:
        raw_date = (title or "").replace(" · ", " ").replace("\u00b7", "").strip()
        for fmt in ("%b %d, %Y %I:%M %p %Z", "%b %d, %Y %I:%M %p UTC", "%b %d, %Y %H:%M %Z"):
            try:
                return datetime.strptime(raw_date, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    def _extract_cursor(self, html: str) -> str | None:
        soup = BeautifulSoup(html, "html.parser")
        show_more = soup.select_one("div.show-more a")
        if show_more and show_more.get("href"):
            match = self.CURSOR_RE.search(show_more["href"])
            if match:
                return match.group(1)
        return None


class SocialFeedService:
    def __init__(self, config: SocialFeedConfig | None = None, store: SocialFeedStore | None = None):
        self.config = config or load_social_feed_config()
        self.store = store or SocialFeedStore(self.config.state_db_path)
        self._first_poll = True

    async def poll_once(self) -> FeedPollResult:
        tweets: list[SocialTweet] = []
        errors: list[str] = []
        scanned = 0
        async with NitterScraper(self.config) as scraper:
            for account in self.config.accounts:
                result = await scraper.scrape_user_timeline(
                    account,
                    is_known_tweet=self.store.is_seen,
                )
                if result.error:
                    errors.append(result.error)
                for tweet in result.tweets:
                    scanned += 1
                    if not self.config.include_retweets and tweet.is_retweet:
                        continue
                    if not self.config.include_replies and tweet.is_reply:
                        continue
                    if tweet.author_username.lower() != account.lower():
                        continue
                    tweets.append(tweet)
                await asyncio.sleep(self.config.delay_between_requests)

        tweets_by_id: dict[str, SocialTweet] = {}
        for tweet in tweets:
            tweets_by_id.setdefault(tweet.tweet_id, tweet)
        unique = list(tweets_by_id.values())
        unique.sort(key=lambda item: item.tweeted_at or datetime.now(timezone.utc))

        if self._first_poll and not self.config.post_existing_on_first_run:
            for tweet in unique:
                self.store.mark_seen(tweet, f"@{tweet.author_username}", posted=False)
            self._first_poll = False
            return FeedPollResult([], scanned=scanned, marked_seen=len(unique), errors=errors)

        self._first_poll = False
        new_tweets = [tweet for tweet in unique if not self.store.is_seen(tweet.tweet_id)]
        skipped = max(0, len(new_tweets) - self.config.max_posts_per_poll)
        for tweet in new_tweets[:skipped]:
            self.store.mark_seen(tweet, f"@{tweet.author_username}", posted=False)
        return FeedPollResult(
            tweets_to_post=new_tweets[skipped:],
            scanned=scanned,
            marked_seen=skipped,
            errors=errors,
        )

    def mark_posted(self, tweet: SocialTweet) -> None:
        self.store.mark_seen(tweet, f"@{tweet.author_username}", posted=True)

    def mark_seen_unposted(self, tweet: SocialTweet) -> None:
        self.store.mark_seen(tweet, f"@{tweet.author_username}", posted=False)

    def recent(self, *, limit: int = 10, author: str | None = None) -> list[dict]:
        return self.store.recent(limit=limit, author=author)


_global_service: SocialFeedService | None = None


def get_social_feed_service() -> SocialFeedService:
    global _global_service
    if _global_service is None:
        _global_service = SocialFeedService()
    return _global_service
