"""
tools/osint/freshness.py — detect stale OSINT payloads.

Why this exists: in production we saw Yeti reply to "aja ko news"
(today's news) with 2022-era COVID stories from the NepalOSINT cache.
The LLM had no way to tell the data was stale because the payloads
don't carry a single normalised `as_of` date.

This module walks arbitrary JSON payloads, extracts the newest plausible
date, and classifies the bundle as fresh / stale / unknown. The plugin
uses the result to:
  1. Annotate the tool message with `[STALE_DATA:<days>]` when the user
     asked about "today" but the data is older than RECENCY_THRESHOLD_DAYS.
  2. Trigger an auto-fallback to internet_search in the same turn.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timezone
from typing import Any, Iterable


# Recency keywords — if *any* appear in the user query, we treat it as a
# time-sensitive request and run the freshness check. Kept in sync with
# context_router.GENERAL_KEYWORDS' recency subset.
_RECENCY_TOKENS: tuple[str, ...] = (
    "today", "tonight", "latest", "breaking", "now", "this morning",
    "this evening", "this afternoon", "just now",
    "आज", "अहिले", "भर्खर", "अभि", "अहिलेसम्म",
    "aja", "aaja", "ahile", "abhi", "bharkhar",
)

# "recent" / "recently" / "last hour/day/week" — softer recency, still
# triggers the check but with a looser threshold (RECENCY_THRESHOLD_DAYS
# is applied uniformly for simplicity; finer tiers can be added later).
_SOFT_RECENCY_TOKENS: tuple[str, ...] = (
    "recent", "recently", "last hour", "last 24", "last day",
    "latest update", "updates today",
)

# Any payload whose newest date is older than this is considered stale
# when the user's query contains a recency keyword.
RECENCY_THRESHOLD_DAYS = 3


def _read_coverage_env() -> date | None:
    """Parse `OSINT_COVERAGE_UNTIL=YYYY-MM-DD` from env.

    Returns None when the env var is unset, empty, or unparseable —
    every caller therefore defaults to "no coverage anchor" behavior.
    This is the opt-in: to activate the coverage-gap path, set the env
    var in production. Without it, the module behaves exactly as it did
    before the gap feature was introduced.
    """
    import os as _os
    raw = (_os.getenv("OSINT_COVERAGE_UNTIL") or "").strip()
    if not raw:
        return None
    try:
        y, m, d = raw.split("-")
        return date(int(y), int(m), int(d))
    except (ValueError, TypeError):
        return None


# Upstream NepalOSINT cache is populated by a scheduler; when the whole
# dataset is frozen past a date, per-payload freshness heuristics miss
# it (individual items look "recent" relative to themselves). Setting
# OSINT_COVERAGE_UNTIL in the environment tells freshness checks the
# true upper bound of coverage — any delta beyond RECENCY_THRESHOLD_DAYS
# surfaces as `coverage_gap=True`, forcing a web fallback.
#
# Default None = opt-in. Railway deploys without the env var behave
# exactly as before this feature existed.
OSINT_COVERAGE_UNTIL: date | None = _read_coverage_env()

# Keys whose values are likely to be dates. Checked first (cheaper than
# scanning every string in every payload).
_DATE_KEYS: frozenset[str] = frozenset({
    "published_at", "publishedAt", "published_date",
    "created_at", "createdAt", "created",
    "updated_at", "updatedAt", "updated",
    "date", "datetime", "timestamp", "ts",
    "pub_date", "pubDate", "posted_at", "story_date",
    "as_of", "asOf", "fetched_at",
})

# ISO-8601-ish: 2026-04-22, 2026-04-22T12:34, 2026/04/22 ...
_ISO_DATE_RE = re.compile(
    r"(?P<y>\d{4})[-/](?P<m>\d{1,2})[-/](?P<d>\d{1,2})"
    r"(?:[Tt ](?P<H>\d{1,2}):(?P<M>\d{2})(?::(?P<S>\d{2}))?)?"
)


def _parse_date(value: Any) -> date | None:
    """Best-effort conversion of a value to a `date`. Returns None on failure."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        # Epoch seconds (or ms — auto-scale).
        ts = float(value)
        if ts > 1e12:
            ts /= 1000.0
        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc).date()
        except (ValueError, OSError, OverflowError):
            return None
    if not isinstance(value, str):
        return None
    m = _ISO_DATE_RE.search(value)
    if not m:
        return None
    try:
        y, mo, d = int(m.group("y")), int(m.group("m")), int(m.group("d"))
        return date(y, mo, d)
    except (ValueError, TypeError):
        return None


def _iter_candidates(payload: Any) -> Iterable[date]:
    """Walk a nested JSON structure, yielding plausible dates."""
    if payload is None:
        return
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(key, str) and key.lower() in _DATE_KEYS:
                d = _parse_date(value)
                if d is not None:
                    yield d
                    continue
            if isinstance(value, (dict, list)):
                yield from _iter_candidates(value)
            elif isinstance(value, str) and len(value) <= 40:
                # Cheap fallback: some payloads put dates into non-canonical
                # keys (e.g. `headline_time`). Only peek at short strings to
                # avoid running the regex over full article bodies.
                d = _parse_date(value)
                if d is not None:
                    yield d
    elif isinstance(payload, list):
        for item in payload:
            yield from _iter_candidates(item)


def newest_date(payloads: dict[str, Any]) -> date | None:
    """Return the newest plausible date across all bundle payloads."""
    newest: date | None = None
    today = date.today()
    for payload in payloads.values():
        for d in _iter_candidates(payload):
            # Guard against obvious garbage (e.g. year 1970 epoch-zero rows
            # or future dates from malformed data).
            if d.year < 2000 or d > today:
                continue
            if newest is None or d > newest:
                newest = d
    return newest


def is_recency_query(query: str) -> bool:
    """True if the query uses a recency keyword."""
    normalized = (query or "").lower()
    if any(tok in normalized for tok in _RECENCY_TOKENS):
        return True
    return any(tok in normalized for tok in _SOFT_RECENCY_TOKENS)


# Sentinel so callers can explicitly disable the coverage check with
# `coverage_until=None`, while an omitted kwarg defaults to the module
# constant (which itself defaults to None — opt-in via env).
_COVERAGE_DEFAULT: Any = object()


def assess_freshness(
    query: str,
    payloads: dict[str, Any],
    *,
    threshold_days: int = RECENCY_THRESHOLD_DAYS,
    coverage_until: Any = _COVERAGE_DEFAULT,
) -> dict[str, Any]:
    """Return `{stale, newest, age_days, required, coverage_gap, gap_days}`.

    `required=True` when the query contains a recency keyword — retained
    for downstream callers that distinguish "asked for today" vs "asked
    anything". Staleness is ALSO decoupled: when coverage_until is set
    (via OSINT_COVERAGE_UNTIL env var) and today exceeds it by more than
    `threshold_days`, `stale=True` regardless of `required` — the whole
    OSINT dataset is behind, and even non-recency queries shouldn't cite
    it as current.

    When neither the env anchor nor an explicit `coverage_until` is
    provided, behavior matches the pre-feature contract: stale only
    fires on recency-tagged queries with aged-out payloads.
    """
    required = is_recency_query(query)
    newest = newest_date(payloads or {})
    today = date.today()

    anchor: date | None
    if coverage_until is _COVERAGE_DEFAULT:
        anchor = OSINT_COVERAGE_UNTIL
    else:
        anchor = coverage_until

    gap_days: int | None = None
    coverage_gap = False
    if anchor is not None:
        gap_days = (today - anchor).days
        coverage_gap = gap_days > threshold_days

    if newest is None:
        return {
            "stale": coverage_gap,
            "newest": None,
            "age_days": None,
            "required": required,
            "coverage_gap": coverage_gap,
            "gap_days": gap_days,
        }
    age = (today - newest).days
    # When no coverage anchor is configured (default, legacy-compatible),
    # keep the pre-feature contract: stale only on recency queries. When
    # an anchor IS configured, fire stale on any aged-out payload so web
    # fallback triggers for every Nepal intent — that's the opt-in
    # behavior change.
    if anchor is not None:
        payload_stale = age > threshold_days
    else:
        payload_stale = required and age > threshold_days
    return {
        "stale": payload_stale or coverage_gap,
        "newest": newest.isoformat(),
        "age_days": age,
        "required": required,
        "coverage_gap": coverage_gap,
        "gap_days": gap_days,
    }
