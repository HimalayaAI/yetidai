"""
core/date_context.py — the rich CURRENT DATE block injected into the
dynamic system prompt.

Why a dedicated module:
  * bot.py was building the block inline; adding today + hijo + "last
    week" made the code messy.
  * We want this logic unit-tested without spinning up discord / sarvam.
  * The block has to be prominent (Sarvam-30B has attention decay on
    very long prompts), consistent, and honest about when BS isn't
    available.

Format (stable — tests depend on it):

    # CURRENT DATE (ground truth — do not invent dates)
    - आज (Today):        2026-04-23 AD  |  BS 2083-01-10  (वैशाख १०, २०८३)
    - हिजो (Yesterday):   2026-04-22 AD  |  BS 2083-01-09  (वैशाख ९, २०८३)
    - गत हप्ता (1 wk ago):2026-04-16 AD  |  BS 2083-01-03  (वैशाख ३, २०८३)

    Rules:
    1. Every AD date in your answer MUST be paired with the BS equivalent
       when you're quoting Nepal-context dates.
    2. Never invent the date — use the block above.
    3. When the user asks "आज कहिले हो?" reply with both calendars.
"""
from __future__ import annotations

from datetime import date, timedelta

from core.nepali_date import format_bs_iso, format_bs_ne


def _fmt_pair(label: str, g: date) -> str:
    ad = g.strftime("%Y-%m-%d")
    bs_iso = format_bs_iso(g)
    bs_ne = format_bs_ne(g)
    if bs_iso and bs_ne:
        return f"- {label}: {ad} AD  |  {bs_iso}  ({bs_ne})"
    return f"- {label}: {ad} AD  |  (BS conversion unavailable)"


def build_date_block(today: date | None = None) -> str:
    """Return the CURRENT DATE section for the dynamic system prompt."""
    today = today or date.today()
    yesterday = today - timedelta(days=1)
    week_ago = today - timedelta(days=7)
    return (
        "# CURRENT DATE (ground truth — do not invent dates)\n"
        f"{_fmt_pair('आज (Today)', today)}\n"
        f"{_fmt_pair('हिजो (Yesterday)', yesterday)}\n"
        f"{_fmt_pair('गत हप्ता (1 week ago)', week_ago)}\n"
        "\n"
        "Rules:\n"
        "1. Every AD date you mention for Nepal-context content MUST be "
        "paired with its BS equivalent when available from the block above.\n"
        "2. Never hallucinate a date. Use the block above. If a date "
        "you need isn't listed, say so honestly.\n"
        "3. When the user asks 'आज कहिले हो?' / 'आजको मिति के हो?' / "
        "'aja ko miti' — reply with BOTH calendars in one short line, "
        "quoting the Today row above verbatim."
    )
