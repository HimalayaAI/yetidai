"""
core/nepali_date.py — Gregorian ↔ Bikram Sambat conversion.

Thin wrapper over the `nepali-datetime` PyPI package (vetted tables
covering BS 1975–2100). Exposes just what bot.py needs: a one-line
Nepali-script BS date string to inject into the dynamic system prompt.

Why a wrapper?
  * Keeps the dependency optional at import time — if the package is
    missing we return None and the caller falls back to AD-only.
  * Gives us a single place to stringify the BS date in the house
    style (Nepali month name + Devanagari digits, e.g. `वैशाख १०, २०८३`).
"""
from __future__ import annotations

from datetime import date

try:
    import nepali_datetime as _nd
    _AVAILABLE = True
except Exception:
    _nd = None  # type: ignore[assignment]
    _AVAILABLE = False


_DEV_DIGITS = str.maketrans("0123456789", "०१२३४५६७८९")


def to_bs(g: date) -> tuple[int, int, int] | None:
    """Return (BS year, month, day) or None if conversion unavailable."""
    if not _AVAILABLE or _nd is None:
        return None
    try:
        bs = _nd.date.from_datetime_date(g)
    except Exception:
        return None
    return bs.year, bs.month, bs.day


def format_bs_ne(g: date) -> str | None:
    """Return `वैशाख १०, २०८३` or None if unavailable."""
    if not _AVAILABLE or _nd is None:
        return None
    try:
        bs = _nd.date.from_datetime_date(g)
        month_name_ne = bs.strftime("%N")
        y = str(bs.year).translate(_DEV_DIGITS)
        d = str(bs.day).translate(_DEV_DIGITS)
        return f"{month_name_ne} {d}, {y}"
    except Exception:
        return None


def format_bs_iso(g: date) -> str | None:
    """Return `BS YYYY-MM-DD` or None if unavailable."""
    bs = to_bs(g)
    if bs is None:
        return None
    return f"BS {bs[0]:04d}-{bs[1]:02d}-{bs[2]:02d}"
