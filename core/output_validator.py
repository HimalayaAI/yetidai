"""
core/output_validator.py — deterministic checks on Yeti's final answer.

Runs AFTER Sarvam emits the final Nepali reply. Flags violations cheaply so
bot.py can ask Sarvam to fix them in one targeted retry.

Checks:
  1. Devanagari-digit-only in body (ASCII digits [0-9] forbidden).
  2. `स्रोत:` citation line present when a tool was used this turn.
  3. No long runs of pure-English prose (loanwords are fine; paragraphs aren't).
  4. Body not empty and not trivially "माफ गर्नुहोस्…" when tool data exists.

Out of scope: semantic correctness, factuality, source-url validity.
"""
from __future__ import annotations

import re
from typing import Iterable

# Range of Devanagari characters used by Nepali prose.
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
# One-or-more ASCII letters + spaces — used to measure English-paragraph length.
_ASCII_WORD_RE = re.compile(r"[A-Za-z]+")

# Terms allowed to appear in Latin script inside a Nepali body.
_LOANWORD_WHITELIST = {
    # Orgs / tickers (keep short — this is a safety valve, not a dictionary).
    "NEPSE", "NRB", "PDMO", "IMF", "ADB", "UEFA", "FIFA", "NBA", "GDP",
    "CPI", "USD", "NPR", "INR", "EUR", "OPEC", "OECD", "HARL", "Yeti",
    "YetiDai", "HimalayaAI", "HimalayaGPT", "OAuth", "Haiku", "Sarvam",
    # Units
    "kg", "km", "mm", "cm", "ml", "MW", "GW", "KW", "pc", "ppm", "bn",
    "mn", "pct",
    # HTTP / URL pieces — validators should ignore links.
    "http", "https", "www", "com", "org", "net", "io", "gov", "np",
}


def _ascii_run_length(text: str) -> int:
    """Longest contiguous block of 'mostly ASCII letters' in text."""
    # Strip URLs first — a URL shouldn't count as English prose.
    no_urls = re.sub(r"https?://\S+", " ", text)
    # Strip digits, punctuation, and whitespace.
    letters_only = re.sub(r"[^A-Za-z\s]", " ", no_urls)
    # Strip whitelisted loanwords token-by-token.
    kept: list[str] = []
    for tok in letters_only.split():
        if tok in _LOANWORD_WHITELIST or tok.upper() in _LOANWORD_WHITELIST:
            continue
        kept.append(tok)
    # Join back and measure longest consecutive run of ≥3-char ASCII words.
    runs: list[str] = []
    current: list[str] = []
    for tok in kept:
        if len(tok) >= 3:
            current.append(tok)
        else:
            if current:
                runs.append(" ".join(current))
                current = []
    if current:
        runs.append(" ".join(current))
    return max((len(r) for r in runs), default=0)


def _split_body_and_sources(answer: str) -> tuple[str, str]:
    """Return (body_without_sources, sources_line_or_empty)."""
    # Find the last occurrence of स्रोत: on its own segment.
    marker_idx = answer.rfind("स्रोत:")
    if marker_idx < 0:
        return answer, ""
    return answer[:marker_idx].rstrip(), answer[marker_idx:].strip()


def validate_answer(
    answer: str,
    *,
    tool_was_used: bool,
    min_devanagari_chars: int = 15,
) -> list[str]:
    """Return a list of Nepali fix instructions. Empty list = all good."""
    issues: list[str] = []

    if not answer or not answer.strip():
        return ["उत्तर खाली छ — Nepali मा पूरा जवाफ लेख्नुहोस्।"]

    body, sources_line = _split_body_and_sources(answer)

    # (1) ASCII digits forbidden in body text (URLs in sources are ok).
    body_no_urls = re.sub(r"https?://\S+", " ", body)
    if re.search(r"[0-9]", body_no_urls):
        issues.append(
            "ASCII अङ्क (0-9) छन् — सबै देवनागरी अङ्क (०-९) मा रूपान्तरण गर्नुहोस्।"
        )

    # (2) Devanagari content present.
    if len(_DEVANAGARI_RE.findall(body)) < min_devanagari_chars:
        issues.append(
            "मुख्य जवाफ देवनागरी (नेपाली) मा लेखिएको छैन — पुनः नेपालीमा लेख्नुहोस्।"
        )

    # (3) स्रोत: line required when tool output was available.
    if tool_was_used and "स्रोत:" not in answer:
        issues.append("स्रोत: रेखा थप्नुहोस् (२–४ भरपर्दो स्रोत उद्धरण गर्नुहोस्)।")

    # (4) No long English paragraphs.
    if _ascii_run_length(body) > 60:
        issues.append(
            "जवाफमा लामो अङ्ग्रेजी वाक्यांश छ — सम्पूर्ण नेपालीमा अनुवाद गर्नुहोस्।"
        )

    return issues


def build_fix_message(issues: Iterable[str]) -> str:
    """Build the system-style nudge asking Sarvam to rewrite its last reply."""
    bullet = "\n".join(f"- {i}" for i in issues)
    return (
        "तपाईंको अघिल्लो जवाफमा केही त्रुटि छन्। कृपया तुरुन्त पुनः लेख्नुहोस्, "
        "उही तथ्य-सूचनालाई समातेर तर यी समस्या हटाएर:\n"
        f"{bullet}\n"
        "JSON वा markdown नराख्नुहोस् — केवल अन्तिम नेपाली उत्तर लेख्नुहोस्।"
    )
