"""Tests for core/bot_helpers — the pure, side-effect-free helpers used by bot.py.

These are deliberately importable without discord/sarvamai/dotenv so they can
run anywhere pytest does.
"""
import asyncio

import pytest

from core.bot_helpers import (
    DISCORD_EMBED_FIELD_VALUE_LIMIT,
    GENERIC_TECH_ERROR,
    chunk_for_discord,
    classify_llm_error,
    ensure_sources_line,
    extract_urls,
    is_bot_apology,
    is_transient_llm_error,
    normalize_digits,
    safe_field_value,
    split_body_and_sources,
    with_turn_id,
)


# ── is_bot_apology ───────────────────────────────────────────────────

def test_is_bot_apology_matches_generic_tech_error():
    assert is_bot_apology(GENERIC_TECH_ERROR)


def test_is_bot_apology_matches_answer_not_ready():
    assert is_bot_apology("माफ गर्नुहोस्, उत्तर तयार गर्न सकिएन।")


def test_is_bot_apology_matches_sarvam_busy():
    assert is_bot_apology(
        "माफ गर्नुहोस्, Sarvam अहिले व्यस्त छ। केही सेकेन्डपछि पुनः प्रयास गर्नुहोस्।"
    )


def test_is_bot_apology_rejects_normal_answer():
    assert not is_bot_apology("नमस्ते! म Yeti हुँ।")


def test_is_bot_apology_rejects_empty():
    assert not is_bot_apology("")


def test_is_bot_apology_tolerates_leading_whitespace():
    # Discord sometimes preserves leading whitespace in quoted replies.
    assert is_bot_apology("   " + GENERIC_TECH_ERROR)


# ── safe_field_value (Discord 1024-char embed cap) ───────────────────

def test_safe_field_value_short_url_unchanged():
    url = "https://example.com/story/123"
    assert safe_field_value(url) == url


def test_safe_field_value_long_url_truncated():
    url = "https://example.com/" + ("a" * 2000)
    clipped = safe_field_value(url)
    assert len(clipped) <= DISCORD_EMBED_FIELD_VALUE_LIMIT
    assert clipped.endswith("…")
    # Truncation should preserve a prefix of the original so users can still
    # recognize the source.
    assert clipped.startswith("https://example.com/")


# ── extract_urls ─────────────────────────────────────────────────────

def test_extract_urls_deduped_and_ordered():
    text = "see https://a.example and https://b.example and https://a.example again"
    assert extract_urls(text) == ["https://a.example", "https://b.example"]


def test_extract_urls_strips_trailing_punctuation():
    assert extract_urls("read https://ex.com/story.") == ["https://ex.com/story"]


def test_extract_urls_handles_empty():
    assert extract_urls(None) == []
    assert extract_urls("") == []


# ── split_body_and_sources ───────────────────────────────────────────

def test_split_body_and_sources_no_marker():
    body, src = split_body_and_sources("केवल मूल जवाफ।")
    assert body == "केवल मूल जवाफ।"
    assert src == ""


def test_split_body_and_sources_with_marker():
    ans = "मूल जवाफ यहाँ छ।\nस्रोत: उदाहरण"
    body, src = split_body_and_sources(ans)
    assert body == "मूल जवाफ यहाँ छ।"
    assert src == "स्रोत: उदाहरण"


# ── chunk_for_discord ────────────────────────────────────────────────

def test_chunk_for_discord_short_text_one_chunk():
    assert chunk_for_discord("छोटो जवाफ") == ["छोटो जवाफ"]


def test_chunk_for_discord_respects_limit():
    text = "x" * 5000
    chunks = chunk_for_discord(text, limit=2000)
    assert all(len(c) <= 2000 for c in chunks)
    assert "".join(chunks) == text


def test_chunk_for_discord_prefers_newline_boundary():
    first = "line one has reasonable length and should stay together"
    second = "line two"
    text = first + "\n" + second
    chunks = chunk_for_discord(text, limit=len(first) + 3)
    # We expect the split to land on the newline, not mid-word.
    assert chunks[0].rstrip() == first
    assert chunks[-1].strip().endswith(second)


# ── normalize_digits ─────────────────────────────────────────────────

def test_normalize_digits_converts_body():
    assert normalize_digits("रु. 2.4 ट्रिलियन") == "रु. २.४ ट्रिलियन"


def test_normalize_digits_preserves_url_digits():
    out = normalize_digits("हेर्नुहोस् https://example.com/2024/03 मा 5 खबर")
    assert "https://example.com/2024/03" in out
    assert "५ खबर" in out


def test_normalize_digits_empty():
    assert normalize_digits("") == ""


def test_normalize_digits_mixed():
    out = normalize_digits("GDP 2% बढ्यो। स्रोत: https://x.io/2025")
    assert "GDP २% बढ्यो" in out
    assert "https://x.io/2025" in out


# ── ensure_sources_line ──────────────────────────────────────────────

def test_ensure_sources_line_noop_when_already_present():
    ans = "मूल जवाफ।\nस्रोत: ex.com"
    assert ensure_sources_line(ans, ["https://ex.com"]) == ans


def test_ensure_sources_line_noop_when_no_urls():
    ans = "मूल जवाफ।"
    assert ensure_sources_line(ans, []) == ans


def test_ensure_sources_line_appends_when_missing():
    ans = "मूल जवाफ।"
    out = ensure_sources_line(ans, ["https://ex.com/1", "https://ex.com/2"])
    assert "स्रोत:" in out
    assert "https://ex.com/1" in out
    assert "https://ex.com/2" in out


def test_ensure_sources_line_caps_url_count():
    urls = [f"https://ex.com/{i}" for i in range(10)]
    out = ensure_sources_line("मूल जवाफ।", urls, max_urls=3)
    assert out.count("https://ex.com/") == 3


# ── is_transient_llm_error ───────────────────────────────────────────

def test_is_transient_on_asyncio_timeout():
    assert is_transient_llm_error(asyncio.TimeoutError())


def test_is_transient_on_rate_limit_status():
    class Fake(Exception):
        status_code = 429
    assert is_transient_llm_error(Fake("429"))


def test_is_transient_on_5xx_status():
    class Fake(Exception):
        status_code = 503
    assert is_transient_llm_error(Fake("503"))


def test_is_transient_on_response_attr():
    class Resp:
        status_code = 502
    class Fake(Exception):
        response = Resp()
    assert is_transient_llm_error(Fake("bad gateway"))


def test_is_transient_on_connect_error_class_name():
    class ConnectError(Exception):
        pass
    assert is_transient_llm_error(ConnectError())


def test_is_transient_rejects_4xx_non_429():
    class Fake(Exception):
        status_code = 404
    assert not is_transient_llm_error(Fake("not found"))


def test_is_transient_rejects_unrelated_exception():
    assert not is_transient_llm_error(ValueError("bad arg"))


# ── classify_llm_error ───────────────────────────────────────────────

def test_classify_timeout_says_sarvam_slow():
    msg = classify_llm_error(asyncio.TimeoutError())
    assert "Sarvam" in msg and "ढिला" in msg


def test_classify_429_says_busy():
    class Fake(Exception):
        status_code = 429
    msg = classify_llm_error(Fake("429"))
    assert "व्यस्त" in msg


def test_classify_5xx_says_service_problem():
    class Fake(Exception):
        status_code = 502
    msg = classify_llm_error(Fake("bad gateway"))
    assert "सेवा" in msg or "समस्या" in msg


def test_classify_connect_error_says_network():
    class ConnectError(Exception):
        pass
    msg = classify_llm_error(ConnectError())
    assert "नेटवर्क" in msg


def test_classify_unknown_falls_back_to_generic():
    assert classify_llm_error(ValueError("weird")) == GENERIC_TECH_ERROR


def test_classify_none_is_generic():
    assert classify_llm_error(None) == GENERIC_TECH_ERROR


@pytest.mark.parametrize("exc", [
    asyncio.TimeoutError(),
    type("X", (Exception,), {"status_code": 429})("429"),
    type("X", (Exception,), {"status_code": 500})("500"),
    type("ConnectError", (Exception,), {})(),
])
def test_classified_message_is_still_apology_shaped(exc):
    """Classified messages must still match BOT_APOLOGY_PREFIXES so they get
    filtered out of replayed history."""
    assert is_bot_apology(classify_llm_error(exc))


# ── with_turn_id ─────────────────────────────────────────────────────

def test_with_turn_id_appends_code():
    out = with_turn_id(GENERIC_TECH_ERROR, "ab12cd34")
    assert GENERIC_TECH_ERROR in out
    assert "ab12cd34" in out
    assert "त्रुटि कोड" in out


def test_with_turn_id_noop_when_missing():
    assert with_turn_id(GENERIC_TECH_ERROR, None) == GENERIC_TECH_ERROR
    assert with_turn_id(GENERIC_TECH_ERROR, "") == GENERIC_TECH_ERROR


def test_with_turn_id_still_matches_apology_filter():
    """Turn-id-decorated apologies must still be detected by is_bot_apology
    so they don't poison the next turn's history."""
    assert is_bot_apology(with_turn_id(GENERIC_TECH_ERROR, "cafef00d"))
