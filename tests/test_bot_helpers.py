"""Tests for core/bot_helpers — the pure, side-effect-free helpers used by bot.py.

These are deliberately importable without discord/OpenAI client/dotenv so they can
run anywhere pytest does.
"""
import asyncio
import unittest

from core.bot_helpers import (
    DISCORD_EMBED_FIELD_VALUE_LIMIT,
    GENERIC_TECH_ERROR,
    TOOL_DEDUP_MARKER,
    TOOL_ERROR_MARKER,
    TOOL_TIMEOUT_MARKER,
    chunk_for_discord,
    classify_llm_error,
    ensure_sources_line,
    extract_urls,
    hash_tool_call,
    has_validator_instruction_leak,
    is_bot_apology,
    is_real_tool_content,
    is_transient_llm_error,
    normalize_digits,
    safe_field_value,
    split_body_and_sources,
    tool_calls_signature,
    with_turn_id,
)


class BotHelpersTests(unittest.TestCase):
    # ── is_bot_apology ───────────────────────────────────────────────

    def test_is_bot_apology_matches_generic_tech_error(self):
        self.assertTrue(is_bot_apology(GENERIC_TECH_ERROR))

    def test_is_bot_apology_matches_answer_not_ready(self):
        self.assertTrue(is_bot_apology("माफ गर्नुहोस्, उत्तर तयार गर्न सकिएन।"))

    def test_is_bot_apology_matches_openai_busy(self):
        self.assertTrue(
            is_bot_apology(
                "माफ गर्नुहोस्, Sarvam अहिले व्यस्त छ। केही सेकेन्डपछि पुनः प्रयास गर्नुहोस्।"
            )
        )

    def test_is_bot_apology_rejects_normal_answer(self):
        self.assertFalse(is_bot_apology("नमस्ते! म Yeti हुँ।"))

    def test_is_bot_apology_rejects_empty(self):
        self.assertFalse(is_bot_apology(""))

    def test_is_bot_apology_tolerates_leading_whitespace(self):
        # Discord sometimes preserves leading whitespace in quoted replies.
        self.assertTrue(is_bot_apology("   " + GENERIC_TECH_ERROR))

    def test_has_validator_instruction_leak_detects_internal_fix_text(self):
        leaked = "मुख्य जवाफ देवनागरी (नेपाली) मा लेखिएको छैन — पुनः नेपालीमा लेख्नुहोस्।"
        self.assertTrue(has_validator_instruction_leak(leaked))

    def test_has_validator_instruction_leak_rejects_normal_answer(self):
        self.assertFalse(has_validator_instruction_leak("नेपालको राजधानी काठमाडौं हो।"))

    # ── safe_field_value (Discord 1024-char embed cap) ───────────────

    def test_safe_field_value_short_url_unchanged(self):
        url = "https://example.com/story/123"
        self.assertEqual(safe_field_value(url), url)

    def test_safe_field_value_long_url_truncated(self):
        url = "https://example.com/" + ("a" * 2000)
        clipped = safe_field_value(url)
        self.assertLessEqual(len(clipped), DISCORD_EMBED_FIELD_VALUE_LIMIT)
        self.assertTrue(clipped.endswith("…"))
        # Truncation should preserve a prefix of the original so users can still
        # recognize the source.
        self.assertTrue(clipped.startswith("https://example.com/"))

    # ── extract_urls ─────────────────────────────────────────────────

    def test_extract_urls_deduped_and_ordered(self):
        text = "see https://a.example and https://b.example and https://a.example again"
        self.assertEqual(extract_urls(text), ["https://a.example", "https://b.example"])

    def test_extract_urls_strips_trailing_punctuation(self):
        self.assertEqual(extract_urls("read https://ex.com/story."), ["https://ex.com/story"])

    def test_extract_urls_handles_empty(self):
        self.assertEqual(extract_urls(None), [])
        self.assertEqual(extract_urls(""), [])

    # ── split_body_and_sources ───────────────────────────────────────

    def test_split_body_and_sources_no_marker(self):
        body, src = split_body_and_sources("केवल मूल जवाफ।")
        self.assertEqual(body, "केवल मूल जवाफ।")
        self.assertEqual(src, "")

    def test_split_body_and_sources_with_marker(self):
        ans = "मूल जवाफ यहाँ छ।\nस्रोत: उदाहरण"
        body, src = split_body_and_sources(ans)
        self.assertEqual(body, "मूल जवाफ यहाँ छ।")
        self.assertEqual(src, "स्रोत: उदाहरण")

    # ── chunk_for_discord ────────────────────────────────────────────

    def test_chunk_for_discord_short_text_one_chunk(self):
        self.assertEqual(chunk_for_discord("छोटो जवाफ"), ["छोटो जवाफ"])

    def test_chunk_for_discord_respects_limit(self):
        text = "x" * 5000
        chunks = chunk_for_discord(text, limit=2000)
        self.assertTrue(all(len(c) <= 2000 for c in chunks))
        self.assertEqual("".join(chunks), text)

    def test_chunk_for_discord_prefers_newline_boundary(self):
        first = "line one has reasonable length and should stay together"
        second = "line two"
        text = first + "\n" + second
        chunks = chunk_for_discord(text, limit=len(first) + 3)
        # We expect the split to land on the newline, not mid-word.
        self.assertEqual(chunks[0].rstrip(), first)
        self.assertTrue(chunks[-1].strip().endswith(second))

    # ── normalize_digits ─────────────────────────────────────────────

    def test_normalize_digits_converts_body(self):
        self.assertEqual(normalize_digits("रु. 2.4 ट्रिलियन"), "रु. २.४ ट्रिलियन")

    def test_normalize_digits_preserves_url_digits(self):
        out = normalize_digits("हेर्नुहोस् https://example.com/2024/03 मा 5 खबर")
        self.assertIn("https://example.com/2024/03", out)
        self.assertIn("५ खबर", out)

    def test_normalize_digits_empty(self):
        self.assertEqual(normalize_digits(""), "")

    def test_normalize_digits_mixed(self):
        out = normalize_digits("GDP 2% बढ्यो। स्रोत: https://x.io/2025")
        self.assertIn("GDP २% बढ्यो", out)
        self.assertIn("https://x.io/2025", out)

    # ── ensure_sources_line ──────────────────────────────────────────

    def test_ensure_sources_line_noop_when_already_present(self):
        ans = "मूल जवाफ।\n\nस्रोत:\n- [ex.com](https://ex.com)"
        self.assertEqual(ensure_sources_line("मूल जवाफ।\nस्रोत: ex.com", ["https://ex.com"]), ans)

    def test_ensure_sources_line_noop_when_no_urls(self):
        ans = "मूल जवाफ।"
        self.assertEqual(ensure_sources_line(ans, []), ans)

    def test_ensure_sources_line_appends_when_missing(self):
        ans = "मूल जवाफ।"
        out = ensure_sources_line(ans, ["https://ex.com/1", "https://ex.com/2"])
        self.assertIn("स्रोत:", out)
        self.assertIn("https://ex.com/1", out)
        self.assertIn("https://ex.com/2", out)

    def test_ensure_sources_line_caps_url_count(self):
        urls = [f"https://ex.com/{i}" for i in range(10)]
        out = ensure_sources_line("मूल जवाफ।", urls, max_urls=3)
        self.assertEqual(out.count("https://ex.com/"), 3)

    # ── is_transient_llm_error ───────────────────────────────────────

    def test_is_transient_on_asyncio_timeout(self):
        self.assertTrue(is_transient_llm_error(asyncio.TimeoutError()))

    def test_is_transient_on_rate_limit_status(self):
        class Fake(Exception):
            status_code = 429

        self.assertTrue(is_transient_llm_error(Fake("429")))

    def test_is_transient_on_5xx_status(self):
        class Fake(Exception):
            status_code = 503

        self.assertTrue(is_transient_llm_error(Fake("503")))

    def test_is_transient_on_response_attr(self):
        class Resp:
            status_code = 502

        class Fake(Exception):
            response = Resp()

        self.assertTrue(is_transient_llm_error(Fake("bad gateway")))

    def test_is_transient_on_connect_error_class_name(self):
        class ConnectError(Exception):
            pass

        self.assertTrue(is_transient_llm_error(ConnectError()))

    def test_is_transient_rejects_4xx_non_429(self):
        class Fake(Exception):
            status_code = 404

        self.assertFalse(is_transient_llm_error(Fake("not found")))

    def test_is_transient_rejects_unrelated_exception(self):
        self.assertFalse(is_transient_llm_error(ValueError("bad arg")))

    # ── classify_llm_error ───────────────────────────────────────────

    def test_classify_timeout_says_backend_slow(self):
        msg = classify_llm_error(asyncio.TimeoutError())
        self.assertIn("Sarvam", msg)
        self.assertIn("ढिला", msg)

    def test_classify_429_says_busy(self):
        class Fake(Exception):
            status_code = 429

        msg = classify_llm_error(Fake("429"))
        self.assertIn("व्यस्त", msg)

    def test_classify_5xx_says_service_problem(self):
        class Fake(Exception):
            status_code = 502

        msg = classify_llm_error(Fake("bad gateway"))
        self.assertTrue("सेवा" in msg or "समस्या" in msg)

    def test_classify_connect_error_says_network(self):
        class ConnectError(Exception):
            pass

        msg = classify_llm_error(ConnectError())
        self.assertIn("नेटवर्क", msg)

    def test_classify_unknown_falls_back_to_generic(self):
        msg = classify_llm_error(ValueError("weird"))
        self.assertTrue(msg.startswith(GENERIC_TECH_ERROR))
        self.assertIn("[debug]", msg)

    def test_classify_none_is_generic(self):
        msg = classify_llm_error(None)
        self.assertTrue(msg.startswith(GENERIC_TECH_ERROR))
        self.assertIn("[debug]", msg)

    def test_classified_message_is_still_apology_shaped(self):
        """Classified messages must still match BOT_APOLOGY_PREFIXES so they get
        filtered out of replayed history."""
        excs = [
            asyncio.TimeoutError(),
            type("X", (Exception,), {"status_code": 429})("429"),
            type("X", (Exception,), {"status_code": 500})("500"),
            type("ConnectError", (Exception,), {})(),
        ]
        for exc in excs:
            with self.subTest(exc=exc):
                self.assertTrue(is_bot_apology(classify_llm_error(exc)))

    # ── with_turn_id ────────────────────────────────────────────────

    def test_with_turn_id_appends_code(self):
        out = with_turn_id(GENERIC_TECH_ERROR, "ab12cd34")
        self.assertIn(GENERIC_TECH_ERROR, out)
        self.assertIn("ab12cd34", out)
        self.assertIn("त्रुटि कोड", out)

    def test_with_turn_id_noop_when_missing(self):
        self.assertEqual(with_turn_id(GENERIC_TECH_ERROR, None), GENERIC_TECH_ERROR)
        self.assertEqual(with_turn_id(GENERIC_TECH_ERROR, ""), GENERIC_TECH_ERROR)

    def test_with_turn_id_still_matches_apology_filter(self):
        """Turn-id-decorated apologies must still be detected by is_bot_apology
        so they don't poison the next turn's history."""
        self.assertTrue(is_bot_apology(with_turn_id(GENERIC_TECH_ERROR, "cafef00d")))

    # ── hash_tool_call ───────────────────────────────────────────────

    def test_hash_tool_call_stable_across_key_order(self):
        """Arg dict key order must not affect the signature — json sort_keys."""
        a = hash_tool_call("search", {"q": "नेपाल", "limit": 5})
        b = hash_tool_call("search", {"limit": 5, "q": "नेपाल"})
        self.assertEqual(a, b)

    def test_hash_tool_call_differs_on_args(self):
        self.assertNotEqual(
            hash_tool_call("search", {"q": "a"}),
            hash_tool_call("search", {"q": "b"}),
        )

    def test_hash_tool_call_differs_on_name(self):
        self.assertNotEqual(
            hash_tool_call("a", {"q": "x"}),
            hash_tool_call("b", {"q": "x"}),
        )

    def test_hash_tool_call_handles_empty_args(self):
        # Empty args must still yield a deterministic hash.
        h = hash_tool_call("search", {})
        self.assertIsInstance(h, str)
        self.assertGreater(len(h), 0)

    def test_hash_tool_call_handles_none_args(self):
        self.assertEqual(hash_tool_call("search", None), hash_tool_call("search", {}))

    # ── tool_calls_signature ─────────────────────────────────────────

    def test_tool_calls_signature_order_invariant(self):
        """Swapping the order of parallel tool_calls must not change the round signature.

        Load-bearing: bot.py uses this to detect "two rounds of identical calls".
        If order changed the signature, every parallel round would look novel.
        """
        a = _FakeToolCall("search", '{"q":"a"}', "1")
        b = _FakeToolCall("osint", '{"subject":"b"}', "2")
        self.assertEqual(tool_calls_signature([a, b]), tool_calls_signature([b, a]))

    def test_tool_calls_signature_detects_repeat(self):
        """Same (name, args) across two rounds → same signature → loop detection fires."""
        r1 = [_FakeToolCall("search", '{"q":"नेपाल"}', "1")]
        r2 = [_FakeToolCall("search", '{"q":"नेपाल"}', "2")]  # new id, same call
        self.assertEqual(tool_calls_signature(r1), tool_calls_signature(r2))

    def test_tool_calls_signature_distinguishes_different_args(self):
        r1 = [_FakeToolCall("search", '{"q":"a"}', "1")]
        r2 = [_FakeToolCall("search", '{"q":"b"}', "2")]
        self.assertNotEqual(tool_calls_signature(r1), tool_calls_signature(r2))

    def test_tool_calls_signature_empty(self):
        self.assertEqual(tool_calls_signature([]), ())

    def test_tool_calls_signature_handles_malformed_args_json(self):
        """Bad JSON in arguments must not crash signature computation."""
        bad = _FakeToolCall("search", "{not-json", "1")
        sig = tool_calls_signature([bad])
        self.assertEqual(len(sig), 1)

    # ── is_real_tool_content ─────────────────────────────────────────

    def test_is_real_tool_content_accepts_good_result(self):
        self.assertTrue(is_real_tool_content(_FakeResult(True, "नेपालमा मुद्रास्फीति ५.२%")))

    def test_is_real_tool_content_rejects_failure(self):
        self.assertFalse(is_real_tool_content(_FakeResult(False, "नेपालमा मुद्रास्फीति ५.२%")))

    def test_is_real_tool_content_rejects_empty(self):
        self.assertFalse(is_real_tool_content(_FakeResult(True, "")))
        self.assertFalse(is_real_tool_content(_FakeResult(True, None)))
        self.assertFalse(is_real_tool_content(_FakeResult(True, "   \n  ")))

    def test_is_real_tool_content_rejects_tool_error_marker(self):
        content = f"{TOOL_ERROR_MARKER} internet_search failed internally."
        self.assertFalse(is_real_tool_content(_FakeResult(False, content)))

    def test_is_real_tool_content_rejects_tool_timeout_marker(self):
        content = f"{TOOL_TIMEOUT_MARKER} osint_lookup exceeded 15s."
        self.assertFalse(is_real_tool_content(_FakeResult(False, content)))

    def test_is_real_tool_content_rejects_dedup_marker_even_on_success(self):
        """A dedup replay must not re-trigger tool_was_used — it wasn't a fresh hit."""
        content = f"{TOOL_DEDUP_MARKER} search already executed.\n(real body here)"
        # Even when success=True (cached result), dedup content signals "already counted".
        self.assertFalse(is_real_tool_content(_FakeResult(True, content)))

    def test_is_real_tool_content_rejects_none(self):
        self.assertFalse(is_real_tool_content(None))


class _FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, name, arguments, call_id="tc_1"):
        self.id = call_id
        self.function = _FakeFunction(name, arguments)


class _FakeResult:
    def __init__(self, success, content):
        self.success = success
        self.content = content


if __name__ == "__main__":
    unittest.main()
