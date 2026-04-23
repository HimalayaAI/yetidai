"""Tests for the URL-shortening helpers.

Motivated by the public-debt transcript where Yeti cited four full
URLs in its स्रोत: block, making the reply visually heavy on Discord.
The fix swaps bare URLs for Discord-markdown links with short labels.
"""
from __future__ import annotations

import unittest

from core.bot_helpers import (
    ensure_sources_line,
    rewrite_sources_as_markdown,
    shorten_for_citation,
)


class ShortenForCitationTests(unittest.TestCase):
    def test_host_only_when_no_path(self) -> None:
        self.assertEqual(
            shorten_for_citation("https://merolagani.com/"),
            "[merolagani.com](https://merolagani.com/)",
        )

    def test_strips_www(self) -> None:
        short = shorten_for_citation("https://www.merolagani.com/news/x")
        self.assertTrue(short.startswith("[merolagani.com"))

    def test_includes_first_path_segment(self) -> None:
        short = shorten_for_citation(
            "https://myrepublica.nagariknetwork.com/news/nepals-public-debt-records-rs-2858-trillion"
        )
        self.assertTrue(short.startswith("["))
        # Label should contain the hostname.
        self.assertIn("myrepublica.nagariknetwork.com", short)
        # And then either '/news' or truncation marker.
        self.assertIn("](https://myrepublica.nagariknetwork.com/news/", short)

    def test_very_long_hostname_truncates(self) -> None:
        url = "https://a-super-long-host-name-that-exceeds-the-cap.example.com/x"
        out = shorten_for_citation(url, max_chars=30)
        # Label fits inside the cap (plus the brackets + paren literals).
        label = out.split("]")[0][1:]
        self.assertLessEqual(len(label), 30)

    def test_non_url_passthrough(self) -> None:
        self.assertEqual(shorten_for_citation(""), "")
        self.assertEqual(shorten_for_citation("just text"), "just text")


class RewriteSourcesTests(unittest.TestCase):
    def test_rewrites_bullet_urls(self) -> None:
        answer = (
            "नेपालको debt बढेको छ।\n\n"
            "स्रोत:\n"
            "- https://myrepublica.nagariknetwork.com/news/nepals-public-debt-records-rs-2858-trillion\n"
            "- https://english.nepalnews.com/s/business/everything-you-need-to-know\n"
        )
        out = rewrite_sources_as_markdown(answer)
        # Each bullet URL becomes a markdown link.
        self.assertIn("[myrepublica.nagariknetwork.com", out)
        self.assertIn("[english.nepalnews.com", out)
        # Full URLs preserved inside the markdown target.
        self.assertIn(
            "](https://myrepublica.nagariknetwork.com/news/nepals-public-debt-records-rs-2858-trillion)",
            out,
        )

    def test_leaves_body_alone(self) -> None:
        # Body URLs (paragraph prose) should NOT be rewritten.
        answer = (
            "NRB ले https://www.nrb.org.np/ मा report प्रकाशित गरेको छ।\n"
            "\nस्रोत:\n"
            "- https://www.nrb.org.np/report\n"
        )
        out = rewrite_sources_as_markdown(answer)
        # Body URL stays bare.
        self.assertIn("NRB ले https://www.nrb.org.np/ मा", out)
        # Source URL is shortened.
        self.assertIn("[nrb.org.np", out)

    def test_preserves_already_short_markdown(self) -> None:
        answer = (
            "summary\n\nस्रोत:\n"
            "- [x](https://x.example.com/path)\n"
        )
        out = rewrite_sources_as_markdown(answer)
        # Already-markdown lines are untouched.
        self.assertIn("- [x](https://x.example.com/path)", out)

    def test_no_sources_block(self) -> None:
        answer = "बिना स्रोतको सामान्य जवाफ।"
        self.assertEqual(rewrite_sources_as_markdown(answer), answer)


class EnsureSourcesLineUsesShortTests(unittest.TestCase):
    def test_injected_sources_are_markdown(self) -> None:
        body = "यस बारे जानकारी।"
        urls = [
            "https://myrepublica.nagariknetwork.com/news/public-debt",
            "https://farsightnepal.com/news/debt-rises",
        ]
        out = ensure_sources_line(body, urls)
        self.assertIn("स्रोत:", out)
        self.assertIn("[myrepublica.nagariknetwork.com", out)
        self.assertIn("[farsightnepal.com", out)
        # No bare URL lines remain.
        self.assertNotIn("\n- https://", out)

    def test_injects_urls_when_header_exists_but_no_urls(self) -> None:
        """The 'स्रोत: नेपालOSINT' case — header is there but empty of URLs."""
        answer = (
            "आजका मुख्य समाचार:\n"
            "१. Neptune IPO बाँडफाँड\n"
            "२. सडक दुर्घटना\n\n"
            "[स्रोत: नेपालOSINT]"
        )
        urls = [
            "https://merolagani.com/NewsDetail.aspx?newsID=125524",
            "https://kantipur.com/news/example",
        ]
        out = ensure_sources_line(answer, urls)
        self.assertIn("[merolagani.com", out)
        self.assertIn("[kantipur.com", out)
        # Body preserved.
        self.assertIn("Neptune IPO", out)

    def test_leaves_alone_when_sources_has_real_urls(self) -> None:
        answer = (
            "Nepal को debt.\n\n"
            "स्रोत:\n- https://myrepublica.nagariknetwork.com/x"
        )
        urls = ["https://farsightnepal.com/other"]
        out = ensure_sources_line(answer, urls)
        # Did NOT replace the existing URL.
        self.assertIn("myrepublica.nagariknetwork.com/x", out)
        # Did NOT inject the unrelated URL either.
        self.assertNotIn("farsightnepal.com", out)


class GovtKeywordRoutingTests(unittest.TestCase):
    def test_decisons_misspelled_still_routes(self) -> None:
        from tools.osint.context_router import route_query
        plan = route_query("nepal govt ko naya decisons k cha")
        self.assertIn("government", plan.intents)

    def test_govt_alone(self) -> None:
        from tools.osint.context_router import route_query
        plan = route_query("govt updates today")
        self.assertIn("government", plan.intents)

    def test_decision_alone(self) -> None:
        from tools.osint.context_router import route_query
        plan = route_query("nepal ko decisions aja")
        self.assertIn("government", plan.intents)


if __name__ == "__main__":
    unittest.main()
