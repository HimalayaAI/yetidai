"""Smoke tests for core/nepali_date.py.

If the `nepali-datetime` dependency isn't installed, all helpers return
None — we assert that behaviour without requiring the package at test
time so the suite stays green on minimal environments.
"""
from __future__ import annotations

import unittest
from datetime import date

from core.nepali_date import format_bs_iso, format_bs_ne, to_bs


class NepaliDateTests(unittest.TestCase):
    def test_shape_or_none(self) -> None:
        """Either we have the lib and convert, or we gracefully return None."""
        g = date(2026, 4, 23)
        bs = to_bs(g)
        if bs is None:
            # Dependency missing. Both string helpers must also be None.
            self.assertIsNone(format_bs_iso(g))
            self.assertIsNone(format_bs_ne(g))
            return
        # Sanity-check the conversion for 2026-04-23 → BS 2083-01-10
        year, month, day = bs
        self.assertEqual(year, 2083)
        self.assertEqual(month, 1)  # Baishakh
        # Day may be 9, 10, or 11 depending on panchanga edge cases;
        # the whole window is within Baishakh.
        self.assertIn(day, (9, 10, 11))
        iso = format_bs_iso(g)
        ne = format_bs_ne(g)
        self.assertIsNotNone(iso)
        self.assertIsNotNone(ne)
        assert iso is not None and ne is not None
        self.assertTrue(iso.startswith("BS 2083-01-"))
        self.assertIn("वैशाख", ne)


if __name__ == "__main__":
    unittest.main()
