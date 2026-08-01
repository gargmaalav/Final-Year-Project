"""Tests for render_window.py — the chatbot EMG-fatigue chart.

Stdlib unittest (no install needed). Run from the repo root:
    .venv/bin/python -m unittest viz.test_render_window -v
    # or, if you add pytest later:  .venv/bin/pytest viz/test_render_window.py

Two groups:
  * input-validation tests — run always (they raise before any data load);
  * data-dependent tests — need the Zenodo dataset at zenodo_biceps/sEMG_data,
    and are skipped (not failed) when it is absent.

The labels-absent test is the regression guard for the crash fixed in cad2e9c:
render_window used to raise TypeError(int(None)) on a trial with no fatigue-label
CSV instead of drawing its advertised grey "no ground-truth labels" fallback.

Trace/frame index integrity of the fatigue-trend line (that it is static and
never enters fig.frames) was verified empirically in the independent review;
these tests assert the observable behaviour of the public render_window() API.
"""
import os
import sys
import unittest
from unittest import mock

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO, "viz"),
           os.path.join(_REPO, "zenodo_biceps"),
           os.path.join(_REPO, "convergence_analysis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import loader  # noqa: E402
from render_window import render_window, DATA_ROOT  # noqa: E402

# Probe once whether the raw dataset is present, cheaply (segment load only).
try:
    loader.load_biceps_segment(DATA_ROOT, 13, "R", target_fs=250, bandpass=True)
    _HAS_DATA = True
except Exception:
    _HAS_DATA = False

_needs_data = unittest.skipUnless(_HAS_DATA, "Zenodo dataset not present at DATA_ROOT")


class InputValidation(unittest.TestCase):
    """These raise before any data is touched, so they run without the dataset."""

    def test_bad_side_raises(self):
        with self.assertRaises(ValueError):
            render_window(13, 120.0, "X")

    def test_subject_none_raises(self):
        with self.assertRaises(ValueError):
            render_window(None, 120.0, "R")

    def test_subject_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            render_window(99, 120.0, "R")


@_needs_data
class HappyPath(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.html = render_window(13, 120.0, "R")

    def test_returns_nontrivial_html(self):
        self.assertIsInstance(self.html, str)
        self.assertIn("plotly-graph-div", self.html)
        self.assertGreater(len(self.html), 100_000)

    def test_has_three_panels(self):
        for marker in ("raw EMG", "Median frequency (MDF)", "FFT spectrum"):
            self.assertIn(marker, self.html, f"panel marker missing: {marker}")

    def test_asked_marker_pins_queried_time(self):
        self.assertIn("asked: 120s", self.html)

    def test_trend_line_present_and_declines(self):
        # S13's MDF declines with fatigue, so the fitted slope must be negative;
        # the legend label reads "fatigue trend -X.X Hz/min".
        self.assertIn("fatigue trend -", self.html)

    def test_payload_stays_optimized(self):
        # guards the frame-trim + basic-bundle optimization (was ~9.4MB, now
        # ~3.1MB); a regression that re-bloats the payload should trip here.
        self.assertLess(len(self.html), 5_000_000)


@_needs_data
class LabelsAbsentFallback(unittest.TestCase):
    """Regression for cad2e9c: a trial with no fatigue-label CSV must render the
    grey fallback, not crash on int(None)."""

    def test_renders_grey_fallback_without_crashing(self):
        with mock.patch.object(loader, "load_fatigue_labels",
                               lambda *a, **k: (None, None)):
            html = render_window(13, 120.0, "R")   # must not raise
        self.assertIn("no ground-truth labels", html)


if __name__ == "__main__":
    unittest.main(verbosity=2)
