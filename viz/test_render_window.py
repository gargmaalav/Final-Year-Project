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

    def test_has_two_panels(self):
        # 2026-08-06 redesign: the FFT panel was dropped as too technical for
        # a non-technical viewer; only the fatigue-over-time hero and the
        # small signal snapshot remain.
        for marker in ("Is the muscle tiring?", "What the signal looks like right now"):
            self.assertIn(marker, self.html, f"panel marker missing: {marker}")

    def test_fft_panel_is_gone(self):
        for marker in ("FFT spectrum", "Frequency mix", "Frequency (Hz)"):
            self.assertNotIn(marker, self.html, f"dropped FFT panel text still present: {marker}")

    def test_asked_marker_pins_queried_time(self):
        self.assertIn("asked: 120s", self.html)

    def test_trend_line_present_and_declines(self):
        # S13's MDF declines with fatigue, so the direction-worded trend
        # sentence must read "slowing down", not the flat/speeding-up wording.
        self.assertIn("Overall, the signal is slowing down over time.", self.html)

    def test_payload_stays_optimized(self):
        # guards the frame-trim + basic-bundle + FFT-panel-removal size (was
        # ~9.4MB, then ~3.1MB, now ~1.2MB); a regression that re-bloats the
        # payload should trip here.
        self.assertLess(len(self.html), 2_500_000)


@_needs_data
class LabelsAbsentFallback(unittest.TestCase):
    """Regression for cad2e9c: a trial with no fatigue-label CSV must render the
    grey fallback, not crash on int(None)."""

    def test_renders_grey_fallback_without_crashing(self):
        with mock.patch.object(loader, "load_fatigue_labels",
                               lambda *a, **k: (None, None)):
            html = render_window(13, 120.0, "R")   # must not raise
        self.assertIn("no fatigue labels for this trial", html)


@_needs_data
class ModelPredictionOverlay(unittest.TestCase):
    """Task 2 (honesty/explainability plan): the model's own prediction renders
    as a distinct series from the ground-truth dots, so a viewer can see where
    the tool agreed/disagreed instead of only ever seeing ground truth."""

    def test_model_preds_add_a_named_trace(self):
        # two windows' worth of predictions, keyed by MDF window-centre time
        preds = {2.0: 0, 122.0: 1}
        html = render_window(13, 120.0, "R", model_preds=preds)
        self.assertIn("Tool's own guess", html)

    def test_no_model_preds_is_backward_compatible(self):
        # existing callers (no model_preds arg) must keep working unchanged
        html = render_window(13, 120.0, "R")
        self.assertIsInstance(html, str)
        self.assertGreater(len(html), 0)


@_needs_data
class DisagreementCallout(unittest.TestCase):
    """Task 3: model_preds reach the JS payload so the select-inspect readout
    can flag a disagreement between the model's guess and ground truth."""

    def test_model_preds_embedded_in_js_payload(self):
        preds = {2.0: 1}
        html = render_window(13, 0.0, "R", model_preds=preds)
        self.assertIn('"model"', html)


@_needs_data
class ReliabilityInTitle(unittest.TestCase):
    """Task 4: the static header above the chart states the deployed model's
    reliability tier for the subject being viewed (moved out of Plotly's own
    title in the 2026-08-06 redesign, same information)."""

    def test_held_out_subject_shows_tier_in_title(self):
        html = render_window(13, 120.0, "R")
        self.assertIn("Highly reliable", html)

    def test_training_subject_shows_disclaimer_in_title(self):
        html = render_window(5, 120.0, "R")
        self.assertIn("Not independently tested for this person", html)


@_needs_data
class PlainLanguageCopy(unittest.TestCase):
    """Task 5: no Hz/min or raw-EMG jargon in headline chart text (numbers may
    still live in hover tooltips/legend colours, just not the legend text)."""

    def test_no_hz_per_min_jargon_in_output(self):
        html = render_window(13, 120.0, "R")
        self.assertNotIn("Hz/min", html)

    def test_no_raw_emg_jargon_in_panel_title(self):
        html = render_window(13, 120.0, "R")
        self.assertNotIn("raw EMG", html)


if __name__ == "__main__":
    unittest.main(verbosity=2)
