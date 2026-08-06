# Chatbot Viz Honesty + Explainability Implementation Plan

> For agentic workers: implement this plan task-by-task with a COMMIT after each task (per CLAUDE.md agent-spawning discipline: subagent-per-task for independent tasks, inline for tightly coupled ones). Steps use checkbox (- [ ]) syntax for tracking.

Goal: Fix a real honesty defect in the existing chatbot chart (it currently shows self-reported ground-truth fatigue colours as if they were the model's prediction), then layer in three plain-language, non-technical-facing UI additions (forecast range, "why we think this", reliability tier) - all built to actually match what the deployed model can support, not what would be nice to show.

Architecture: All changes live in viz/render_window.py (the 3-panel Plotly chart) and models/serve.py / models/classify.py (the FastAPI bridge + inference). No new files for panels - extend the existing chart via new traces/annotations. One new small static lookup module for per-subject reliability tiers. No dataset or model retraining today - this plan works within the deployed binary LSTM (models/fatigue_model.pt, trained on subjects 1-11 per models/train_model.py:77) and the existing ground-truth CSVs.

Tech Stack: Python, Plotly (go.Scatter, go.Frame), FastAPI, existing loader/core modules from zenodo_biceps/ and convergence_analysis/.

## Global Constraints

- Non-technical audience: no "MDF", "FFT", "confidence interval", "LOSO", "accuracy %", "Hz/min" as headline text. Plain sentences only. Numbers may live in hover tooltips.
- Dataset is fixed: Zenodo biceps only (1259 Hz native), subjects 1-13, no new data collection.
- The deployed model (models/fatigue_model.pt, loaded by models/classify.py) is binary only (fatigue_label: 0 = non-fatigue, 1 = fatigue - models/classify.py:66), trained on subjects 1-11 only (models/train_model.py:69,77). It is NOT the 3-class sklearn model from zenodo_biceps/classify_biceps.py (91%/73.1% headline numbers) - those numbers describe a different, undeployed model and must never be quoted as "the chatbot's accuracy."
- Forecast/trend line beyond the recording's own span is a genuine forecast and only beats a naive baseline in 4 of 13 subjects - must never be shown with confident/future-tense copy.
- convergence_analysis/forecast.py and core.FS reassignment are both off-limits (unfiltered OpenBCI drift data / the fs trap - see root CLAUDE.md). Every call in this plan passes fs explicitly.
- No new left-sidebar/form fields. Chart stays chat-embedded HTML only.

---

## File Structure

- Modify: viz/render_window.py - add model-prediction series, disagreement callout, reliability tier text, forecast band, explainability sentence; fix existing jargon (Overall trend: Hz/min legend, "raw EMG" panel title).
- Modify: viz/test_render_window.py - tests for each addition.
- Create: viz/reliability_tiers.py - static per-subject reliability lookup (own file: it's a data table + one lookup function, not chart-rendering logic).
- Modify: models/serve.py - clamp out-of-range subject/t_start to a graceful error instead of a 500/KeyError; /classify must also return the model's prediction at the requested point for the new comparison series (it already computes this - expose it).
- Modify: models/openwebui_tool_reference.py - default every get_fatigue param including subject, add two example sentences to the docstring (Open WebUI builds the LLM's function schema from it).

---

### Task 1: Compute deployed-model reliability tiers from real LOSO/held-out numbers

Files:
- Create: viz/reliability_tiers.py
- Test: viz/test_reliability_tiers.py

Interfaces:
- Consumes: nothing (static table, hand-populated from the held-out eval already run: S12: acc=0.854, S13: acc=0.976, subjects 1-11 were used to train the deployed LSTM so no fair held-out score exists for them).
- Produces: reliability_tier(subject: int) -> str returning one of "Highly reliable", "Somewhat reliable", "Not independently tested for this person" - used by Task 4.

This directly implements the fable review's salvage of idea 3: tiers are only meaningful for subjects 12 and 13 (the two the deployed model never trained on). For subjects 1-11, showing any tier framed as "how well the model generalizes" would be a training-set score dressed as a generalization claim - so those get an honest disclaimer string instead of a fabricated tier.

- [ ] Step 1: Write the failing test

```python
# viz/test_reliability_tiers.py
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reliability_tiers import reliability_tier


class ReliabilityTiers(unittest.TestCase):
    def test_held_out_subject_13_is_highly_reliable(self):
        # S13 acc=0.976 in the real held-out eval (models/eval_deployed.py)
        self.assertEqual(reliability_tier(13), "Highly reliable")

    def test_held_out_subject_12_is_somewhat_reliable(self):
        # S12 acc=0.854
        self.assertEqual(reliability_tier(12), "Somewhat reliable")

    def test_training_subject_gets_disclaimer_not_a_tier(self):
        # Subjects 1-11 trained the deployed model (models/train_model.py:77) -
        # no fair held-out score exists for them.
        self.assertEqual(
            reliability_tier(5),
            "Not independently tested for this person",
        )

    def test_out_of_range_subject_raises(self):
        with self.assertRaises(ValueError):
            reliability_tier(99)


if __name__ == "__main__":
    unittest.main()
```

- [ ] Step 2: Run test to verify it fails

Run: .venv/bin/python -m unittest viz.test_reliability_tiers -v
Expected: FAIL with ModuleNotFoundError: No module named 'reliability_tiers'

- [ ] Step 3: Write minimal implementation

```python
# viz/reliability_tiers.py
"""Per-subject reliability tier for the DEPLOYED binary LSTM
(models/fatigue_model.pt, trained on subjects 1-11 per models/train_model.py:77).

Subjects 1-11 trained the model - there is no fair held-out score for them, so
they get an honest disclaimer instead of a fabricated "reliable" label. Only
subjects 12 and 13 were held out and independently evaluated
(models/eval_deployed.py): S12 acc=0.854, S13 acc=0.976.

Do NOT confuse this with the 91%/73.1% LOSO numbers from
zenodo_biceps/classify_biceps.py - those describe a different, undeployed
sklearn 3-class model. This module is about the model actually answering
chatbot queries.
"""
from __future__ import annotations

ALL_SUBJECTS = range(1, 14)

TRAIN_SUBJECTS = frozenset(range(1, 12))  # mirrors models/train_model.py:77

_HELD_OUT_ACC = {12: 0.854, 13: 0.976}   # models/eval_deployed.py, live-verified

DISCLAIMER = "Not independently tested for this person"


def reliability_tier(subject: int) -> str:
    """Plain-language reliability label for the deployed model on this subject.

    Highly reliable    - held-out accuracy >= 0.90
    Somewhat reliable   - held-out accuracy >= 0.70 and < 0.90
    disclaimer string   - subject trained the model (1-11); no fair score exists
    """
    if subject not in ALL_SUBJECTS:
        raise ValueError(f"subject must be 1-13, got {subject}")
    if subject in TRAIN_SUBJECTS:
        return DISCLAIMER
    acc = _HELD_OUT_ACC[subject]
    if acc >= 0.90:
        return "Highly reliable"
    if acc >= 0.70:
        return "Somewhat reliable"
    return "Less reliable for this person"
```

- [ ] Step 4: Run test to verify it passes

Run: .venv/bin/python -m unittest viz.test_reliability_tiers -v
Expected: PASS (4 tests)

- [ ] Step 5: Commit

Stage viz/reliability_tiers.py and viz/test_reliability_tiers.py, commit message: "feat(viz): add per-subject reliability tier lookup"

---

### Task 2: Expose the model's own prediction alongside ground truth on the MDF panel

Files:
- Modify: models/serve.py - /render endpoint (find with grep -n "def render\|/render" models/serve.py)
- Modify: viz/render_window.py:581-608 (render_window function signature + _chart_html)
- Test: viz/test_render_window.py

Interfaces:
- Consumes: models.classify.classify(subject, t_start, side) -> dict with keys mdf_hz, fatigue_label (0/1), confidence (already exists, models/classify.py:56-121).
- Produces: render_window(subject, t_start, side, model_preds: dict[float, int] | None = None) - model_preds maps each MDF-window centre time (matching mdf_t from loader.mdf_trend) to the model's binary prediction (0/1) at that window. When None (e.g. called standalone without serve.py), the new series is simply omitted - this keeps viz/test_render_window.py's existing tests, which call render_window() directly, working unmodified.

This is the fable review's top suggestion (C1): today the coloured dots on panel 2 are 100% ground-truth labels (_dominant_label(), render_window.py:276-292), and a non-technical user reading green/orange/red dots next to the chatbot's answer reasonably assumes those dots ARE the model's prediction. They are not. Adding the model's own binary call as a second, visually distinct series (small X markers, not filled dots) makes the tool's real accuracy visible instead of implied.

- [ ] Step 1: Write the failing test

```python
# add to viz/test_render_window.py, inside a new class

class ModelPredictionOverlay(unittest.TestCase):
    """Task 2: the model's own prediction renders as a distinct series from
    the ground-truth dots, so a viewer can see where the tool agreed/disagreed."""

    @_needs_data
    def test_model_preds_add_a_named_trace(self):
        # two windows' worth of predictions, keyed by MDF window-centre time
        preds = {2.0: 0, 122.0: 1}
        html = render_window(13, 120.0, "R", model_preds=preds)
        self.assertIn("Tool's own guess", html)

    @_needs_data
    def test_no_model_preds_is_backward_compatible(self):
        # existing callers (no model_preds arg) must keep working unchanged
        html = render_window(13, 120.0, "R")
        self.assertIsInstance(html, str)
        self.assertGreater(len(html), 0)
```

- [ ] Step 2: Run test to verify it fails

Run: .venv/bin/python -m unittest viz.test_render_window.ModelPredictionOverlay -v
Expected: FAIL - render_window() got an unexpected keyword argument 'model_preds'

- [ ] Step 3: Write minimal implementation

In viz/render_window.py, change the signature and thread model_preds down to _chart_html:

```python
def render_window(subject: int, t_start: float, side: str = "R",
                  model_pred: dict | None = None,
                  model_preds: dict[float, int] | None = None) -> str:
    """... (existing docstring stays; append:)

    model_preds: optional {window_centre_time: predicted_label} from the
    DEPLOYED model (models/classify.py), one entry per MDF window. When given,
    renders as a second series ("Tool's own guess") distinct from the
    ground-truth dots, so the chart shows where the model agrees/disagrees
    with the subject's self-report instead of only ever showing ground truth.
    """
    side = _validate_side(side)
    if subject is None:
        raise ValueError("subject is required (the all-subjects overview was removed)")
    if subject not in ALL_SUBJECTS:
        raise ValueError(f"subject must be 1-13, got {subject}")

    seg, fs, lab_t, lab_v = _load_subject(subject, side)
    t_start = min(max(float(t_start), 0.0), float(seg.t[-1]))
    return _chart_html(
        seg, fs, t_start,
        chart_label=f"S{subject} {side} biceps",
        length_tag=f"subject {subject} recording",
        title_prefix=f"Subject {subject} ({side} biceps)",
        lab_t=lab_t, lab_v=lab_v, model_preds=model_preds)
```

Add model_preds=None to _chart_html's signature (render_window.py:316-318) and, right after the existing ground-truth loop that adds panel-2 dots (render_window.py:432-439), add:

```python
    # --- model's own prediction, as a distinct series from ground truth ---
    # Small X markers (not filled dots) so it reads visually as "a guess",
    # never mistaken for the same kind of mark as the ground-truth dots.
    if model_preds:
        mp_t, mp_v, mp_lab = [], [], []
        for i, tc in enumerate(mdf_t):
            key = min(model_preds.keys(), key=lambda k: abs(k - float(tc)))
            if abs(key - float(tc)) <= STEP_SEC:  # only plot a genuine match
                mp_t.append(float(tc)); mp_v.append(float(mdf_v[i]))
                mp_lab.append(int(model_preds[key]))
        if mp_t:
            mp_colors = ["#e74c3c" if l else "#2ecc71" for l in mp_lab]
            fig.add_trace(go.Scatter(
                x=mp_t, y=mp_v, mode="markers", name="Tool's own guess",
                marker=dict(size=9, symbol="x", color=mp_colors,
                            line=dict(width=1.5, color=mp_colors))),
                row=2, col=1)
```

In models/serve.py, find the /render handler and, before calling render_window, build model_preds by calling classify.classify() once per MDF window centre (reuse loader.mdf_trend the same way render_window.py does - same fs, WIN_SEC, STEP_SEC constants):

```python
# models/serve.py, inside the /render handler, before render_window(...):
import loader as _loader  # already imported elsewhere in serve.py - reuse that import
seg, fs, _, _ = viz.render_window._load_subject(subject, side)  # or equivalent already-loaded seg/fs
mdf_t, _, _ = _loader.mdf_trend(seg, fs=fs, win_sec=viz.render_window.WIN_SEC,
                                 step_sec=viz.render_window.STEP_SEC)
model_preds = {}
for tc in mdf_t:
    try:
        r = classify.classify(subject, float(tc), side)
        model_preds[float(tc)] = r["fatigue_label"]
    except Exception:
        pass  # a single window's failure must not blank the whole overlay
html = render_window(subject, t_start, side, model_preds=model_preds)
```

- [ ] Step 4: Run test to verify it passes

Run: .venv/bin/python -m unittest viz.test_render_window -v
Expected: PASS, all tests including the two new ones

- [ ] Step 5: Commit

Stage viz/render_window.py, viz/test_render_window.py, models/serve.py, commit message: "feat(viz): overlay the deployed model's own prediction on the fatigue chart"

---

### Task 3: Plain-language disagreement callout

Files:
- Modify: viz/render_window.py (_SELECT_INSPECT JS block, render_window.py:162-261)
- Test: viz/test_render_window.py

Interfaces:
- Consumes: the model_preds data already threaded into the chart's JS payload via _select_inspect_html (Task 2's mp_lab/mp_t arrays need to reach the same __VIZ_DATA__ JSON blob _select_inspect_html already builds at render_window.py:264-273).
- Produces: an extra sentence in the existing #__viz_readout box (no new DOM element) when the selected span contains at least one window where the model's guess and the ground-truth dominant label disagree.

This is fable's C2 - falls directly out of Task 2's data. Reuses the existing box-select readout mechanism instead of adding new UI chrome.

- [ ] Step 1: Write the failing test

```python
class DisagreementCallout(unittest.TestCase):
    @_needs_data
    def test_disagreement_data_embedded_when_model_preds_given(self):
        preds = {2.0: 1}  # deliberately opposite of whatever ground truth is at t=2s
        html = render_window(13, 0.0, "R", model_preds=preds)
        self.assertIn('"model"', html)  # model predictions reach the JS payload
```

- [ ] Step 2: Run test to verify it fails

Run: .venv/bin/python -m unittest viz.test_render_window.DisagreementCallout -v
Expected: FAIL - AssertionError: '"model"' not found in html

- [ ] Step 3: Write minimal implementation

Extend _select_inspect_html's payload (render_window.py:264-273) to carry the model series:

```python
def _select_inspect_html(mdf_t, mdf_v, mdf_labels, model_lab=None) -> str:
    data = {
        "t": [round(float(v), 2) for v in mdf_t],
        "v": [round(float(v), 2) for v in mdf_v],
        "lab": [int(l) for l in mdf_labels],
        "model": [int(l) if l is not None else -1
                  for l in (model_lab if model_lab is not None else [None] * len(mdf_t))],
        "names": {str(k): v for k, v in LABEL_NAME.items()},
        "colors": {str(k): v for k, v in LABEL_COLOR.items()},
    }
    payload = json.dumps(data).replace("</", "<\\/")
    return _SELECT_INSPECT.replace("__VIZ_DATA__", payload)
```

In _chart_html, build model_lab aligned to mdf_t (reuse the same nearest-match loop from Task 2, extracted once) and pass it to _select_inspect_html(mdf_t, mdf_v, mdf_labels, model_lab) at the call site (render_window.py:578).

In the _SELECT_INSPECT JS inspect() function (render_window.py:204-229), after computing dom/name/col, add:

```javascript
    var disagreeN = 0;
    for (var m = 0; m < idx.length; m++) {
      var mv = D.model[idx[m]];
      if (mv !== -1 && mv !== D.lab[idx[m]]) disagreeN++;
    }
    var disagreeMsg = '';
    if (disagreeN > 0) {
      disagreeMsg = ' &nbsp;|&nbsp; <span style="color:#e0a030">the tool\'s own '
        + 'guess disagreed with what the person reported for ' + disagreeN
        + ' of ' + idx.length + ' moments here</span>';
    }
```

and append + disagreeMsg to the existing box.innerHTML = ... assignment.

- [ ] Step 4: Run test to verify it passes

Run: .venv/bin/python -m unittest viz.test_render_window -v
Expected: PASS

- [ ] Step 5: Commit

Stage viz/render_window.py, viz/test_render_window.py, commit message: "feat(viz): plain-language callout when model disagrees with self-report"

---

### Task 4: "How reliable is this reading" tier in the chart title

Files:
- Modify: viz/render_window.py:483-489 (_title), render_window() signature
- Test: viz/test_render_window.py

Interfaces:
- Consumes: viz.reliability_tiers.reliability_tier(subject: int) -> str (Task 1).
- Produces: chart title string includes the tier text.

- [ ] Step 1: Write the failing test

```python
class ReliabilityInTitle(unittest.TestCase):
    @_needs_data
    def test_held_out_subject_shows_tier_in_title(self):
        html = render_window(13, 120.0, "R")
        self.assertIn("Highly reliable", html)

    @_needs_data
    def test_training_subject_shows_disclaimer_in_title(self):
        html = render_window(5, 120.0, "R")
        self.assertIn("Not independently tested for this person", html)
```

- [ ] Step 2: Run test to verify it fails

Run: .venv/bin/python -m unittest viz.test_render_window.ReliabilityInTitle -v
Expected: FAIL - tier text not in title

- [ ] Step 3: Write minimal implementation

Add the import near the top of viz/render_window.py:

```python
from reliability_tiers import reliability_tier
```

Thread subject into _chart_html (it already has chart_label/title_prefix derived from subject at the render_window() call site - add a plain subject: int | None = None param to _chart_html, None for render_segment's uploaded-recording path where no subject/tier applies) and update _title:

```python
    def _title(k):
        tier_txt = f"  |  {reliability_tier(subject)}" if subject is not None else ""
        return (f"Muscle Fatigue - {title_prefix}  |  "
                f"at {mdf_t[k]:.0f}s into the session, signal frequency {frame_mdf[k]:.1f} Hz"
                f"{tier_txt}")
```

Pass subject=subject at render_window()'s call into _chart_html (render_window.py:603-608).

- [ ] Step 4: Run test to verify it passes

Run: .venv/bin/python -m unittest viz.test_render_window -v
Expected: PASS

- [ ] Step 5: Commit

Stage viz/render_window.py, viz/test_render_window.py, commit message: "feat(viz): surface deployed-model reliability tier in chart title"

---

### Task 5: Fix the trend line's tense and remove Hz/min jargon from headline text

Files:
- Modify: viz/render_window.py:441-459 (fitted trend line + legend label), :420 (panel-1 title)
- Test: viz/test_render_window.py

Interfaces:
- No new interfaces - corrects existing headline copy the fable review flagged as jargon violations under the non-technical constraint ("Overall trend: -X.X Hz/min" and "raw EMG").

This is a same-day, low-risk fix independent of the other tasks - do it whenever convenient, but before calling any of this "non-technical-facing."

- [ ] Step 1: Write the failing test

```python
class PlainLanguageCopy(unittest.TestCase):
    @_needs_data
    def test_no_hz_per_min_jargon_in_output(self):
        html = render_window(13, 120.0, "R")
        self.assertNotIn("Hz/min", html)

    @_needs_data
    def test_no_raw_emg_jargon_in_panel_title(self):
        html = render_window(13, 120.0, "R")
        self.assertNotIn("raw EMG", html)
```

- [ ] Step 2: Run test to verify it fails

Run: .venv/bin/python -m unittest viz.test_render_window.PlainLanguageCopy -v
Expected: FAIL - both jargon strings present

- [ ] Step 3: Write minimal implementation

render_window.py:452-453, replace:

```python
        _trend_lbl = ("Overall trend: roughly flat" if abs(_slope_hz_min) < 0.05
                      else f"Overall trend: {_slope_hz_min:+.1f} Hz/min")
```
with:
```python
        _direction = ("staying about the same" if abs(_slope_hz_min) < 0.05
                      else "slowing down over time" if _slope_hz_min < 0
                      else "speeding up over time")
        _trend_lbl = f"Overall: the signal is {_direction}"
```
(the precise _slope_hz_min value is still available for a hover tooltip if wanted later - just not in the legend text.)

render_window.py:420, replace:
```python
            f"{chart_label}: the muscle's raw signal right now (raw EMG, a 4-second close-up)",
```
with:
```python
            f"{chart_label}: the muscle's signal right now (a 4-second close-up)",
```

- [ ] Step 4: Run test to verify it passes

Run: .venv/bin/python -m unittest viz.test_render_window -v
Expected: PASS

- [ ] Step 5: Commit

Stage viz/render_window.py, viz/test_render_window.py, commit message: "fix(viz): remove Hz/min and raw-EMG jargon from headline chart text"

---

### Task 6: Chatbot tool robustness - defaults, clamping, example sentences (drop the sidebar fields safely)

Files:
- Modify: models/openwebui_tool_reference.py:42-54 (get_fatigue docstring + signature)
- Modify: models/serve.py (/classify, /render handlers)
- Test: none (Open WebUI reference file is a paste-in copy, not imported/tested by this repo per its own header comment) - verify manually per Step 3.

Interfaces:
- Consumes: nothing new.
- Produces: get_fatigue(subject: int = 13, t_start: float = 0.0, side: str = "R") - every param now has a default so a partial LLM extraction (small local model, Legacy function-calling) still produces a valid call instead of erroring. /classify and /render return a clean 4xx JSON body instead of a 500/KeyError on out-of-range subject.

This directly addresses the fable review's flagged risk in dropping the sidebar form fields: the small local model in Legacy mode is more likely to mis-extract a param than a human filling a form, so every param needs a safe default and the server needs to fail gracefully instead of 500ing.

- [ ] Step 1: Update the tool signature + docstring

```python
# models/openwebui_tool_reference.py
    def get_fatigue(self, subject: int = 13, t_start: float = 0.0, side: str = "R"):
        """
        Get the muscle-fatigue state predicted by the EMG deep-learning model for
        a subject at a given time in their recording, with an interactive chart
        of that window. Call this whenever the user asks whether a subject is
        fatigued, or about the fatigue state at a specific time point.

        Examples:
        - "is subject 13 fatigued at 2 minutes?" -> get_fatigue(13, 120, "R")
        - "check subject 5's left arm at the start" -> get_fatigue(5, 0, "L")

        :param subject: subject id (1-13 in the dataset). Default 13 if unclear.
        :param t_start: time in seconds into the recording (e.g. 120). Default 0 if unclear.
        :param side: which arm, "R" or "L" (default "R")
        :return: (chart, summary) if the chart renders, else summary alone
        """
```

- [ ] Step 2: Clamp server-side instead of erroring

In models/serve.py, find the /classify handler (grep -n "def classify\|/classify" models/serve.py) and wrap the subject validation:

```python
from fastapi import HTTPException

# inside the /classify and /render handlers, before calling classify()/render_window():
if not (1 <= subject <= 13):
    raise HTTPException(status_code=400,
                         detail=f"subject must be 1-13, got {subject}. Try subject 13.")
if side.upper() not in ("R", "L"):
    raise HTTPException(status_code=400,
                         detail=f"side must be 'R' or 'L', got {side!r}.")
```

(The bridge already returns readable error text to the LLM via openwebui_tool_reference.py:65-66's f"ERROR {r.status_code}: {r.text}" path - this just makes that text useful instead of a raw traceback.)

- [ ] Step 3: Manual verification

Run python models/serve.py, then request GET http://localhost:8000/classify?subject=99&t_start=0&side=R
Expected: 400 with a plain-English detail message, not a 500 or unhandled traceback.

- [ ] Step 4: Commit

Stage models/openwebui_tool_reference.py, models/serve.py, commit message: "fix(chatbot): default every get_fatigue param, clamp bad input to a plain error"

---

## Deferred (not today - needs Ray's judgment)

- Forecast band (original idea 1): fable flagged real statistical problems (autocorrelated OLS residuals from 50%-overlapping windows understate the interval; extrapolating past the recording carries the 4/13 weak-forecast risk). Cheapest honest version is a NEW small panel/sentence stating the forecast plainly beats a naive guess for only about a third of people tested - not a shaded "likely range" band. Needs a decision on exact wording/placement before implementing; not blocked on anything above.
- Explainability sentence (original idea 2): needs a decision on which referent it explains (model prediction vs ground truth vs raw current-window spectrum) - Task 2's overlay makes this answerable by inspection once built, so revisit after Task 2 ships.
- Switching the deployed model to a real 3-class model, so the model's own prediction can show "Getting tired" and not just binary fresh/fatigued - bigger scope, not a today item.

---

## Self-Review

Spec coverage: Idea 3 (Task 1+4), idea 1 (deferred with reasoning, not silently dropped), idea 2 (deferred, unblocked by Task 2), fable's A1/A2 fix (Task 2+3), jargon fixes (Task 5), chatbot flow (Task 6) - all addressed or explicitly deferred with a named reason, none silently skipped.

Placeholder scan: No TBD/TODO; every step has real code referencing real line numbers from the current files.

Type consistency: model_preds: dict[float, int] is the same shape from Task 2 (produced in serve.py) through Task 3 (consumed in _select_inspect_html) - checked.
