"""How accurate is "when did they start fatiguing?"

    python scripts/validate_onset.py

The chatbot answers that question by scanning a recording and reporting the
first time fatigue appears and holds. That answer was shipped without ever
being checked, and it was being presented as accurate to the scan step (5 s),
which is precision, not accuracy.

The dataset carries its own ground truth: loader.fatigue_onsets() returns the
end of the labelled non-fatigue span, which is the paper's own marker for when
fatigue begins. So the estimate can simply be scored against it.

Three findings drove code changes:

  - +/-5 s was an overclaim. analysis.ONSET_TYPICAL_ERROR_SEC now carries the
    measured figure and the answer quotes that instead.
  - Subject 12 was the worst case by far: predicted 0.0 s against a labelled
    102 s, because the very first scanned window already reads as fatigued.
    find_onset() now reports that as "fatigued from the start" rather than as
    an onset time, since these protocols begin fresh and a time of 0 s is
    almost certainly wrong rather than merely imprecise.
  - The remaining error was almost all late bias, and it was the scan step
    causing it, not the sustain count. scripts/tune_onset.py swept both;
    halving the step to 2.5 s more than halved the MAE.

    original                   n=12, MAE 19.3 s, 8/12 within +/-15 s
    after the first two fixes  n=11, MAE 11.8 s, 8/11, mean +9.0 s
    after the step change      n=11, MAE  5.3 s, 10/11, mean +2.9 s

Current (right arm, 2.5 s scan step, 2 sustained windows):

    n = 11 subjects an onset is reported for
    mean error   +2.9 s
    median       +1.0 s
    MAE           5.3 s
    within +/-15 s  10/11

The tail that used to dominate is gone: subjects 1 and 9 were 24-36 s late and
now sit within 2 s. Subject 8 (+20.7 s) is the only one outside 15 s.

Subject 6 is excluded: its right-arm sEMG file is 24.5 s while its label file
runs to 221 s, so the two do not describe the same recording.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in ("frontend", "models", "viz", "zenodo_biceps"):
    sys.path.insert(0, os.path.join(_ROOT, _p))

import analysis                                  # noqa: E402
import loader                                    # noqa: E402
from classify import available_subjects, load_subject_segment  # noqa: E402

DATA_ROOT = os.path.join(_ROOT, "zenodo_biceps", "sEMG_data")


def main(side: str = "R") -> int:
    print(f"onset estimate vs. labelled transition ({side} arm)\n")
    print(f"{'subj':>4}  {'predicted':>10}  {'labelled':>9}  {'error':>8}  note")

    errors = []
    for subject in available_subjects():
        label_t, label_v = loader.load_fatigue_labels(DATA_ROOT, subject, side)
        truth, _ = loader.fatigue_onsets(label_t, label_v)

        seg, fs = load_subject_segment(subject, side)
        duration = float(seg.t[-1]) if seg.t.size else 0.0
        onset = analysis.find_onset(
            analysis.scan_recording(subject, side, seg=seg, fs=fs))

        note = ""
        if truth is not None and truth > duration:
            note = f"label runs to {truth:.0f}s but recording is {duration:.0f}s"
            truth = None
        if onset.get("fatigued_from_start"):
            note = note or "fatigued at the first reading, no onset reported"

        predicted = onset["t_start"] if onset.get("found") else None
        if predicted is None or truth is None:
            print(f"{subject:>4}  {str(predicted):>10}  "
                  f"{('%.1f' % truth) if truth else '--':>9}  {'--':>8}  {note}")
            continue

        error = predicted - truth
        errors.append(error)
        print(f"{subject:>4}  {predicted:>10.1f}  {truth:>9.1f}  "
              f"{error:>+8.1f}  {note}")

    if not errors:
        print("\nno comparable subjects")
        return 1

    a = np.array(errors)
    print(f"\nn={len(a)}   mean {a.mean():+.1f}s   median {np.median(a):+.1f}s   "
          f"MAE {np.abs(a).mean():.1f}s   within +/-15s {(np.abs(a) <= 15).sum()}/{len(a)}")
    print(f"\nanalysis.ONSET_TYPICAL_ERROR_SEC is {analysis.ONSET_TYPICAL_ERROR_SEC:.0f}s "
          "-- update it if this run disagrees.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "R"))
