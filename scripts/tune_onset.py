"""What should the onset scan step and sustain count actually be?

    python scripts/tune_onset.py

analysis.SCAN_STEP_SEC = 5 and SUSTAIN_WINDOWS = 2 were picked by judgement,
not measurement: a 5 s step felt like a reasonable latency/resolution trade,
and requiring 2 consecutive fatigued readings felt like enough to reject an
isolated flip from an ~88%-accurate classifier. Both are defensible guesses.
Neither had been checked against the dataset's own labels.

scripts/validate_onset.py scores one setting. This sweeps them, using the same
ground truth (loader.fatigue_onsets: the end of the labelled non-fatigue span)
and the same exclusions, so the numbers are directly comparable.

What the settings trade against each other:

  - a smaller step means finer resolution, but it also means the `sustain`
    windows span less time, so a shorter noise burst can satisfy them
  - a larger sustain rejects more noise but necessarily reports fatigue LATE,
    since it waits for confirmation before calling it

The reported error is signed on purpose. A setting that reports every onset
30 s late has the same MAE as one that scatters +/-30 s, and they are not
equally good: a consistent late bias is a threshold problem, scatter is not.
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

STEPS = [2.5, 5.0, 10.0]
SUSTAINS = [1, 2, 3, 4]


def _truth(subject: int, side: str, duration: float):
    """The labelled transition, or None when it can't be compared.

    Subject 6's right-arm sEMG file is 24.5 s while its label file runs to
    221 s -- the two do not describe the same recording, so scoring against it
    would be measuring a data inconsistency, not the estimator.
    """
    label_t, label_v = loader.load_fatigue_labels(DATA_ROOT, subject, side)
    truth, _ = loader.fatigue_onsets(label_t, label_v)
    if truth is None or truth > duration:
        return None
    return truth


def main(side: str = "R") -> int:
    subjects = available_subjects()

    # Scan once per (subject, step). The scan is the expensive part; every
    # sustain value is then just a different read of the same flags.
    print(f"scanning {len(subjects)} subjects at {len(STEPS)} step sizes...")
    scans: dict[tuple[int, float], dict] = {}
    truths: dict[int, float | None] = {}
    for subject in subjects:
        seg, fs = load_subject_segment(subject, side)
        duration = float(seg.t[-1]) if seg.t.size else 0.0
        truths[subject] = _truth(subject, side, duration)
        for step in STEPS:
            scans[(subject, step)] = analysis.scan_recording(
                subject, side, seg=seg, fs=fs, step=step)

    print(f"\nonset error by setting ({side} arm), n = subjects an onset is "
          "reported for\n")
    print(f"{'step':>5}  {'sustain':>7}  {'n':>3}  {'mean':>7}  {'median':>7}  "
          f"{'MAE':>6}  {'<=15s':>6}  {'from-start':>10}  {'none':>4}")

    rows = []
    for step in STEPS:
        for sustain in SUSTAINS:
            errors, from_start, none_found = [], 0, 0
            for subject in subjects:
                onset = analysis.find_onset(scans[(subject, step)], sustain=sustain)
                if onset.get("fatigued_from_start"):
                    from_start += 1
                    continue
                if not onset.get("found"):
                    none_found += 1
                    continue
                truth = truths[subject]
                if truth is None:
                    continue
                errors.append(onset["t_start"] - truth)

            if not errors:
                continue
            a = np.array(errors)
            row = {
                "step": step, "sustain": sustain, "n": len(a),
                "mean": a.mean(), "median": float(np.median(a)),
                "mae": np.abs(a).mean(),
                "within15": int((np.abs(a) <= 15).sum()),
                "from_start": from_start, "none": none_found,
            }
            rows.append(row)
            current = " <- current" if (step == analysis.SCAN_STEP_SEC
                                        and sustain == analysis.SUSTAIN_WINDOWS) else ""
            print(f"{step:>5.1f}  {sustain:>7}  {row['n']:>3}  {row['mean']:>+7.1f}  "
                  f"{row['median']:>+7.1f}  {row['mae']:>6.1f}  "
                  f"{row['within15']:>3}/{row['n']:<2}  {from_start:>10}  "
                  f"{none_found:>4}{current}")

    # Ranked on MAE, but "fatigued from the start" is a non-answer, not a free
    # pass -- a setting that reports nothing for half the subjects can post a
    # flattering MAE on the rest, so the count is shown alongside.
    best = min(rows, key=lambda r: r["mae"])
    print(f"\nlowest MAE: step {best['step']:.1f}s, sustain {best['sustain']} "
          f"-- MAE {best['mae']:.1f}s, mean {best['mean']:+.1f}s, "
          f"{best['within15']}/{best['n']} within 15s, "
          f"{best['from_start']} reported as fatigued from the start")
    print(f"currently in analysis.py: step {analysis.SCAN_STEP_SEC:.1f}s, "
          f"sustain {analysis.SUSTAIN_WINDOWS}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "R"))
