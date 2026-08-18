# Calibrating an athlete the model has never seen

How `classify_upload()` normalises a recording with no stored baseline, why the
first version of it was wrong, and what the constants in `classify.py` are set
from. Mirrors `zenodo_biceps/PIPELINE_HANDOVER.md` / `LSTM_HANDOVER.md` in
purpose: the method and the numbers behind one design decision.

## The problem

`train_model.py` normalises every subject against their own fresh baseline:

```python
base = X[y == 0]                 # every non-fatigue window in the recording
mu, sd = base.mean(0), base.std(0)
```

`classify()` reuses the `mu, sd` saved in `fatigue_model.pt`, so the transform
at query time is identical to the one at training time. That only works for the
13 dataset subjects. A real athlete uploading their own recording has no stored
baseline and no fatigue labels, so `mu, sd` has to be estimated from the
recording itself — `compute_fresh_baseline()`.

"Fresh" is approximated by elapsed time instead of by label: windows inside the
first `FRESH_SEC` seconds. That is reasonable for this protocol, since every
trial starts unfatigued.

## Why the first version was wrong

The original implementation used a 15 second baseline, giving 6 windows at
4 s / 2 s. `sd` is a variance estimate, and 6 samples under-estimate it. Every
z-score then inflates, the LSTM sees an out-of-distribution input, and it reads
"extreme" — which for this model means fatigue.

Scored against ground-truth labels on all 13 subjects, every window:

| baseline | accuracy | agrees with `classify()` | predicts "fatigue" | mean confidence |
|---|---:|---:|---:|---:|
| stored (`classify()`) | 0.836 | — | 0.362 | 0.926 |
| computed, 15 s | 0.750 | 0.696 | 0.657 | 0.929 |
| *ground truth* | — | — | *0.521* | — |

Three things stand out. Accuracy drops ~9 points. Nearly a third of window
labels flip. And **mean confidence does not move** — the model is exactly as
sure when it is wrong, so nothing downstream can detect the degradation.

Feeding subject 13 back through the upload path as a CSV, the two paths
disagreed on 5 of 8 windows. At t=60 s both computed an identical MDF of
60.8 Hz and still returned opposite labels, which isolates the cause to the
baseline rather than to any parsing or resampling difference.

## Choosing FRESH_SEC

Sweeping the baseline length over all 13 subjects:

| `fresh_sec` | accuracy | agreement | predicts "fatigue" |
|---:|---:|---:|---:|
| 10 | 0.648 | 0.562 | 0.769 |
| 15 | 0.750 | 0.696 | 0.657 |
| 30 | 0.753 | 0.801 | 0.520 |
| **60** | **0.811** | **0.903** | 0.393 |
| 90 | 0.758 | 0.900 | 0.308 |

60 s is the optimum. Below it there are too few windows for a stable `sd`;
above it the baseline starts absorbing genuinely fatigued windows and drags
`mu` toward the fatigued state, which suppresses detection (the fatigue rate
falls to 0.308 against a true 0.521).

Note the trend is driven by sample count, not by the purity of "fresh" — 60 s
includes some already-fatiguing windows for the faster subjects and still wins,
because a usable variance estimate matters more than a perfectly clean mean.

## The self-normalisation trap

A second failure mode is not about accuracy at all. If the baseline spans most
of the recording, the recording is normalised against itself: `mu` lands on its
own mean, every window looks average, and the verdict is meaningless whatever
the athlete's true state.

The original `len(feats) < 3` guard did not catch this. An 8 second clip
produces exactly 3 windows, passes, and gets classified — so the guard that was
supposed to reject short uploads instead guaranteed a "fresh" verdict for them.

Measured directly: taking the final 20 / 40 / 60 s of 8 subjects' recordings —
moments where both ground truth and `classify()` say *fatigued* — and feeding
each in as an upload, **6 of 24 clips came back "not fatigued"**, several above
90% confidence.

This is not fixable by tuning. A short clip does not contain a fresh reference,
so the honest response is to refuse. Hence `MIN_BASELINE_FRACTION`: the
recording must be at least 2.5x the baseline span, i.e. ~150 s minimum.

`MIN_BASELINE_FRACTION` was set from the same sweep. At 3.0 two 178 s
recordings were refused for no measurable benefit; 2.5 admits 12 of the 13
subjects with slightly better accuracy.

## Where it landed

```python
FRESH_SEC             = 60.0   # baseline window
MIN_FRESH_WINDOWS     = 12     # was 3 -- too few to estimate sd
MIN_BASELINE_FRACTION = 2.5    # recording must be >= 2.5x the baseline span
```

On the 12 subjects long enough to self-calibrate:

| | accuracy | agreement with `classify()` |
|---|---:|---:|
| stored baseline | 0.822 | — |
| self-calibrated | 0.796 | 0.895 |

A ~2.6 point accuracy cost and ~90% label agreement, versus ~9 points and 70%
before. Subject 13, the held-out demo subject, went from 49% to 88% agreement.

## What this means for a demo

The self-calibrated path is **still measurably worse** than the calibrated one,
so `classify_upload()` returns a `calibration` block and the frontend prints a
provenance line under every uploaded-file answer. Claiming parity between the
two paths would not be supportable.

For the honest headline numbers, quote `classify()` on the dataset subjects.
The upload path is the demonstration that the "uncalibrated athlete" case is
handled, not a second accuracy result.

## Reproducing

```bash
python models/test_classify.py        # includes the agreement regression guard
```

`test_upload_tracks_calibrated_path()` fails if mean agreement drops below 80%,
which is what stops `FRESH_SEC` being quietly shortened again.

## Open follow-up

The proper fix is a short dedicated calibration recording — ask the athlete for
~60 s of fresh contraction once, store their `mu, sd` the way training does for
the 13 subjects, and reuse it for every later session. That removes the
estimation problem entirely rather than bounding it, and matches how the
system would work with the team's own OpenBCI rig.
