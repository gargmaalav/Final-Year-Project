# Zenodo Biceps Pipeline Handover

Date: 2026-06-13

## What Changed

The classifier pipeline was updated to improve results without weakening the
validation method.

- Added majority/purity-based window labels. A window is labelled by the
  dominant fatigue label inside that window, and mixed windows can be dropped.
- Added transition-boundary cleanup using `--transition-margin-sec`. The current
  reported experiments use a 4 second margin around label changes.
- Added causal temporal fatigue features using only current and past windows:
  previous-window delta, rolling-history delta, and rolling slope.
- Added `--target-fs 250` support to the classifier so public Delsys data can be
  downsampled before feature extraction to mimic the OpenBCI/HYFY 250 Hz target.
- Kept validation as leave-one-subject-out (LOSO), so no windows from the test
  subject appear in training.
- Replaced the hard-coded MDF/forecast summary with live recomputation in
  `make_summary.py`.
- Added `make_classifier_report.py` to create a compact results table from saved
  metrics JSON files.

## Best Current Results

All results use LOSO validation and subject-relative fresh-baseline
normalisation.

| Task | FS | Features | Best model | Accuracy | Macro-F1 |
|---|---:|---|---|---:|---:|
| 3-class Fresh / Transition / Fatigued | 250 Hz | temporal, 4 s transition margin | RF | 0.705 | 0.643 |
| 3-class Fresh / Transition / Fatigued | 1259 Hz | temporal, 4 s transition margin | RF | 0.731 | 0.675 |
| Binary Non-fatigue / Fatigue | 250 Hz | temporal, transition dropped, 4 s margin | RF | 0.896 | 0.854 |
| Binary Non-fatigue / Fatigue | 1259 Hz | temporal, transition dropped, 4 s margin | RF | 0.910 | 0.876 |

The most defensible headline result is the 250 Hz binary classifier:

> Non-fatigue vs fatigue classification reaches 89.6% LOSO accuracy after
> downsampling to 250 Hz, using causal temporal features and honest
> subject-held-out validation.

The 3-class task improved but is still below the original 75% target at 250 Hz.
That is likely because the transition label is inherently ambiguous and
self-reported rather than a clean electrophysiological boundary.

## MDF And Forecast Summary

The MDF analysis still supports fatigue physiology:

- Native 1259 Hz: 11/13 subjects show negative MDF slope.
- Downsampled 250 Hz: 11/13 subjects show negative MDF slope.

The spectrum forecast should be framed cautiously:

- Native 1259 Hz: forecast beats the naive repeat-last baseline in 4/13 subjects.
- Downsampled 250 Hz: forecast beats baseline in 2/13 subjects.

This means the forecast is useful as a visualisation/demo of spectral trajectory,
but it should not be presented as the strongest quantitative ML result.

## How To Reproduce

From the repository root:

```powershell
python zenodo_biceps\classify_biceps.py --root zenodo_biceps\sEMG_data --side R --target-fs 250 --label-mode 3class --json-out zenodo_biceps\out\metrics_3class_250hz_static.json

python zenodo_biceps\classify_biceps.py --root zenodo_biceps\sEMG_data --side R --target-fs 250 --label-mode 3class --temporal --transition-margin-sec 4 --json-out zenodo_biceps\out\metrics_3class_250hz_temporal_m4.json

python zenodo_biceps\classify_biceps.py --root zenodo_biceps\sEMG_data --side R --target-fs 250 --label-mode binary_drop_transition --temporal --transition-margin-sec 4 --json-out zenodo_biceps\out\metrics_binary_250hz_temporal_m4.json

python zenodo_biceps\make_summary.py --root zenodo_biceps\sEMG_data --side R
python zenodo_biceps\make_summary.py --root zenodo_biceps\sEMG_data --side R --target-fs 250
python zenodo_biceps\make_classifier_report.py
```

The raw/extracted dataset is intentionally ignored by Git.

## Current Problems

- The public dataset is biceps/dynamic dumbbell curls, while the final project
  target is forearm grip using OpenBCI/HYFY-style 250 Hz hardware.
- Subject 6 is too short and becomes single-class after windowing, so it is
  skipped for classifier validation.
- Some subjects have sparse transition labels, which makes 3-class performance
  unstable.
- The transition class is self-reported and does not necessarily line up exactly
  with EMG feature changes.
- The forecast model usually does not beat a naive baseline, especially after
  downsampling.
- The code currently reports model performance, but does not yet export a final
  trained deployment model.

## Sensible Next Steps

1. Treat binary fatigue detection as the main robust result for now.
2. Keep 3-class as an experimental result unless better labels or a better
   transition definition are developed.
3. Try label alternatives honestly:
   - transition merged into fatigue,
   - transition merged into fresh,
   - larger transition margins,
   - ordinal/regression-style fatigue score instead of hard 3-class labels.
4. Add feature-importance reporting for the best RF models so the report can
   explain which signals matter most.
5. Add a saved-model export path after the model configuration is chosen.
6. When real forearm-grip data is collected, keep the same LOSO/participant-held
   validation rule and compare against the public-dataset baseline.
7. Build a small `requirements.txt` or environment note so teammates can run the
   pipeline without missing `scikit-learn`.
