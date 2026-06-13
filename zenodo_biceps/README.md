# Zenodo biceps sEMG — drop-in replacement for the broken OpenBCI capture

Open biceps-brachii sEMG dataset standing in for the rig that failed to gather
usable data. Feeds the existing `convergence_analysis/core.py` forecast maths.

- **Source:** Sensors 2024, *A Comprehensive Dataset of Surface EMG and
  Self-Perceived Fatigue Levels for Muscle Fatigue Analysis* — Zenodo
  [10.5281/zenodo.14182446](https://doi.org/10.5281/zenodo.14182446), CC BY 4.0.
- **Fit:** 13 subjects, 1259 Hz raw sEMG, biceps brachii both arms, dynamic
  dumbbell curls to fatigue, 0/1/2 self-perceived-fatigue labels. Trial 5 =
  R biceps, trial 6 = L biceps (the dedicated biceps-curl fatigue trials).

## Why these numbers beat the rig
- 1259 Hz (vs the OpenBCI 250 Hz that read 1000 Hz in the header). Nyquist 629 Hz
  covers the full EMG band, so MDF / spectral-shape forecasting is valid.
- Raw unfiltered CSV provided, so we control filtering (fixes the no-filter
  drift caveat). The authors' recipe: Butterworth 20-450 Hz, 4 s window / 2 s step.

## Get the data (3.3 GB)
```bash
cd zenodo_biceps
curl -L "https://zenodo.org/records/14182446/files/sEMG_data.zip?download=1" -o sEMG_data.zip
curl -L "https://zenodo.org/records/14182446/files/self_perceived_fatigue_index.zip?download=1" -o labels.zip
unzip -q sEMG_data.zip && unzip -q labels.zip
# -> sEMG_data/Subject_1../  and  self_perceived_fatigue_index/subject_1../
```

## Run
```bash
python run_biceps.py --root sEMG_data --subject 5 --side R              # native 1259 Hz
python run_biceps.py --root sEMG_data --subject 5 --side R --target-fs 250  # mimic OpenBCI rig
```
Prints the MDF-decline slope (fatigue => negative) and the frequency-shape
forecast backtest match%, and writes `out/S{n}_{side}_biceps.png`.

## Files
- `loader.py` — CSV -> `core.Segment`; fs passed explicitly to every core call
  (never `core.FS = 1259`, which would silently mis-scale MDF). fs-aware
  `mdf_trend` because `core.mdf_trend` hardcodes FS=250 / N_CH=8.
- `run_biceps.py` — MDF trend + `core.spectrum_backtest`/`forward` on the biceps
  channel, fatigue ground-truth overlaid.

## Scope note
This set is biceps + 3 deltoid heads per arm, NOT 8 co-located biceps electrodes.
The biceps-fatigue path (MDF decline + spectrum forecast on the biceps channel)
is the right fit. The 8-channel convergence detector in `core.py` is N/A here —
confirm what the original 8 OpenBCI channels actually measured before assuming
the loader must output 8 channels.
