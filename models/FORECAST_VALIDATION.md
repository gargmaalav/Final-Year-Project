# Forecasting fatigue with deep learning

The "prediction using deep learning" half of the supervisor's brief: what was
measured, why the task had to be reframed before any model was trained, and
what the numbers actually support. Same purpose as
`CALIBRATION_VALIDATION.md` — the method and the evidence behind one design
decision.

## The task as stated is not solvable

The brief asks for predicting future *signal values*. Before training anything,
that assumption was tested directly on subject 13 (218 s, 250 Hz) with an
AR(10) model fitted on the first half and scored on the held-out second half:

| target | horizon | R² |
|---|---|---:|
| raw signal | 4 ms (1 sample) | **+0.518** |
| raw signal | 100 ms | **−0.000** |
| raw signal | 1 s | **−0.001** |
| RMS envelope | 100 ms | +0.595 |
| RMS envelope | 1 s | −0.430 |

Autocorrelation of the bandpassed signal: **+0.084 at 20 ms, +0.002 at 100 ms,
+0.006 at 500 ms.**

A negative R² means *worse than predicting the mean*. The +0.518 at one sample
is the 20–450 Hz Butterworth's own smoothing, not physiology — it vanishes by
100 ms. Surface EMG is band-limited stochastic noise modulated by activation:
there is no structure beyond a few milliseconds for any architecture to learn.

The quantity underneath it *is* forecastable. Median frequency declines
steadily and significantly as the muscle fatigues (subject 13: −2.68 Hz/min,
R²=0.40, p<0.001). So the task here is **predict the fatigue level ahead**, not
the waveform — which is also the question the chatbot is actually asked.

Measured on one subject; the autocorrelation result is a property of the
filter and the signal class, not of that recording.

## Task definition

Given a causal sequence of the last 20 feature windows (40 s of history) ending
at time *t*, predict **MDF(t+h) − MDF(t)** for h ∈ {10, 30, 60} s.

Predicting the *change* rather than the absolute level is what makes it
transferable. These subjects sit anywhere from 53 to 78 Hz mean MDF, so a model
asked for absolute values would have to guess a held-out athlete's baseline,
which it never sees.

Features are the same 8 the classifier uses (RMS/MAV/WL/VAR/ZC/SSC/MDF/MNF),
z-scored against the recording's own first 60 s — the same label-free
fresh-baseline transform as `classify_upload()`, so the forecaster is
deployable on an uploaded recording. Elapsed time is appended as a 9th feature.

## Baselines

All evaluated causally, on identical samples:

- **persistence** — predict no change. The bar anything must clear.
- **ols_full** — fit OLS on all MDF history up to *t*, extrapolate. **This is
  what `fatigue_forecast.py` shipped.**
- **ols_recent** — same, fitted on the last 60 s only.
- **drift** — mean Hz/s slope of the training subjects × h. A one-number model.

## Results

Leave-one-subject-out over 12 subjects (subject 6 is too short), 1227 samples,
LSTM retrained per fold and averaged over 3 seeds, early-stopped on an inner
validation split drawn from training subjects only.

**MAE in Hz, lower is better:**

| method | 10 s | 30 s | 60 s |
|---|---:|---:|---:|
| persistence | 3.05 | 3.34 | 4.31 |
| ols_full *(shipped)* | 3.14 | 4.27 | 6.52 |
| ols_recent | 2.91 | 3.98 | 6.36 |
| drift | 3.04 | 3.30 | 4.01 |
| **lstm** | **2.91** | **3.32** | **4.08** |

**Skill vs persistence, and paired Wilcoxon p:**

| method | 10 s | 30 s | 60 s |
|---|---:|---:|---:|
| ols_full | −2.9% (p=0.850) | **−27.8% (p=0.003)** | **−51.3% (p<0.001)** |
| ols_recent | +4.6% (p=0.233) | **−18.9% (p=0.005)** | **−47.6% (p<0.001)** |
| drift | +0.4% (p=0.266) | +1.4% (p=0.970) | +7.0% (p=0.424) |
| lstm | **+4.6% (p=0.027)** | +0.8% (p=0.519) | +5.4% (p=0.424) |

Window-to-window MDF jitter is **1.96 Hz**. No forecaster can beat that floor,
and persistence at 10 s is already 3.05 Hz — the headroom is about 1 Hz.

## What this shows

**1. The shipped forecast was worse than doing nothing.** Extrapolating a
full-history OLS line is significantly worse than assuming no change at 30 s
(−28%, p=0.003) and 60 s (−51%, p<0.001). MDF drops steeply early then
flattens; a line fitted over the whole recording keeps projecting the early
slope long after the signal has levelled off. This is the most useful result
here, and it is a defect in code that was already in front of users.

**2. The LSTM gives a small but real improvement — at 10 s only.** +4.6%,
p=0.027. At 30 s and 60 s it is statistically indistinguishable from
persistence.

**3. Deep learning did not beat a simple approach, and that was expected.**
1227 samples across 12 subjects is thin. The honest reading is that the
sequence model *matches* the best simple method everywhere and beats it
narrowly at short range. It is used because it is never worse than persistence,
whereas the OLS line is much worse — not because it is dramatically better.

This mirrors the classification finding in `LSTM_HANDOVER.md`: the LSTM matches
Random Forest using 8 raw features instead of 32 hand-crafted ones. The value
is in what it removes, not in a higher number.

## What shipped

`fatigue_forecast.forecast_fatigue()` now:

- keeps the OLS **slope / R² / p-value** — those correctly *describe* the
  observed trend, and that was never the broken part
- replaces the **projected values** with the trained forecaster when
  `models/forecast_model.pt` is present, falling back to OLS when it is not,
  so a fresh clone still works
- reports `method` as `"lstm"` or `"ols"` so callers can tell which produced
  the numbers
- uses the **measured LOSO error** as the uncertainty band, not an OLS residual
  band — the inner band is the typical error, the outer the 95% interval
- holds the projection flat beyond 60 s. Nothing beat "no further change" at
  long range, so the model is not allowed to invent one; the band keeps
  widening.

The 0 Hz floor and `MAX_HORIZON_SEC` cap are retained as backstops for the
fallback path.

## Limitations

- **12 subjects, one muscle, one protocol.** Every subject starts fresh and
  fatigues monotonically. A recording with rest periods is out of distribution.
- **Validated to 60 s.** Beyond that the projection is held flat by
  construction and has not been tested.
- The checkpoint in `models/forecast_model.pt` is trained on **all** subjects
  and must never be scored on them. The LOSO table above is the honest
  out-of-subject estimate.
- The 10 s win (p=0.027) is uncorrected for testing three horizons. Under a
  Bonferroni correction (α=0.017) it would not clear. It is reported as a
  small real effect, not a strong one.

## Reproducing

```bash
# LOSO evaluation (~15 min on CPU)
python models/forecast_lstm.py --root zenodo_biceps/sEMG_data --seeds 3 \
    --json-out zenodo_biceps/out/metrics_forecast_lstm_250hz.json

# retrain the deployable checkpoint
python models/forecast_lstm.py --root zenodo_biceps/sEMG_data \
    --horizons 10 20 30 45 60 --skip-loso --save-model models/forecast_model.pt

# guards, including the causality check
python models/test_classify.py
```

`test_forecaster_cannot_see_the_future()` asserts that truncating the recording
at the query time changes the prediction by **0.0000 Hz**. That is the guard
that stops a future-leaking forecaster from producing impressive nonsense —
the same class of error as the time-ordered-split artifact documented in
`LSTM_HANDOVER.md`.

## For the supervisor

Two things worth raising:

1. **The target should be restated** from "predict future signal values" to
   "predict future spectral/fatigue features", with the autocorrelation table
   above as the justification.
2. **The dataset limits what deep learning can show here.** ~1200 samples is
   not enough for a sequence model to pull away from a well-chosen baseline.
   The result that *does* matter is that the previous forecasting method was
   measurably worse than assuming nothing changes, which would not have been
   found without building this benchmark.
