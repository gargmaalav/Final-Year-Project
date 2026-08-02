# Changes on branch `feat/chatbot-integration`

Written the same way as `CHANGES_maalav-chatbot.md`: for a teammate (or an AI
assistant) reviewing this branch's diff against `maalav-chatbot`. It explains
**why** each change exists — the code comments and the two validation docs do
the "what" and the "how much".

## Why this branch exists

`maalav-chatbot` built the dedicated Streamlit frontend the integration
contract always intended, and that decision stands — this branch keeps all of
it. What it did not have was any measurement. The upload path, the forecast,
and the query parser were each verified by "it ran and produced output", which
cannot detect a wrong answer.

Scoring them against the 13 labelled subjects found three defects, all of which
produced confident, plausible-looking, wrong output. Nothing here is a rewrite;
every change is to a specific measured failure.

## 1. The upload path was over-predicting fatigue

`classify_upload()` normalised an uploaded recording against its own first
**15 seconds**. That is 6 windows, and `sd` is a variance estimate — 6 samples
under-estimate it, every z-score inflates, and the LSTM reads "extreme", which
for this model means fatigue.

Scored on all 13 subjects, every window:

| baseline | accuracy | agrees with `classify()` | predicts fatigue | confidence |
|---|---:|---:|---:|---:|
| stored (`classify()`) | 0.836 | — | 0.362 | 0.926 |
| computed, 15 s | 0.750 | 0.696 | 0.657 | 0.929 |
| *ground truth* | — | — | *0.521* | — |

The dangerous column is the last one. **Confidence does not move.** The model
is exactly as sure when it is wrong, so nothing downstream could have caught
this.

Fixed by sweeping the baseline length: 60 s is the optimum (0.811 accuracy,
0.903 agreement). Below it there are too few windows for a stable `sd`; above
it the baseline starts absorbing genuinely fatigued windows.

A second failure was worse. The `len(feats) < 3` guard was supposed to reject
short uploads — but an 8 s clip produces exactly 3 windows, passes, and gets
normalised against itself, so every window looks "average" and the verdict is
meaningless. Measured: taking the final 20/40/60 s of 8 subjects' recordings,
moments where ground truth says *fatigued*, **6 of 24 came back "not fatigued"**,
several above 90% confidence. Short recordings are now refused, because a short
clip genuinely does not contain a fresh reference.

Full method and numbers: `models/CALIBRATION_VALIDATION.md`.

## 2. The query parser invented parameters

`parse_query("what about the left arm?")` returned `{subject: 13,
t_start: 120.0}` — numbers that appear nowhere in the question. They passed
every range check, and `build_prompt` never stated which window it used, so the
answer looked normal and described a moment the user never asked about.

Now every extracted field must be **grounded in the literal user text**. If a
number is not in the message it is carried forward from the previous turn
instead of invented, and the frontend prints a provenance line under each
answer saying which subject/time/side actually produced it.

## 3. The forecast described a different moment than the classification

`forecast_fatigue()` fitted on the whole recording and projected from its end,
while the classification described `t_start`. Ask about t=20 s and you got a
fatigue reading for 20 s next to a forecast starting at 200 s, presented as one
answer. The forecast is now anchored to the queried time.

Left unbounded, the same OLS line predicted **−106 Hz** at a one-hour horizon.
MDF is a frequency and cannot be negative, so the projection is floored at zero
and the horizon capped.

## 4. The forecast method itself was wrong (this is the significant one)

Anchoring and clamping fixed the obvious symptoms. Benchmarking found the
method was the problem.

Before training anything, whether raw sEMG is predictable at all was tested
directly (AR(10), fitted on the first half of subject 13, scored on the held-out
second half). Autocorrelation of the bandpassed signal: **+0.084 at 20 ms,
+0.002 at 100 ms**. One-step R² collapses from +0.518 at 4 ms — which is the
20–450 Hz filter's own smoothing, not physiology — to −0.000 at 100 ms.
Negative R² means worse than predicting the mean.

**"Predict the next signal value" is not a solvable task on sEMG**, for any
architecture. The median frequency underneath it is forecastable, and it is
what the chatbot is actually asked about. So the task is predicting the fatigue
level ahead, not the waveform. This is worth raising with the supervisor.

`models/forecast_lstm.py` (new) trains an LSTM on the same 8 features the
classifier uses, predicting **MDF(t+h) − MDF(t)** — the change, not the level,
because subjects sit anywhere from 53 to 78 Hz and a model asked for absolute
values would have to guess a held-out athlete's baseline it never sees.
Leave-one-subject-out, 12 subjects, 1227 samples, 3 seeds per fold, early
stopping on an inner split drawn from training subjects only.

MAE in Hz, and skill against simply assuming nothing changes:

| method | 10 s | 30 s | 60 s |
|---|---:|---:|---:|
| assume no change | 3.05 | 3.34 | 4.31 |
| **OLS line (what shipped)** | 3.14 | **4.27** | **6.52** |
| OLS, last 60 s only | 2.91 | 3.98 | 6.36 |
| mean drift rate | 3.04 | 3.30 | 4.01 |
| **LSTM** | **2.91** | 3.32 | 4.08 |

| skill vs "no change" | 10 s | 30 s | 60 s |
|---|---:|---:|---:|
| OLS line | −2.9% (p=0.85) | **−27.8% (p=0.003)** | **−51.3% (p<0.001)** |
| LSTM | **+4.6% (p=0.027)** | +0.8% (p=0.52) | +5.4% (p=0.42) |

**The shipped forecast was significantly worse than doing nothing.** MDF drops
steeply early then flattens; a line fitted over the whole recording keeps
projecting the early slope long after the signal has levelled off.

Be honest about the other half of this: the LSTM beats "no change" only at
10 s, and only at p=0.027 uncorrected for three horizons. Everywhere else it
ties. It is used because it is never *worse*, not because it is dramatically
better. ~1200 samples across 12 subjects is thin for a sequence model, and this
mirrors the classification finding in `LSTM_HANDOVER.md` — the LSTM matches
Random Forest on 8 raw features instead of 32 hand-crafted ones. The value is
in what it removes.

Full method, limitations and reproduction: `models/FORECAST_VALIDATION.md`.

### What that changed in `models/fatigue_forecast.py`

Maalav's file, reworked rather than replaced:

- **kept** the OLS slope / R² / p-value — those correctly *describe* the
  observed trend, which was never the broken part
- **replaced** the projected values with the trained forecaster, falling back
  to OLS when `models/forecast_model.pt` is absent so a fresh clone still works
- added a `method` field (`"lstm"` / `"ols"`) so callers can always tell which
  produced a number
- the uncertainty band is now the **measured out-of-subject error**, not an OLS
  residual band computed around a line that does not extrapolate. Inner band =
  typical error, outer = 95%.
- the projection is held flat beyond 60 s. Nothing beat "no further change" at
  long range, so the model is not allowed to invent one; the band keeps widening.

## Smaller fixes

- `frontend/upload.py` — headerless CSVs are sniffed rather than losing their
  first row; non-numeric columns raise a clear error instead of producing NaN;
  sample rates are snapped to integers (1259.29 vs 1259 drift).
- `frontend/charts.py` — charts embed the vendored plotly.js instead of a CDN
  `<script src>`, so they render offline. The forecast chart now also plots the
  **measured MDF points**, necessary because the trend line and the forecast now
  come from two different models and need not meet — without the data
  underneath, that step reads as a rendering glitch.
- `frontend/history.py` — dropped a persisted key that was never written.
- `frontend/app.py` — segment loads are cached; a forecast is only computed when
  one was actually asked for.
- `requirements.txt` (new) and a README section documenting **both** frontends.

## What was deliberately not done

- **`models/serve.py` / Open WebUI was left in place.** The supervisor asked for
  Open WebUI + function calling. `maalav-chatbot` moved away from it because the
  3B model's tool-calling was unreliable, which is a fair engineering call, but
  the directive stands. Both frontends consume the same `classify()`, and the
  README documents both. Neither replaces the other.
- **No classifier retraining.** The binary LSTM (88.7%) is untouched. Every
  headline accuracy number in the report is unaffected by this branch.
- The `--predict-next` classifier (~68%, 3-class) is still not wired into the
  app; it predicts one 2 s window ahead, not a continuous horizon.

## Tests

```bash
python models/test_classify.py        # 21 checks
python viz/test_render_window.py      # 9 checks
```

Two guards matter more than the rest:

`test_upload_tracks_calibrated_path()` fails if the self-calibrated path drops
below 80% label agreement with the calibrated one. That is what stops
`FRESH_SEC` being quietly shortened again. Currently 93.2%.

`test_forecaster_cannot_see_the_future()` truncates the recording at the query
time and asserts the prediction changes by **0.0000 Hz**. A forecaster that can
see future windows produces impressive numbers that mean nothing — the same
class of error as the time-ordered-split artifact documented in
`LSTM_HANDOVER.md`.

## Honest status

The upload path is still **measurably worse** than the calibrated one (0.796 vs
0.822 accuracy, ~90% agreement), and `classify_upload()` returns a `calibration`
block so the frontend can say so. For headline numbers, quote `classify()` on
the dataset subjects. The upload path demonstrates that the uncalibrated-athlete
case is *handled*, not a second accuracy result.
