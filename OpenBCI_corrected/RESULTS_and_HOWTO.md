# OpenBCI LSTM — corrected scripts + empirical results

Follow-up to `CLAUDE_REVIEW_OpenBCI_LSTM.md`. This folder turns the review's
findings into runnable code and reports the numbers that came out.

## What's here

| File | Purpose |
|---|---|
| `f1_spectral_check.py` | PSD of "valid" vs "invalid" segments |
| `f1_spectral_check.png` | The plot it produces (waveform + PSD, valid vs invalid) |
| `emg_burst_scan.py` | Scans the whole recording for EMG-band bursts vs railing |
| `emg_burst_scan.png` | Band power over time (green = "valid" segments) |
| `train_openbci_lstm_forecast_v2.m` | Forecast LSTM with persistence/linear baselines + leave-one-segment-out CV |
| `train_openbci_lstm_classifier_v2.m` | Classifier with segment-level split + logistic baseline (no window leakage) |

## How to run

1. Extract `OpenBCI_LSTM_Handoff.zip` somewhere (gives you
   `manual_segments_output/`, `classification_segments_output/`, etc.).
2. Copy the two `*_v2.m` files into that extracted folder (so they sit beside
   the data folders) and run them from there in MATLAB.
   - Needs Deep Learning Toolbox; the classifier also needs Statistics and
     Machine Learning Toolbox (`fitclinear`).
   - Set `trainLSTM = false` at the top of either script to get the baselines
     only in seconds, without training any network.
3. `f1_spectral_check.py` runs from the extracted folder too:
   `pip install numpy scipy pandas matplotlib` then `python3 f1_spectral_check.py`.

> Note: the `.m` scripts were written and their logic verified against the data
> with equivalent Python (numbers below), but they were not executed in MATLAB
> on my side — run them to reproduce. The Python-measured numbers are real.

---

## Finding 1 — RESOLVED: mostly railed; the weak EMG that exists is buried under drift

Montage is now known: **all 8 electrodes on one bicep, subject doing bicep
flexures, no formal protocol.** That removes the "common-mode artifact" worry —
8 electrodes on the same muscle *should* correlate, so high correlation is
expected, not suspicious. The real problems are acquisition and preprocessing.

**1. The amplifier railed for most of the session.** At least one of the 8
channels is pegged at the ±187500 µV rail (Cyton full scale at gain 24) in
**70.5% of all samples.** Saturated input = poor electrode contact / impedance /
gain. This dominates the data.

**2. The apparent "EMG bursts" are railing, not muscle.** The strongest raw
20–100 Hz windows (t = 5, 115, 117, 145, 316 s) are all clipped at exactly
187500 µV with 33–100% of samples at the rail. Clipping is a square wave; it
dumps energy into every band, so railed segments *look* like high-frequency EMG
but are saturation artifact.

**3. In the genuinely clean (non-railed) data there IS some real EMG — but weak
and drift-swamped.** Restricting to the 157 of 548 one-second windows with <5%
samples near the rail, and measuring 20–100 Hz power with the 50 Hz mains line
excluded (NZ mains sits mid-band):
   - Median EMG-band / low-band (<5 Hz) ratio is **0.03** — drift dominates.
   - But the best clean windows show real broadband EMG: t = 439 s has
     EMG/low = **0.35** with the 50 Hz line only ~3% of that band power (so it is
     broadband muscle activity, not powerline). Also at t = 416, 418, 494 s.
   - These clean-EMG windows cluster at **385–494 s**, overlapping the pipeline's
     "valid" region — so this is NOT a case of real EMG being thrown out as
     "invalid." The signal is there, just small.

So: there is *some* genuine bicep EMG, but it never rises above ~35% of the
baseline-drift power even at its best, and there is no high-contrast rest-vs-flex
structure to learn from. As recorded and preprocessed, it is not reliably
learnable.

**Implication — fix acquisition AND preprocessing before more model work:**
- **Acquisition:** fix electrode contact / skin prep / impedance so channels sit
  in a sane µV range and stop railing (this is the 70% problem).
- **Preprocessing:** the current pipeline applies **no high-pass filter**, so
  sub-5 Hz drift swamps everything. A high-pass (~10–20 Hz) on the clean data
  would expose the EMG that is already weakly present. This is a code-side fix,
  not only a re-record.
- **Protocol:** run a timed rest/flex cycle (e.g. 5 s rest / 5 s flex) so EMG
  bursts are high-contrast and labellable, and define "valid pattern" by
  EMG-band activity during flex — not by raw correlation.

Caveat: native ~222 Hz caps observation at ~110 Hz, so >110 Hz is invisible —
but the 20–110 Hz band IS observable, which is where the above is measured.

---

## Finding 5 — Forecast baselines: the LSTM does not beat trivial methods

Leave-one-segment-out forecast RMSE in Z space (measured in Python; the MATLAB
script reproduces and adds the trained LSTM):

```
segment 10 (the LSTM's reported test):
    LSTM (reported)        0.1945
    persistence            0.2896
    linear extrapolation   0.1696   <- a one-line fit BEATS the 128-unit LSTM

LOSO mean over segments:
    persistence            0.197  (std 0.137)
    linear extrapolation   0.200  (std 0.165)
```

The LSTM's 0.1945 is the same ballpark as "repeat the last value" and worse than
a straight-line fit on its own test segment. The headline RMSE means nothing
without these references, and with them it is unimpressive — expected, because a
sub-1 Hz signal is almost perfectly predictable by trivial methods.

---

## Finding 2 + 5 — Classifier leakage and baseline

Logistic regression on three hand features `[mean pairwise corr, mean channel
std, saturation fraction]` (measured in Python; MATLAB script reproduces and
adds the LSTM under the same segment-level CV):

```
(a) RANDOM-window split (same leaky protocol as the original):
    logistic, 3 features          88.1%
    logistic, correlation alone   85.7%
    original LSTM                 97.6%

(b) LEAVE-ONE-SEGMENT-OUT (honest, no overlap leakage):
    logistic, 3 features          80.0%
```

Reading: most of the score is reachable with three trivial features and no
network — the class is literally defined by correlation. The gap up to 97.6% is
largely the window-overlap leakage (87.5% overlap split randomly). Under an
honest segment-level split even the simple baseline is ~80%, and that still
overstates real discrimination because the invalid class is dominated by railed
data (Finding 3, not fixed here).

---

## Deferred (need decisions/data, not code)

- **Finding 3** — rebuild a fair "invalid" class of clean-but-uncorrelated
  windows. Needs re-running the Python extractor. Partly moot until Finding 1's
  montage question is answered.
- **Finding 4** — cross-session generalisation. Needs more recordings.
- **Finding 8** — cross-session normalisation rule. Decide alongside Finding 4.
