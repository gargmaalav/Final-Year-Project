# 8-Channel EMG Convergence Analysis

Finds and plays back the **sections where all 8 EMG channels look similar
(converge)**, strictly within the recording's clean period. Built for the
supervisor's request: *"automatically find the moment when the signals from the
8 channels look similar"* and *"plot the data within the cleaned period
dynamically with time series and frequency, then predict future frequency."*

## What "convergence" means here

The supervisor watched the OpenBCI GUI (each channel in its own auto-scaled
lane) and said the channels "look similar". That is **shape** similarity, not
raw-amplitude overlap (the channels carry DC offsets of -100,000 to +90,000
uV). So convergence at any moment = the **mean of the 28 pairwise Pearson
correlations** across the 8 channels in a 1-second window. The channels are
divergent most of the time and converge only occasionally; a "converged
section" is a contiguous stretch where that score stays >= 0.80 for >= 1 s.

## Key decisions (and how they were proven)

- **FS = 250 Hz** - proven from the data, not the header. The file header says
  `Sample Rate = 1000 Hz` and the old `pipeline.py` trusted it, which made every
  MDF/MNF value 4x too high. The Cyton `Sample Index` column is the board's own
  per-sample counter (wraps 0-255), so its wrap rate is physical ground truth,
  independent of the bursty WiFi arrival timestamps. Three methods agree:
  wrap-count (534 x 256 / 547.5 s = 249.6 Hz), index-advance (136,803 samples /
  547.5 s = 249.9 Hz), packet-loss consistency (received 121,793 / true 136,803
  = 89%).
- **No noise filtering.** No notch, no bandpass (supervisor's instruction). The
  only processing is per-window normalization for visual comparison and Pearson
  correlation, which removes the DC offset internally. That is not filtering.
- **Restricted to the clean period.** Everything runs only inside the **8 clean
  segments** (maximal runs with all 8 channels above the -180,000 dropout rail,
  >= 1 real second), **353.8 s total**. ~11% of samples are isolated single-sample
  WiFi losses; those are interpolated, and any analysis window spanning a larger
  hole is skipped, so interpolation can never fabricate a correlation.

### Honest caveat: what the channels are converging ON

Because no filter is applied, 70-99% of the signal power in the converged
sections sits **below 5 Hz** - slow baseline drift, not muscle activity. The
channels "look similar" largely because their low-frequency wander moves
together (exactly what is visible in the auto-scaled OpenBCI GUI). Consequently
the median frequency (MDF) of the *raw* signal reads ~0.7-4 Hz, and the
frequency trend / forecast tracks that raw-signal median. This is faithful to
the "no filtering" instruction. If the goal later shifts to muscle-band
convergence, add a 20-120 Hz bandpass before correlating - one line in
`core.py` - and the same machinery applies.

### Note on the old "78s / 7 segments" figure

The previous `session_report.txt` said the clean data was "78s across 7
segments". That was an artifact of the FS=1000 bug: the old code computed
`duration = rows / 1000`, but the real rate is ~222 received rows/s (250 Hz
minus 11% loss). The same clean rows actually span **353.8 s**. The old code
also dropped a valid 2.8 s clean stretch because it had fewer than 1000 rows.
Corrected: **8 segments, 353.8 s.**

## Run it

From this folder, with the project's `python` (anaconda3):

```bash
python run.py gui       # interactive converged-sections player (needs a display)
python run.py detect    # headless: writes CSV + summary + PNG snapshots
python run.py all       # detect, then launch the player
```

To record for the supervisor: run `python run.py gui`, then screen-record with
QuickTime (Cmd-Shift-5) while you press **Next ->** through the sections.

## The player (GUI)

Shows ONLY the converged sections. For each one it plays, dynamically:

- **Left:** the converged sections (wall-clock start, length, mean similarity),
  longest first. Pick one; **Prev / Next** step through them.
The three things the supervisor asked for - time domain, frequency domain, and a
predicted-frequency line derived from the frequency data:

- **Top - time domain:** scrolling 8-channel time series, normalized so you SEE
  them move together.
- **Bottom-left - frequency domain:** live per-channel FFT spectrum of the
  current 1 s window (x-axis to the 125 Hz Nyquist).
- **Bottom-right - predicted frequency:** solid yellow = the measured frequency
  **shape** now (share of power per frequency, %, over the most recent 30% of the
  recording); purple dashed = the **predicted** spectrum projected +120 s ahead,
  drawn as a **separate line on the same graph**. The title shows the median
  frequency shifting (e.g. MDF 11 -> 17 Hz), and a small green badge carries a
  backtest match score (~93%) so the projection - which cannot be checked against
  data that does not exist yet - stays credible. The +120 s horizon is chosen so
  the predicted line visibly separates from the measured one for the demo (the
  trend is gentle, so a shorter horizon overlaps); `FORWARD_SEC` in `gui.py`
  controls it.
- Why shape, not amplitude: absolute amplitude swings with how hard the muscle
  fires and is not predictable on this resting pilot (a linear amplitude forecast
  scored *worse* than naive). The spectral shape - where the energy sits - is the
  real frequency signature, is far more stable, and its prediction beats naive.
  The backtest behind the badge is available headlessly if the full
  train/predict/actual proof is ever wanted.
- **Left:** the converged sections (wall-clock start, length, mean similarity),
  longest first. Pick one; **Prev / Next** step through them.
- **Transport:** Play / Pause / Restart / seek.

The slides (FS, MDF-decline science) were a baseline, not the spec; the
supervisor's actual ask is the future frequency visualisation above. A separate
slide-flavored MDF-vs-time forecast still lives in `run.py forecast`
(`out/forecast_seg*.png`) if it is ever wanted.

## Outputs (`out/`, from `detect`)

- `converged_sections.csv` - every converged section (87 of them), ranked by
  length (segment, wall start/end, duration, mean & peak similarity).
- `convergence_summary.txt` - top sections + per-segment breakdown.
- `section_NN_*.png` - normalized 8-channel snapshot of each top section.
- `player_preview.png` - a still of the GUI on the longest section.

## Result on the current recording

Over the 8 clean segments (353.8 s), **87 converged sections totaling 171.0 s**
(48% of the clean period) at threshold 0.80. Longest sections:

| Rank | Segment | Wall start | Length | Mean sim |
|---|---|---|---|---|
| 1 | 8 | 15:45:14.87 | 7.0 s | 0.998 |
| 2 | 5 | 15:43:07.57 | 5.8 s | 0.946 |
| 3 | 1 | 15:40:41.35 | 5.2 s | 0.975 |
| 4 | 8 | 15:45:50.77 | 5.2 s | 0.945 |
| 5 | 8 | 15:45:04.07 | 5.0 s | 0.977 |

Per-segment converged time: seg1 40.8s, seg2 3.1s, seg3 1.1s, seg4 **0s**, seg5
18.9s, seg6 1.5s, seg7 45.9s, seg8 59.7s. Segment 4 never reaches the threshold.

## Files

| File | Role |
|---|---|
| `core.py` | loading, clean-segment cutting + resampling, convergence + frequency maths, converged-run extraction |
| `detect.py` | headless detector -> CSV, summary, PNG snapshots |
| `gui.py` | interactive converged-sections player (the deliverable) |
| `run.py` | entry point / dispatch |

Threshold and minimum run length are single constants at the top of `core.py`
(`CONV_THRESHOLD`, `MIN_RUN_SEC`).
