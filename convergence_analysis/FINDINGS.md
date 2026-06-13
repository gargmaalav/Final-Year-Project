# EMG Convergence Analysis - Findings for Supervisor

**Prepared:** 2026-06-01 (Rayyan Abzal)
**Data:** `data/OpenBCI-RAW-2026-03-21_15-37-11.txt` (8-channel forearm EMG, ~9 min)
**Code:** `convergence_analysis/` (run `python run.py gui | detect | forecast`)

This addresses the two supervisor requests:
1. Automatically find the moments where the 8 channels look similar (converge).
2. Plot the cleaned period dynamically (time series + frequency), then predict
   future frequency.

---

## 1. Sample rate was wrong: it is 250 Hz, not 1000 Hz

The file header says `Sample Rate = 1000 Hz` and the earlier `pipeline.py`
trusted it. That is wrong, and it made every frequency value 4x too high.

The Cyton `Sample Index` column is the board's own per-sample counter (wraps
0-255), so its rate is physical ground truth, independent of the bursty WiFi
arrival timestamps. Three independent methods all give 250 Hz:

| Method | Result |
|---|---|
| Sample-Index wrap count | 534 wraps x 256 / 547.5 s = **249.6 Hz** |
| Index-advance / wall time | 136,803 samples / 547.5 s = **249.9 Hz** |
| Packet-loss consistency | received 121,793 / true 136,803 = 89% (11% lost) |

This matches the presentation slides (System Architecture + Signal Pipeline both
say 250 Hz). **All analysis below uses FS = 250 Hz.**

---

## 2. The clean period is 353.8 s, not 78 s

The old `session_report.txt` reported "78 s clean across 7 segments". That was an
artifact of the same FS bug: the old code computed `duration = rows / 1000`, but
the real arrival rate is ~222 rows/s (250 Hz minus 11% packet loss). The same
clean rows actually span **353.8 s**. The old code also dropped a valid 2.8 s
clean stretch for having fewer than 1000 rows.

Corrected clean period: **8 segments, 353.8 s** (contiguous runs with all 8
channels above the -180,000 uV dropout rail, >= 1 real second). Isolated
single-sample WiFi gaps (~11%) are interpolated; any analysis window spanning a
larger hole is skipped, so interpolation never fabricates a result.

### Scope: the full recording, not the 7 s demo window

To be clear about what we analyse: this uses the **whole clean recording (354 s,
8 segments)**, which is the same data the original feature pipeline
(`pipeline_full.py`) ran on - its 7 segment plots span 15:39:10 to 15:46:19, the
full session. The "78 s" was only the FS-bug miscount of that same data.

The short ~7 s slice (15:42:28-15:42:35) that appears elsewhere came from
`raw_plotting.py`, which is a throwaway **demonstration** script (its own
docstring: "purely to show why mean-centering is needed"). It was never the
analysis basis, and there is no scientific reason to anchor on it. We use the
full 354 s because (a) it is consistent with the real pipeline, (b) the
frequency forecast needs the long 75-122 s segments to be meaningful, and (c) it
yields 87 converged sections instead of a handful. The convergence GUI still
surfaces the short "similar" portions - it just finds them automatically across
the whole recording rather than from one hand-picked window.

**Caveat for the meeting:** this 354 s is resting / mixed pilot data with no
Fresh/Moderate/Fatigued protocol markers. It is right for demonstrating
convergence and the pipeline, but the fatigue science (MDF decline, below) needs
S2's structured grip protocol.

---

## 3. Convergence: where the 8 channels look similar

**Definition.** The channels carry large DC offsets (-100,000 to +90,000 uV), so
"look similar" means **shape**, not raw amplitude - exactly what you saw in the
auto-scaled OpenBCI GUI. Convergence in a 1 s window = the **mean of the 28
pairwise Pearson correlations** across the 8 channels (Pearson removes the DC
offset internally; this is not filtering). A "converged section" is a contiguous
stretch where that score stays >= 0.80 for >= 1 s.

**No filtering applied** (per your instruction).

**Result:** over the 8 clean segments, **87 converged sections totaling 158.1 s**
(44.7% of the clean period — union of overlapping section boundaries). Longest sections:

| Rank | Segment | Wall start | Length | Mean similarity |
|---|---|---|---|---|
| 1 | 8 | 15:45:14.87 | 7.0 s | 0.998 |
| 2 | 5 | 15:43:07.57 | 5.8 s | 0.946 |
| 3 | 1 | 15:40:41.35 | 5.2 s | 0.975 |
| 4 | 8 | 15:45:50.77 | 5.2 s | 0.945 |
| 5 | 8 | 15:45:04.07 | 5.0 s | 0.977 |

Per-segment converged time (union): seg1 36.4 s, seg2 3.1 s, seg3 1.1 s,
**seg4 0 s**, seg5 17.4 s, seg6 1.5 s, seg7 41.9 s, seg8 56.7 s. Segment 4
never reaches the threshold - the channels stay divergent there.

Full list in `out/converged_sections.csv`.

### GUI layout (two panels)

**Top — normalized channels.** Each channel is z-scored over the full converged section:

```
z = (channel – channel_mean) / channel_std
```

This removes each electrode's DC offset so the plot compares **shape**, not raw amplitude. The 8 traces move together visually; a green playhead advances during playback. Without z-scoring the large offsets (−100,000 to +90,000 µV) would push channels off screen. Z-scoring is display normalization only — the convergence metric (Pearson correlation) does the same thing mathematically.

**Bottom — predicted frequency (full width).** The spectral shape (share of power per Hz bin) averaged over the most recent 30% of the whole recording (yellow solid), projected 120 s ahead via per-bin linear trend (purple dashed). Both normalized to unit area so Y-axis reads share of power (%). Backtest on held-out last 30%: model 93% vs naive 86% match. Title carries the MDF of each curve (median of the power distribution). Note: on unfiltered data MDF tracks the 2–3 Hz baseline drift, not the muscle band — this is labelled in the panel.

Run `python run.py gui` and press Play. Use Next -> to step through all 14 listed sections.

---

## 4. Frequency forecast - and an open question for you

The "predict future frequency" half is built: per long segment, an OLS linear
regression of median frequency (MDF) over time, with R^2, a slope confidence
interval, a significance test, and a projection +20 s ahead with 95% confidence
and prediction bands (`out/forecast_seg*.png`, `frequency_forecast.csv`).

**The open question.** Slide 12 of our presentation states the key science:
*MDF shifts DOWNWARD as a muscle fatigues, and MDF decline rate is the strongest
predictor.* But on this recording, with **no filter**, MDF reads ~0.7-4 Hz and
trends slightly **upward** (e.g. seg1 +0.021 Hz/s, R^2 0.23):

| Segment | MDF slope | R^2 | significant? |
|---|---|---|---|
| 1 (122 s) | +0.021 Hz/s | 0.23 | yes (p<0.001) |
| 5 (38 s) | +0.001 Hz/s | 0.00 | no |
| 7 (82 s) | +0.016 Hz/s | 0.05 | yes |
| 8 (75 s) | +0.016 Hz/s | 0.02 | yes |

The reason: with no filter, **70-99% of the signal power sits below 5 Hz**
(baseline drift), so the unfiltered MDF tracks drift, not the muscle band. The
channels also "converge" largely on this shared low-frequency wander.

This is faithful to the "no filtering" instruction, but it means the unfiltered
MDF is **not** the fatigue indicator from the slides. The slides' MDF-decline
science needs the **filtered 20-120 Hz muscle band** (the same bandpass + 50 Hz
notch already shown in the Signal Pipeline slide).

**Recommendation / question:** keep the convergence view unfiltered (as
instructed), but should the **frequency/fatigue forecast** run on the filtered
20-120 Hz muscle-band MDF so it matches the slides' science? One line in
`core.py` (`mdf_trend`) switches it on; I have left it unfiltered and clearly
labelled pending your call.

---

## Files

- GUI: `python run.py gui` - converged-sections player (screen-record this)
- `out/converged_sections.csv` - all 87 sections, ranked
- `out/convergence_summary.txt` - top sections + per-segment breakdown
- `out/forecast_seg*.png`, `out/frequency_forecast.csv` - MDF forecasts
- `README.md` - full method and rationale
