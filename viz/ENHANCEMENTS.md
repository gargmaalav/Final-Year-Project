# Chart enhancement notes

Ideas surfaced while polishing the chatbot chart for the supervisor demo
(2026-07-14). One implemented, two parked with rationale so we do not re-litigate
them. Example stills for all three live in `viz/ex_*.png`; the decision gallery is
`viz/enhancement_examples.html`.

## Implemented

### MDF fatigue trend + slope (Hz/min)
Least-squares line through the whole MDF trend on panel 2, slope shown in the
legend as `fatigue trend -X.X Hz/min`. Turns the coloured dots into the actual
quantitative fatigue metric (median frequency falls as the muscle fatigues).
- Where: `render_window.py`, static trace added before `base = len(fig.data)` so
  it never enters `fig.frames` (the line does not change as playback scrubs).
- Colour `#58a6ff` dashed, `hoverinfo=skip` (does not steal hover from the dots).
- Trustworthy because FS=250 is confirmed wired correctly (not the old 4x bug).
- Real slopes: S13 -2.7 Hz/min, S2 -6.2 Hz/min, S1 -5.7 Hz/min.

## Parked (keep as notes, do NOT build yet)

### 1. Spectrogram + MDF overlay  [strongest deferred idea]
Whole recording as a time-frequency heatmap with the MDF line drawn on top; the
spectral energy visibly compresses downward as the muscle fatigues, and MDF is
the median of each column so the line and heatmap agree on screen. Highest visual
payoff, zero examiner risk (standard signal processing), reads on video with no
narration.
- Cost: one `scipy.signal.spectrogram` call (nperseg=256, noverlap=192), a
  downsampled heatmap (~300x60 bins) is negligible against the 3.1 MB budget.
- Build sketch: add as a 4th panel, or replace the single-instant FFT panel
  (panel 3) which only shows one window. Frequency axis uses the same fs.
- Example: `viz/ex_spectrogram_s2.png` (clearest), `ex_spectrogram_s13.png`.
- Status 2026-07-14: Ray chose to ship the slope only for now; this stays parked.

### 2. Model fatigue-probability curve  [GATED - inspect before shipping]
Plot the classifier's confidence as a curve across the whole recording. Tempting
but risky: the model was trained `binary_drop_transition`, so it never saw the
transition zone. The curve may flicker or look erratic exactly where ground truth
is transitioning - a viva question we do not want to hand the examiner.
- Rule of thumb this produced: signal-derived visuals (spectrogram, MDF slope)
  are safe to bolt on; model-derived ones you build, eyeball, then keep only if
  the curve is actually clean. No example rendered on purpose.

## Also considered, skipped
- More interactive gadgets: scrub/playback + box-select is already strong; the
  deliverable is a video, so a striking static visual beats an interaction the
  presenter has to demonstrate.
- No 3D, no framework rewrite, no heavy architecture (data fits in one figure).
