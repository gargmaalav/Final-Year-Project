# Decisions

Architecture and methodology decisions, newest first. Each entry: context, decision,
consequences.

## 6. Open WebUI function calling: use Legacy mode, not Native/"Default"

- **Date:** 2026-07-12
- **Context:** The chatbot integration calls `models/serve.py` through Open WebUI
  function calling. In Native (labelled "Default") mode the customized Open WebUI
  fork injects builtin calendar/automation tools, and the small model misfires,
  calling `create_calendar_event` instead of the project's classify/render tools.
- **Decision:** Set Open WebUI's function-calling mode explicitly to **Legacy**.
- **Consequences:** The real classify/render tool fires and the chart renders. No
  infrastructure or model change was needed; this is a settings-level fix,
  live-verified in both directions.

## 5. Chatbot visualization: single-subject only, scrub/playback is fast-follow

- **Date:** 2026-07-13
- **Context:** An all-13-subjects overview was tried, but 13 overlaid MDF lines were
  an unreadable spaghetti plot. `signal_viewer.py` (the tool the supervisor
  previewed) has a scrub slider + Play button; `render_window` produces a single
  static Plotly figure.
- **Decision:** Ship the **single-subject** view as the deliverable
  (`viz/render_window.py`): raw EMG window coloured by fatigue label, MDF-over-time,
  FFT of the current window, with native Plotly hover/zoom/pan. Drop the
  13-subject overview. Treat scrub/playback as an explicit fast-follow, not a silent
  substitution.
- **Consequences:** A readable, embeddable chart for the chatbot. The interactive
  scrub/playback of `signal_viewer.py` does not carry over to the embedded figure yet.

## 4. Classifier configuration: normalized + causal-temporal + transition margin

- **Date:** 2026-06
- **Context:** Cross-subject fatigue classification on the Zenodo set, validated
  leave-one-subject-out (LOSO), needs per-subject calibration and clean labels near
  the fresh/fatigued boundary.
- **Decision:** Run `classify_biceps.py` with subject-baseline normalization, causal
  temporal features, and a transition-boundary exclusion margin
  (`--norm --temporal --transition-margin-sec 4`).
- **Consequences:** Best validated results ~91% binary and ~73.1% three-class under
  LOSO. Normalization alone lands ~67%; the temporal + margin combination is what
  reaches the three-class figure. Spectrum forecasting beats a naive baseline on a
  minority of subjects, not universally.

## 3. Sample-rate handling: pass fs explicitly, never mutate the global

- **Date:** 2026-06
- **Context:** `core.py` binds `fs=FS` (250) as a default argument at definition
  time. The Zenodo data is 1259 Hz. Reassigning `core.FS` does not rebind those
  defaults but does desync `mdf_trend`'s window length from its inner
  `median_frequency`, throwing MDF ~5x off (the same class of bug as an earlier
  "frequencies 4x too high" defect).
- **Decision:** Never reassign `core.FS`. Pass `fs` explicitly to every core call for
  1259 Hz data, and keep an fs-aware `mdf_trend` in `zenodo_biceps/loader.py`.
- **Consequences:** Correct MDF and spectra on the Zenodo path. Any new consumer of
  `core` on non-250 Hz data must pass `fs` explicitly.

## 2. Primary dataset: Zenodo biceps set over OpenBCI self-capture

- **Date:** 2026-06
- **Context:** The original OpenBCI self-capture had a header reporting 1000 Hz
  against a true 250 Hz rate, and its "valid" segments were sub-1 Hz baseline drift
  rather than real EMG. Simple linear/logistic baselines beat the model on it.
- **Decision:** Use the open Zenodo biceps-brachii sEMG dataset
  (DOI 10.5281/zenodo.14182446, Sensors 2024, CC BY 4.0) as the primary source for
  building and validating the pipeline. Keep OpenBCI only in corrected form for
  reference.
- **Consequences:** A clean, published, reproducible dataset to validate against.
  Own age-band recordings still need to be collected separately.

## 1. Muscle target: biceps brachii

- **Date:** 2026-06-13
- **Context:** No clean, public forearm sEMG fatigue dataset exists. A validated
  public biceps set does.
- **Decision:** Target the **biceps brachii**. Use Zenodo 14182446 to validate the
  pipeline; any self-collected data follows the same biceps-curl fatigue protocol to
  stay comparable.
- **Consequences:** The whole pipeline (loader, classifier, forecast, viz) is built
  around biceps trials (Zenodo trials 5 and 6). A forearm pivot would require a new
  dataset and re-validation.
