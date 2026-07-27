# Changes on branch `maalav-chatbot`

This document is written for an AI assistant (or a teammate) reviewing this
branch's diff against `feat/aryan-classify` / `main`. It explains **why**
each change exists, not just what changed — the code comments do the "what."

## Why this branch exists

The project's own integration contract (`README.md`, agreed before building)
assigns three roles: Aryan (`models/` — `classify()`), Rayyan (`viz/` —
`render_window()`), and Maalav (`frontend/` — chat UI + LLM wiring,
`build_prompt()`). Only `frontend/.gitkeep` existed before this branch — the
frontend was never built. Instead, `feat/aryan-classify` wired the model
straight into **Open WebUI** using Ollama's tool-calling.

That Open WebUI integration turned out to be unreliable: `llama3.2:3b`'s
native/legacy function-calling hallucinated tool calls on this stack
(`create_calendar_event`, `search_calendar_events`, fabricated JSON — none of
them real, confirmed by live testing). An unmerged branch,
`feat/rayyan-chatbot-viz`, had already diagnosed the same root cause and
proposed switching Open WebUI to Legacy mode, but that fix was never cleanly
verified end-to-end.

Rather than keep depending on a small local model's tool-calling reliability,
this branch builds the **dedicated frontend the contract always intended**:
Python code calls `classify()` / `render_window()` directly — no LLM
tool-calling involved at all. The LLM's only job is phrasing an answer (and,
for sport/plan questions, adding its own general knowledge) from numbers it's
already been handed. It can never skip the tool call, mis-call it, or invent
a subject/confidence number, because it's never given the chance to.

## Architecture

```
User question (Streamlit chat_input, with a native "+" file-attach button)
  -> extract.py: LLM-based subject/time/side extraction (dataset queries)
     OR regex-based time/horizon extraction (upload queries — much simpler
     task, doesn't need another LLM round-trip)
  -> models/classify.py: classify() for the 13 dataset subjects,
     classify_upload() for an uploaded recording
  -> models/fatigue_forecast.py: forecast_fatigue() — MDF trend projection
  -> viz/render_window.py (dataset subjects) or frontend/charts.py (uploads)
     for the chart
  -> frontend/prompt.py / frontend/recommend.py: build the grounding prompt
  -> Ollama chat completion -> final answer (+ optional recommendation block)
  -> Streamlit: chat text + chart(s), persisted to frontend/.chat_history/
```

## New/changed files

### `models/classify.py` (modified)

Refactored to share its window-building/LSTM-inference core
(`_classify_window`) between two paths:
- `classify(subject, t_start, side)` — unchanged public signature and
  behavior, still the dataset-subject path (uses the per-subject baseline
  stored in `fatigue_model.pt`).
- `classify_upload(seg, fs, t_start, ...)` (new) — implements the
  "uncalibrated athlete" TODO that used to just be a comment. Computes a
  fresh baseline from a recording's own first ~15 seconds
  (`compute_fresh_baseline`) instead of ground-truth fatigue labels, which
  an uploaded recording will never have. Mirrors `train_model.py`'s own
  `mu, sd = base.mean(0), base.std(0)` baseline math, just choosing "fresh"
  by elapsed time instead of by label.

Verified behavior-preserving: `classify(13, 60.0, "R")` returns the exact
same result before and after the refactor.

### `models/fatigue_forecast.py` (new)

`forecast_fatigue(seg, fs, horizon_sec)` projects the MDF (fatigue marker)
trend forward using `convergence_analysis.core.forecast_regression` — an
existing, validated OLS-trend-with-confidence/prediction-bands function
Rayyan built for the exact "predict future frequency" ask, but only ever run
previously on unfiltered legacy OpenBCI data where `FINDINGS.md` documents
the MDF was tracking baseline drift, not muscle fatigue. This applies the
same regression to `loader.mdf_trend()`'s correctly bandpassed muscle-band
MDF, which this project already computes for classification and for
`render_window`'s chart.

Named `fatigue_forecast.py`, not `forecast.py`, because
`convergence_analysis/forecast.py` also exists and ends up on `sys.path`
once `loader.py` pulls `convergence_analysis` in — a plain `forecast.py`
name collided with it (hit and fixed during this session's testing).

A separate, already-trained "predict next window's label" classifier exists
in `zenodo_biceps/lstm_classify_biceps.py --predict-next`
(~68% LOSO accuracy, 3-class, `out/metrics_lstm_predictnext_3class_250hz_m4.json`)
but is **not** wired into this app — it's meaningfully weaker than the
deployed binary current-window model (88.7%) and only predicts one window
(2s) ahead, not a continuous horizon. Noted here so it isn't "rediscovered"
as a gap; it was a deliberate choice.

### `viz/render_window.py` + `viz/vendor/` (pulled in, unmodified)

Brought over from the unmerged `feat/rayyan-chatbot-viz` branch via
`git checkout origin/feat/rayyan-chatbot-viz -- viz/` — additive, no
conflicts with any file on this branch. Renders the 3-panel (raw EMG / MDF
trend / FFT) interactive Plotly chart for a dataset subject query. Untouched
by this branch; still Rayyan's file.

### `frontend/app.py` (new)

The Streamlit app. Orchestrates the whole pipeline for both the dataset path
and the upload path, plus:
- Sidebar: chat history list (switch/delete), "+ New chat", model picker,
  sample-rate input (for single-column uploads), an optional "sport/goal"
  note used to personalize recommendations, a disclaimer, "Clear all
  history" (two-step confirm).
- Regenerate button: re-runs just the LLM phrasing step(s) on the last turn
  (reusing already-computed `classify()`/`forecast()` results, not
  recomputing them — the phrasing call is what has real variance).
- File upload via `st.chat_input(accept_file=True, file_type=["csv"])`,
  Streamlit's native attach control (confirmed supported in the installed
  1.60.0).

### `frontend/extract.py` (new)

- `parse_query(text)` — the original LLM-based `{subject, t_start, side}`
  JSON extraction for dataset queries (unchanged from the first version of
  this frontend).
- `extract_t_start_seconds(text)` / `extract_horizon_seconds(text)` (new) —
  deterministic regex extraction for the upload path, where there's no
  dataset subject/side to resolve, so a full LLM extraction call is
  unnecessary overhead for a much simpler task.

### `frontend/upload.py` (new)

`parse_uploaded_csv(file, sample_rate_hz)` reads a two-column
(`time_s, signal`, sample rate inferred) or single-column (sample rate
required) CSV. `load_uploaded_segment(...)` calls `loader.to_segment(...)`
(already fully generic — zero dependency on the Zenodo file format) to
resample + bandpass it into the same `core.Segment` shape the rest of the
pipeline consumes.

### `frontend/charts.py` (new)

Lightweight Plotly panels for recordings `render_window.py` doesn't know how
to load: an uploaded file's raw signal + MDF trend, and the forecast
trend/confidence/prediction bands. Deliberately simpler than
`render_window.py`'s animated 3-panel chart, which stays subject-specific
and untouched.

### `frontend/history.py` (new)

Multi-conversation chat history, persisted as one JSON file per chat under
`frontend/.chat_history/` (gitignored — local per machine). **Chart HTML is
deliberately not persisted** — one chart rendered 3.1 MB of HTML in this
session's own testing, and saving several per chat would bloat the folder
fast. A reloaded past chat shows its saved text/recommendation but no chart
underneath.

### `frontend/recommend.py` (new)

`wants_recommendation(text)` — deterministic keyword gate (`sport`,
`recommend`, `plan`, `diet`, `gym`, `training`, etc.), no extra LLM call just
to detect intent. `build_recommendation_prompt(...)` hands the LLM the real
measured numbers (fatigue state, MDF, confidence, forecast trend) and is
explicit that sport-fit/plan suggestions draw on the LLM's own general
knowledge — there is no labeled data anywhere in this project connecting
EMG patterns to sport suitability, so a trained classifier for that would
have no ground truth to learn from. Always closes with a one-line disclaimer
that this is educational, not medical/professional advice.

### `frontend/prompt.py` (modified)

`build_prompt()` gained an optional trailing `forecast` argument (backward
compatible — the contract's two-arg signature still works). Also hardened
against a real failure mode hit during testing: `llama3.2:3b` sometimes
refused to state the plain fatigue result ("I can't provide information
that could be used to make medical diagnoses...") when the same user message
also asked a sport/plan question, even though this prompt never asks it to
give advice. Fixed by explicitly framing the result as a factual sensor
reading and telling it to ignore any recommendation sub-question (that's
`recommend.py`'s job).

### `frontend/llm.py` (modified)

`chat()` takes an optional `model` argument (was hardcoded); added
`list_models()` (calls Ollama's `/api/tags`) to populate the sidebar picker.

### `.gitignore` (modified)

Added `frontend/.chat_history/` and `.webui_secret_key` (a locally-generated
Open WebUI session secret that was sitting untracked in the repo).

## What was verified this session

- `classify()` regression-checked identical before/after the refactor.
- Full upload pipeline (`parse_uploaded_csv` -> `load_uploaded_segment` ->
  `classify_upload` -> `forecast_fatigue`) run end-to-end in Python with a
  synthetic CSV — correct output, no errors.
- Error paths: a <8s recording correctly raises the "need a longer
  recording" error; a single-column file with no sample rate correctly
  raises a clear error instead of crashing.
- Live in the browser: the golden-path dataset query, a combined
  fatigue+recommendation query (confirmed the disclaimer renders, confirmed
  a plain fatigue query does *not* trigger the recommendation block), the
  Regenerate button, chat history surviving a full Streamlit process
  restart (with chart correctly absent on reload, as designed).

**Not verified**: an actual drag-and-drop/file-picker upload through the
browser UI itself — the browser automation used this session can't drive a
native OS file dialog. The upload *logic* is verified directly in Python;
the UI's file-attach control is confirmed present, but a real end-to-end
click-through by a human is still worth doing (see the testing guide).

## Deliberately deferred, not done in this pass

- Exporting a chat/report as PDF.
- Retraining any model (the existing binary LSTM is unchanged).
- Token-by-token streaming / a "stop generating" button (the pipeline makes
  a few short blocking calls, not one long stream).
- Copy-to-clipboard on messages (needs custom JS in Streamlit, low value
  here).
