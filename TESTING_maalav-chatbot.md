# Testing the `maalav-chatbot` branch

This branch replaces the Open WebUI + Ollama tool-calling chatbot with a
dedicated Streamlit frontend that calls our own `classify()` / `render_window()`
code directly (no LLM tool-calling in the loop for the actual fatigue
numbers). It also adds: file upload (classify your own EMG recording), a
fatigue-trend forecast, sport/plan recommendations, and ChatGPT-style chat
history.

You do **not** need Open WebUI, Docker, or Python 3.11 specifically for this
branch — any recent Python 3.x with the packages below works.

---

## 1. Get the code

```bash
git clone <repo-url>
cd Final-Year-Project
git checkout maalav-chatbot
git pull
```

## 2. Get the dataset

Same as before: download Zenodo 14182446 and place it at
`zenodo_biceps/sEMG_data/` (gitignored, not in the repo). If you already have
this from earlier testing, you're set.

## 3. Get the trained model file

`models/fatigue_model.pt` is gitignored. Either:
- **Easiest:** get the `.pt` file from a teammate and drop it in `models/`.
- **Or** regenerate it yourself (needs the dataset from step 2):
  ```bash
  python models/train_model.py
  ```

## 4. Python dependencies

```bash
pip install torch numpy scipy pandas streamlit plotly requests
```

(`fastapi`/`uvicorn` are only needed if you also want to run the old
`models/serve.py` HTTP bridge — not required for this branch's frontend.)

## 5. Ollama

```bash
ollama pull llama3.2:3b
```
Make sure Ollama is running (`ollama serve`, or it may already be running as
a background service — check with `ollama list`).

## 6. Run it

```bash
streamlit run frontend/app.py
```

It should open `http://localhost:8501` automatically (or open it manually).

---

## Things to test

### A. Golden path (dataset subject)

Ask: **"Is subject 13 fatigued at 60 seconds on the right side?"**

Expected: a real answer citing non-fatigue and ~96.6% confidence, an
interactive 3-panel chart (EMG window / MDF trend / FFT) underneath, and a
second smaller chart showing the fatigue-trend forecast with shaded
confidence/prediction bands.

### B. Forecast with a custom horizon

Ask: **"Is subject 5 fatigued at 100 seconds, and what will it look like in
the next 40 seconds?"**

Expected: the answer mentions the trend direction, and the forecast panel's
projection extends further than in test A (custom horizon picked up).

### C. Upload your own recording

Click the **"+" button inside the chat input box** (bottom-left of the text
box) and attach a CSV. Two formats work:
- **Two columns**, `time_s,signal` — sample rate is inferred automatically.
- **One column** (signal only) — set the "Sample rate for single-column
  uploads (Hz)" field in the sidebar first to match your recording device.

Ask something like **"Am I fatigued?"** with no time specified — it should
classify the most recent part of the recording. Try again with **"...at 20
seconds"** to target a specific point.

Expected: a real classification computed from *your* recording (not a
canned response), a raw-signal + MDF-trend chart, and a forecast panel. If
your recording is shorter than ~8 seconds, expect a clear "need a longer
recording" message instead of a crash or a silently-wrong answer.

**This is the one thing not fully verified yet** — please specifically
confirm the "+" attach control and the resulting classification work
smoothly for you; a synthetic CSV was tested programmatically but a real
click-through hadn't been done by a human as of this branch being pushed.

### D. Sport / training plan recommendation

Ask: **"Subject 13 at 60 seconds on the right side — what sport would suit
them and what gym plan should they follow?"**

Expected: the plain fatigue answer first, then a visually distinct info box
with sport-fit and training/diet suggestions, ending with a disclaimer that
this is educational, not medical/professional advice. Try the sidebar
"Sport/goal" field (e.g. "competitive rock climbing") and ask again — the
suggestions should take it into account.

Then ask a **plain** fatigue question with no sport/plan/diet/gym keywords
in it, and confirm the recommendation box does *not* appear.

### E. Chat history

- Ask a couple of questions, then click **"+ New chat"** — confirm a fresh
  chat starts and the old one is still listed in the sidebar with an
  auto-generated title.
- Click back into the old chat — confirm the text (and recommendation, if
  any) reloads, but note the chart is intentionally *not* restored (a
  design tradeoff — chart HTML got large enough that persisting it wasn't
  worth the disk usage; see "What was verified" below for the reasoning).
- Fully stop and restart `streamlit run frontend/app.py` — confirm your
  chats are still listed after a real restart, not just a page refresh.
- Try deleting a chat (🗑) and "Clear all history" (it asks for confirmation
  first).

### F. Regenerate + model picker

- Click **"🔄 Regenerate"** under the last answer — confirm it produces a
  reworded (but still consistent) answer without noticeably re-running the
  whole pipeline.
- If you have more than one Ollama model pulled, switch the sidebar "Model"
  dropdown and ask another question — confirm the new model is actually
  used (response style should change).

---

## If something breaks

- **`serve.py`-style errors about a missing baseline / dataset** → same two
  usual culprits as before: dataset not at the expected path, or
  `fatigue_model.pt` missing. Both show up as errors in the terminal running
  `streamlit run`, so check there first.
- **Ollama connection errors** → confirm `ollama list` works and
  `llama3.2:3b` is pulled.
- **Upload errors** → read the in-chat error message first, it's meant to be
  specific (missing sample rate, recording too short, unreadable CSV, etc).
- Anything else: check the terminal running Streamlit for a traceback and
  share it — that's usually more informative than the in-browser message.

---

## What changed on this branch (for reference / for your AI assistant)

The rest of this document is a copy of `CHANGES_maalav-chatbot.md` from the
repo root, included here so it travels with these testing steps. If you're
pointing an AI assistant at this branch to review the diff, this section is
written for that.

<!-- BEGIN CHANGES_maalav-chatbot.md -->

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

<!-- END CHANGES_maalav-chatbot.md -->
