# Architecture

EMG-based biceps fatigue analysis: classify a subject's fatigue state from surface
EMG, forecast how the signal's spectrum evolves over a recording, and surface both
through an interactive chatbot visualization.

## Data sources (two sources, two different sample rates)

The single most important thing to get right in this repo: **there are two EMG
sources sampled at different rates, and neither is the number in the file header.**

| Source | True sample rate | Role | Location |
|--------|------------------|------|----------|
| Zenodo biceps sEMG (DOI 10.5281/zenodo.14182446) | 1259 Hz (native) | Primary / validation dataset | `zenodo_biceps/` |
| OpenBCI self-capture | 250 Hz (header says 1000) | Secondary, drift-prone | `OpenBCI_corrected/`, `Graphs OpenBCI RAW/` |

- The **Zenodo set** is the primary source: 13 subjects, one CSV per trial, four
  muscles per arm with interleaved time/EMG columns. Trials 5 and 6 are the
  dedicated biceps-brachii fatigue trials. Raw signal is in Volts, sampled at
  1259 Hz. This is what the classifier and forecast are validated on.
- The **OpenBCI capture** was the original plan but its recorded header reports
  1000 Hz while the true rate is 250 Hz, and its "valid" segments turned out to be
  sub-1 Hz drift rather than real EMG (baselines beat the model on it). Kept only
  in corrected form for reference.

### The sample-rate trap

`convergence_analysis/core.py` binds `fs=FS` (FS = 250) as a **default argument at
definition time**. Reassigning the module global `core.FS` does *not* rebind those
defaults, but it *does* change the window length that `core.mdf_trend` reads from
the global while the inner `median_frequency` call stays at 250, silently throwing
MDF off by ~5x. So the Zenodo path (1259 Hz) **passes `fs` explicitly to every core
call** rather than mutating the global. See the docstring at the top of
`zenodo_biceps/loader.py` for the full explanation.

## Pipeline

```
raw EMG (Zenodo CSV, 1259 Hz)
  -> zenodo_biceps/loader.py        # CSV -> core.Segment, fs passed explicitly
  -> convergence_analysis/core.py   # bandpass 20-450 Hz, windowed features, MDF
       |-- classify_biceps.py       # fatigue-state classification (LOSO validated)
       |-- spectrum_backtest/forward # spectrum-over-time forecasting
  -> viz/render_window.py           # interactive Plotly figure for the chatbot
  -> models/serve.py                # FastAPI bridge -> Open WebUI chatbot
```

## Components

| Path | Responsibility |
|------|----------------|
| `convergence_analysis/core.py` | Shared signal pipeline: `Segment`, bandpass filter, `median_frequency`, `channel_spectrum`, `collect_session_spectra`, `spectrum_backtest`, `spectrum_forward`, `mdf_trend`. FS default 250. |
| `convergence_analysis/gui.py` | Desktop GUI over the convergence pipeline. |
| `zenodo_biceps/loader.py` | Zenodo CSV -> `core.Segment`; owns the 1259 Hz handling and an fs-aware `mdf_trend`. |
| `zenodo_biceps/classify_biceps.py` | Cross-subject fatigue classifier, leave-one-subject-out validation, subject-baseline calibration. |
| `zenodo_biceps/lstm_classify_biceps.py`, `transformer_classify_biceps.py` | Deep-learning classifier variants. |
| `viz/render_window.py` | `render_window(subject, t_start, side) -> interactive Plotly HTML`. Single-subject panels: raw EMG coloured by fatigue label, MDF-over-time, FFT of the current window. |
| `viz/signal_viewer.py` | Single-subject scrub/playback viewer (the tool previewed with the supervisor). |
| `models/serve.py` | FastAPI wrapper exposing `classify()` and `render_window()` over HTTP (`http://localhost:8000`) so the Open WebUI chatbot can reach them via function calling. |
| `models/openwebui_tool_reference.py` | The Open WebUI tool that calls the serve.py API. |

## Chatbot serving bridge

Open WebUI cannot import project code directly, so `models/serve.py` runs a small
FastAPI service on `localhost:8000`. Open WebUI runs natively on the same host
(`.venv/bin/open-webui serve`, no Docker) and reaches `/classify`, `/render`, and
`/health` through function calling. This is the "link the deep-learning algorithms
to the chatbot frontend" integration.

## Team split

- **Rayyan** - chatbot subject, data visualization (`viz/`, `render_window`).
- **Maalav** - frontend + custom-GPT model integration.
- **Aryan** - Transformer models + the `classify()` / chatbot bridge (Open WebUI tool).

See `docs/decisions.md` for the reasoning behind the dataset choice, sample-rate
handling, and classifier configuration.
