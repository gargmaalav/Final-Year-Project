# Signal-Driven AI Systems — EMG Chatbot

Final year project, AUT ENSE891. Team: Maalav, Aryan, Rayyan.

## Architecture

```
Browser (viz/chatbot_ui.html, vanilla JS)
    |  POST /turn (one message in, one answer out)
    |  POST /chart (render a figure, on demand)
models/serve.py  (FastAPI)
    |
frontend/turn.py  ──>  frontend/intent.py + extract.py   (what was asked, what it names)
                  ──>  models/classify.py                (the fatigue reading itself)
                  ──>  models/fatigue_forecast.py         (optional trend projection)
                  ──>  frontend/interpret.py + prompt.py  (plain-language facts -> LLM prompt)
                  ──>  Ollama (local LLM)                 (phrases the answer, invents nothing)
                  ──>  viz/render_window.py / charts.py   (the figure, built only if asked for)
```

The LLM is handed measured numbers and told to phrase them — it never sees a
chart, never runs the classifier itself, and its prose is post-processed
(`frontend/interpret.py`) to strip anything it might have invented.

## Running it

```bash
python -m pip install -r requirements.txt
python models/train_model.py          # once, produces models/fatigue_model.pt
```

Also needed: [Ollama](https://ollama.com) running locally with a chat model
pulled (`ollama pull llama3.2:3b`), and the Zenodo dataset at
`zenodo_biceps/sEMG_data/` (see Dataset, below).

```bash
python models/serve.py                # UI at http://localhost:8000
```

That's the one entry point — it serves `viz/chatbot_ui.html` and the `/turn`
and `/chart` API it calls. The server warms Ollama's prompt cache at startup
(`[warm-up] ready in ~Ns`) so the first real question isn't the slow one.

`models/openwebui_tool_reference.py` is a second, optional way to reach
`classify()` via Open WebUI's function calling, kept working but not required
to run the chatbot above.

## Directory structure

```
zenodo_biceps/   the EMG dataset + pipeline (loader, classifier, core)
models/          classify(), the LSTM fatigue forecaster, serve.py (the API + server)
frontend/        turn.py (turn-handling engine) + intent/extract/interpret/prompt/recommend
viz/             chatbot_ui.html (the UI) + render_window.py/charts.py (the figures)
docs/            design docs and decisions log
```

`frontend/app.py` is the project's original Streamlit frontend. It's no
longer used (see `docs/decisions.md`) but kept around until its logic is
fully accounted for elsewhere — don't build against it.

## Tests

```bash
python frontend/test_answers.py       # the facts handed to the model say the right thing
python frontend/test_understanding.py # a question resolves to the right intent/window
python models/test_classify.py        # classify() contract + calibration guards
python viz/test_render_window.py      # chart rendering
```

`models/CALIBRATION_VALIDATION.md` documents how an athlete with no stored
baseline is calibrated, and the measurements behind the constants that control
it. Read it before changing anything in `compute_fresh_baseline()`.

`models/FORECAST_VALIDATION.md` covers the fatigue forecaster: why "predict the
next signal value" is not a solvable task on sEMG, the leave-one-subject-out
benchmark against persistence and OLS baselines, and the finding that the
straight-line forecast previously used was significantly *worse* than assuming
no change (−28% at 30 s, −51% at 60 s).

## Dataset

Zenodo 14182446 — 13 subjects, biceps brachii sEMG, 1259 Hz.
Download and place at `zenodo_biceps/sEMG_data/` (gitignored — not committed).
