# Signal-Driven AI Systems — EMG Chatbot

Final year project, AUT ENSE891. Team: Maalav, Aryan, Rayyan.

## Architecture

```
User query
    |
Maalav frontend  ──>  Aryan classify()   ──>  structured features
                  ──>  Rayyan render_window()  ──>  PNG panel
                  ──>  Stock LLM (Ollama / API) grounded with features
                  ──>  Text answer + visual panel displayed side-by-side
```

LLM reads structured features, not images. No model training required.

## Running it

```bash
python -m pip install -r requirements.txt
python models/train_model.py          # once, produces models/fatigue_model.pt
```

Also needed: Ollama running locally with a chat model pulled (default
`llama3.2:3b`), and the Zenodo dataset at `zenodo_biceps/sEMG_data/`.

There are **two frontends over the same `classify()`**, both supported:

```bash
streamlit run frontend/app.py         # Streamlit chat UI (dataset + file upload)
python models/serve.py                # HTTP bridge for Open WebUI function calling
```

The supervisor asked for Open WebUI + function calling, which is what
`models/serve.py` and `models/openwebui_tool_reference.py` provide. The
Streamlit app is a second frontend that calls `classify()` directly in Python
instead of relying on the LLM to invoke a tool — the 3B model proved
unreliable at deciding to call the tool. Neither replaces the other, and both
consume the same contract function.

## Directory structure

```
zenodo_biceps/   existing EMG pipeline (loader, classifier, core) — do not reorganise
viz/             Rayyan: render_window() + grounding bridge
models/          Aryan: LSTM classifier, classify()/classify_upload(), serve.py
frontend/        Maalav: Streamlit chat UI + LLM wiring
```

## Tests

```bash
python models/test_classify.py        # classify() contract + calibration guards
python viz/test_render_window.py      # chart rendering
```

`models/CALIBRATION_VALIDATION.md` documents how an athlete with no stored
baseline is calibrated, and the measurements behind the constants that control
it. Read it before changing anything in `compute_fresh_baseline()`.

## Integration contract

Agree these signatures before building — a mismatch here breaks the whole system silently.

### viz/render_window.py

```python
def render_window(subject: int, t_start: float, side: str = "R") -> str:
    """Return interactive Plotly chart as an HTML string (no full_html wrapper).
    Embed via innerHTML in JS or st.components.html() in Streamlit.
    If Maalav uses React + plotly.js, switch return to fig.to_json() instead."""
```

Produced by: Rayyan. Consumed by: Maalav.

### models/classify.py

```python
def classify(subject: int, t_start: float, side: str = "R") -> dict:
    """Return {'mdf_hz': float, 'fatigue_label': int, 'confidence': float}"""
```

Produced by: Aryan. Consumed by: Maalav.

### frontend/prompt.py

```python
def build_prompt(features: dict, user_query: str) -> str:
    """Build the LLM system prompt from pipeline features + user question.
    Constrain LLM to only use provided feature values — no hallucinated numbers."""
```

Produced by: Maalav.

## Branch convention

- `main` — stable, reviewed, merged only via PR
- `feat/rayyan-visualization` — render_window panel + grounding bridge
- `feat/aryan-ml` — Transformer / LSTM classifier
- `feat/maalav-frontend` — chat UI + LLM wiring

Open a PR into main when your feature is working end-to-end.

## Dataset

Zenodo 14182446 — 13 subjects, biceps brachii sEMG, 1259 Hz.
Download and place at `zenodo_biceps/sEMG_data/` (gitignored — not committed).
