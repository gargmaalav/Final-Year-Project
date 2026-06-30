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

## Directory structure

```
zenodo_biceps/   existing EMG pipeline (loader, classifier, core) — do not reorganise
viz/             Rayyan: render_window() + grounding bridge
models/          Aryan: Transformer / LSTM classifier
frontend/        Maalav: chat UI + LLM wiring
```

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
