"""
HTTP wrapper around classify() -- also serves viz/chatbot_ui.html.
===================================================================

Three consumers of this API:
  1. viz/chatbot_ui.html -- served at "/" by this file. Calls POST /turn per
     message; that route runs frontend/turn.py's handle_turn(), which is
     Maalav/Aryan's Streamlit turn-handling logic (intent routing, param
     extraction, follow-ups, comparisons, recommendations, upload
     calibration) minus Streamlit -- see docs/decisions.md for why the UI
     changed but this logic didn't.
  2. Open WebUI, as an optional alternate frontend -- calls /classify via
     function calling (models/openwebui_tool_reference.py).
  3. GET /models -- Ollama models available, for the UI's model picker.

One-time setup:
    pip install fastapi uvicorn requests
    # needs `ollama serve` running with a model pulled, e.g.
    # `ollama run llama3.2:3b` once before your first query.

Run (from the repo root):
    python models/serve.py            # UI at http://localhost:8000

Test in a browser:
    http://localhost:8000/                          # chatbot UI
    http://localhost:8000/classify?subject=13&t_start=120
    http://localhost:8000/health
"""
from __future__ import annotations

import os
import sys
import threading

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
import uvicorn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "frontend"))
from classify import classify  # noqa: E402  our contract function
import turn as turn_engine     # noqa: E402  frontend/turn.py

app = FastAPI(title="EMG Fatigue classify() API")

_STATE = {0: "non-fatigue", 1: "fatigue", 2: "fatigue"}
_UI_PATH = os.path.join(REPO_ROOT, "viz", "chatbot_ui.html")

# One process, one demo server -- see frontend/turn.py's module docstring for
# why an in-memory dict (keyed by the session_id the UI already assigns each
# chat) is an acceptable substitute for what was st.session_state.
SESSIONS: dict[str, dict] = {}


def _session(session_id: str) -> dict:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = turn_engine.new_session()
    return SESSIONS[session_id]


@app.on_event("startup")
def _warm_ollama() -> None:
    """Pay the first-question costs at boot instead of in front of the reader.

    Loading the model and evaluating the prompt for the first time together
    cost ~20 s on the opening question and ~2 s on every one after. Done in a
    daemon thread so the server still binds its port immediately -- a question
    asked during the warm-up is not blocked by it, it just does not benefit.
    """
    threading.Thread(target=turn_engine.warm_up, daemon=True,
                     name="ollama-warmup").start()


class _UploadShim:
    """Adapts a FastAPI UploadFile's already-read bytes to the .name/.size/
    .getvalue() interface frontend/upload.py's parse_uploaded_csv expects
    (that interface matches Streamlit's UploadedFile, which is what it was
    originally written against)."""
    def __init__(self, name: str, data: bytes):
        self.name = name
        self.size = len(data)
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


@app.get("/")
def chatbot_ui():
    """Serve the standalone chatbot UI (viz/chatbot_ui.html).

    Sent no-store: this is a single hand-edited file with no build step and no
    content hash in its URL, so a cached copy is indistinguishable from the
    current one. Without this the browser kept serving an older UI after the
    file changed -- fixes looked like they had not been applied at all, and
    the only way out was a manual hard-reload the reader had no reason to
    suspect they needed. It is one local file on a demo server; re-reading it
    per request costs nothing.
    """
    return FileResponse(_UI_PATH, media_type="text/html",
                        headers={"Cache-Control": "no-store, must-revalidate"})


@app.get("/models")
def list_models_endpoint():
    """Ollama models available for the chat answer, for the UI's model picker."""
    from llm import list_models, MODEL as default_model  # noqa: E402
    names = list_models()
    return {"models": names, "default": default_model if default_model in names else names[0]}


@app.post("/turn")
def turn_endpoint(session_id: str = Form(...), user_text: str = Form(""),
                  model: str = Form(""), sample_rate: float | None = Form(None),
                  athlete_note: str = Form(""), file: UploadFile | None = File(None)):
    """One user turn in, one finalized answer out -- see frontend/turn.py."""
    session = _session(session_id)
    if model:
        session["model"] = model
    session["sample_rate"] = sample_rate
    session["athlete_note"] = athlete_note or ""

    uploaded = None
    if file is not None and file.filename:
        raw = _run_sync(file.read())
        uploaded = _UploadShim(file.filename, raw)

    final = turn_engine.handle_turn(session, user_text, uploaded)
    # Only the JSON-safe, UI-relevant subset -- "features"/"forecast" carry
    # numpy scalars the model already used to build chart_html and content,
    # so the UI has no need of them raw and they aren't JSON-serialisable
    # as-is.
    return {
        "content": final.get("content"),
        "chart_ref": final.get("chart_ref"),
        "chart_caption": final.get("chart_caption"),
        "forecast_chart_html": final.get("forecast_chart_html"),
        "forecast_chart_caption": final.get("forecast_chart_caption"),
        "recommendation": final.get("recommendation"),
        "provenance": final.get("provenance"),
        "suggestions": final.get("suggestions"),
        "window": final.get("window"),
    }


def _run_sync(coro):
    """FastAPI's sync def endpoints run in a threadpool, off the event loop,
    so a plain asyncio.run() here (rather than `async def` + `await`) is
    safe and keeps turn_endpoint a normal blocking function like the rest of
    frontend/turn.py's call chain."""
    import asyncio
    return asyncio.run(coro)


@app.post("/chart")
def chart_endpoint(session_id: str = Form(...), source: str = Form("dataset"),
                   subject: int | None = Form(None),
                   t_start: float | None = Form(None),
                   side: str = Form("R")):
    """Draw one figure, on request.

    Charts used to be rendered for every reading whether or not anyone looked
    at one. Measured, that cost ~2 s and a 3.3 MB payload per answer, and it
    was not free even on a background thread -- Plotly figure building is CPU
    work competing with the Ollama process generating the answer on a machine
    with no GPU (a turn with a cold chart took 10.2 s against 8.3 s without).
    So POST /turn now returns a `chart_ref` describing what COULD be drawn,
    the UI shows a "Show graph" button, and this route runs only when it is
    pressed.
    """
    ref = {"source": source, "subject": subject, "t_start": t_start,
           "side": side}
    html = turn_engine.render_chart_ref(_session(session_id), ref)
    if html is None:
        raise HTTPException(status_code=404,
                            detail="That figure could not be drawn.")
    return {"chart_html": html}


@app.get("/classify")
def classify_endpoint(subject: int, t_start: float, side: str = "R"):
    """Return {mdf_hz, fatigue_label, confidence, fatigue_state} for one window."""
    try:
        result = classify(subject, t_start, side)
    except KeyError as e:                     # e.g. no baseline for that subject
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:                     # bad time, missing data, etc.
        raise HTTPException(status_code=400, detail=str(e))
    # add a human-readable label so the LLM doesn't have to decode the int
    result["fatigue_state"] = _STATE.get(result["fatigue_label"],
                                         str(result["fatigue_label"]))
    return result


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    # 0.0.0.0 so the Docker container (or a teammate on the LAN) can reach it
    uvicorn.run(app, host="0.0.0.0", port=8000)
