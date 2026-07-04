"""
HTTP wrapper around classify() so the Open WebUI chatbot can reach it.
=====================================================================

Open WebUI runs inside Docker and can't import this project's code directly, so
it calls this small API on the host via function calling
(http://host.docker.internal:8000). This is the "link your deep-learning
algorithms to the chatbot frontend" bridge the supervisor asked for.

One-time setup:
    pip install fastapi uvicorn

Run (from the repo root):
    python models/serve.py            # serves on http://localhost:8000

Test in a browser:
    http://localhost:8000/classify?subject=13&t_start=120
    http://localhost:8000/health
"""
from __future__ import annotations

import os
import sys

from fastapi import FastAPI, HTTPException
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classify import classify  # noqa: E402  our contract function

app = FastAPI(title="EMG Fatigue classify() API")

_STATE = {0: "non-fatigue", 1: "fatigue", 2: "fatigue"}


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
    # 0.0.0.0 so the Docker container can reach it via host.docker.internal
    uvicorn.run(app, host="0.0.0.0", port=8000)
