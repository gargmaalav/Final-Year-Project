"""
HTTP wrapper around classify() so the Open WebUI chatbot can reach it.
=====================================================================

Open WebUI can't import this project's code directly, so it calls this small
API via function calling. Open WebUI runs natively on this host (verified
2026-07-12: `.venv/bin/open-webui serve`, no Docker in this repo), so it
reaches this at http://localhost:8000. This is the "link your deep-learning
algorithms to the chatbot frontend" bridge the supervisor asked for.

One-time setup:
    pip install fastapi uvicorn

Run (from the repo root):
    python models/serve.py            # serves on http://localhost:8000

Test in a browser:
    http://localhost:8000/classify?subject=13&t_start=120
    http://localhost:8000/render?subject=13&t_start=120
    http://localhost:8000/health

Upload endpoints (POST, multipart form) take an arbitrary EMG recording
instead of a dataset subject id -- see classify_upload_endpoint()/
render_upload_endpoint() below for the CSV formats accepted.
"""
from __future__ import annotations

import io
import os
import sys

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
import uvicorn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "viz"))
sys.path.insert(0, os.path.join(REPO_ROOT, "zenodo_biceps"))
from classify import classify, classify_upload       # noqa: E402  contract fns
from render_window import (                          # noqa: E402
    render_window, render_segment, TARGET_FS, WIN_SEC, STEP_SEC, _load_subject)
import loader                                         # noqa: E402  to_segment, mdf_trend

app = FastAPI(title="EMG Fatigue classify() API")

_STATE = {0: "non-fatigue", 1: "fatigue", 2: "fatigue"}


def _parse_csv_upload(raw: bytes, sample_rate_hz: float | None):
    """(t_native, x_native, fs_native) from a two- or one-column EMG CSV.

    Two columns (any header, or none): first = time in seconds, second =
    signal -- sample rate is inferred from the median time step. One column:
    signal only, `sample_rate_hz` is required (the caller's own recording
    device rate; we have no header to read it from, and headerless single-
    column files are ambiguous about units without it).
    """
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise ValueError(f"could not read CSV: {e}")
    if df.shape[1] < 1:
        raise ValueError("CSV has no columns")

    if df.shape[1] >= 2:
        t_native = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(float)
        x_native = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(float)
        if np.isnan(t_native).any() or np.isnan(x_native).any():
            raise ValueError("non-numeric values in the time/signal columns")
        dt = np.median(np.diff(t_native))
        if dt <= 0:
            raise ValueError("time column is not monotonically increasing")
        fs_native = 1.0 / dt
    else:
        if not sample_rate_hz or sample_rate_hz <= 0:
            raise ValueError(
                "single-column CSV: pass sample_rate_hz (your recording "
                "device's sample rate) -- it can't be inferred with no time column")
        x_native = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(float)
        if np.isnan(x_native).any():
            raise ValueError("non-numeric values in the signal column")
        fs_native = float(sample_rate_hz)
        t_native = np.arange(x_native.size) / fs_native

    return t_native, x_native, fs_native


def _load_upload_segment(raw: bytes, sample_rate_hz: float | None):
    t_native, x_native, fs_native = _parse_csv_upload(raw, sample_rate_hz)
    seg = loader.to_segment(t_native, x_native, fs=fs_native,
                            target_fs=TARGET_FS, bandpass=True)
    return seg, int(seg.eff_fs)


@app.get("/classify")
def classify_endpoint(subject: int, t_start: float, side: str = "R"):
    """Return {mdf_hz, fatigue_label, confidence, fatigue_state} for one window."""
    if not (1 <= subject <= 13):
        raise HTTPException(status_code=400,
                            detail=f"subject must be 1-13, got {subject}. Try subject 13.")
    if side.upper() not in ("R", "L"):
        raise HTTPException(status_code=400,
                            detail=f"side must be 'R' or 'L', got {side!r}.")
    try:
        result = classify(subject, t_start, side)
    except KeyError as e:                     # e.g. no baseline for that subject
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:                     # bad time, missing data, etc.
        raise HTTPException(status_code=400, detail=str(e))
    # add a human-readable label so the LLM doesn't have to decode the int
    result["fatigue_state"] = _STATE.get(result["fatigue_label"],
                                         str(result["fatigue_label"]))
    # provenance: exactly which query this answer grounds, so a chatbot reply
    # can state it rather than leave the reader to trust an invented-sounding
    # number (mirrors the grounding fix teammates made on the Streamlit branch)
    result["provenance"] = {"subject": subject, "t_start": t_start, "side": side,
                            "source": "dataset"}
    return result


@app.post("/classify_upload")
async def classify_upload_endpoint(file: UploadFile = File(...),
                                   t_start: float = Form(0.0),
                                   sample_rate_hz: float | None = Form(None),
                                   baseline_sec: float = Form(60.0)):
    """Same shape as /classify, for an uploaded recording (no dataset subject).

    CSV formats: two columns (time_s, signal -- rate inferred) or one column
    (signal only -- pass sample_rate_hz). Needs >= baseline_sec + 4s of
    recording for a fresh calibration; shorter uploads are refused with a
    clear message rather than silently normalised against themselves.
    """
    raw = await file.read()
    try:
        seg, fs = _load_upload_segment(raw, sample_rate_hz)
        result = classify_upload(seg, fs, t_start, baseline_sec=baseline_sec)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    result["fatigue_state"] = _STATE.get(result["fatigue_label"],
                                         str(result["fatigue_label"]))
    result["provenance"] = {"t_start": t_start, "source": "upload",
                            "filename": file.filename,
                            "recording_sec": round(float(seg.t[-1]), 1)}
    return result


@app.get("/render")
def render_endpoint(subject: int, t_start: float = 0, side: str = "R"):
    """Return {"html": ...} interactive Plotly chart for one subject's window.

    Also runs classify() once per MDF window (same WIN_SEC/STEP_SEC grid
    render_window() plots) so the chart can overlay the model's OWN
    prediction at every window, not just the single asked-about point --
    see viz.render_window's model_preds param (honesty/explainability plan
    Task 2: ground-truth dots alone read, to a non-technical viewer, as if
    they were the model's call).
    """
    if not (1 <= subject <= 13):
        raise HTTPException(status_code=400,
                            detail=f"subject must be 1-13, got {subject}. Try subject 13.")
    if side.upper() not in ("R", "L"):
        raise HTTPException(status_code=400,
                            detail=f"side must be 'R' or 'L', got {side!r}.")
    try:
        seg, fs, _, _ = _load_subject(subject, side)
        mdf_t, _, _ = loader.mdf_trend(seg, fs=fs, win_sec=WIN_SEC, step_sec=STEP_SEC)
        model_preds = {}
        for tc in mdf_t:
            try:
                r = classify(subject, float(tc), side)
                model_preds[float(tc)] = r["fatigue_label"]
            except Exception:
                pass  # a single window's failure must not blank the whole overlay
    except Exception:
        model_preds = None   # overlay is additive: chart still renders without it
    try:
        html = render_window(subject, t_start, side, model_preds=model_preds)
    except (KeyError, FileNotFoundError) as e:  # no data for that subject/side
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:                       # bad subject/side/time, etc.
        raise HTTPException(status_code=400, detail=str(e))
    return {"html": html}


@app.post("/render_upload")
async def render_upload_endpoint(file: UploadFile = File(...),
                                 t_start: float = Form(0.0),
                                 sample_rate_hz: float | None = Form(None)):
    """Return {"html": ...} interactive chart for an uploaded recording."""
    raw = await file.read()
    try:
        seg, fs = _load_upload_segment(raw, sample_rate_hz)
        html = render_segment(seg, fs, t_start)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"html": html}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    # 0.0.0.0 so any other host/container on the network can still reach it
    uvicorn.run(app, host="0.0.0.0", port=8000)
