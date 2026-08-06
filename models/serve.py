"""
HTTP wrapper around classify() -- also serves the standalone chatbot UI.
=========================================================================

Two consumers of this API:
  1. viz/chatbot_ui.html -- served at "/" by this file. A self-contained
     frontend (no build step, no LLM-side tool-calling) that calls
     /classify, /render, /answer, /forecast, and the upload variants
     directly. This is the primary demo UI.
  2. Open WebUI, as an optional alternate frontend -- it calls this same
     API via function calling (models/openwebui_tool_reference.py). Kept
     working; not required to run the UI in (1).

One-time setup:
    pip install fastapi uvicorn requests
    # requests is new: /answer and /answer_upload call Ollama's HTTP API
    # directly (http://localhost:11434). Also needs `ollama serve` running
    # with a model pulled (default: llama3.1:8b) -- `ollama run llama3.1:8b`
    # once before your first query, or the model's cold-load can trip the
    # 60s request timeout on the very first classify.

Run (from the repo root):
    python models/serve.py            # UI at http://localhost:8000

Test in a browser:
    http://localhost:8000/                          # chatbot UI
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
import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
import uvicorn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "viz"))
sys.path.insert(0, os.path.join(REPO_ROOT, "zenodo_biceps"))
from classify import classify, classify_upload, classify_from_segment  # noqa: E402
from render_window import (                          # noqa: E402
    render_window, render_segment, TARGET_FS, WIN_SEC, STEP_SEC, _load_subject, DATA_ROOT)
from fatigue_forecast import forecast_fatigue         # noqa: E402  MDF trend projection
import loader                                         # noqa: E402  to_segment, mdf_trend

app = FastAPI(title="EMG Fatigue classify() API")

_STATE = {0: "non-fatigue", 1: "fatigue", 2: "fatigue"}

_UI_PATH = os.path.join(REPO_ROOT, "viz", "chatbot_ui.html")


@app.get("/")
def chatbot_ui():
    """Serve the standalone chatbot UI (viz/chatbot_ui.html)."""
    return FileResponse(_UI_PATH, media_type="text/html")


# --- Ollama-backed chat answers ---------------------------------------------
# Same grounding pattern as models/openwebui_tool_reference.py's get_fatigue():
# hand the LLM the real classify() numbers and tell it to use only those, not
# free-form knowledge. Native Ollama here (not through Open WebUI's chat loop)
# so viz/chatbot_ui.html can be a standalone frontend with no Open WebUI
# dependency.
OLLAMA_BASE = "http://localhost:11434"
DEFAULT_MODEL = "llama3.1:8b"


def _grounding_text(subject, side, t_start, result, source, calibration=None):
    subject_desc = f"subject {subject}" if isinstance(subject, int) else f"the uploaded recording ({subject})"
    lines = [
        f"{subject_desc}, {side} arm, t={t_start}s, source={source}: "
        f"fatigue_state={result.get('fatigue_state')}, "
        f"fatigue_label={result['fatigue_label']}, "
        f"median_frequency={result['mdf_hz']:.1f} Hz, "
        f"confidence={result['confidence']:.2f}.",
    ]
    if calibration:
        lines.append(f"calibration: fresh, first {calibration.get('baseline_sec')}s of this recording "
                     "(no stored per-subject baseline for an upload).")
    if result["confidence"] < 0.6:
        lines.append("Confidence is below 60%, which is low for this model -- say explicitly that "
                     "this reading is uncertain and should be treated as indicative, not a firm verdict.")
    lines.append("Use only these values in your answer -- do not invent numbers, do not add a "
                "baseline/delta figure unless one was given above, and answer in 2-4 sentences.")
    return " ".join(lines)


@app.get("/models")
def list_models():
    """Ollama models available for the chat answer, for the UI's model picker."""
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        r.raise_for_status()
        names = [m["name"] for m in r.json().get("models", [])]
        if names:
            return {"models": names, "default": DEFAULT_MODEL if DEFAULT_MODEL in names else names[0]}
    except requests.RequestException:
        pass
    return {"models": [DEFAULT_MODEL], "default": DEFAULT_MODEL}


def _ask_ollama(model: str, question: str, grounding: str) -> str:
    payload = {
        "model": model or DEFAULT_MODEL,
        "messages": [
            {"role": "system", "content": "You are a concise assistant reporting EMG fatigue "
                                          "readings for a final-year research project. " + grounding},
            {"role": "user", "content": question},
        ],
        "stream": False,
    }
    r = requests.post(f"{OLLAMA_BASE}/api/chat", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


_FORECAST_JSON_KEYS = ("t_observed", "y_observed", "t_future", "y_future",
                      "ci_lo", "ci_hi", "pi_lo", "pi_hi")


def _jsonable_forecast(forecast: dict | None) -> dict | None:
    """forecast_fatigue() returns numpy arrays; FastAPI's default JSON
    encoder can't serialise those, so convert the array fields to plain
    lists and drop anything else non-serialisable (e.g. nested numpy floats
    inside "lstm")."""
    if not forecast:
        return None
    if forecast.get("ok") is False:
        return {"ok": False}
    out = {"ok": True, "summary": forecast["summary"],
          "horizon_sec": forecast["horizon_sec"],
          "clipped_at_zero": forecast["clipped_at_zero"],
          "method": forecast["method"],
          "slope_hz_per_min": float(forecast["slope"]) * 60.0,
          "r2": float(forecast["r2"]), "p_value": float(forecast["p_value"])}
    for k in _FORECAST_JSON_KEYS:
        if k in forecast:
            out[k] = np.asarray(forecast[k]).tolist()
    return out


def _forecast_grounding_line(forecast: dict | None) -> str:
    """Append a forecast summary to the grounding, if one was computed.

    forecast_fatigue() already writes a plain-language "summary" sentence
    that states the observed trend and, separately, the projected value --
    hand that straight to the model rather than re-deriving it, so the
    LLM's forecast claim and the chart's forecast claim can never drift
    apart from each other.
    """
    if not forecast:
        return ""
    if forecast.get("ok") is False:
        return ("A forecast was requested but there isn't enough MDF history "
                "yet to fit a trend -- say so, don't invent one.")
    return "Forecast: " + forecast["summary"]


@app.get("/answer")
def answer_endpoint(subject: int, t_start: float, side: str = "R",
                    question: str = "", model: str = DEFAULT_MODEL,
                    horizon_sec: float | None = None):
    """Classify the window, then have Ollama phrase the result (grounded).

    horizon_sec: if given, also run forecast_fatigue() for that many seconds
    ahead of t_start and fold its summary into the grounding text.
    """
    if not (1 <= subject <= 13):
        raise HTTPException(status_code=400,
                            detail=f"subject must be 1-13, got {subject}. Try subject 13.")
    if side.upper() not in ("R", "L"):
        raise HTTPException(status_code=400,
                            detail=f"side must be 'R' or 'L', got {side!r}.")
    try:
        result = classify(subject, t_start, side)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    result["fatigue_state"] = _STATE.get(result["fatigue_label"], str(result["fatigue_label"]))
    grounding = _grounding_text(subject, side, t_start, result, "dataset")

    forecast = None
    if horizon_sec:
        seg = loader.load_biceps_segment(DATA_ROOT, subject, side,
                                         target_fs=TARGET_FS, bandpass=True)
        fs = int(getattr(seg, "eff_fs", TARGET_FS))
        forecast = forecast_fatigue(seg, fs, horizon_sec=horizon_sec, t_end=t_start)
        grounding += " " + _forecast_grounding_line(forecast)

    q = question or f"Is subject {subject} fatigued at {t_start}s on the {side} side?"
    try:
        text = _ask_ollama(model, q, grounding)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Ollama unreachable: {e}")
    return {"text": text, "result": result, "forecast": _jsonable_forecast(forecast)}


@app.post("/answer_upload")
async def answer_upload_endpoint(file: UploadFile = File(...),
                                 t_start: float = Form(0.0),
                                 sample_rate_hz: float | None = Form(None),
                                 baseline_sec: float = Form(60.0),
                                 question: str = Form(""),
                                 model: str = Form(DEFAULT_MODEL),
                                 horizon_sec: float | None = Form(None)):
    """Same as /answer, for an uploaded recording."""
    raw = await file.read()
    try:
        seg, fs = _load_upload_segment(raw, sample_rate_hz)
        result = classify_upload(seg, fs, t_start, baseline_sec=baseline_sec)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    result["fatigue_state"] = _STATE.get(result["fatigue_label"], str(result["fatigue_label"]))
    grounding = _grounding_text(file.filename, "R", t_start, result, "upload",
                                calibration=result.get("calibration"))

    forecast = None
    if horizon_sec:
        forecast = forecast_fatigue(seg, fs, horizon_sec=horizon_sec, t_end=t_start)
        grounding += " " + _forecast_grounding_line(forecast)

    q = question or f"Am I fatigued in {file.filename} at {t_start}s?"
    try:
        text = _ask_ollama(model, q, grounding)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Ollama unreachable: {e}")
    return {"text": text, "result": result, "forecast": _jsonable_forecast(forecast)}


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
        # classify_from_segment() reuses the seg/fs already loaded above --
        # classify() itself reloads the whole subject recording from disk
        # EVERY call (~3s), which made this loop take minutes per chart when
        # it ran classify() once per window (100+ windows). Same segment
        # object as classify() would load internally (both use TARGET_FS).
        model_preds = {}
        for tc in mdf_t:
            try:
                r = classify_from_segment(subject, seg, fs, float(tc))
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


@app.get("/forecast")
def forecast_endpoint(subject: int, t_start: float, side: str = "R",
                      horizon_sec: float = 20.0):
    """MDF trend projection (models/fatigue_forecast.py), independent of
    /answer -- lets the UI show the forecast even when Ollama is down."""
    if not (1 <= subject <= 13):
        raise HTTPException(status_code=400,
                            detail=f"subject must be 1-13, got {subject}. Try subject 13.")
    if side.upper() not in ("R", "L"):
        raise HTTPException(status_code=400,
                            detail=f"side must be 'R' or 'L', got {side!r}.")
    try:
        seg = loader.load_biceps_segment(DATA_ROOT, subject, side,
                                         target_fs=TARGET_FS, bandpass=True)
    except (KeyError, FileNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    fs = int(getattr(seg, "eff_fs", TARGET_FS))
    forecast = forecast_fatigue(seg, fs, horizon_sec=horizon_sec, t_end=t_start)
    return {"forecast": _jsonable_forecast(forecast)}


@app.post("/forecast_upload")
async def forecast_upload_endpoint(file: UploadFile = File(...),
                                   t_start: float = Form(0.0),
                                   sample_rate_hz: float | None = Form(None),
                                   horizon_sec: float = Form(20.0)):
    """Same as /forecast, for an uploaded recording."""
    raw = await file.read()
    try:
        seg, fs = _load_upload_segment(raw, sample_rate_hz)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    forecast = forecast_fatigue(seg, fs, horizon_sec=horizon_sec, t_end=t_start)
    return {"forecast": _jsonable_forecast(forecast)}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    # 0.0.0.0 so any other host/container on the network can still reach it
    uvicorn.run(app, host="0.0.0.0", port=8000)
