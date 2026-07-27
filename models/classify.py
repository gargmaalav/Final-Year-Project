"""
classify(subject, t_start, side) -> {mdf_hz, fatigue_label, confidence}
======================================================================

The team integration contract (chatbot_plan 2026-06-15 / README on main):
  Produced by: Aryan.  Consumed by: the chatbot frontend via FUNCTION CALLING
  (the supervisor's directive -- the LLM calls this to ground its answers).

Uses the deep-learning LSTM saved by models/train_model.py (run that once
first). See train_model.py for the LSTM-now / Transformer-later rationale.

This file is a thin wrapper: it reuses classify_biceps.window_features and the
LSTM model class -- no new modelling logic.

Also implements the "uncalibrated athlete" TODO: classify_upload() classifies
a recording that has no pre-stored per-subject baseline (e.g. a user-uploaded
EMG file) by computing a fresh baseline from the recording's own early
window(s) instead of from ground-truth fatigue labels.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZB = os.path.join(REPO_ROOT, "zenodo_biceps")
sys.path.insert(0, ZB)
import classify_biceps as cb         # noqa: E402  window_features, BASE_FEATS
import lstm_classify_biceps as lstm  # noqa: E402  LSTMClassifier
import loader                        # noqa: E402

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "fatigue_model.pt")
DATA_ROOT = os.path.join(ZB, "sEMG_data")

_BUNDLE = None
_MODEL = None


def _load():
    """Load the saved model + metadata once and cache it."""
    global _BUNDLE, _MODEL
    if _MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"{MODEL_PATH} missing -- run `python models/train_model.py` first")
        _BUNDLE = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        a = _BUNDLE["arch"]
        _MODEL = lstm.LSTMClassifier(
            n_features=a["n_features"], hidden_size=a["hidden_size"],
            num_layers=a["num_layers"], n_classes=a["n_classes"],
            dropout=a["dropout"])
        _MODEL.load_state_dict(_BUNDLE["state_dict"])
        _MODEL.eval()
    return _BUNDLE, _MODEL


def _classify_window(seg, fs: int, t_start: float, mu: np.ndarray, sd: np.ndarray,
                     cfg: dict, base_feats: list[str], model) -> dict:
    """Shared inference body: build the causal window sequence at `t_start`,
    normalise with the given (mu, sd), and run the LSTM. Used by both the
    dataset path (classify()) and the upload path (classify_upload())."""
    x = seg.data[:, 0]
    t = np.asarray(seg.t, float)
    win = max(2, int(round(cfg["win_sec"] * fs)))
    step = max(1, int(round(cfg["step_sec"] * fs)))
    seq_len = cfg["seq_len"]

    if x.size < win:
        raise ValueError(
            f"recording ({x.size / fs:.1f}s) is shorter than one "
            f"{cfg['win_sec']:.0f}s window")

    # index of the window starting at t_start (clamped into range)
    cur = int(np.searchsorted(t, t_start))
    cur = min(max(cur, 0), x.size - win)

    # causal sequence of seq_len windows ending at the current one, repeating
    # the earliest window when the recording starts (matches build_sequences'
    # replicate-padding)
    starts = [max(0, cur - step * k) for k in range(seq_len - 1, -1, -1)]
    rows, cur_feat = [], None
    for j, st in enumerate(starts):
        feat = cb.window_features(x[st:st + win], fs)
        vec = np.array([feat[k] for k in base_feats], float)
        rows.append((vec - mu) / sd)        # normalise each timestep
        if j == len(starts) - 1:
            cur_feat = feat                  # current window's raw features
    seq = np.asarray(rows, float)[None, :, :]   # (1, seq_len, n_features)

    # predict label + confidence (softmax over logits)
    with torch.no_grad():
        logits = model(torch.tensor(seq, dtype=torch.float32))
        prob = torch.softmax(logits, dim=1)[0].numpy()
    label = int(prob.argmax())

    return {
        "mdf_hz": float(cur_feat["mdf"]),
        "fatigue_label": label,
        "confidence": float(prob[label]),
    }


def classify(subject: int, t_start: float, side: str = "R") -> dict:
    """Classify the EMG window starting at `t_start` for one subject.

    Args:
        subject: subject id (1-13 in the Zenodo dataset).
        t_start: window start time in seconds.
        side: "R" or "L".

    Returns:
        {"mdf_hz": float, "fatigue_label": int, "confidence": float}
        fatigue_label: 0 = non-fatigue, 1 = fatigue (see bundle["label_meaning"]).
    """
    bundle, model = _load()
    cfg = bundle["config"]
    base_feats = bundle["base_feats"]

    # subject fresh-baseline (the SAME transform training used)
    bl = bundle["baselines"].get(subject) or bundle["baselines"].get(str(subject))
    if bl is None:
        # a real, uncalibrated athlete has no fresh baseline yet -- see
        # classify_upload() below, which computes one from a fresh recording.
        raise KeyError(f"no fresh-baseline calibration stored for subject {subject}")
    mu, sd = np.array(bl["mu"], float), np.array(bl["sd"], float)

    # load this subject's (downsampled, band-passed) signal
    seg = loader.load_biceps_segment(DATA_ROOT, subject, side,
                                     target_fs=cfg["target_fs"], bandpass=True)
    fs = int(getattr(seg, "eff_fs", loader.FS_NATIVE))
    return _classify_window(seg, fs, t_start, mu, sd, cfg, base_feats, model)


def compute_fresh_baseline(seg, fs: int, cfg: dict, base_feats: list[str],
                           fresh_sec: float = 15.0) -> dict:
    """Baseline mu/sd for a recording with no ground-truth fatigue labels.

    Mirrors train_model.py's `base = X[y==0]; mu, sd = base.mean(0),
    base.std(0)`, but since an uploaded recording has no per-window fatigue
    labels, "fresh" is approximated as every window whose center falls in
    the recording's first `fresh_sec` seconds -- true of all 13 training
    subjects too, since every trial starts unfatigued.
    """
    x = seg.data[:, 0]
    t = np.asarray(seg.t, float)
    win = max(2, int(round(cfg["win_sec"] * fs)))
    step = max(1, int(round(cfg["step_sec"] * fs)))
    n_fresh_samples = int(np.searchsorted(t, fresh_sec))

    feats = []
    start = 0
    while start + win <= min(n_fresh_samples, x.size):
        feat = cb.window_features(x[start:start + win], fs)
        feats.append([feat[k] for k in base_feats])
        start += step

    if len(feats) < 3:
        raise ValueError(
            f"only {len(feats)} fresh window(s) found in the first "
            f"{fresh_sec:.0f}s -- need at least 3; upload a longer recording "
            "or lower the fresh-baseline duration")

    X = np.array(feats, float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    return {"mu": mu.tolist(), "sd": sd.tolist()}


def classify_upload(seg, fs: int, t_start: float, fresh_sec: float = 15.0,
                    baseline: dict | None = None) -> tuple[dict, dict]:
    """Classify a window in a recording that has no stored per-subject
    baseline (e.g. a user-uploaded EMG file) -- implements the "uncalibrated
    athlete" TODO by computing a fresh baseline from the recording's own
    early window(s) instead of ground-truth fatigue labels.

    Pass a previously-returned `baseline` back in on follow-up questions
    about the same recording to skip recomputing it.

    Returns (result, baseline): `result` is the same shape classify() returns,
    `baseline` is {"mu": [...], "sd": [...]} for reuse.
    """
    bundle, model = _load()
    cfg = bundle["config"]
    base_feats = bundle["base_feats"]

    if baseline is None:
        baseline = compute_fresh_baseline(seg, fs, cfg, base_feats, fresh_sec)
    mu, sd = np.array(baseline["mu"], float), np.array(baseline["sd"], float)

    result = _classify_window(seg, fs, t_start, mu, sd, cfg, base_feats, model)
    return result, baseline


if __name__ == "__main__":
    # smoke test -- needs fatigue_model.pt (train_model.py) + the dataset present.
    # subject 13 is held out of training, so this is a genuine unseen-subject demo.
    import json
    print(json.dumps(classify(13, 120.0, "R"), indent=2))
