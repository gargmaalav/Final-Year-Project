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
    seq_len = cfg["seq_len"]

    # subject fresh-baseline (the SAME transform training used)
    bl = bundle["baselines"].get(subject) or bundle["baselines"].get(str(subject))
    if bl is None:
        # TODO: a real, uncalibrated athlete has no fresh baseline yet. Needs a
        #       short "fresh" calibration recording before classify() can
        #       normalise their windows. For now only trained subjects are valid.
        raise KeyError(f"no fresh-baseline calibration stored for subject {subject}")
    mu, sd = np.array(bl["mu"], float), np.array(bl["sd"], float)

    # load this subject's (downsampled, band-passed) signal
    seg = loader.load_biceps_segment(DATA_ROOT, subject, side,
                                     target_fs=cfg["target_fs"], bandpass=True)
    fs = int(getattr(seg, "eff_fs", loader.FS_NATIVE))
    x = seg.data[:, 0]
    t = np.asarray(seg.t, float)
    win = max(2, int(round(cfg["win_sec"] * fs)))
    step = max(1, int(round(cfg["step_sec"] * fs)))

    if x.size < win:
        raise ValueError(f"subject {subject} recording shorter than one window")

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


if __name__ == "__main__":
    # smoke test -- needs fatigue_model.pt (train_model.py) + the dataset present.
    # subject 13 is held out of training, so this is a genuine unseen-subject demo.
    import json
    print(json.dumps(classify(13, 120.0, "R"), indent=2))
