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


def _subject_baseline(bundle: dict, subject: int) -> tuple[np.ndarray, np.ndarray]:
    """The stored fresh-baseline (mu, sd) for one subject.

    Raises KeyError when the subject has none -- a real, uncalibrated athlete
    is handled by classify_upload() below, which computes one from a fresh
    recording instead.
    """
    bl = bundle["baselines"].get(subject) or bundle["baselines"].get(str(subject))
    if bl is None:
        raise KeyError(f"no fresh-baseline calibration stored for subject {subject}")
    return np.array(bl["mu"], float), np.array(bl["sd"], float)


def mdf_reference_from(mu, sd, base_feats: list[str]) -> dict:
    """The fresh median-frequency reference implied by a baseline.

    A raw "51.2 Hz" tells a reader nothing -- subjects sit anywhere from 59 to
    81 Hz when fresh, so the same number is unremarkable for one person and a
    steep fall for another. What is interpretable is the distance from *that
    person's own* fresh state, which the stored baseline already contains: mu
    is their fresh mean and sd their fresh spread, both in Hz for the mdf
    feature. No extra computation -- this is read straight out of the bundle.
    """
    i = base_feats.index("mdf")
    spread = float(sd[i])
    return {"fresh_mdf": float(mu[i]), "sd_mdf": spread if spread > 0 else None}


def subject_reference(subject: int) -> dict:
    """The stored fresh reference for one of the dataset subjects."""
    bundle, _ = _load()
    mu, sd = _subject_baseline(bundle, subject)
    return mdf_reference_from(mu, sd, bundle["base_feats"])


def upload_reference(baseline: dict) -> dict:
    """The same, for a recording calibrated against its own fresh window."""
    bundle, _ = _load()
    return mdf_reference_from(baseline["mu"], baseline["sd"],
                              bundle["base_feats"])


def available_subjects() -> list[int]:
    """Subject ids that have a stored calibration, so callers can say what
    exists rather than hard-coding "1-13" and going stale if the bundle
    changes."""
    bundle, _ = _load()
    out = set()
    for key in bundle["baselines"]:
        try:
            out.add(int(key))
        except (TypeError, ValueError):
            continue
    return sorted(out)


def load_subject_segment(subject: int, side: str = "R"):
    """The (downsampled, band-passed) signal for one subject, plus its rate.

    Split out of classify() so callers that need many windows from the same
    recording can load it once -- see classify_many().
    """
    bundle, _ = _load()
    seg = loader.load_biceps_segment(DATA_ROOT, subject, side,
                                     target_fs=bundle["config"]["target_fs"],
                                     bandpass=True)
    return seg, int(getattr(seg, "eff_fs", loader.FS_NATIVE))


def classify(subject: int, t_start: float, side: str = "R", seg=None,
             fs: int | None = None) -> dict:
    """Classify the EMG window starting at `t_start` for one subject.

    Args:
        subject: subject id (1-13 in the Zenodo dataset).
        t_start: window start time in seconds.
        side: "R" or "L".
        seg, fs: an already-loaded segment for this subject/side. Optional and
            purely a cost optimisation -- omit them and the recording is
            loaded here, exactly as before.

    Returns:
        {"mdf_hz": float, "fatigue_label": int, "confidence": float}
        fatigue_label: 0 = non-fatigue, 1 = fatigue (see bundle["label_meaning"]).
    """
    bundle, model = _load()
    cfg = bundle["config"]
    base_feats = bundle["base_feats"]

    # subject fresh-baseline (the SAME transform training used)
    mu, sd = _subject_baseline(bundle, subject)

    if seg is None:
        seg, fs = load_subject_segment(subject, side)
    return _classify_window(seg, fs, t_start, mu, sd, cfg, base_feats, model)


def classify_many(subject: int, t_starts, side: str = "R", seg=None,
                  fs: int | None = None) -> list[dict]:
    """classify() at several times in one recording, loading it only once.

    Scanning a whole recording through classify() re-reads and re-resamples a
    multi-MB CSV per window, which makes questions like "when did they start
    fatiguing?" cost tens of seconds. Each returned dict is classify()'s, plus
    the "t_start" it was measured at.
    """
    bundle, model = _load()
    cfg = bundle["config"]
    base_feats = bundle["base_feats"]
    mu, sd = _subject_baseline(bundle, subject)

    if seg is None:
        seg, fs = load_subject_segment(subject, side)

    out = []
    for t_start in t_starts:
        result = _classify_window(seg, fs, float(t_start), mu, sd, cfg,
                                  base_feats, model)
        result["t_start"] = float(t_start)
        out.append(result)
    return out


# Calibration constants for the uncalibrated-athlete path. These are not
# arbitrary -- see CALIBRATION_VALIDATION.md for the sweep that set them.
FRESH_SEC = 60.0        # baseline window; 15 s cost ~9pp accuracy, 60 s costs ~2pp
MIN_FRESH_WINDOWS = 12  # a 15 s baseline gives 6 windows -- far too few for a stable sd
MIN_BASELINE_FRACTION = 2.5   # recording must be >= this x the baseline span
                              # (2.5 scores 12/13 subjects; 3.0 needlessly
                              #  refused two 178 s recordings for no gain)


def compute_fresh_baseline(seg, fs: int, cfg: dict, base_feats: list[str],
                           fresh_sec: float = FRESH_SEC) -> dict:
    """Baseline mu/sd for a recording with no ground-truth fatigue labels.

    Mirrors train_model.py's `base = X[y==0]; mu, sd = base.mean(0),
    base.std(0)`, but since an uploaded recording has no per-window fatigue
    labels, "fresh" is approximated as every window falling entirely within
    the recording's first `fresh_sec` seconds -- reasonable for this protocol,
    since every trial starts unfatigued.

    Two guards, both added after measuring the failure modes on all 13 labelled
    subjects (CALIBRATION_VALIDATION.md):

    1. sd is a variance estimate, so it needs samples. A 15 s baseline yields
       6 windows and systematically UNDER-estimates sd, which inflates every
       z-score and makes the LSTM read "extreme" -> it over-predicts fatigue
       (66% of windows vs a true rate of 52%) at undiminished confidence.
    2. If the baseline spans most of the recording, the recording is being
       normalised against itself: mu lands on the recording's own mean, so
       every window looks "average" and the answer is meaningless regardless
       of the athlete's true state. Refusing is the only honest option -- a
       short clip genuinely does not contain the information needed.

    Raises ValueError with a user-facing message when either guard trips.
    """
    x = seg.data[:, 0]
    t = np.asarray(seg.t, float)
    win = max(2, int(round(cfg["win_sec"] * fs)))
    step = max(1, int(round(cfg["step_sec"] * fs)))
    duration = float(t[-1]) if t.size else 0.0

    # guard 2 first: it gives the clearer error when the recording is just short
    if duration < fresh_sec * MIN_BASELINE_FRACTION:
        raise ValueError(
            f"this recording is {duration:.0f}s long, but a trustworthy fresh "
            f"baseline needs the first {fresh_sec:.0f}s to be a small part of "
            f"it (at least {fresh_sec * MIN_BASELINE_FRACTION:.0f}s total). "
            "Below that the recording is normalised against itself and the "
            "fatigue reading is meaningless -- please upload a longer recording.")

    n_fresh_samples = int(np.searchsorted(t, fresh_sec))
    feats = []
    start = 0
    while start + win <= min(n_fresh_samples, x.size):
        feat = cb.window_features(x[start:start + win], fs)
        feats.append([feat[k] for k in base_feats])
        start += step

    if len(feats) < MIN_FRESH_WINDOWS:
        raise ValueError(
            f"only {len(feats)} baseline window(s) in the first {fresh_sec:.0f}s "
            f"-- need at least {MIN_FRESH_WINDOWS} for a stable estimate of the "
            "athlete's fresh spread. Please upload a longer recording.")

    X = np.array(feats, float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    return {"mu": mu.tolist(), "sd": sd.tolist(),
            "fresh_sec": float(fresh_sec), "n_windows": int(len(feats)),
            "source": "computed-from-recording"}


def classify_upload(seg, fs: int, t_start: float, fresh_sec: float = FRESH_SEC,
                    baseline: dict | None = None) -> tuple[dict, dict]:
    """Classify a window in a recording that has no stored per-subject
    baseline (e.g. a user-uploaded EMG file) -- implements the "uncalibrated
    athlete" TODO by computing a fresh baseline from the recording's own
    early windows instead of ground-truth fatigue labels.

    Pass a previously-returned `baseline` back in on follow-up questions
    about the same recording to skip recomputing it.

    Returns (result, baseline). `result` has classify()'s three contract keys
    plus "calibration", which flags that this reading used a self-computed
    baseline rather than a stored one -- callers should surface that, because
    it is measurably less accurate than the calibrated path (see
    CALIBRATION_VALIDATION.md: 81% vs 84% on the labelled subjects).
    """
    bundle, model = _load()
    cfg = bundle["config"]
    base_feats = bundle["base_feats"]

    if baseline is None:
        baseline = compute_fresh_baseline(seg, fs, cfg, base_feats, fresh_sec)
    mu, sd = np.array(baseline["mu"], float), np.array(baseline["sd"], float)

    result = _classify_window(seg, fs, t_start, mu, sd, cfg, base_feats, model)
    result["calibration"] = {
        "kind": "self-calibrated",
        "fresh_sec": baseline.get("fresh_sec", fresh_sec),
        "n_windows": baseline.get("n_windows"),
        "note": ("baseline computed from this recording's own first "
                 f"{baseline.get('fresh_sec', fresh_sec):.0f}s, not from a "
                 "stored per-athlete calibration"),
    }
    return result, baseline


if __name__ == "__main__":
    # smoke test -- needs fatigue_model.pt (train_model.py) + the dataset present.
    # subject 13 is held out of training, so this is a genuine unseen-subject demo.
    import json
    print(json.dumps(classify(13, 120.0, "R"), indent=2))
