"""
LSTM MDF forecaster: predict the fatigue marker `horizon` seconds ahead.
=======================================================================

The "prediction using deep learning" half of the supervisor's brief, reframed
from "predict the next signal value" to "predict the next FATIGUE LEVEL".

Why the reframe: raw sEMG is band-limited noise. Measured on subject 13 with a
held-out AR(10) baseline, the autocorrelation of the bandpassed signal is
+0.084 at 20 ms and +0.002 at 100 ms, and one-step-ahead R^2 collapses from
+0.52 at 4 ms (which is the 20-450 Hz filter's own smoothing, not physiology)
to -0.000 at 100 ms. There is no linearly-predictable structure to learn
beyond a few milliseconds, so no architecture can forecast future samples.
The median frequency underneath it, however, drifts down steadily and
significantly as the muscle fatigues -- that is the forecastable quantity, and
it is the one the chatbot is actually asked about.

What this replaces: models/fatigue_forecast.py fits an OLS straight line
through the MDF history and extends it. That is what ships today, and left
unbounded it predicts -106 Hz at a one-hour horizon (hence MAX_HORIZON_SEC and
the clamp at zero). This trains a sequence model on the same feature stream the
classifier uses and asks whether it beats that line out-of-subject.

Task: given a causal sequence of the last `--seq-len` feature windows ending at
time t, predict MDF(t + h) - MDF(t) for each h in `--horizons`. Predicting the
CHANGE rather than the absolute level is what makes the task transferable --
absolute MDF is subject-specific (these subjects sit anywhere from ~55 to
~95 Hz), so a model asked for absolute values would have to guess a held-out
athlete's baseline, which it cannot see.

Baselines, all evaluated causally on the identical samples:
  persistence  -- predict no change. The bar any forecaster must clear.
  ols_full     -- fit OLS on all MDF history up to t, extrapolate to t+h.
                  This is what models/fatigue_forecast.py does today.
  ols_recent   -- same, but fit only on the last --ols-window seconds.
  drift        -- the mean Hz/s drift rate of the TRAINING subjects x h.
                  A one-number learned model; if the LSTM cannot beat this,
                  the sequence information is not being used.

Validation is leave-one-subject-out, same as classify_biceps/lstm_classify_
biceps, and the LSTM is retrained per fold over `--seeds` seeds so a win has to
survive seed noise on a dataset this small.

Usage:
    python models/forecast_lstm.py --root zenodo_biceps/sEMG_data
    python models/forecast_lstm.py --root zenodo_biceps/sEMG_data --horizons 10 30 60
    python models/forecast_lstm.py --root zenodo_biceps/sEMG_data \
        --json-out zenodo_biceps/out/metrics_forecast_lstm_250hz.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch import nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(REPO_ROOT, "zenodo_biceps"),
           os.path.join(REPO_ROOT, "convergence_analysis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import loader             # noqa: E402
import classify_biceps as cb  # noqa: E402  window_features, BASE_FEATS

# Baseline span for the per-subject z-score. Matches models/classify.py's
# FRESH_SEC, which was set by measurement (CALIBRATION_VALIDATION.md) -- a
# shorter baseline under-estimates sd and inflates every z-score.
FRESH_SEC = 60.0
MIN_HISTORY_WINDOWS = 5   # OLS needs some history before it can fit anything


# ---------------------------------------------------------------------------
# per-subject feature + MDF series
# ---------------------------------------------------------------------------
def segment_series(seg, fs: int, win_sec: float, step_sec: float) -> dict | None:
    """Any Segment -> {t, X (n, 8), mdf (n,)} over uniform causal windows.

    Uses the same windowing and the same 8 base features as the classifier, so
    a forecast and a classification of the same moment describe the same data.
    Shared by training and by inference, so the two can never drift apart.
    """
    x = seg.data[:, 0]
    win = max(2, int(round(win_sec * fs)))
    step = max(1, int(round(step_sec * fs)))

    t_centers, rows = [], []
    start = 0
    while start + win <= x.size:
        feat = cb.window_features(x[start:start + win], fs)
        rows.append([feat[k] for k in cb.BASE_FEATS])
        t_centers.append(start / fs + win_sec / 2.0)
        start += step

    if len(rows) < MIN_HISTORY_WINDOWS * 3:
        return None

    X = np.asarray(rows, float)
    t = np.asarray(t_centers, float)
    mdf = X[:, cb.BASE_FEATS.index("mdf")].copy()
    return {"t": t, "X": X, "mdf": mdf, "fs": fs}


def subject_series(root: str, subject: int, side: str, win_sec: float,
                   step_sec: float, target_fs: int | None) -> dict | None:
    """One Zenodo subject -> segment_series()."""
    seg = loader.load_biceps_segment(root, subject, side,
                                     target_fs=target_fs, bandpass=True)
    fs = int(getattr(seg, "eff_fs", loader.FS_NATIVE))
    series = segment_series(seg, fs, win_sec, step_sec)
    if series is not None:
        series["subject"] = subject
    return series


def fresh_zscore(X: np.ndarray, t: np.ndarray, fresh_sec: float = FRESH_SEC) -> np.ndarray:
    """Z-score every feature against the recording's own first `fresh_sec`.

    Deliberately label-free: an athlete uploading a recording has no fatigue
    labels, so a forecaster that needs them could never be deployed. This is
    the same fresh-baseline transform classify_upload() uses.
    """
    fresh = X[t <= fresh_sec]
    if fresh.shape[0] < 3:
        fresh = X[: max(3, X.shape[0] // 4)]
    mu, sd = fresh.mean(0), fresh.std(0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def irreducible_noise(mdf: np.ndarray, k: int = 5) -> float:
    """Std of MDF about a local moving average -- the window-to-window jitter
    no forecaster can predict. Sets the floor on any achievable MAE."""
    if mdf.size < k:
        return float("nan")
    kernel = np.ones(k) / k
    smooth = np.convolve(mdf, kernel, mode="same")
    edge = k // 2
    resid = (mdf - smooth)[edge:-edge] if mdf.size > 2 * edge else (mdf - smooth)
    return float(np.std(resid))


# ---------------------------------------------------------------------------
# supervised samples
# ---------------------------------------------------------------------------
def build_samples(series: dict, seq_len: int, horizon_steps: list[int],
                  use_time: bool) -> dict:
    """Causal sequences -> delta-MDF targets at each horizon.

    sample i uses windows [i-seq_len+1 .. i] (replicate-padded at the start,
    never looking forward) and targets mdf[i+h] - mdf[i] for each h.
    Samples are only emitted where every horizon has a real future window, so
    all horizons are scored on exactly the same moments.
    """
    X, t, mdf = series["X"], series["t"], series["mdf"]
    Z = fresh_zscore(X, t)
    if use_time:
        # elapsed time matters: fatigue is a function of how long you have been
        # contracting. Scaled by 100 s to sit in the same range as the z-scores.
        Z = np.hstack([Z, (t / 100.0).reshape(-1, 1)])

    n = Z.shape[0]
    max_h = max(horizon_steps)
    idx = [i for i in range(MIN_HISTORY_WINDOWS, n - max_h)]
    if not idx:
        return {}

    seqs = np.empty((len(idx), seq_len, Z.shape[1]), float)
    for j, i in enumerate(idx):
        start = i - seq_len + 1
        if start < 0:
            pad = np.repeat(Z[:1], -start, axis=0)
            seqs[j] = np.vstack([pad, Z[: i + 1]])
        else:
            seqs[j] = Z[start: i + 1]

    idx = np.asarray(idx)
    targets = np.stack([mdf[idx + h] - mdf[idx] for h in horizon_steps], axis=1)
    return {"seq": seqs, "y": targets, "idx": idx,
            "t": t, "mdf": mdf, "subject": series["subject"]}


# ---------------------------------------------------------------------------
# non-learned baselines, all causal
# ---------------------------------------------------------------------------
def _ols_delta(t_hist: np.ndarray, y_hist: np.ndarray, t_now: float,
               horizon_sec: float, y_now: float) -> float:
    """OLS on history only, extrapolated to t_now+horizon, as a delta.

    Clamped so the projected MDF cannot fall below 0 Hz -- the same floor
    models/fatigue_forecast.py applies, so this baseline is a faithful stand-in
    for what the chatbot ships rather than a strawman.
    """
    if t_hist.size < 3:
        return 0.0
    a, b = np.polyfit(t_hist, y_hist, 1)
    projected = max(0.0, a * (t_now + horizon_sec) + b)
    return float(projected - y_now)


def baseline_predictions(sample: dict, horizon_steps: list[int], step_sec: float,
                         ols_window_sec: float) -> dict[str, np.ndarray]:
    """persistence / ols_full / ols_recent predictions for one subject."""
    t, mdf, idx = sample["t"], sample["mdf"], sample["idx"]
    n_h = len(horizon_steps)
    preds = {"persistence": np.zeros((idx.size, n_h)),
             "ols_full": np.zeros((idx.size, n_h)),
             "ols_recent": np.zeros((idx.size, n_h))}

    for j, i in enumerate(idx):
        t_now, y_now = float(t[i]), float(mdf[i])
        t_all, y_all = t[: i + 1], mdf[: i + 1]
        recent = t_all >= t_now - ols_window_sec
        t_rec, y_rec = t_all[recent], y_all[recent]
        for k, h in enumerate(horizon_steps):
            hs = h * step_sec
            preds["ols_full"][j, k] = _ols_delta(t_all, y_all, t_now, hs, y_now)
            preds["ols_recent"][j, k] = _ols_delta(t_rec, y_rec, t_now, hs, y_now)
    return preds


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
class MDFForecaster(nn.Module):
    """Same LSTM shape as the fatigue classifier, with a regression head:
    one output per horizon, in Hz of MDF change."""

    def __init__(self, n_features: int, hidden_size: int = 64,
                 num_layers: int = 1, n_horizons: int = 3, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, n_horizons)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


def train_forecaster(model: MDFForecaster, X: np.ndarray, y: np.ndarray,
                     epochs: int, lr: float, batch_size: int, device,
                     val: tuple[np.ndarray, np.ndarray] | None = None,
                     patience: int = 10) -> MDFForecaster:
    """Train, early-stopping on `val` if given.

    `val` must come from TRAINING subjects only -- never the held-out test
    subject. Without it the model overfits badly on a dataset this small
    (one fold's MAE went to 6.23 Hz, worse than predicting no change at all).
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # Huber, not MSE: MDF has occasional single-window spikes and squared error
    # lets a handful of them dominate the gradient on a dataset this small.
    crit = nn.SmoothL1Loss()
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)
    if val is not None:
        Xv = torch.tensor(val[0], dtype=torch.float32, device=device)
        yv = torch.tensor(val[1], dtype=torch.float32, device=device)
    n = X_t.shape[0]

    best_loss, best_state, since_best = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            b = perm[i:i + batch_size]
            opt.zero_grad()
            loss = crit(model(X_t[b]), y_t[b])
            loss.backward()
            opt.step()

        if val is None:
            continue
        model.eval()
        with torch.no_grad():
            v = float((model(Xv) - yv).abs().mean())
        if v < best_loss - 1e-4:
            best_loss, since_best = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            since_best += 1
            if since_best >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict(model: MDFForecaster, X: np.ndarray, device) -> np.ndarray:
    model.eval()
    return model(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy()


# ---------------------------------------------------------------------------
# LOSO
# ---------------------------------------------------------------------------
def run_loso(samples: dict, horizon_steps: list[int], step_sec: float,
             ols_window_sec: float, hidden: int, layers: int, dropout: float,
             epochs: int, lr: float, batch_size: int, seeds: int, device) -> dict:
    subs = sorted(samples)
    n_h = len(horizon_steps)
    methods = ["persistence", "ols_full", "ols_recent", "drift", "lstm"]
    # per method -> per horizon -> list of per-subject MAE
    mae = {m: [[] for _ in range(n_h)] for m in methods}
    rmse = {m: [[] for _ in range(n_h)] for m in methods}
    per_subject = {}

    for test_s in subs:
        others = [s for s in subs if s != test_s]
        # inner validation for early stopping, drawn from the TRAINING subjects
        # only. Rotating which ones by test_s keeps every subject out of its own
        # validation set without ever touching the held-out fold.
        n_val = max(1, len(others) // 6)
        offset = subs.index(test_s)
        val_s = [others[(offset + i) % len(others)] for i in range(n_val)]
        train_s = [s for s in others if s not in val_s]

        Xtr = np.concatenate([samples[s]["seq"] for s in train_s], axis=0)
        ytr = np.concatenate([samples[s]["y"] for s in train_s], axis=0)
        Xval = np.concatenate([samples[s]["seq"] for s in val_s], axis=0)
        yval = np.concatenate([samples[s]["y"] for s in val_s], axis=0)
        te = samples[test_s]
        Xte, yte = te["seq"], te["y"]

        preds = baseline_predictions(te, horizon_steps, step_sec, ols_window_sec)

        # drift: mean Hz/s slope of every non-test subject, x horizon. Given all
        # of them rather than the reduced training split, so the LSTM's early
        # stopping is never an advantage over this baseline.
        slopes = []
        for s in others:
            ts, ms = samples[s]["t"], samples[s]["mdf"]
            slopes.append(np.polyfit(ts, ms, 1)[0])
        drift_rate = float(np.mean(slopes))
        preds["drift"] = np.array([[drift_rate * h * step_sec for h in horizon_steps]]
                                  ).repeat(Xte.shape[0], axis=0)

        # LSTM, averaged over seeds so a win has to survive initialisation noise
        seed_preds = []
        for seed in range(seeds):
            torch.manual_seed(seed)
            model = MDFForecaster(n_features=Xtr.shape[-1], hidden_size=hidden,
                                  num_layers=layers, n_horizons=n_h,
                                  dropout=dropout).to(device)
            model = train_forecaster(model, Xtr, ytr, epochs, lr, batch_size,
                                     device, val=(Xval, yval))
            seed_preds.append(predict(model, Xte, device))
        preds["lstm"] = np.mean(seed_preds, axis=0)

        subj_row = {}
        for m in methods:
            err = preds[m] - yte
            m_mae = np.abs(err).mean(axis=0)
            m_rmse = np.sqrt((err ** 2).mean(axis=0))
            for k in range(n_h):
                mae[m][k].append(float(m_mae[k]))
                rmse[m][k].append(float(m_rmse[k]))
            subj_row[m] = {"mae": [float(v) for v in m_mae],
                           "rmse": [float(v) for v in m_rmse]}
        subj_row["n_samples"] = int(Xte.shape[0])
        subj_row["noise_floor_hz"] = irreducible_noise(te["mdf"])
        per_subject[test_s] = subj_row

        print(f"  test S{test_s:<2} n={Xte.shape[0]:<4} "
              + "  ".join(f"{m}={np.mean(subj_row[m]['mae']):.2f}" for m in methods))

    summary = {}
    for m in methods:
        summary[m] = {
            "mae_per_horizon": [float(np.mean(v)) for v in mae[m]],
            "mae_std_per_horizon": [float(np.std(v)) for v in mae[m]],
            "rmse_per_horizon": [float(np.mean(v)) for v in rmse[m]],
        }
    # skill vs persistence: >0 means better than assuming nothing changes
    base = summary["persistence"]["mae_per_horizon"]
    for m in methods:
        summary[m]["skill_vs_persistence"] = [
            float(1.0 - summary[m]["mae_per_horizon"][k] / base[k]) if base[k] > 0 else 0.0
            for k in range(n_h)]

    # The differences here are ~0.2 Hz on 12 subjects. Without a paired test
    # there is no way to tell a real improvement from fold noise, and claiming
    # one would not be supportable.
    from scipy.stats import wilcoxon
    for m in methods:
        pvals = []
        for k in range(n_h):
            a, b = np.array(mae[m][k]), np.array(mae["persistence"][k])
            if m == "persistence" or np.allclose(a, b):
                pvals.append(1.0)
                continue
            pvals.append(float(wilcoxon(a, b).pvalue))
        summary[m]["p_vs_persistence"] = pvals
    return {"summary": summary, "per_subject": per_subject, "subjects": subs}


# ---------------------------------------------------------------------------
# inference for the shipped chatbot
# ---------------------------------------------------------------------------
DEPLOYED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "forecast_model.pt")
_DEPLOYED = None


def load_deployed(path: str | None = None):
    """Load and cache the saved forecaster. Returns (model, config) or None
    if no checkpoint has been trained yet -- callers must handle None and fall
    back, so the chatbot still works on a fresh clone."""
    global _DEPLOYED
    if _DEPLOYED is None:
        p = path or DEPLOYED_PATH
        if not os.path.exists(p):
            return None
        ck = torch.load(p, map_location="cpu", weights_only=False)
        a = ck["arch"]
        model = MDFForecaster(n_features=a["n_features"], hidden_size=a["hidden_size"],
                              num_layers=a["num_layers"], n_horizons=a["n_horizons"],
                              dropout=a["dropout"])
        model.load_state_dict(ck["state_dict"])
        model.eval()
        _DEPLOYED = (model, ck["config"])
    return _DEPLOYED


def predict_delta(seg, fs: int, t_end: float | None = None,
                  path: str | None = None) -> dict | None:
    """Predicted MDF change (Hz) at each trained horizon, from history only.

    Everything up to `t_end` is used and nothing after it, so this can be
    called for a moment in the middle of a recording without leaking the
    future. Returns None when the model is missing or there is too little
    history to build one input sequence.
    """
    loaded = load_deployed(path)
    if loaded is None:
        return None
    model, cfg = loaded

    series = segment_series(seg, fs, cfg["win_sec"], cfg["step_sec"])
    if series is None:
        return None
    t, X, mdf = series["t"], series["X"], series["mdf"]
    if t_end is not None:
        keep = t <= float(t_end)
        t, X, mdf = t[keep], X[keep], mdf[keep]
    if t.size < MIN_HISTORY_WINDOWS:
        return None

    Z = fresh_zscore(X, t, cfg["fresh_sec"])
    if cfg["use_time"]:
        Z = np.hstack([Z, (t / 100.0).reshape(-1, 1)])

    seq_len = cfg["seq_len"]
    start = Z.shape[0] - seq_len
    if start < 0:
        seq = np.vstack([np.repeat(Z[:1], -start, axis=0), Z])
    else:
        seq = Z[start:]

    with torch.no_grad():
        delta = model(torch.tensor(seq[None, :, :], dtype=torch.float32))[0].numpy()

    return {"horizons_sec": list(cfg["horizons_sec"]),
            "delta_hz": [float(v) for v in delta],
            "mdf_now": float(mdf[-1]), "t_now": float(t[-1]),
            "n_history_windows": int(t.size)}


# ---------------------------------------------------------------------------
def _save_deployable(args, samples: dict, horizon_steps: list[int],
                     horizons_sec: list[float], device) -> None:
    """Retrain on every subject and save a checkpoint the chatbot can load.

    The LOSO numbers above are the honest out-of-subject estimate of how this
    performs; this checkpoint uses all of them, so it must never be scored on
    the same subjects it saw. Everything needed to rebuild the input transform
    at inference time is stored alongside the weights.
    """
    if not args.save_model:
        return
    subs = sorted(samples)
    X = np.concatenate([samples[s]["seq"] for s in subs], axis=0)
    y = np.concatenate([samples[s]["y"] for s in subs], axis=0)

    # a small held-out slice purely to stop training, matching the LOSO folds
    val_s = subs[::5]
    Xv = np.concatenate([samples[s]["seq"] for s in val_s], axis=0)
    yv = np.concatenate([samples[s]["y"] for s in val_s], axis=0)

    torch.manual_seed(0)
    model = MDFForecaster(n_features=X.shape[-1], hidden_size=args.hidden,
                          num_layers=args.layers, n_horizons=len(horizon_steps),
                          dropout=args.dropout).to(device)
    model = train_forecaster(model, X, y, args.epochs, args.lr,
                             args.batch_size, device, val=(Xv, yv))

    torch.save({
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "arch": {"n_features": int(X.shape[-1]), "hidden_size": args.hidden,
                 "num_layers": args.layers, "n_horizons": len(horizon_steps),
                 "dropout": args.dropout},
        "config": {"win_sec": args.win, "step_sec": args.step,
                   "target_fs": args.target_fs or None, "seq_len": args.seq_len,
                   "use_time": args.use_time, "fresh_sec": FRESH_SEC,
                   "base_feats": list(cb.BASE_FEATS),
                   "horizons_sec": horizons_sec},
        "target": "delta_mdf_hz",
        "trained_on_subjects": subs,
        "note": ("predicts MDF(t+h) - MDF(t) in Hz. Trained on ALL subjects -- "
                 "for out-of-subject performance see the LOSO table in "
                 "models/FORECAST_VALIDATION.md, not this checkpoint."),
    }, args.save_model)
    print(f"saved deployable model -> {args.save_model} "
          f"({X.shape[0]} samples, {len(subs)} subjects)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(REPO_ROOT, "zenodo_biceps", "sEMG_data"))
    ap.add_argument("--side", choices=["R", "L"], default="R")
    ap.add_argument("--win", type=float, default=4.0)
    ap.add_argument("--step", type=float, default=2.0)
    ap.add_argument("--target-fs", type=int, default=250,
                    help="downsample before features (250 = OpenBCI bridge rate)")
    ap.add_argument("--horizons", type=float, nargs="+", default=[10.0, 30.0, 60.0],
                    help="forecast horizons in seconds")
    ap.add_argument("--seq-len", type=int, default=20,
                    help="causal history fed to the LSTM, in windows "
                         "(20 x 2 s step = 40 s of history)")
    ap.add_argument("--ols-window", type=float, default=60.0,
                    help="history span for the ols_recent baseline, seconds")
    ap.add_argument("--no-time", dest="use_time", action="store_false",
                    help="drop the elapsed-time feature")
    ap.set_defaults(use_time=True)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--save-model", default=None,
                    help="after evaluating, retrain on ALL subjects and save a "
                         "deployable checkpoint here (e.g. models/forecast_model.pt)")
    ap.add_argument("--skip-loso", action="store_true",
                    help="only train + save the deployable model, no LOSO scoring")
    args = ap.parse_args()

    target_fs = args.target_fs or None
    horizon_steps = [max(1, int(round(h / args.step))) for h in args.horizons]
    horizons_sec = [h * args.step for h in horizon_steps]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"horizons: {horizons_sec} s  ({horizon_steps} windows ahead)")

    samples, noise = {}, []
    for s in range(1, 14):
        series = subject_series(args.root, s, args.side, args.win, args.step, target_fs)
        if series is None:
            print(f"S{s}: too short, skipped")
            continue
        sam = build_samples(series, args.seq_len, horizon_steps, args.use_time)
        if not sam:
            print(f"S{s}: no samples at this horizon, skipped")
            continue
        samples[s] = sam
        nf = irreducible_noise(series["mdf"])
        noise.append(nf)
        print(f"S{s}: {series['mdf'].size} windows -> {sam['seq'].shape[0]} samples "
              f"x {sam['seq'].shape[1]} steps x {sam['seq'].shape[2]} feats  "
              f"MDF {series['mdf'].mean():.1f} Hz  noise {nf:.2f} Hz")

    if len(samples) < 3:
        print("not enough subjects for LOSO")
        return 1

    total = sum(v["seq"].shape[0] for v in samples.values())
    print(f"\n{len(samples)} subjects, {total} samples total")
    print(f"mean window-to-window MDF noise: {np.mean(noise):.2f} Hz "
          "(no forecaster can beat this)")

    if args.skip_loso:
        res = None
    else:
        print(f"\n=== LOSO forecast ({len(samples)} folds x {args.seeds} seeds) ===")
        res = run_loso(samples, horizon_steps, args.step, args.ols_window,
                       args.hidden, args.layers, args.dropout, args.epochs,
                       args.lr, args.batch_size, args.seeds, device)

    if res is None:
        _save_deployable(args, samples, horizon_steps, horizons_sec, device)
        return 0

    print(f"\nMAE (Hz), lower is better -- mean over {len(samples)} held-out subjects")
    head = "  ".join(f"{h:>6.0f}s" for h in horizons_sec)
    print(f"{'method':<14} {head}     skill vs persistence")
    print("-" * (16 + len(head) + 26))
    for m, r in res["summary"].items():
        row = "  ".join(f"{v:>6.2f}" for v in r["mae_per_horizon"])
        skill = "  ".join(f"{v:>+6.1%}" for v in r["skill_vs_persistence"])
        print(f"{m:<14} {row}     {skill}")

    print("\npaired Wilcoxon vs persistence (p < 0.05 = a real difference)")
    print(f"{'method':<14} {head}")
    print("-" * (16 + len(head)))
    for m, r in res["summary"].items():
        if m == "persistence":
            continue
        row = "  ".join(f"{v:>6.3f}" for v in r["p_vs_persistence"])
        print(f"{m:<14} {row}")

    if args.json_out:
        payload = {
            "config": {
                "root": args.root, "side": args.side, "win": args.win,
                "step": args.step, "target_fs": target_fs,
                "horizons_sec": horizons_sec, "horizon_steps": horizon_steps,
                "seq_len": args.seq_len, "use_time": args.use_time,
                "ols_window_sec": args.ols_window, "hidden": args.hidden,
                "layers": args.layers, "epochs": args.epochs, "seeds": args.seeds,
                "target": "delta_mdf_hz",
            },
            "n_samples": total,
            "mdf_noise_floor_hz": float(np.mean(noise)),
            "results": res,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nsaved {args.json_out}")

    _save_deployable(args, samples, horizon_steps, horizons_sec, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
