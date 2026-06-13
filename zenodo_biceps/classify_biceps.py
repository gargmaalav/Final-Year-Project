"""
Classifier: biceps fatigue state from Zenodo 14182446.
=======================================================

Honest cross-subject classifier with:
  - leave-one-subject-out validation
  - subject-baseline calibration from known fresh windows
  - majority/purity window labels
  - optional transition-boundary exclusion
  - causal temporal fatigue features

Usage:
    python classify_biceps.py --root sEMG_data --side R
    python classify_biceps.py --root sEMG_data --side R --label-mode binary_drop_transition
    python classify_biceps.py --root sEMG_data --side R --target-fs 250 --temporal
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loader  # noqa: E402
import core    # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
BASE_FEATS = ["rms", "mav", "wl", "var", "zc", "ssc", "mdf", "mnf"]


@dataclass
class SubjectData:
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    dropped_mixed: int
    dropped_boundary: int


# ---------------------------------------------------------------------------
# per-window features
# ---------------------------------------------------------------------------
def window_features(x: np.ndarray, fs: int) -> dict[str, float]:
    x = np.asarray(x, float)
    dx = np.diff(x)
    rms = float(np.sqrt(np.mean(x ** 2)))
    mav = float(np.mean(np.abs(x)))
    wl = float(np.sum(np.abs(dx)))
    var = float(np.var(x))
    zc = int(np.sum((x[:-1] * x[1:] < 0)))
    ssc = int(np.sum(np.diff(np.sign(dx)) != 0))
    mdf = float(core.median_frequency(x, fs=fs))
    freqs, pxx = core.channel_spectrum(x.reshape(-1, 1), fs=fs)
    pxx = pxx[:, 0] if pxx.ndim == 2 else pxx
    psum = pxx.sum()
    mnf = float((freqs * pxx).sum() / psum) if psum > 0 else 0.0
    return dict(rms=rms, mav=mav, wl=wl, var=var, zc=zc, ssc=ssc,
                mdf=mdf, mnf=mnf)


def label_transitions(lab_t: np.ndarray, lab_v: np.ndarray) -> np.ndarray:
    """Times where the discrete label changes."""
    idx = np.flatnonzero(np.diff(lab_v) != 0) + 1
    return lab_t[idx].astype(float)


def window_label(t0: float, t1: float, lab_t: np.ndarray, lab_v: np.ndarray,
                 min_purity: float) -> int | None:
    """Majority label inside a window; discard if too mixed/ambiguous."""
    inside = (lab_t >= t0) & (lab_t < t1)
    if not inside.any():
        center = (t0 + t1) / 2.0
        return int(lab_v[int(np.argmin(np.abs(lab_t - center)))])

    vals, counts = np.unique(lab_v[inside].astype(int), return_counts=True)
    best_i = int(np.argmax(counts))
    purity = counts[best_i] / counts.sum()
    if purity < min_purity:
        return None
    return int(vals[best_i])


def near_transition(t_center: float, transitions: np.ndarray,
                    margin_sec: float) -> bool:
    if margin_sec <= 0 or transitions.size == 0:
        return False
    return bool(np.min(np.abs(transitions - t_center)) <= margin_sec)


def rolling_slope(values: np.ndarray, step_sec: float) -> float:
    """Causal least-squares slope over a short history."""
    if values.size < 2:
        return 0.0
    t = np.arange(values.size, dtype=float) * step_sec
    return float(np.polyfit(t, values, 1)[0])


def add_temporal_features(X: np.ndarray, step_sec: float,
                          base_names: list[str],
                          history: int) -> tuple[np.ndarray, list[str]]:
    """Add only current/past-window trajectory features.

    For every base feature, add:
      - d1: current minus previous window
      - dN: current minus value N windows ago
      - slopeN: causal rolling slope over the last N windows
    """
    if X.size == 0:
        return X, base_names

    d1 = np.zeros_like(X)
    d1[1:] = X[1:] - X[:-1]

    d_hist = np.zeros_like(X)
    slopes = np.zeros_like(X)
    for i in range(X.shape[0]):
        start = max(0, i - history + 1)
        hist = X[start:i + 1]
        d_hist[i] = X[i] - hist[0]
        for j in range(X.shape[1]):
            slopes[i, j] = rolling_slope(hist[:, j], step_sec)

    names = (
        base_names
        + [f"{n}_d1" for n in base_names]
        + [f"{n}_d{history}" for n in base_names]
        + [f"{n}_slope{history}" for n in base_names]
    )
    return np.hstack([X, d1, d_hist, slopes]), names


def apply_label_mode(X: np.ndarray, y: np.ndarray,
                     mode: str) -> tuple[np.ndarray, np.ndarray]:
    if mode == "3class":
        return X, y
    if mode == "binary_drop_transition":
        keep = y != 1
        return X[keep], (y[keep] == 2).astype(int)
    if mode == "binary_transition_fatigued":
        return X, (y >= 1).astype(int)
    if mode == "binary_transition_fresh":
        return X, (y == 2).astype(int)
    raise ValueError(f"unknown label mode: {mode}")


def subject_windows(root: str, subject: int, side: str,
                    win_sec: float, step_sec: float,
                    target_fs: int | None = None,
                    min_label_purity: float = 0.80,
                    transition_margin_sec: float = 0.0,
                    temporal: bool = False,
                    temporal_history: int = 5,
                    label_mode: str = "3class") -> SubjectData | None:
    seg = loader.load_biceps_segment(root, subject, side,
                                     target_fs=target_fs, bandpass=True)
    fs = int(getattr(seg, "eff_fs", loader.FS_NATIVE))
    lab_t, lab_v = loader.load_fatigue_labels(root, subject, side)
    if lab_t is None:
        return None

    lab_t = np.asarray(lab_t, float)
    lab_v = np.asarray(lab_v, int)
    transitions = label_transitions(lab_t, lab_v)
    win = max(2, int(round(win_sec * fs)))
    step = max(1, int(round(step_sec * fs)))
    x = seg.data[:, 0]

    rows, ys = [], []
    dropped_mixed = 0
    dropped_boundary = 0
    start = 0
    while start + win <= x.size:
        t0 = float(seg.t[start])
        t1 = float(seg.t[start + win - 1])
        tc = (t0 + t1) / 2.0

        if near_transition(tc, transitions, transition_margin_sec):
            dropped_boundary += 1
            start += step
            continue

        y = window_label(t0, t1, lab_t, lab_v, min_label_purity)
        if y is None:
            dropped_mixed += 1
            start += step
            continue

        feat = window_features(x[start:start + win], fs)
        rows.append([feat[k] for k in BASE_FEATS])
        ys.append(y)
        start += step

    X = np.array(rows, float)
    y = np.array(ys, int)
    feature_names = list(BASE_FEATS)
    if X.size and temporal:
        X, feature_names = add_temporal_features(
            X, step_sec=step_sec, base_names=feature_names,
            history=temporal_history)
    X, y = apply_label_mode(X, y, label_mode)

    return SubjectData(X=X, y=y, feature_names=feature_names,
                       dropped_mixed=dropped_mixed,
                       dropped_boundary=dropped_boundary)


# ---------------------------------------------------------------------------
# classifiers
# ---------------------------------------------------------------------------
def build_classifiers():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    return {
        "RF": make_pipeline(
            StandardScaler(),
            RandomForestClassifier(n_estimators=500, random_state=0,
                                   n_jobs=-1, class_weight="balanced"),
        ),
        "SVM": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=1.0, gamma="scale",
                class_weight="balanced", random_state=0),
        ),
        "KNN": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=5, weights="distance",
                                 metric="euclidean"),
        ),
    }


def run_loso(subs, X_by, y_by, clf_factory):
    from sklearn.metrics import accuracy_score, f1_score

    accs, f1s, all_true, all_pred = [], [], [], []
    for test_s in subs:
        train_s = [s for s in subs if s != test_s]
        Xtr = np.vstack([X_by[s] for s in train_s])
        ytr = np.concatenate([y_by[s] for s in train_s])
        Xte, yte = X_by[test_s], y_by[test_s]
        clf = clf_factory()
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        accs.append(accuracy_score(yte, pred))
        f1s.append(f1_score(yte, pred, average="macro",
                            zero_division=0))
        all_true.append(yte)
        all_pred.append(pred)
    return dict(
        mean_acc=float(np.mean(accs)),
        std_acc=float(np.std(accs)),
        mean_f1=float(np.mean(f1s)),
        per_fold_acc=[float(v) for v in accs],
        per_fold_f1=[float(v) for v in f1s],
        all_true=np.concatenate(all_true),
        all_pred=np.concatenate(all_pred),
    )


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------
def save_comparison_figure(results: dict, label_mode: str, norm: bool,
                           target_fs: int | None, temporal: bool,
                           margin_sec: float):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    os.makedirs(OUT, exist_ok=True)
    names = list(results.keys())
    accs = [results[n]["mean_acc"] for n in names]
    stds = [results[n]["std_acc"] for n in names]
    f1s = [results[n]["mean_f1"] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(
        f"LOSO Classifier - Zenodo 14182446 Biceps\n"
        f"{label_mode}, {'subject-norm' if norm else 'no-norm'}, "
        f"{target_fs or loader.FS_NATIVE}Hz, "
        f"{'temporal' if temporal else 'static'}, margin={margin_sec:g}s",
        fontsize=10,
    )

    ax = axes[0]
    bars = ax.bar(names, accs, yerr=stds, capsize=5,
                  color=["#4c72b0", "#dd8452", "#55a868"], alpha=0.85)
    ax.axhline(0.75, color="red", linestyle="--", linewidth=1.0,
               label="75% target")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean LOSO Accuracy")
    ax.set_title("Accuracy")
    ax.legend(fontsize=8)
    for b, v in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9)

    ax = axes[1]
    bars = ax.bar(names, f1s, color=["#4c72b0", "#dd8452", "#55a868"],
                  alpha=0.85)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean LOSO Macro-F1")
    ax.set_title("Macro-F1")
    for b, v in zip(bars, f1s):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9)

    best_name = names[int(np.argmax(accs))]
    best_res = results[best_name]
    yt, yp = best_res["all_true"], best_res["all_pred"]
    labs = sorted(np.unique(yt).tolist())
    cm = confusion_matrix(yt, yp, labels=labs)
    denom = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, denom, out=np.zeros_like(cm, dtype=float),
                        where=denom != 0)
    ax = axes[2]
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(labs))); ax.set_xticklabels([str(l) for l in labs])
    ax.set_yticks(range(len(labs))); ax.set_yticklabels([str(l) for l in labs])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion ({best_name})")
    for i in range(len(labs)):
        for j in range(len(labs)):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if cm_norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fs_tag = f"_{target_fs}hz" if target_fs is not None else ""
    temp_tag = "_temporal" if temporal else ""
    margin_tag = f"_m{int(margin_sec)}s" if margin_sec else ""
    fname = os.path.join(
        OUT, f"classifier_comparison_{label_mode}{fs_tag}{temp_tag}{margin_tag}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {fname}")
    return fname


def json_ready(results: dict) -> dict:
    out = {}
    for name, res in results.items():
        out[name] = {k: v for k, v in res.items()
                     if k not in {"all_true", "all_pred"}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--side", choices=["R", "L"], default="R")
    ap.add_argument("--win", type=float, default=4.0)
    ap.add_argument("--step", type=float, default=2.0)
    ap.add_argument("--target-fs", type=int, default=None,
                    help="downsample before feature extraction, e.g. 250")
    ap.add_argument("--label-mode", choices=[
        "3class",
        "binary_drop_transition",
        "binary_transition_fatigued",
        "binary_transition_fresh",
    ], default="3class")
    ap.add_argument("--binary", action="store_true",
                    help="alias for --label-mode binary_drop_transition")
    ap.add_argument("--min-label-purity", type=float, default=0.80,
                    help="discard windows whose majority label fraction is lower")
    ap.add_argument("--transition-margin-sec", type=float, default=0.0,
                    help="discard windows centered this close to a label change")
    ap.add_argument("--temporal", action="store_true",
                    help="add causal rolling delta/slope fatigue features")
    ap.add_argument("--temporal-history", type=int, default=5,
                    help="number of past/current windows used by temporal features")
    ap.add_argument("--no-norm", dest="norm", action="store_false",
                    help="disable subject fresh-baseline calibration")
    ap.add_argument("--json-out", default=None,
                    help="optional metrics JSON path")
    ap.set_defaults(norm=True)
    args = ap.parse_args()

    if args.binary:
        args.label_mode = "binary_drop_transition"

    subs, X_by, y_by = [], {}, {}
    feature_names = None
    dropped = {}
    for s in range(1, 14):
        data = subject_windows(
            args.root, s, args.side, args.win, args.step,
            target_fs=args.target_fs,
            min_label_purity=args.min_label_purity,
            transition_margin_sec=args.transition_margin_sec,
            temporal=args.temporal,
            temporal_history=args.temporal_history,
            label_mode=args.label_mode,
        )
        if data is None or data.X.size == 0:
            print(f"S{s}: no labels / no windows, skipped")
            continue

        X, y = data.X, data.y
        if args.norm:
            if args.label_mode == "3class":
                base = X[y == 0]
            else:
                base = X[y == 0]
            if base.shape[0] >= 3:
                mu, sd = base.mean(0), base.std(0)
                sd[sd == 0] = 1.0
                X = (X - mu) / sd

        if np.unique(y).size < 2:
            print(f"S{s}: single-class after filtering, skipped")
            continue

        subs.append(s)
        X_by[s] = X
        y_by[s] = y
        feature_names = data.feature_names
        dropped[s] = {
            "mixed": data.dropped_mixed,
            "boundary": data.dropped_boundary,
        }
        dist = {int(k): int(v)
                for k, v in zip(*np.unique(y, return_counts=True))}
        print(f"S{s}: {X.shape[0]} windows  class dist {dist}  "
              f"dropped mixed={data.dropped_mixed} boundary={data.dropped_boundary}")

    if len(subs) < 2:
        print("not enough labelled subjects for LOSO")
        return

    print(f"\n=== LOSO ({args.label_mode}, "
          f"{'norm' if args.norm else 'no-norm'}, "
          f"{args.target_fs or loader.FS_NATIVE}Hz, "
          f"{'temporal' if args.temporal else 'static'}) ===")
    print(f"features: {len(feature_names or [])}")

    classifiers = build_classifiers()
    results = {}
    for clf_name, clf_template in classifiers.items():
        print(f"\n--- {clf_name} ---")

        import copy
        def make_factory(tmpl):
            def factory():
                return copy.deepcopy(tmpl)
            return factory

        res = run_loso(subs, X_by, y_by, make_factory(clf_template))
        results[clf_name] = res
        for i, test_s in enumerate(subs):
            print(f"  test S{test_s:<2}  acc={res['per_fold_acc'][i]:.3f}  "
                  f"macroF1={res['per_fold_f1'][i]:.3f}")
        print(f"  mean acc  = {res['mean_acc']:.3f} +/- {res['std_acc']:.3f}")
        print(f"  mean F1   = {res['mean_f1']:.3f}")

    print(f"\n{'Classifier':<12} {'Mean Acc':>10} {'Std Acc':>10} {'Mean F1':>10}")
    print("-" * 44)
    for name, res in results.items():
        print(f"{name:<12} {res['mean_acc']:>10.3f} {res['std_acc']:>10.3f} "
              f"{res['mean_f1']:>10.3f}")

    figure = save_comparison_figure(
        results, args.label_mode, args.norm, args.target_fs,
        args.temporal, args.transition_margin_sec)

    payload = {
        "config": {
            "root": args.root,
            "side": args.side,
            "win": args.win,
            "step": args.step,
            "target_fs": args.target_fs or loader.FS_NATIVE,
            "label_mode": args.label_mode,
            "min_label_purity": args.min_label_purity,
            "transition_margin_sec": args.transition_margin_sec,
            "temporal": args.temporal,
            "temporal_history": args.temporal_history,
            "subject_norm": args.norm,
            "n_features": len(feature_names or []),
        },
        "subjects": subs,
        "dropped_windows": dropped,
        "results": json_ready(results),
        "figure": figure,
    }
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"JSON saved: {args.json_out}")


if __name__ == "__main__":
    main()
