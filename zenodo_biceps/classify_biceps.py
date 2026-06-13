"""
Classifier: biceps fatigue STATE (0 non-fatigue / 1 transition / 2 fatigue).
============================================================================

Companion to run_biceps.py. LOSO cross-subject classification with RF, SVM,
and KNN (slide 9 spec). Outputs a bar-chart comparison figure.

Usage:
    python classify_biceps.py --root sEMG_data --side R
    python classify_biceps.py --root sEMG_data --side R --binary
    python classify_biceps.py --root sEMG_data --side R --no-norm
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loader  # noqa: E402
import core    # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
LABELS = {0: "non-fatigue", 1: "transition", 2: "fatigue"}


# ---------------------------------------------------------------------------
# per-window features
# ---------------------------------------------------------------------------
def window_features(x: np.ndarray, fs: int) -> dict:
    x = np.asarray(x, float)
    dx = np.diff(x)
    rms = float(np.sqrt(np.mean(x ** 2)))
    mav = float(np.mean(np.abs(x)))
    wl  = float(np.sum(np.abs(dx)))
    var = float(np.var(x))
    zc  = int(np.sum((x[:-1] * x[1:] < 0)))
    ssc = int(np.sum(np.diff(np.sign(dx)) != 0))
    mdf = float(core.median_frequency(x, fs=fs))
    freqs, pxx = core.channel_spectrum(x.reshape(-1, 1), fs=fs)
    pxx = pxx[:, 0] if pxx.ndim == 2 else pxx
    psum = pxx.sum()
    mnf = float((freqs * pxx).sum() / psum) if psum > 0 else 0.0
    return dict(rms=rms, mav=mav, wl=wl, var=var, zc=zc, ssc=ssc,
                mdf=mdf, mnf=mnf)


FEATS = ["rms", "mav", "wl", "var", "zc", "ssc", "mdf", "mnf"]


def label_at(t_center: float, lab_t: np.ndarray, lab_v: np.ndarray) -> int:
    i = int(np.argmin(np.abs(lab_t - t_center)))
    return int(lab_v[i])


def subject_windows(root: str, subject: int, side: str,
                    win_sec: float, step_sec: float):
    seg = loader.load_biceps_segment(root, subject, side, bandpass=True)
    fs = int(getattr(seg, "eff_fs", loader.FS_NATIVE))
    lab_t, lab_v = loader.load_fatigue_labels(root, subject, side)
    if lab_t is None:
        return None, None
    win  = max(2, int(round(win_sec * fs)))
    step = max(1, int(round(step_sec * fs)))
    x = seg.data[:, 0]
    rows, ys = [], []
    start = 0
    while start + win <= x.size:
        w  = x[start:start + win]
        tc = seg.t[start] + win_sec / 2.0
        feat = window_features(w, fs)
        rows.append([feat[k] for k in FEATS])
        ys.append(label_at(tc, lab_t, lab_v))
        start += step
    return np.array(rows, float), np.array(ys, int)


# ---------------------------------------------------------------------------
# classifiers (slide 9 spec: RF, SVM, KNN)
# ---------------------------------------------------------------------------
def build_classifiers():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    return {
        "RF":  make_pipeline(StandardScaler(),
                             RandomForestClassifier(n_estimators=300,
                                                    random_state=0,
                                                    n_jobs=-1)),
        "SVM": make_pipeline(StandardScaler(),
                             SVC(kernel="rbf", C=1.0, gamma="scale",
                                 class_weight="balanced", random_state=0)),
        "KNN": make_pipeline(StandardScaler(),
                             KNeighborsClassifier(n_neighbors=5,
                                                  weights="distance",
                                                  metric="euclidean")),
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
        f1s.append(f1_score(yte, pred, average="macro"))
        all_true.append(yte)
        all_pred.append(pred)
    return dict(
        mean_acc=float(np.mean(accs)),
        std_acc=float(np.std(accs)),
        mean_f1=float(np.mean(f1s)),
        per_fold_acc=accs,
        per_fold_f1=f1s,
        all_true=np.concatenate(all_true),
        all_pred=np.concatenate(all_pred),
    )


# ---------------------------------------------------------------------------
# comparison figure
# ---------------------------------------------------------------------------
def save_comparison_figure(results: dict, label_kind: str, norm: bool):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    os.makedirs(OUT, exist_ok=True)
    names = list(results.keys())
    accs  = [results[n]["mean_acc"] for n in names]
    stds  = [results[n]["std_acc"]  for n in names]
    f1s   = [results[n]["mean_f1"]  for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(
        f"LOSO Classifier Comparison - Zenodo 14182446 Biceps\n"
        f"({'binary 0v2' if label_kind == 'binary' else '3-class'}, "
        f"{'subject-norm' if norm else 'no-norm'})",
        fontsize=11,
    )

    # accuracy bar
    ax = axes[0]
    bars = ax.bar(names, accs, yerr=stds, capsize=5,
                  color=["#4c72b0", "#dd8452", "#55a868"], alpha=0.85)
    ax.axhline(0.75, color="red", linestyle="--", linewidth=1.0,
               label=">75% target")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean LOSO Accuracy")
    ax.set_title("Accuracy (mean +/- std)")
    ax.legend(fontsize=8)
    for b, v in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9)

    # macro-F1 bar
    ax = axes[1]
    bars = ax.bar(names, f1s, color=["#4c72b0", "#dd8452", "#55a868"],
                  alpha=0.85)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean LOSO Macro-F1")
    ax.set_title("Macro-F1")
    for b, v in zip(bars, f1s):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9)

    # confusion matrix for best model
    best_name = names[int(np.argmax(accs))]
    best_res  = results[best_name]
    yt, yp    = best_res["all_true"], best_res["all_pred"]
    labs      = sorted(np.unique(yt).tolist())
    cm        = confusion_matrix(yt, yp, labels=labs)
    cm_norm   = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    ax = axes[2]
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(labs))); ax.set_xticklabels([str(l) for l in labs])
    ax.set_yticks(range(len(labs))); ax.set_yticklabels([str(l) for l in labs])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion (best: {best_name}, pooled)")
    for i in range(len(labs)):
        for j in range(len(labs)):
            ax.text(j, i, f"{cm_norm[i,j]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if cm_norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fname = os.path.join(OUT, f"classifier_comparison_{label_kind}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {fname}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--side", choices=["R", "L"], default="R")
    ap.add_argument("--win",  type=float, default=4.0)
    ap.add_argument("--step", type=float, default=2.0)
    ap.add_argument("--binary", action="store_true",
                    help="collapse to 0 vs 2 only (drop transition)")
    ap.add_argument("--no-norm", dest="norm", action="store_false",
                    help="disable subject-relative baseline normalization "
                         "(not recommended for 3-class)")
    ap.set_defaults(norm=True)
    args = ap.parse_args()

    # build subject pool
    subs, X_by, y_by = [], {}, {}
    for s in range(1, 14):
        X, y = subject_windows(args.root, s, args.side, args.win, args.step)
        if X is None or X.size == 0:
            print(f"S{s}: no labels / no windows, skipped")
            continue
        if args.norm:
            base = X[y == 0]
            if base.shape[0] >= 3:
                mu, sd = base.mean(0), base.std(0)
                sd[sd == 0] = 1.0
                X = (X - mu) / sd
        if args.binary:
            keep = y != 1
            X, y = X[keep], y[keep]
            y = (y == 2).astype(int)
        if np.unique(y).size < 2:
            print(f"S{s}: single-class after filtering, skipped")
            continue
        subs.append(s)
        X_by[s] = X
        y_by[s] = y
        dist = {int(k): int(v)
                for k, v in zip(*np.unique(y, return_counts=True))}
        print(f"S{s}: {X.shape[0]} windows  class dist {dist}")

    if len(subs) < 2:
        print("not enough labelled subjects for LOSO")
        return

    label_kind = "binary" if args.binary else "3class"
    norm_tag   = "norm" if args.norm else "no-norm"
    print(f"\n=== LOSO ({label_kind}, {norm_tag}) ===")

    classifiers = build_classifiers()
    results = {}

    for clf_name, clf_template in classifiers.items():
        print(f"\n--- {clf_name} ---")

        # need a factory so each fold gets a fresh unfitted clf
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

    # summary table
    print(f"\n{'Classifier':<12} {'Mean Acc':>10} {'Std Acc':>10} {'Mean F1':>10}")
    print("-" * 44)
    for name, res in results.items():
        print(f"{name:<12} {res['mean_acc']:>10.3f} {res['std_acc']:>10.3f} "
              f"{res['mean_f1']:>10.3f}")

    save_comparison_figure(results, label_kind, args.norm)


if __name__ == "__main__":
    main()
