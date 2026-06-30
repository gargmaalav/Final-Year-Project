"""
Transformer sequence classifier: biceps fatigue STATE, built on classify_biceps.py.
=====================================================================================

Adds a second deep-learning option alongside RF/SVM/KNN and the LSTM, per the
supervisor's "this break: LSTM ... next semester: transformer" plan.

Reuses the SAME data pipeline as classify_biceps.py and lstm_classify_biceps.py
so results are directly comparable on a like-for-like basis:
  - loader.load_biceps_segment with the same --target-fs downsampling
    (default 250 Hz, the validated OpenBCI/HYFY bridge rate)
  - majority/purity window labeling + transition-margin trimming
    (classify_biceps.subject_windows, same label modes)
  - subject-relative fresh-baseline normalisation
  - leave-one-subject-out (LOSO) validation, no leakage

What's new: each 4 s window's 8 base EMG features (RMS/MAV/WL/VAR/ZC/SSC/MDF/MNF)
becomes one timestep, exactly as for the LSTM. A small encoder-only Transformer
consumes a CAUSAL sequence of the last `--seq-len` windows (current window +
history, default 6) and either:
  - classifies the CURRENT window's fatigue state (default), reading out the
    encoder's output at the final ("current window") sequence position --
    the direct analogue of the LSTM's final hidden state.
  - predicts the NEXT window's fatigue state (--predict-next).

No causal attention mask is applied: the input sequence already only contains
the current + past windows, so full self-attention within it cannot see the
future.

Optional --finetune mirrors lstm_classify_biceps.py's transfer-learning
experiment (same caveat applies: a time-ordered calibration/eval split is an
artifact on this monotonic-fatigue dataset -- see TRANSFORMER_HANDOVER.md).

Usage:
    python transformer_classify_biceps.py --root sEMG_data --side R
    python transformer_classify_biceps.py --root sEMG_data --side R --label-mode binary_drop_transition
    python transformer_classify_biceps.py --root sEMG_data --side R --predict-next
    python transformer_classify_biceps.py --root sEMG_data --side R --json-out out/metrics_transformer_3class_250hz_m4.json
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loader  # noqa: E402
import classify_biceps as cb  # noqa: E402  -- reuse the validated data pipeline

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


# ---------------------------------------------------------------------------
# per-window features -> causal sequences (identical to lstm_classify_biceps.py)
# ---------------------------------------------------------------------------
def build_sequences(X: np.ndarray, y: np.ndarray, seq_len: int,
                    predict_next: bool) -> tuple[np.ndarray, np.ndarray]:
    """Causal sequences of `seq_len` windows -> (sequences, labels).

    sequences[i] = X[i-seq_len+1 .. i], replicate-padded at the start of the
    session so every window (including the first few) yields a full-length
    sequence without looking into the future.

    If predict_next, sequences[i] is paired with y[i+1] (next window's label)
    and the final window is dropped (it has no "next").
    """
    n, n_feat = X.shape
    seqs = np.empty((n, seq_len, n_feat), dtype=float)
    for i in range(n):
        start = i - seq_len + 1
        if start < 0:
            pad = np.repeat(X[:1], -start, axis=0)
            seqs[i] = np.vstack([pad, X[: i + 1]])
        else:
            seqs[i] = X[start: i + 1]

    if predict_next:
        return seqs[:-1], y[1:]
    return seqs, y


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
class TransformerClassifier(nn.Module):
    def __init__(self, n_features: int, seq_len: int, d_model: int = 32,
                 nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 64,
                 n_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x):  # x: (batch, seq_len, n_features)
        h = self.input_proj(x) + self.pos_embed
        h = self.encoder(h)
        return self.fc(h[:, -1])  # output at the "current window" position


def class_weights(y: np.ndarray, n_classes: int, device) -> torch.Tensor:
    """sklearn-style 'balanced' weights, so CrossEntropyLoss matches the
    class_weight="balanced" used by the RF/SVM baselines."""
    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    w = counts.sum() / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float32, device=device)


def train_model(model: TransformerClassifier, X: np.ndarray, y: np.ndarray,
               n_classes: int, epochs: int, lr: float, device,
               batch_size: int = 64) -> TransformerClassifier:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss(weight=class_weights(y, n_classes, device))
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    n = X_t.shape[0]
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = crit(model(X_t[idx]), y_t[idx])
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def predict(model: TransformerClassifier, X: np.ndarray, device) -> np.ndarray:
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    return model(X_t).argmax(dim=1).cpu().numpy()


# ---------------------------------------------------------------------------
# leave-one-subject-out
# ---------------------------------------------------------------------------
def run_loso_transformer(subs, Seq_by, y_by, n_classes, seq_len, d_model, nhead,
                         num_layers, dim_feedforward, dropout, epochs, lr,
                         batch_size, device, seed) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    accs, f1s, all_true, all_pred = [], [], [], []
    for test_s in subs:
        train_s = [s for s in subs if s != test_s]
        Xtr = np.concatenate([Seq_by[s] for s in train_s], axis=0)
        ytr = np.concatenate([y_by[s] for s in train_s])
        Xte, yte = Seq_by[test_s], y_by[test_s]

        torch.manual_seed(seed)
        model = TransformerClassifier(
            n_features=Xtr.shape[-1], seq_len=seq_len, d_model=d_model,
            nhead=nhead, num_layers=num_layers, dim_feedforward=dim_feedforward,
            n_classes=n_classes, dropout=dropout).to(device)
        model = train_model(model, Xtr, ytr, n_classes, epochs, lr, device, batch_size)
        pred = predict(model, Xte, device)

        accs.append(accuracy_score(yte, pred))
        f1s.append(f1_score(yte, pred, average="macro", zero_division=0))
        all_true.append(yte)
        all_pred.append(pred)

    return dict(
        mean_acc=float(np.mean(accs)),
        std_acc=float(np.std(accs)),
        mean_f1=float(np.mean(f1s)),
        per_fold_acc=[float(v) for v in accs],
        per_fold_f1=[float(v) for v in f1s],
        all_true=np.concatenate(all_true).tolist(),
        all_pred=np.concatenate(all_pred).tolist(),
    )


def run_loso_transformer_finetune(subs, Seq_by, y_by, n_classes, seq_len, d_model,
                                  nhead, num_layers, dim_feedforward, dropout,
                                  epochs, lr, batch_size, device, seed,
                                  calib_frac, ft_epochs, ft_lr):
    """Pretrain on n-1 subjects, then fine-tune a copy on an early slice of the
    held-out subject's own (time-ordered) sequences. Returns (zero-shot,
    fine-tuned, used_subjects) -- folds where the calibration slice is
    single-class are skipped from BOTH so the two stay comparable.

    NOTE: as with lstm_classify_biceps.py, a time-ordered 40/60 split is an
    artifact on this monotonic-fatigue dataset -- see TRANSFORMER_HANDOVER.md.
    """
    from sklearn.metrics import accuracy_score, f1_score

    zeroshot, finetuned, used = [], [], []
    for test_s in subs:
        train_s = [s for s in subs if s != test_s]
        Xtr = np.concatenate([Seq_by[s] for s in train_s], axis=0)
        ytr = np.concatenate([y_by[s] for s in train_s])

        Xte_full, yte_full = Seq_by[test_s], y_by[test_s]
        n_calib = max(1, int(round(len(Xte_full) * calib_frac)))
        X_calib, y_calib = Xte_full[:n_calib], yte_full[:n_calib]
        X_eval, y_eval = Xte_full[n_calib:], yte_full[n_calib:]
        if X_eval.size == 0 or np.unique(y_calib).size < 2:
            print(f"  test S{test_s:<2}  skipped (calibration slice single-class)")
            continue

        torch.manual_seed(seed)
        base = TransformerClassifier(
            n_features=Xtr.shape[-1], seq_len=seq_len, d_model=d_model,
            nhead=nhead, num_layers=num_layers, dim_feedforward=dim_feedforward,
            n_classes=n_classes, dropout=dropout).to(device)
        base = train_model(base, Xtr, ytr, n_classes, epochs, lr, device, batch_size)

        pred_zs = predict(base, X_eval, device)
        zeroshot.append((accuracy_score(y_eval, pred_zs),
                        f1_score(y_eval, pred_zs, average="macro", zero_division=0)))

        ft_model = copy.deepcopy(base)
        ft_model = train_model(ft_model, X_calib, y_calib, n_classes,
                               ft_epochs, ft_lr, device, batch_size)
        pred_ft = predict(ft_model, X_eval, device)
        finetuned.append((accuracy_score(y_eval, pred_ft),
                         f1_score(y_eval, pred_ft, average="macro", zero_division=0)))
        used.append(test_s)

        print(f"  test S{test_s:<2}  zero-shot acc={zeroshot[-1][0]:.3f} F1={zeroshot[-1][1]:.3f}"
              f"   |   fine-tuned acc={finetuned[-1][0]:.3f} F1={finetuned[-1][1]:.3f}")

    def summarize(pairs):
        arr = np.array(pairs)
        return dict(
            mean_acc=float(arr[:, 0].mean()), std_acc=float(arr[:, 0].std()),
            mean_f1=float(arr[:, 1].mean()),
            per_fold_acc=[float(v) for v in arr[:, 0]],
            per_fold_f1=[float(v) for v in arr[:, 1]],
        )

    return summarize(zeroshot), summarize(finetuned), used


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--side", choices=["R", "L"], default="R")
    ap.add_argument("--win", type=float, default=4.0)
    ap.add_argument("--step", type=float, default=2.0)
    ap.add_argument("--target-fs", type=int, default=250,
                    help="downsample before feature extraction (default 250, "
                         "the validated OpenBCI/HYFY bridge rate; pass 0 for "
                         "native 1259 Hz)")
    ap.add_argument("--label-mode", choices=[
        "3class",
        "binary_drop_transition",
        "binary_transition_fatigued",
        "binary_transition_fresh",
    ], default="3class")
    ap.add_argument("--binary", action="store_true",
                    help="alias for --label-mode binary_drop_transition")
    ap.add_argument("--min-label-purity", type=float, default=0.80)
    ap.add_argument("--transition-margin-sec", type=float, default=4.0,
                    help="discard windows centered this close to a label "
                         "change (4s matches classify_biceps's best config)")
    ap.add_argument("--seq-len", type=int, default=6,
                    help="causal window history length fed to the Transformer "
                         "(current window + history)")
    ap.add_argument("--predict-next", action="store_true",
                    help="predict the NEXT window's label instead of the "
                         "current one (the 'prediction' half of the brief)")
    ap.add_argument("--d-model", type=int, default=32)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2,
                    help="number of Transformer encoder layers")
    ap.add_argument("--ff-dim", type=int, default=64,
                    help="feed-forward dimension inside each encoder layer")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--no-norm", dest="norm", action="store_false",
                    help="disable subject fresh-baseline calibration")
    ap.set_defaults(norm=True)
    ap.add_argument("--finetune", action="store_true",
                    help="also report a transfer-learning (pretrain + "
                         "per-subject fine-tune) result, separately")
    ap.add_argument("--calib-frac", type=float, default=0.4)
    ap.add_argument("--ft-epochs", type=int, default=20)
    ap.add_argument("--ft-lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", default=None,
                    help="optional metrics JSON path (same schema as "
                         "classify_biceps.py / lstm_classify_biceps.py so "
                         "make_classifier_report.py picks it up)")
    args = ap.parse_args()

    if args.binary:
        args.label_mode = "binary_drop_transition"
    target_fs = args.target_fs or None
    n_classes = 3 if args.label_mode == "3class" else 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    subs, Seq_by, y_by = [], {}, {}
    dropped = {}
    n_features = None
    for s in range(1, 14):
        data = cb.subject_windows(
            args.root, s, args.side, args.win, args.step,
            target_fs=target_fs,
            min_label_purity=args.min_label_purity,
            transition_margin_sec=args.transition_margin_sec,
            temporal=False,   # raw per-window features; the Transformer learns
                              # the temporal pattern itself from the sequence
            label_mode=args.label_mode,
        )
        if data is None or data.X.size == 0:
            print(f"S{s}: no labels / no windows, skipped")
            continue

        X, y = data.X, data.y
        if args.norm:
            base = X[y == 0]
            if base.shape[0] >= 3:
                mu, sd = base.mean(0), base.std(0)
                sd[sd == 0] = 1.0
                X = (X - mu) / sd

        if np.unique(y).size < 2:
            print(f"S{s}: single-class after filtering, skipped")
            continue

        seqs, labels = build_sequences(X, y, args.seq_len, args.predict_next)
        if labels.size == 0 or np.unique(labels).size < 2:
            print(f"S{s}: single-class after sequencing, skipped")
            continue

        subs.append(s)
        Seq_by[s] = seqs
        y_by[s] = labels
        n_features = seqs.shape[-1]
        dropped[s] = {"mixed": data.dropped_mixed, "boundary": data.dropped_boundary}
        dist = {int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}
        print(f"S{s}: {seqs.shape[0]} sequences x {seqs.shape[1]} steps x "
              f"{seqs.shape[2]} feats  class dist {dist}  "
              f"dropped mixed={data.dropped_mixed} boundary={data.dropped_boundary}")

    if len(subs) < 2:
        print("not enough labelled subjects for LOSO")
        return

    fs_label = target_fs or loader.FS_NATIVE
    task_label = "predict-next" if args.predict_next else "classify-current"
    print(f"\n=== Transformer LOSO ({args.label_mode}, {fs_label}Hz, "
          f"seq-len={args.seq_len}, {task_label}) ===")
    print(f"features per step: {n_features}")

    res = run_loso_transformer(subs, Seq_by, y_by, n_classes, args.seq_len,
                               args.d_model, args.nhead, args.layers,
                               args.ff_dim, args.dropout, args.epochs, args.lr,
                               args.batch_size, device, args.seed)
    for i, test_s in enumerate(subs):
        print(f"  test S{test_s:<2}  acc={res['per_fold_acc'][i]:.3f}  "
              f"macroF1={res['per_fold_f1'][i]:.3f}")
    print(f"  mean acc  = {res['mean_acc']:.3f} +/- {res['std_acc']:.3f}")
    print(f"  mean F1   = {res['mean_f1']:.3f}")

    results = {"Transformer": res}

    if args.finetune:
        print("\n--- transfer learning (pretrain on 11 subjects + per-subject "
              "fine-tune on an early calibration slice) ---")
        zs, ft, used = run_loso_transformer_finetune(
            subs, Seq_by, y_by, n_classes, args.seq_len, args.d_model, args.nhead,
            args.layers, args.ff_dim, args.dropout, args.epochs, args.lr,
            args.batch_size, device, args.seed, args.calib_frac, args.ft_epochs,
            args.ft_lr)
        print(f"\n{'':<12} {'Mean Acc':>10} {'Std Acc':>10} {'Mean F1':>10}")
        print("-" * 44)
        print(f"{'zero-shot':<12} {zs['mean_acc']:>10.3f} {zs['std_acc']:>10.3f} {zs['mean_f1']:>10.3f}")
        print(f"{'fine-tuned':<12} {ft['mean_acc']:>10.3f} {ft['std_acc']:>10.3f} {ft['mean_f1']:>10.3f}")
        print(f"(transfer-learning evaluated on {len(used)}/{len(subs)} subjects "
              f"-- folds with a single-class calibration slice are skipped)")
        results["Transformer_transfer_zeroshot"] = zs
        results["Transformer_transfer_finetuned"] = ft

    print(f"\n{'Model':<28} {'Mean Acc':>10} {'Std Acc':>10} {'Mean F1':>10}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<28} {r['mean_acc']:>10.3f} {r['std_acc']:>10.3f} {r['mean_f1']:>10.3f}")

    if args.json_out:
        temporal_label = f"Transformer(seq={args.seq_len})"
        if args.predict_next:
            temporal_label += ",predict-next"
        payload = {
            "config": {
                "root": args.root, "side": args.side, "win": args.win,
                "step": args.step, "target_fs": target_fs,
                "label_mode": args.label_mode,
                "min_label_purity": args.min_label_purity,
                "transition_margin_sec": args.transition_margin_sec,
                "temporal": temporal_label,
                "seq_len": args.seq_len,
                "predict_next": args.predict_next,
                "n_features": n_features,
                "d_model": args.d_model, "nhead": args.nhead,
                "layers": args.layers, "ff_dim": args.ff_dim,
                "epochs": args.epochs,
            },
            "subjects": subs,
            "dropped_windows": dropped,
            "results": results,
        }
        os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nsaved {args.json_out}")


if __name__ == "__main__":
    main()
