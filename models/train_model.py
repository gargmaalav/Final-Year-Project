"""
Train and save a deployable DEEP-LEARNING fatigue model (LSTM).
==============================================================

The research scripts run LOSO and discard the trained model. This trains ONE
LSTM on ALL subjects and saves it so classify.py can score a query window.
Built entirely on the existing, validated pipeline (classify_biceps.py +
lstm_classify_biceps.py) -- this file is glue, not new modelling.

Run once:
    python models/train_model.py
Produces models/fatigue_model.pt, consumed by models/classify.py.

MODEL CHOICE -- LSTM now, Transformer later
-------------------------------------------
We deploy the LSTM, not the Transformer, on the current 13-subject dataset:
  - LSTM beats Transformer on the headline task (binary LOSO 88.7% vs 87.1%;
    see zenodo_biceps/TRANSFORMER_HANDOVER.md).
  - Transformers are data-hungry; with few subjects the LSTM's stronger
    sequential inductive bias wins, while attention has too little data to
    learn temporal structure from scratch.
WHEN TO SWITCH: once the team's own OpenBCI forearm-grip data is collected
(more subjects / longer sessions), retrain using transformer_classify_biceps.py's
encoder -- attention scales better with data and can exploit a longer seq_len.
This also matches the supervisor's "LSTM this break, Transformer next semester"
roadmap. Swapping is cheap: both models share this exact pipeline, so it's a
one-import change here plus loading TransformerClassifier in classify.py.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

# --- reuse the existing validated pipeline ---------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZB = os.path.join(REPO_ROOT, "zenodo_biceps")
sys.path.insert(0, ZB)
import classify_biceps as cb         # noqa: E402  subject_windows, BASE_FEATS
import lstm_classify_biceps as lstm  # noqa: E402  build_sequences, LSTMClassifier, train_model
import loader                        # noqa: E402

DATA_ROOT = os.path.join(ZB, "sEMG_data")  # gitignored -- see README
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "fatigue_model.pt")

# --- config (mirrors the validated LSTM run) -------------------------------
SIDE = "R"
WIN_SEC = 4.0
STEP_SEC = 2.0
TARGET_FS = 250                       # the validated OpenBCI/HYFY bridge rate
LABEL_MODE = "binary_drop_transition"  # 0 = non-fatigue, 1 = fatigue
SEQ_LEN = 6
HIDDEN = 64
LAYERS = 1
DROPOUT = 0.0
EPOCHS = 30
LR = 1e-3
BATCH = 64
SEED = 0
# DECISION: margin = 0 so the training window grid is contiguous, matching how
# classify() must build sequences at query time (no labels available live, so it
# can't drop transition windows). The reported 88.7% used a 4 s margin for a
# clean eval; this deployable model trades a hair of that for train/serve
# consistency.
TRANSITION_MARGIN = 0.0
ALL_SUBJECTS = range(1, 14)
# DECISION: hold out the demo subjects. Train only on TRAIN_SUBJECTS so the
# chatbot can be demoed on subjects 12-13 the model never saw -- a genuine
# unseen-subject prediction, the same condition as the 88.7% LOSO number.
# Baselines are still saved for ALL 13 (per-subject calibration is not leakage),
# so classify() can normalise the held-out demo subjects.
# PRODUCTION NOTE: once the team's own OpenBCI data exists (query targets are
# then genuinely new athletes), train on all available subjects instead.
TRAIN_SUBJECTS = range(1, 12)   # subjects 1-11
DEMO_SUBJECTS = (12, 13)        # held out for honest chatbot demos


def build_training_set(root: str):
    """Return (train sequences, labels, per-subject baselines).

    Baselines are computed for ALL usable subjects (so classify() can normalise
    the held-out demo subjects), but only TRAIN_SUBJECTS contribute training data.
    """
    seqs_all, y_all, baselines = [], [], {}
    for s in ALL_SUBJECTS:
        data = cb.subject_windows(
            root, s, SIDE, WIN_SEC, STEP_SEC,
            target_fs=TARGET_FS, transition_margin_sec=TRANSITION_MARGIN,
            temporal=False, label_mode=LABEL_MODE,
        )
        if data is None or data.X.size == 0:
            print(f"S{s}: skipped (no windows)")
            continue

        X, y = data.X, data.y
        # subject fresh-baseline normalisation, saved for EVERY usable subject so
        # classify() applies the SAME transform at query time (incl. held-out demos)
        base = X[y == 0]
        if base.shape[0] < 3:
            print(f"S{s}: skipped (too few fresh windows for a baseline)")
            continue
        mu, sd = base.mean(0), base.std(0)
        sd[sd == 0] = 1.0
        baselines[s] = {"mu": mu.tolist(), "sd": sd.tolist()}

        if s not in TRAIN_SUBJECTS:
            print(f"S{s}: held out for demo (baseline saved, not trained on)")
            continue

        seqs, labels = lstm.build_sequences((X - mu) / sd, y, SEQ_LEN,
                                            predict_next=False)
        if labels.size == 0 or np.unique(labels).size < 2:
            print(f"S{s}: skipped (single-class after sequencing)")
            continue
        seqs_all.append(seqs)
        y_all.append(labels)
        print(f"S{s}: {seqs.shape[0]} training sequences")

    return np.concatenate(seqs_all, 0), np.concatenate(y_all), baselines


def main():
    if not os.path.isdir(DATA_ROOT):
        sys.exit(f"dataset not found at {DATA_ROOT}\n"
                 f"  -> download Zenodo 14182446 and place it there (see README)")

    X, y, baselines = build_training_set(DATA_ROOT)
    n_features = X.shape[-1]
    n_classes = 2 if LABEL_MODE.startswith("binary") else 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\ndevice: {device}")
    print(f"trained on {len(X)} sequences from subjects {list(TRAIN_SUBJECTS)}")
    print(f"baselines saved for {len(baselines)} subjects "
          f"(demo held-out: {list(DEMO_SUBJECTS)})")

    torch.manual_seed(SEED)
    model = lstm.LSTMClassifier(n_features=n_features, hidden_size=HIDDEN,
                                num_layers=LAYERS, n_classes=n_classes,
                                dropout=DROPOUT).to(device)
    model = lstm.train_model(model, X, y, n_classes, EPOCHS, LR, device, BATCH)

    bundle = {
        "model_kind": "LSTM",
        "state_dict": model.state_dict(),
        "arch": {"n_features": n_features, "hidden_size": HIDDEN,
                 "num_layers": LAYERS, "n_classes": n_classes, "dropout": DROPOUT},
        "base_feats": list(cb.BASE_FEATS),   # 8 feature names, in order
        "baselines": baselines,              # per-subject fresh mu/sd (all 13)
        "train_subjects": list(TRAIN_SUBJECTS),
        "demo_subjects": list(DEMO_SUBJECTS),
        "config": {"side": SIDE, "win_sec": WIN_SEC, "step_sec": STEP_SEC,
                   "target_fs": TARGET_FS, "label_mode": LABEL_MODE,
                   "seq_len": SEQ_LEN},
        "label_meaning": {0: "non-fatigue", 1: "fatigue"},
    }
    torch.save(bundle, MODEL_PATH)
    print(f"\nsaved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
