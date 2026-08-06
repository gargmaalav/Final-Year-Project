"""
Evaluate the DEPLOYED model (models/fatigue_model.pt) on the held-out demo
subjects it never trained on.
=====================================================================

This is the honest accuracy of what classify() actually serves. It is distinct
from the LOSO numbers in the handover docs, which used a different config
(32 features / 4 s transition margin). The deployed model is the LSTM on 8 raw
features, margin 0, trained on subjects 1-11 -- so its real quality has to be
measured separately, here, on subjects 12-13.

Run:
    python models/eval_deployed.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZB = os.path.join(REPO_ROOT, "zenodo_biceps")
sys.path.insert(0, ZB)
import classify_biceps as cb         # noqa: E402
import lstm_classify_biceps as lstm  # noqa: E402
import loader                        # noqa: E402

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "fatigue_model.pt")
DATA_ROOT = os.path.join(ZB, "sEMG_data")


def eval_subject(bundle, model, s):
    """Return (true, pred) label arrays for one subject, or None if unusable."""
    cfg = bundle["config"]
    bl = bundle["baselines"].get(s) or bundle["baselines"].get(str(s))
    if bl is None:
        return None
    mu, sd = np.array(bl["mu"], float), np.array(bl["sd"], float)

    data = cb.subject_windows(
        DATA_ROOT, s, cfg["side"], cfg["win_sec"], cfg["step_sec"],
        target_fs=cfg["target_fs"], transition_margin_sec=0.0,
        temporal=False, label_mode=cfg["label_mode"])
    if data is None or data.X.size == 0:
        return None

    seqs, labels = lstm.build_sequences((data.X - mu) / sd, data.y,
                                        cfg["seq_len"], predict_next=False)
    if labels.size == 0 or np.unique(labels).size < 2:
        return None

    with torch.no_grad():
        pred = model(torch.tensor(seqs, dtype=torch.float32)).argmax(1).numpy()
    return labels, pred


def main():
    bundle = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    a = bundle["arch"]
    model = lstm.LSTMClassifier(
        n_features=a["n_features"], hidden_size=a["hidden_size"],
        num_layers=a["num_layers"], n_classes=a["n_classes"],
        dropout=a["dropout"])
    model.load_state_dict(bundle["state_dict"])
    model.eval()

    demo = bundle.get("demo_subjects", [12, 13])
    print(f"Deployed model ({bundle['model_kind']}), "
          f"trained on {bundle['train_subjects']}")
    print(f"Evaluated on held-out subjects {demo} "
          f"(label mode: {bundle['config']['label_mode']})\n")

    all_t, all_p = [], []
    for s in demo:
        out = eval_subject(bundle, model, s)
        if out is None:
            print(f"S{s}: skipped (no usable windows)")
            continue
        yt, yp = out
        print(f"S{s}: {len(yt)} windows  acc={accuracy_score(yt, yp):.3f}  "
              f"macro-F1={f1_score(yt, yp, average='macro', zero_division=0):.3f}")
        all_t.append(yt)
        all_p.append(yp)

    if all_t:
        yt, yp = np.concatenate(all_t), np.concatenate(all_p)
        print(f"\nPooled held-out: acc={accuracy_score(yt, yp):.3f}  "
              f"macro-F1={f1_score(yt, yp, average='macro', zero_division=0):.3f}")
        print("confusion matrix (rows=true, cols=pred):")
        print(confusion_matrix(yt, yp))


if __name__ == "__main__":
    main()
