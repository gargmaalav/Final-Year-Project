# LSTM Sequence Classifier - Handover

Date: 2026-06-14
Branch: `aryan/lstm-sequence-classifier`

## What this adds

Per the supervisor's "this break: LSTM pattern classification and prediction"
plan. Nothing in the existing pipeline was changed - this is a new file
(`lstm_classify_biceps.py`) that reuses `classify_biceps.py`'s validated data
pipeline directly (`subject_windows`):

- same loader, `--target-fs 250` downsampling (the validated OpenBCI/HYFY
  bridge rate)
- same majority/purity window labeling + 4 s transition-margin trimming
- same label modes (`3class`, `binary_drop_transition`, ...)
- same subject-relative fresh-baseline normalisation
- same leave-one-subject-out (LOSO) validation, no leakage

What's new: each 4 s window's 8 base features (RMS/MAV/WL/VAR/ZC/SSC/MDF/MNF)
becomes one LSTM timestep. A causal sequence of the current + 5 previous
windows (`--seq-len 6`, the same span as classify_biceps's
`--temporal-history 5`) feeds a single-layer LSTM (hidden=64) + linear head.

Two modes:
- **classify-current** (default): same task RF/SVM/KNN solve
- **`--predict-next`**: predicts the label of the NEXT window (2 s ahead) -
  the "prediction" half of the supervisor's brief
- optional **`--finetune`**: pretrain + per-subject fine-tune transfer-learning
  experiment (see caveat below)

## Results (LOSO, 250 Hz, subject-relative norm, 4 s transition margin)

| Task | Best classical (RF, 32 feats) | LSTM (8 raw feats, seq=6) |
|---|---:|---:|
| Binary fatigue detection | 89.6% acc / 0.854 F1 | 88.7% acc / 0.843 F1 |
| 3-class | 70.5% acc / 0.643 F1 | 68.4% acc / 0.638 F1 |
| 3-class, predict 2s ahead | - | 68.4% acc / 0.632 F1 |

**Headline takeaway:** the LSTM, using only the 8 raw per-window features (no
hand-engineered deltas/slopes), gets within ~1-2pp of RF's 32-feature result
on both tasks - it learns a comparable temporal pattern end-to-end instead of
being told the pattern. And predicting the fatigue state 2 seconds into the
future is essentially as accurate as classifying the current window (68.4%
either way) - a genuine "prediction" result.

The per-fold pattern matches the classical pipeline too: **Subject 7 is the
weakest fold for the LSTM on both tasks** (acc 0.55 binary / 0.42 3-class),
the same subject that's weakest for RF and the only one (besides S3) whose
MDF doesn't decline. The model's errors track the same physiologically-odd
subject as the classical pipeline, not random noise.

## Important caveat: `--finetune` is not meaningful on this dataset (yet)

The optional transfer-learning experiment (pretrain on 11 subjects, fine-tune
on the first 40% of the held-out subject's session, evaluate on the last 60%)
reported **100% accuracy on every fold that ran - do not present this
number, it's an artifact.**

Because fatigue progresses monotonically within a session (early windows =
fresh, late windows = fatigued), a time-ordered 40/60 split makes the "last
60%" eval slice single-class for almost every subject: 8/12 folds were
skipped outright (calibration slice was already single-class), and the 4 that
ran had single-class eval slices too. A model that always predicts "fatigued"
scores perfectly there. The `--finetune` flag/code is left in the script for
future use, but a time-ordered split is the wrong design for this label
distribution - see suggestions below.

## Suggestions

1. **Lead with the binary 250 Hz result for Monday** (89.6% RF / 88.7% LSTM) -
   both are honest LOSO numbers at the validated deployment rate.
2. Frame the LSTM result as **"deep learning validates the hand-crafted
   feature set"** rather than "LSTM beats RF" - it doesn't, yet, but it's
   close without any manual feature engineering.
3. For genuine transfer learning, don't use a time-ordered calibration/eval
   split given the monotonic label structure. Better options: (a) an
   interleaved/stratified within-subject split for fine-tune vs. eval, or (b)
   fine-tune on a handful of labeled windows spread across the WHOLE session
   (few-shot) rather than only the first 40%.
4. `seq-len=6` and `hidden=64` were only lightly tuned (5 vs 30 vs 60 epochs
   tried, 30 kept). There's room for a proper sweep (hidden size, num_layers,
   dropout, seq-len) - but don't tune against the LOSO numbers themselves,
   that's the same kind of leakage as the transfer-learning issue above. Use
   a nested/inner validation split if pursuing this.
5. Next semester's Transformer step can reuse this exact data pipeline
   (`subject_windows` -> causal sequences) - same framing, swap the LSTM for
   an attention-based encoder.
6. Possible follow-up: feed the LSTM's learned hidden state into the RF as
   extra columns alongside the 32 hand-crafted features - a simple way to
   combine learned and engineered temporal representations.

## How to reproduce

```
python lstm_classify_biceps.py --root sEMG_data --side R --label-mode binary_drop_transition --json-out out/metrics_lstm_binary_250hz_m4.json
python lstm_classify_biceps.py --root sEMG_data --side R --label-mode 3class --json-out out/metrics_lstm_3class_250hz_m4.json
python lstm_classify_biceps.py --root sEMG_data --side R --label-mode 3class --predict-next --json-out out/metrics_lstm_predictnext_3class_250hz_m4.json
python make_classifier_report.py
```
