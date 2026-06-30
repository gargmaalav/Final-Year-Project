# Transformer Sequence Classifier - Handover

Date: 2026-06-15
Branch: `aryan/lstm-sequence-classifier`

## What this adds

The final step of the supervisor's plan: *"this break: LSTM pattern
classification and prediction ... next semester: transformer."*
`transformer_classify_biceps.py` is a near-mirror of `lstm_classify_biceps.py`
- same data pipeline, same LOSO protocol, same CLI shape, same JSON schema -
with the LSTM swapped for a small encoder-only Transformer:

- same loader, `--target-fs 250` downsampling, majority/purity window
  labeling + 4 s transition-margin trimming, subject-relative fresh-baseline
  normalisation, and LOSO validation as classify_biceps / lstm_classify_biceps
- same 8 base per-window features (RMS/MAV/WL/VAR/ZC/SSC/MDF/MNF) as one
  timestep, same causal `seq-len=6` (current window + 5 history)
- **model**: `input_proj` (8 -> d_model=32) + learned positional embedding +
  2-layer `nn.TransformerEncoder` (4 heads, ff_dim=64, dropout=0.1) + linear
  head reading out the encoder's output at the final ("current window")
  sequence position - the direct analogue of the LSTM's final hidden state
- no causal attention mask needed: the sequence already only contains
  current+past windows, so full self-attention inside it can't see the future
- same two modes as the LSTM: classify-current (default) and `--predict-next`
- same `--finetune` transfer-learning option (off by default, same caveat as
  below)

## Results (LOSO, 250 Hz, subject-relative norm, 4 s transition margin)

| Task | RF (32 feats) | LSTM (8 raw feats, seq=6) | Transformer (8 raw feats, seq=6) |
|---|---:|---:|---:|
| Binary fatigue detection | 89.6% acc / 0.854 F1 | 88.7% acc / 0.843 F1 | 87.1% acc / 0.820 F1 |
| 3-class | 70.5% acc / 0.643 F1 | 68.4% acc / 0.638 F1 | 68.5% acc / 0.634 F1 |
| 3-class, predict 2s ahead | - | 68.4% acc / 0.632 F1 | 69.5% acc / 0.636 F1 |

**Headline takeaway:** the Transformer lands in the same band as the LSTM on
all three tasks - about 1-3pp below RF on binary, and statistically
indistinguishable from the LSTM on both 3-class tasks (even a touch ahead on
predict-next: 69.5% vs 68.4%). Swapping the sequence model's architecture
(recurrent -> attention) barely moves the result. That's evidence the
**ceiling here is set by the dataset size and the feature/label pipeline, not
by which sequence architecture reads the 6-window history** - both deep models
get within a few points of RF's 32-feature result using only the 8 raw
per-window features, with no hand-engineered deltas/slopes.

The per-fold pattern again matches the classical pipeline and the LSTM:
**Subject 7 is the weakest binary fold for the Transformer too** (acc 0.588,
macro-F1 0.559 - close to the LSTM's 0.553/0.503 for the same subject). For
3-class, Subjects 5 and 7 are again the two hardest folds (acc 0.44 and 0.47
respectively), exactly as for the LSTM (0.40 and 0.42). The errors track the
same physiologically-odd subjects across RF, LSTM, and Transformer - not
random noise, and not an architecture-specific weakness.

## Important caveat: `--finetune` is not meaningful on this dataset (yet)

Carried over unchanged from `lstm_classify_biceps.py` and not re-run here:
the optional transfer-learning experiment (pretrain on 11 subjects, fine-tune
on the first 40% of the held-out subject's session, evaluate on the last 60%)
is an artifact on this dataset. Because fatigue progresses monotonically
within a session, a time-ordered 40/60 split makes the "last 60%" eval slice
single-class for almost every subject, so a model that always predicts
"fatigued" scores perfectly there. The flag/code is present
(`--finetune`) for future use once a non-time-ordered calibration/eval split
is designed (see Suggestions in `LSTM_HANDOVER.md`, point 3 - the same fix
applies here).

## Suggestions

1. **Lead with the binary 250 Hz result** for the report (89.6% RF / 88.7%
   LSTM / 87.1% Transformer) - all three are honest LOSO numbers at the
   validated deployment rate, and the small, consistent gap RF > LSTM >
   Transformer is itself a finding worth stating: at 13 subjects, more
   model flexibility doesn't buy more accuracy once the features and
   labelling are fixed.
2. Frame the overall LSTM + Transformer story as **"two different deep
   architectures both validate the hand-crafted feature set, and neither
   beats it"** - the interesting result is that the *temporal information*
   in the 8 raw features is enough, however it's processed, to get within a
   few points of RF's 32 hand-engineered features.
3. `d_model=32`, `nhead=4`, `layers=2`, `ff_dim=64`, `dropout=0.1` and
   `epochs=30` were chosen to be roughly parameter-comparable to the LSTM's
   `hidden=64` (both ~15-20k params) and were not tuned. As with the LSTM,
   there's room for a sweep (d_model, nhead, layers, dropout, seq-len) - but
   use a nested/inner validation split, not the LOSO numbers themselves, to
   avoid the same kind of leakage flagged for the LSTM's `--finetune`.
4. Possible follow-up: an ablation that removes the positional embedding, or
   widens `seq-len` beyond 6, to see whether the Transformer can extract more
   signal from a longer history than the LSTM can (attention doesn't decay
   with distance the way recurrence does) - this would be a genuine
   "transformer advantage" angle if it shows up.
5. This completes the supervisor's break -> next-semester progression
   (RF/SVM/KNN -> LSTM -> Transformer) on the same validated pipeline and
   public dataset. When real forearm-grip OpenBCI/HYFY data is collected, all
   three scripts can be re-run unchanged (just point `--root` at the new data)
   for a like-for-like comparison against this baseline.

## How to reproduce

```
python transformer_classify_biceps.py --root sEMG_data --side R --label-mode binary_drop_transition --json-out out/metrics_transformer_binary_250hz_m4.json
python transformer_classify_biceps.py --root sEMG_data --side R --label-mode 3class --json-out out/metrics_transformer_3class_250hz_m4.json
python transformer_classify_biceps.py --root sEMG_data --side R --label-mode 3class --predict-next --json-out out/metrics_transformer_predictnext_3class_250hz_m4.json
python make_classifier_report.py
```
