"""Per-subject reliability tier for the DEPLOYED binary LSTM
(models/fatigue_model.pt, trained on subjects 1-11 per models/train_model.py:77).

Subjects 1-11 trained the model - there is no fair held-out score for them, so
they get an honest disclaimer instead of a fabricated "reliable" label. Only
subjects 12 and 13 were held out and independently evaluated
(models/eval_deployed.py): S12 acc=0.854, S13 acc=0.976.

Do NOT confuse this with the 91%/73.1% LOSO numbers from
zenodo_biceps/classify_biceps.py - those describe a different, undeployed
sklearn 3-class model. This module is about the model actually answering
chatbot queries.
"""
from __future__ import annotations

ALL_SUBJECTS = range(1, 14)

TRAIN_SUBJECTS = frozenset(range(1, 12))  # mirrors models/train_model.py:77

_HELD_OUT_ACC = {12: 0.854, 13: 0.976}   # models/eval_deployed.py, live-verified

DISCLAIMER = "Not independently tested for this person"


def reliability_tier(subject: int) -> str:
    """Plain-language reliability label for the deployed model on this subject.

    Highly reliable    - held-out accuracy >= 0.90
    Somewhat reliable   - held-out accuracy >= 0.70 and < 0.90
    disclaimer string   - subject trained the model (1-11); no fair score exists
    """
    if subject not in ALL_SUBJECTS:
        raise ValueError(f"subject must be 1-13, got {subject}")
    if subject in TRAIN_SUBJECTS:
        return DISCLAIMER
    acc = _HELD_OUT_ACC[subject]
    if acc >= 0.90:
        return "Highly reliable"
    if acc >= 0.70:
        return "Somewhat reliable"
    return "Less reliable for this person"
