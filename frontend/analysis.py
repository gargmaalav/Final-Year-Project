"""
Whole-recording and cross-subject analysis.
===========================================

Everything here is built on models/classify.py's classify_many(), which scores
many windows from a single loaded recording. Nothing new is modelled: these
are the same LSTM predictions the single-window path returns, just gathered
across a recording or across subjects and summarised.

A note on cost, because it shapes the design. One classify() call re-reads and
re-resamples a multi-MB CSV, so a naive "score every window of every subject"
answer takes minutes on a laptop. Two things keep it usable: the recording is
loaded once per subject and reused, and scans are taken at a coarse step
(SCAN_STEP_SEC) rather than at every window. The step is a deliberate
resolution/latency trade -- it is stated in the answers so nobody reads more
precision into an onset time than was actually measured.
"""
from __future__ import annotations

import numpy as np

from classify import (classify, classify_many, load_subject_segment,
                      subject_reference)

SCAN_STEP_SEC = 5.0        # resolution of a whole-recording scan
SUSTAIN_WINDOWS = 2        # consecutive fatigue readings before calling it onset
FATIGUE_LABELS = {1, 2}    # matches app.py's _FATIGUE_STATE mapping


def _duration(seg) -> float:
    t = np.asarray(seg.t, float)
    return float(t[-1]) if t.size else 0.0


def scan_times(duration: float, step: float = SCAN_STEP_SEC) -> list[float]:
    """Evenly spaced sample points across a recording."""
    if duration <= 0:
        return [0.0]
    n = max(2, int(duration // step) + 1)
    return [round(i * duration / (n - 1), 3) for i in range(n)]


def scan_recording(subject: int, side: str = "R", seg=None, fs: int | None = None,
                   step: float = SCAN_STEP_SEC) -> dict:
    """Fatigue readings across one whole recording.

    Returns {"subject", "side", "duration", "step", "readings"} where each
    reading is classify()'s dict plus its t_start.
    """
    if seg is None:
        seg, fs = load_subject_segment(subject, side)
    duration = _duration(seg)
    times = scan_times(duration, step)
    readings = classify_many(subject, times, side, seg=seg, fs=fs)
    return {"subject": subject, "side": side, "duration": duration,
            "step": step, "readings": readings}


# Measured against the dataset's own ground-truth labels (the end of the
# non-fatigue span, per loader.fatigue_onsets): MAE 11.8 s over the 11 subjects
# an onset is actually reported for, 8/11 within +/-15 s.
#
# This number matters because it is NOT the scan step. A 5 s step gives 5 s of
# *precision*, and quoting that as the answer's accuracy overstated it by more
# than twofold. Reproduce with scripts/validate_onset.py.
ONSET_MAE_SEC = 11.8
ONSET_TYPICAL_ERROR_SEC = 12.0


def find_onset(scan: dict, sustain: int = SUSTAIN_WINDOWS) -> dict:
    """First time fatigue appears and stays.

    A single flipped window is not an onset -- the classifier is ~88% accurate
    per window, so isolated flips are expected and reporting one as "the
    moment fatigue set in" would be reading noise as signal. `sustain`
    consecutive fatigued readings are required.
    """
    readings = scan["readings"]
    flags = [r["fatigue_label"] in FATIGUE_LABELS for r in readings]
    base = {"sustain": sustain, "step": scan["step"],
            "typical_error": ONSET_TYPICAL_ERROR_SEC,
            "fraction_fatigued": sum(flags) / len(flags) if flags else 0.0}

    for i in range(len(flags) - sustain + 1):
        if all(flags[i:i + sustain]):
            r = readings[i]
            # Already fatigued at the very first reading. These protocols start
            # fresh, so this is not an onset at t=0 -- it means the transition
            # happened before anything was measured, or the classifier is wrong
            # early. Subject 12 does this: reporting "fatigue set in at 0 s"
            # against a ground truth of 102 s was the single worst error in the
            # validation, and stating it as a time is what made it wrong.
            if i == 0:
                return {**base, "found": False, "fatigued_from_start": True}
            return {**base, "found": True, "t_start": r["t_start"],
                    "confidence": r["confidence"], "mdf_hz": r["mdf_hz"],
                    "fatigued_from_start": False}

    return {**base, "found": False, "fatigued_from_start": False}


def summarise(scan: dict) -> dict:
    """How one recording develops from start to finish."""
    readings = scan["readings"]
    mdf = [r["mdf_hz"] for r in readings]
    flags = [r["fatigue_label"] in FATIGUE_LABELS for r in readings]
    onset = find_onset(scan)

    return {
        "subject": scan["subject"], "side": scan["side"],
        "duration": scan["duration"], "step": scan["step"],
        "n_readings": len(readings),
        "mdf_start": mdf[0] if mdf else None,
        "mdf_end": mdf[-1] if mdf else None,
        "mdf_drop": (mdf[0] - mdf[-1]) if len(mdf) >= 2 else None,
        "mdf_min": min(mdf) if mdf else None,
        "mdf_max": max(mdf) if mdf else None,
        "fraction_fatigued": sum(flags) / len(flags) if flags else 0.0,
        "onset": onset,
        "readings": readings,
    }


def compare_sides(subject: int, t_start: float | None = None) -> dict:
    """The same subject's two arms, at the same moment in each recording.

    t_start=None reads the end of the shorter recording, so the two are
    compared at a moment that exists in both.
    """
    out = {}
    segs = {}
    for side in ("R", "L"):
        seg, fs = load_subject_segment(subject, side)
        segs[side] = (seg, fs)

    shortest = min(_duration(seg) for seg, _ in segs.values())
    when = shortest if t_start is None else min(float(t_start), shortest)

    for side, (seg, fs) in segs.items():
        out[side] = classify(subject, when, side, seg=seg, fs=fs)
        out[side]["t_start"] = when
    return {"kind": "sides", "subject": subject, "t_start": when,
            "duration": shortest, "results": out}


# These are fatigue-to-exhaustion protocols and the recordings are nowhere
# near the same length -- 24.5 s (subject 6, right) to 491 s (subject 11,
# right). So "read every subject at the same absolute second" is the wrong
# comparison: pinning it to the shortest recording compares one subject at
# 100% of their effort against another at 5% of theirs, and reads everyone as
# unfatigued. Comparing at the same *fraction* of each person's own recording
# puts them at a comparable point in their own effort instead.
LATE_FRACTION = 0.9        # "late in the effort"
EARLY_FRACTION = 0.05      # "early, still fresh"
MIN_MEANINGFUL_SEC = 60.0  # below this a recording can't show a fatigue arc


def _load_all(subjects: list[int], side: str) -> tuple[dict, dict]:
    loaded, durations = {}, {}
    for s in subjects:
        seg, fs = load_subject_segment(s, side)
        loaded[s] = (seg, fs)
        durations[s] = _duration(seg)
    return loaded, durations


def compare_subjects(subjects: list[int], t_start: float | None = None,
                     side: str = "R", fraction: float = LATE_FRACTION) -> dict:
    """Several subjects read at a comparable point in each recording.

    With `t_start` given, every subject is read at that absolute second
    (clamped to their own recording, which is noted). Without one, each is
    read at the same fraction of their own recording -- see LATE_FRACTION.
    """
    loaded, durations = _load_all(subjects, side)
    results, clamped = {}, []

    for s, (seg, fs) in loaded.items():
        if t_start is None:
            when = durations[s] * fraction
        else:
            when = min(float(t_start), durations[s])
            if when < float(t_start):
                clamped.append(s)
        r = classify(s, when, side, seg=seg, fs=fs)
        r["t_start"] = when
        r["duration"] = durations[s]
        # Each subject's own fresh level, because absolute median frequency is
        # NOT comparable between people -- fresh subjects here span 59 to 81 Hz,
        # so "5 is at 70 Hz and 9 is at 61 Hz" says nothing about who is more
        # fatigued. Only the distance from each person's own fresh state does.
        try:
            ref = subject_reference(s)
            r["fresh_mdf"] = ref["fresh_mdf"]
            r["drop_percent"] = ((ref["fresh_mdf"] - r["mdf_hz"])
                                 / ref["fresh_mdf"] * 100.0) if ref["fresh_mdf"] else None
        except Exception:
            r["fresh_mdf"] = r["drop_percent"] = None
        results[s] = r

    return {"kind": "subjects", "subjects": subjects, "side": side,
            "t_start": t_start, "fraction": None if t_start else fraction,
            "durations": durations, "clamped": clamped,
            "short": [s for s in subjects if durations[s] < MIN_MEANINGFUL_SEC],
            "results": results}


def rank_subjects(subjects: list[int], side: str = "R") -> dict:
    """Order subjects by how much their fatigue marker fell over the effort.

    Ranked on the drop in MDF between an early and a late reading, rather than
    on the classifier's confidence at one moment. MDF drop is the physical
    fatigue signal the whole project is built on, and a confidence score is
    not a severity score -- the calibration work showed confidence barely
    moves even when the label is wrong, so ordering by it would be ranking
    noise.

    Two readings per subject rather than a full scan each, to keep this fast
    enough to sit behind a chat message.
    """
    loaded, durations = _load_all(subjects, side)
    results = {}

    for s, (seg, fs) in loaded.items():
        if durations[s] < MIN_MEANINGFUL_SEC:
            continue          # too short to show an arc; excluded, and said so
        early = classify(s, durations[s] * EARLY_FRACTION, side, seg=seg, fs=fs)
        late = classify(s, durations[s] * LATE_FRACTION, side, seg=seg, fs=fs)
        results[s] = {
            "mdf_early": early["mdf_hz"], "mdf_late": late["mdf_hz"],
            "mdf_drop": early["mdf_hz"] - late["mdf_hz"],
            "fatigue_label": late["fatigue_label"],
            "confidence": late["confidence"],
            "duration": durations[s],
            "t_early": durations[s] * EARLY_FRACTION,
            "t_late": durations[s] * LATE_FRACTION,
        }

    ranked = sorted(results, key=lambda s: -results[s]["mdf_drop"])
    return {"kind": "ranking", "side": side, "ranked": ranked,
            "results": results, "durations": durations,
            "excluded": [s for s in subjects if s not in results],
            "early_fraction": EARLY_FRACTION, "late_fraction": LATE_FRACTION}


# ---------------------------------------------------------------------------
# definitions
# ---------------------------------------------------------------------------
# Written against what this project actually does, not general textbook
# phrasing -- including the honest caveat on confidence, which the calibration
# work showed does NOT move when the model is wrong.
DEFINITIONS = {
    "median frequency": (
        "Median frequency (MDF) is the frequency that splits an EMG window's "
        "power in half -- half the signal's power sits below it, half above. "
        "It is this project's fatigue marker: as a muscle fatigues, the "
        "conduction velocity of its fibres falls and the signal's power "
        "shifts towards lower frequencies, so MDF drops. It is measured in "
        "Hz, and in this dataset subjects sit anywhere from about 53 to 78 Hz "
        "when fresh, which is why the model is given the change in MDF rather "
        "than its absolute value."),
    "fatigue": (
        "Fatigue here is a label the classifier assigns to a 2-second window "
        "of EMG signal: fatigued or not fatigued. It is inferred from how the "
        "window's features compare with the same person's fresh baseline, not "
        "from anything the person reported feeling."),
    "confidence": (
        "Confidence is the model's own probability for the label it chose "
        "(the softmax score), so 86% means the network put 86% of its "
        "probability on that answer. Worth being straight about what it is "
        "not: it is not accuracy, and it does not reliably drop when the "
        "model is wrong. Measured on the labelled subjects, confidence stayed "
        "essentially unchanged between correct and incorrect readings."),
    "EMG": (
        "EMG (electromyography) records the electrical activity muscle fibres "
        "produce when they contract. This project uses surface EMG from the "
        "biceps, band-passed to 20-450 Hz to keep the muscle band and drop "
        "movement artefacts and mains noise."),
    "the forecast": (
        "The forecast projects the fatigue marker (MDF) forward from the "
        "moment being asked about. It uses an LSTM trained to predict the "
        "*change* in MDF over a horizon, validated leave-one-subject-out. "
        "Beyond 60 seconds it holds the projection flat, because nothing "
        "tested beat simply assuming no further change at that range."),
    "the baseline": (
        "The baseline is a per-person reference taken from their fresh, "
        "unfatigued windows -- a mean and spread for each feature, used to "
        "express later windows as how far they have moved from that person's "
        "own fresh state. The 13 dataset subjects have a stored baseline. An "
        "uploaded recording has none, so one is computed from its own first "
        "60 seconds, which is measurably less accurate."),
}


def define(term: str) -> str | None:
    return DEFINITIONS.get(term)
