"""
Loader: Zenodo sEMG biceps dataset -> core.Segment objects.
============================================================

Replaces the broken OpenBCI capture with the open biceps-brachii sEMG set
(Zenodo 10.5281/zenodo.14182446, Sensors 2024, CC BY 4.0). It turns one trial
CSV into the SAME `core.Segment` object the convergence/forecast pipeline
already consumes, so `core.collect_session_spectra`, `core.spectrum_backtest`
and `core.spectrum_forward` run UNCHANGED on this data.

Dataset facts (from the authors' notebook + protocol.xlsx):
    - 13 subjects, sEMG_data/Subject_{1..13}/, one CSV per trial.
    - Each CSV holds 4 muscles of one arm, columns interleaved:
        time cols = [0,2,4,6], emg cols = [1,3,5,7]   (each Delsys sensor
        carries its OWN time column).
    - Trial 5 = R BICEPS BRACHII, trial 6 = L BICEPS BRACHII (the dedicated
        biceps-curl fatigue trials). get_prime_mover(trial) selects which of
        the 4 pairs is the fatigued muscle.
    - Raw sEMG sampled at 1259 Hz, signal in Volts.
    - Authors' MDF recipe: 4th-order Butterworth 20-450 Hz bandpass, then a
        4 s window / 2 s step sliding median frequency (30 bpm cadence -> 2 s
        cycle, so a 4 s window spans two cycles).

The fs trap (why we never set core.FS = 1259):
    core's `median_frequency`, `channel_spectrum`, `collect_session_spectra`,
    `spectrum_*` all bound `fs=FS` (=250) at def time. Reassigning the module
    global does NOT rebind those defaults; it WOULD, however, change the window
    length that `core.mdf_trend` reads from the global while its inner
    `median_frequency` call stays at 250 -> MDF silently ~5x off (the same
    family as the old "pipeline.py freq 4x too high" bug). So we pass fs=1259
    explicitly to every core call, and ship an fs-aware mdf_trend here.
"""
from __future__ import annotations

import os
import re
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# make `core` importable regardless of CWD (it lives in convergence_analysis/)
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "convergence_analysis"))
import core  # noqa: E402  (core.Segment, bandpass_filter, channel_spectrum, ...)

FS_NATIVE = 1259  # Hz, the dataset's true sample rate
EMG_INDEX = [1, 3, 5, 7]   # column of each muscle's signal
TIME_INDEX = [0, 2, 4, 6]  # column of each muscle's own time vector
_EPOCH = datetime(2020, 1, 1)  # arbitrary fixed wall-clock base (no Date.now)

# trial number -> agonist muscle (verbatim from the authors' notebook)
TRIAL_TO_MUSCLE = {
    1: "R DELTOID ANTERIOR", 2: "L DELTOID ANTERIOR",
    3: "R DELTOID POSTERIOR", 4: "L DELTOID POSTERIOR",
    5: "R BICEPS BRACHII", 6: "L BICEPS BRACHII",
    7: "R DELTOID MEDIUS", 8: "L DELTOID MEDIUS",
    9: "R DELTOID ANTERIOR C", 10: "L DELTOID ANTERIOR C",
    11: "R DELTOID POSTERIOR C", 12: "L DELTOID POSTERIOR C",
}
BICEPS_TRIAL = {"R": 5, "L": 6}  # the two uni-articular biceps-curl trials


def get_prime_mover(trial: int) -> int:
    """Which of the 4 (time,emg) pairs in a trial CSV is the agonist muscle.

    Verbatim from the authors' notebook. e.g. trial 5 -> 0 (R biceps lives in
    cols time[0]/emg[1]); trial 6 -> 1 (L biceps in cols time[2]/emg[3]).
    """
    mapping = {1: 1, 2: 0, 3: 3, 4: 2, 5: 0, 6: 1,
               7: 2, 8: 3, 9: 1, 10: 0, 11: 3, 12: 2}
    return mapping[trial]


def _trial_of(filename: str) -> int:
    """Trial number = the LAST integer in the filename (authors' convention)."""
    return int(re.findall(r"\d+", filename)[-1])


def _clean_pair(time_col: np.ndarray, emg_col: np.ndarray):
    """Drop NaN/inf padding and return strictly-increasing (t, x) in seconds.

    The authors pad short trials with NaN and zero out inf/NaN samples. We drop
    any row whose time or signal is non-finite, then de-duplicate / sort the
    time base so np.interp onto a uniform grid is well-defined.
    """
    t = np.asarray(time_col, dtype=float)
    x = np.asarray(emg_col, dtype=float)
    ok = np.isfinite(t) & np.isfinite(x)
    t, x = t[ok], x[ok]
    order = np.argsort(t, kind="stable")
    t, x = t[order], x[order]
    keep = np.concatenate([[True], np.diff(t) > 0])  # strictly increasing time
    return t[keep], x[keep]


def load_trial_channel(csv_path: str, trial: int | None = None,
                       pair_index: int | None = None):
    """Read one trial CSV and return (t_native, x_native) for one muscle.

    Pass `trial` to auto-select that trial's agonist (the fatigued muscle), or
    `pair_index` (0..3) to pull a specific one of the 4 muscles in the file.
    Times are the sensor's own clock in seconds; signal is raw, unfiltered V.
    """
    if pair_index is None:
        if trial is None:
            trial = _trial_of(os.path.basename(csv_path))
        pair_index = get_prime_mover(trial)
    df = pd.read_csv(csv_path, delimiter=",", header=0)
    vals = df.values
    t = vals[:, TIME_INDEX[pair_index]]
    x = vals[:, EMG_INDEX[pair_index]]
    return _clean_pair(t, x)


def to_segment(t_native: np.ndarray, x_native: np.ndarray,
               fs: int = FS_NATIVE, target_fs: int | None = None,
               bandpass: bool = True, lo_hz: float = 20.0,
               hi_hz: float = 450.0, idx: int = 1) -> "core.Segment":
    """Resample one channel onto a uniform grid and wrap as a core.Segment.

    fs        : the data's native rate (1259 Hz for this set).
    target_fs : if set and != fs, the channel is anti-alias resampled to this
                rate (e.g. 250 to mimic the OpenBCI rig for comparability).
                The returned Segment then carries `eff_fs == target_fs`.
    bandpass  : apply the authors' 4th-order 20-450 Hz Butterworth (clamped
                below the working Nyquist when downsampled).
    Returns a (n, 1) single-channel Segment; gap_mask is all-False (the Delsys
    capture has no dropout rails, unlike the OpenBCI export).
    """
    eff_fs = fs
    # uniform grid at native fs using the sensor's own time base
    t_uni = np.arange(t_native[0], t_native[-1] + 0.5 / fs, 1.0 / fs)
    x_uni = np.interp(t_uni, t_native, x_native)

    if target_fs is not None and target_fs != fs:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(int(round(target_fs)), int(round(fs)))
        x_uni = resample_poly(x_uni, int(round(target_fs)) // g, int(round(fs)) // g)
        eff_fs = int(round(target_fs))
        t_uni = np.arange(x_uni.size) / eff_fs

    if bandpass:
        nyq = eff_fs / 2.0
        hi = min(hi_hz, nyq * 0.99) if hi_hz is not None else None
        x_uni = core.bandpass_filter(x_uni.reshape(-1, 1), eff_fs, lo_hz, hi)[:, 0]

    data = x_uni.reshape(-1, 1).astype(float)
    t_rel = np.arange(data.shape[0]) / eff_fs
    seg = core.Segment(
        idx=idx, data=data, t=t_rel,
        gap_mask=np.zeros(data.shape[0], dtype=bool),
        wall_start=_EPOCH, n_raw=int(t_native.size),
    )
    seg.eff_fs = eff_fs  # attach so the runner never re-guesses the rate
    return seg


def find_biceps_csv(subject_dir: str, side: str = "R") -> str:
    """Path to the biceps trial CSV (trial 5 for R, 6 for L) in a Subject dir."""
    want = BICEPS_TRIAL[side.upper()]
    for f in sorted(os.listdir(subject_dir)):
        if f.endswith(".csv") and _trial_of(f) == want:
            return os.path.join(subject_dir, f)
    raise FileNotFoundError(f"no trial {want} csv in {subject_dir}")


def load_biceps_segment(root: str, subject: int, side: str = "R",
                        fs: int = FS_NATIVE, target_fs: int | None = None,
                        bandpass: bool = True) -> "core.Segment":
    """One-call: subject number + side -> biceps Segment ready for core funcs."""
    sub_dir = os.path.join(root, f"subject_{subject}")
    csv = find_biceps_csv(sub_dir, side)
    trial = BICEPS_TRIAL[side.upper()]
    t, x = load_trial_channel(csv, trial=trial)
    return to_segment(t, x, fs=fs, target_fs=target_fs, bandpass=bandpass)


# ---------------------------------------------------------------------------
# Fatigue ground-truth labels (0 = non-fatigue, 1 = transition, 2 = fatigue)
# ---------------------------------------------------------------------------
def load_fatigue_labels(root: str, subject: int, side: str = "R"):
    """Return (label_time, label_value) arrays for the matching biceps trial.

    Labels live in self_perceived_fatigue_index/subject_{n}/ alongside the
    sEMG tree. Returns (None, None) if that folder is not present.
    """
    want = BICEPS_TRIAL[side.upper()]
    base = os.path.dirname(root.rstrip("/"))
    lab_dir = os.path.join(base, "self_perceived_fatigue_index", f"subject_{subject}")
    if not os.path.isdir(lab_dir):
        return None, None
    for f in sorted(os.listdir(lab_dir)):
        if f.endswith(".csv") and _trial_of(f) == want:
            df = pd.read_csv(os.path.join(lab_dir, f), delimiter=",", header=0)
            return df.iloc[:, 0].to_numpy(float), df.iloc[:, 1].to_numpy(float)
    return None, None


def fatigue_onsets(label_time, label_value):
    """(transition_t, fatigue_t): last-0 time and last-1 time, per the authors.

    These are the two dashed vertical markers in the paper's MDF figures:
    end of the non-fatigue (label 0) span, and end of the transition (label 1)
    span. Returns (None, None) if labels are absent.
    """
    if label_time is None:
        return None, None
    t = np.asarray(label_time, float)
    v = np.asarray(label_value, float)
    trans = float(t[np.where(v == 0)[0][-1]]) if (v == 0).any() else None
    fatig = float(t[np.where(v == 1)[0][-1]]) if (v == 1).any() else None
    return trans, fatig


# ---------------------------------------------------------------------------
# fs-aware MDF trend (core.mdf_trend hardcodes FS=250 and N_CH=8; do not use it)
# ---------------------------------------------------------------------------
def mdf_trend(seg: "core.Segment", fs: int | None = None,
              win_sec: float = 4.0, step_sec: float = 2.0):
    """Per-window median frequency over time for every channel in `seg`.

    fs-correct re-implementation of core.mdf_trend: window length uses the
    Segment's effective rate, and median_frequency is called WITH that same fs,
    so the two never disagree. Defaults match the dataset protocol (4 s / 2 s).
    Returns (t_centers, mean_mdf, mdf[W, k]).
    """
    if fs is None:
        fs = int(getattr(seg, "eff_fs", FS_NATIVE))
    win = max(2, int(round(win_sec * fs)))
    step = max(1, int(round(step_sec * fs)))
    k = seg.data.shape[1]
    t_centers, per_ch = [], []
    start = 0
    while start + win <= seg.data.shape[0]:
        if not seg.gap_mask[start:start + win].any():
            w = seg.data[start:start + win]
            t_centers.append(seg.t[start] + win_sec / 2.0)
            per_ch.append([core.median_frequency(w[:, c], fs=fs) for c in range(k)])
        start += step
    t_centers = np.array(t_centers)
    per_ch = np.array(per_ch) if per_ch else np.zeros((0, k))
    mean_mdf = per_ch.mean(axis=1) if per_ch.size else np.array([])
    return t_centers, mean_mdf, per_ch
