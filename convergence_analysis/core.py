"""
Core data + maths for 8-channel EMG convergence analysis.
=========================================================

Pure computation: load the OpenBCI export, cut it into the *clean* segments
(the "certain period" the supervisor asked us to stay inside), and score how
SIMILAR the 8 channels look at each moment. Everything downstream operates only
on these clean segments -- nothing is ever computed on dropout regions.

What "similar / converged" means (decided with supervisor + Ray):
    The supervisor watched the OpenBCI GUI, where every channel sits in its own
    auto-scaled lane, and said the channels "look similar". That is SHAPE
    similarity, not raw-amplitude overlap (impossible here: the channels carry
    DC offsets of -100,000 to +90,000 uV). So convergence in a short window is
    the mean of the 28 pairwise Pearson correlations across the 8 channels.
    Pearson centres and scales each channel internally, so the DC offset is
    removed automatically -- this is NOT noise filtering. No notch/bandpass is
    applied anywhere.

Sample rate -- resolved from the data, not the header:
    The file header says "Sample Rate = 1000 Hz", and the old pipeline.py
    trusted that. It is WRONG. The Cyton "Sample Index" column is the board's
    own per-sample counter that wraps 0-255, so its wrap rate is physical ground
    truth, independent of the bursty WiFi arrival timestamps. Three methods all
    agree on 250 Hz:
        - wrap count      : 534 wraps x 256 / 547.5 s = 249.6 Hz
        - index-advance   : 136,900 wrap-corrected samples / 547.5 s = 250.0 Hz
        - packet loss      : received 121,793 / true 136,900 = 89% (~11% lost)
    So FS = 250. The wall-clock arrival timestamps are delivered in WiFi bursts
    (many rows share one millisecond) and are useless for sample spacing; we use
    the wrap-corrected Sample Index as the clock instead.

Packet loss handling:
    ~11% of samples are lost, almost all as ISOLATED single-sample gaps (the
    index steps by +2). Interpolating one missing sample inside a 250-sample
    window is standard EMG practice and negligible. Larger holes are rare; any
    analysis window that spans a hole bigger than MAX_GAP_FILL samples is
    skipped, so interpolation can never fabricate a correlation.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy import stats
from scipy.signal import butter, filtfilt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FS = 250                       # Hz, proven from the Sample Index wrap rate
N_CH = 8
DROPOUT_TH = -180000           # uV; at/below this a sample is a saturation rail
MIN_SEG_SEC = 1.0              # ignore clean runs shorter than this
MAX_GAP_FILL = 2               # missing samples we silently interpolate across;
#                                a window spanning a bigger hole is skipped

WIN_SEC = 1.0                  # convergence / FFT window length
STEP_SEC = 0.1                 # how far the window hops each score sample
CONV_THRESHOLD = 0.8           # mean-pairwise-corr above this = "converged".
#                                0.8 = "clearly similar". Lower to 0.6 for a
#                                looser definition; one constant, used everywhere.
MOMENT_MIN_SEP_SEC = 1.5       # min spacing between reported convergence moments
MIN_RUN_SEC = 1.0              # a converged *run* must last at least this long

TS_FMT = "%Y-%m-%d %H:%M:%S.%f"

CHANNEL_COLORS = [
    "#f5a623", "#4ec9ec", "#3ddc97", "#a78bfa",
    "#ff6b9d", "#ffd93d", "#5eead4", "#fb7185",
]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class Segment:
    """One clean stretch of recording, resampled onto a uniform 250 Hz grid.

    data      : (n, 8) uV on the uniform grid
    t         : (n,) seconds from segment start, exact 1/FS spacing
    gap_mask  : (n,) True where this uniform sample sits inside a hole larger
                than MAX_GAP_FILL (i.e. interpolated across a real gap) -- such
                samples are excluded from any analysis window
    wall_start: wall-clock datetime of the first sample (file_t0 + index/FS)
    n_raw     : received rows that formed this segment (before resampling)
    """
    idx: int
    data: np.ndarray
    t: np.ndarray
    gap_mask: np.ndarray
    wall_start: datetime
    n_raw: int

    @property
    def n(self) -> int:
        return self.data.shape[0]

    @property
    def dur(self) -> float:
        return self.n / FS

    def wall_at(self, t_rel: float) -> str:
        """HH:MM:SS.cc wall-clock string at t_rel seconds into the segment."""
        return (self.wall_start + timedelta(seconds=t_rel)).strftime(
            "%H:%M:%S.%f")[:-4]

    def label(self) -> str:
        a = self.wall_start.strftime("%H:%M:%S")
        b = (self.wall_start + timedelta(seconds=self.dur)).strftime("%H:%M:%S")
        return f"Seg {self.idx}  {a} -> {b}  ({self.dur:.1f}s)"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _parse_raw(path: str):
    """Read the OpenBCI .txt export.

    Returns (sample_idx[N], channels[N, 8], file_t0) where sample_idx is the
    raw 0-255 counter and file_t0 is the first formatted timestamp.
    """
    sidx: list[float] = []
    rows: list[list[float]] = []
    t0: datetime | None = None
    with open(path) as f:
        for line in f:
            if line.startswith("%") or line.startswith("Sample"):
                continue
            parts = line.strip().split(",")
            if len(parts) < 25:
                continue
            try:
                si = float(parts[0])
                vals = [float(parts[i + 1]) for i in range(N_CH)]
                ts = datetime.strptime(parts[-1].strip(), TS_FMT)
            except (ValueError, IndexError):
                continue
            if t0 is None:
                t0 = ts
            sidx.append(si)
            rows.append(vals)
    if not rows:
        raise ValueError(f"No data rows parsed from {path}")
    return np.array(sidx), np.array(rows), t0


def _wrap_corrected(sidx: np.ndarray) -> np.ndarray:
    """Cumulative true sample number from the 0-255 wrapping Sample Index."""
    step = np.diff(sidx)
    step = np.where(step < 0, step + 256, step)      # undo the 0-255 wrap
    return np.concatenate([[0.0], np.cumsum(step)])


def _rail_runs(rail_ok: np.ndarray):
    """Index ranges [s, e) of consecutive received rows that are all rail-clean."""
    runs = []
    s = None
    for i, ok in enumerate(rail_ok):
        if ok and s is None:
            s = i
        elif not ok and s is not None:
            runs.append((s, i)); s = None
    if s is not None:
        runs.append((s, len(rail_ok)))
    return runs


def load_clean_segments(path: str) -> list[Segment]:
    """Load the export and return clean segments resampled to a uniform 250 Hz.

    A segment is a maximal run of consecutive received rows where all 8 channels
    are above the saturation rail and which lasts at least MIN_SEG_SEC. Each
    segment is resampled onto an exact 1/FS grid using the wrap-corrected Sample
    Index as the clock; isolated lost samples are interpolated, and any uniform
    sample that falls inside a hole larger than MAX_GAP_FILL is flagged in
    gap_mask so analysis windows can skip it.
    """
    sidx, chans, file_t0 = _parse_raw(path)
    samp_no = _wrap_corrected(sidx)
    rail_ok = np.all(chans > DROPOUT_TH, axis=1)

    segments: list[Segment] = []
    seg_i = 0
    for s, e in _rail_runs(rail_ok):
        sn = samp_no[s:e] - samp_no[s]            # true sample offset in segment
        dur = sn[-1] / FS
        if dur < MIN_SEG_SEC:
            continue
        seg_i += 1

        t_recv = sn / FS                          # true time of each received row
        t_uni = np.arange(0.0, t_recv[-1] + 0.5 / FS, 1.0 / FS)
        data_uni = np.empty((t_uni.size, N_CH))
        for c in range(N_CH):
            data_uni[:, c] = np.interp(t_uni, t_recv, chans[s:e, c])

        # mark uniform samples that fall inside a hole > MAX_GAP_FILL samples
        gap_mask = np.zeros(t_uni.size, dtype=bool)
        miss = np.diff(sn).astype(int) - 1        # missing samples between rows
        for k in np.where(miss > MAX_GAP_FILL)[0]:
            lo, hi = t_recv[k], t_recv[k + 1]
            gap_mask |= (t_uni > lo) & (t_uni < hi)

        wall_start = file_t0 + timedelta(seconds=samp_no[s] / FS)
        segments.append(Segment(
            idx=seg_i, data=data_uni, t=t_uni, gap_mask=gap_mask,
            wall_start=wall_start, n_raw=e - s,
        ))
    return segments


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
def bandpass_filter(data: np.ndarray, fs: int = FS,
                    lo_hz: float = 20.0, hi_hz: float = None) -> np.ndarray:
    """Zero-phase 4th-order Butterworth filter applied column-wise.

    lo_hz: highpass cutoff in Hz (None = no highpass)
    hi_hz: lowpass cutoff in Hz (None = no lowpass, i.e. passes up to Nyquist)
    Returns a new array; never mutates the input.
    """
    nyq = fs / 2.0
    if lo_hz is not None and hi_hz is not None:
        Wn = [lo_hz / nyq, hi_hz / nyq]
        btype = "bandpass"
    elif lo_hz is not None:
        Wn = lo_hz / nyq
        btype = "highpass"
    elif hi_hz is not None:
        Wn = hi_hz / nyq
        btype = "lowpass"
    else:
        return data.copy()
    b, a = butter(4, Wn, btype=btype)
    return filtfilt(b, a, data, axis=0)


def filter_segments(segments: list, lo_hz: float = 20.0,
                    hi_hz: float = None) -> list:
    """Return new Segment list with filtered data. Original segments unchanged."""
    return [replace(seg, data=bandpass_filter(seg.data, FS, lo_hz, hi_hz))
            for seg in segments]


# ---------------------------------------------------------------------------
# Convergence maths
# ---------------------------------------------------------------------------
def mean_pairwise_corr(window: np.ndarray) -> float:
    """Mean of the 28 pairwise Pearson correlations across 8 channels.

    window: (n_samples, 8). Flat (zero-variance) channels are dropped before
    correlating; fewer than 2 usable channels -> 0. Result in [-1, 1]; ~1 means
    all channels move together (converged).
    """
    stds = window.std(axis=0)
    keep = stds > 1e-9
    if keep.sum() < 2:
        return 0.0
    c = np.corrcoef(window[:, keep], rowvar=False)
    iu = np.triu_indices_from(c, k=1)
    vals = c[iu]
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if vals.size else 0.0


def cross_channel_dispersion(window: np.ndarray) -> float:
    """Mean over time of the cross-channel std after per-channel centring.

    Lower = the 8 traces sit closer together once their DC offset is removed.
    Reported alongside correlation; not used for detection.
    """
    centred = window - window.mean(axis=0, keepdims=True)
    return float(centred.std(axis=1).mean())


def _window_has_gap(seg: "Segment", start: int, win: int) -> bool:
    return bool(seg.gap_mask[start:start + win].any())


def scan_segment(seg: Segment, win_sec: float = WIN_SEC,
                 step_sec: float = STEP_SEC):
    """Slide a window across a segment -> (t_centers, corr_scores, dispersions).

    Windows that span a real data hole are skipped entirely (not scored).
    """
    win = max(2, int(round(win_sec * FS)))
    step = max(1, int(round(step_sec * FS)))
    t_centers, corr, disp = [], [], []
    start = 0
    while start + win <= seg.n:
        if not _window_has_gap(seg, start, win):
            w = seg.data[start:start + win]
            t_centers.append(seg.t[start] + win_sec / 2.0)
            corr.append(mean_pairwise_corr(w))
            disp.append(cross_channel_dispersion(w))
        start += step
    return np.array(t_centers), np.array(corr), np.array(disp)


def find_moments(t: np.ndarray, scores: np.ndarray,
                 threshold: float = CONV_THRESHOLD,
                 min_sep_sec: float = MOMENT_MIN_SEP_SEC):
    """Convergence moments above threshold, greedily spaced apart by score.

    Returns [(t_center, score), ...] sorted by score descending, with no two
    moments closer than min_sep_sec (non-maximum suppression).
    """
    if t.size == 0:
        return []
    order = np.argsort(scores)[::-1]
    chosen: list[tuple[float, float]] = []
    for i in order:
        if scores[i] < threshold:
            break
        ti = t[i]
        if all(abs(ti - tc) >= min_sep_sec for tc, _ in chosen):
            chosen.append((float(ti), float(scores[i])))
    return chosen


@dataclass
class ConvergedRun:
    """A contiguous stretch where the 8 channels stay similar.

    seg_idx   : which clean segment it lives in
    t_start   : seconds into the segment where the converged stretch begins
    t_end     : seconds into the segment where it ends
    mean_score: average convergence score over the run
    peak_score: best convergence score in the run
    """
    seg_idx: int
    t_start: float
    t_end: float
    mean_score: float
    peak_score: float

    @property
    def dur(self) -> float:
        return self.t_end - self.t_start


def converged_runs(seg: Segment, threshold: float = CONV_THRESHOLD,
                   min_run_sec: float = MIN_RUN_SEC,
                   win_sec: float = WIN_SEC, step_sec: float = STEP_SEC):
    """Contiguous runs in a segment where the score stays >= threshold.

    Walks the windowed convergence scores in time order and groups consecutive
    windows that (a) clear the threshold and (b) are temporally adjacent (no gap
    bigger than ~1.5 steps, so a skipped-over data hole always breaks a run).
    Each run is widened by half a window at each end (the score is at the window
    centre) and clipped to the segment. Runs shorter than min_run_sec are
    dropped. Returns a list of ConvergedRun, sorted by start time.
    """
    t, corr, _disp = scan_segment(seg, win_sec, step_sec)
    if t.size == 0:
        return []
    half = win_sec / 2.0
    max_step = step_sec * 1.5
    runs: list[ConvergedRun] = []
    i = 0
    n = t.size
    while i < n:
        if corr[i] < threshold:
            i += 1
            continue
        j = i
        while (j + 1 < n and corr[j + 1] >= threshold
               and (t[j + 1] - t[j]) <= max_step):
            j += 1
        t0 = max(0.0, t[i] - half)
        t1 = min(seg.dur, t[j] + half)
        seg_scores = corr[i:j + 1]
        if t1 - t0 >= min_run_sec:
            runs.append(ConvergedRun(
                seg_idx=seg.idx, t_start=float(t0), t_end=float(t1),
                mean_score=float(seg_scores.mean()),
                peak_score=float(seg_scores.max()),
            ))
        i = j + 1
    return runs


def all_converged_runs(segments, threshold: float = CONV_THRESHOLD,
                       min_run_sec: float = MIN_RUN_SEC):
    """Converged runs across every segment, paired with their Segment.

    Returns [(Segment, ConvergedRun), ...] in chronological order.
    """
    out = []
    for seg in segments:
        for run in converged_runs(seg, threshold, min_run_sec):
            out.append((seg, run))
    return out


# ---------------------------------------------------------------------------
# Frequency maths
# ---------------------------------------------------------------------------
def channel_spectrum(window: np.ndarray, fs: int = FS):
    """One-sided amplitude spectrum per channel. Returns (freqs, mags[8, F])."""
    centred = window - window.mean(axis=0, keepdims=True)
    mags = np.abs(rfft(centred, axis=0)).T          # (8, F)
    freqs = rfftfreq(window.shape[0], 1.0 / fs)
    return freqs, mags


def _smooth_bins(v: np.ndarray, smooth_bins: int = 3) -> np.ndarray:
    """Light moving-average across frequency bins (cosmetic only)."""
    if smooth_bins > 1 and v.size >= smooth_bins:
        k = np.ones(smooth_bins) / smooth_bins
        return np.convolve(v, k, mode="same")
    return v


def collect_session_spectra(segments, win_sec: float = WIN_SEC,
                            step_sec: float = STEP_SEC, fs: int = FS):
    """Channel-averaged amplitude spectrum of every clean window, with its time.

    Slides a 1 s window across every clean segment (skipping data holes) and
    returns (freqs, specs[W, F], times[W]) where times are seconds from the first
    clean sample on one session timeline. Shared by the forecast/backtest.
    """
    win = max(2, int(round(win_sec * fs)))
    step = max(1, int(round(step_sec * fs)))
    freqs = rfftfreq(win, 1.0 / fs)
    if not segments:
        return freqs, np.zeros((0, freqs.size)), np.zeros(0)
    t0 = segments[0].wall_start
    specs, times = [], []
    for seg in segments:
        offset = (seg.wall_start - t0).total_seconds()
        start = 0
        while start + win <= seg.n:
            if not _window_has_gap(seg, start, win):
                _f, mags = channel_spectrum(seg.data[start:start + win], fs)
                specs.append(mags.mean(axis=0))          # channel-averaged (F,)
                times.append(offset + (start + win / 2.0) / fs)
            start += step
    return freqs, np.array(specs), np.array(times)


def spectrum_forward(segments, recent_frac: float = 0.30, horizon_sec: float = 60.0,
                     win_sec: float = WIN_SEC, step_sec: float = STEP_SEC,
                     fs: int = FS, smooth_bins: int = 3):
    """Forward prediction of the future frequency SHAPE from the actual signal.

    The literal "predict the future from the actual measured frequency" view:

      1. Collect each clean window's channel-averaged spectrum, normalise to unit
         area -> spectral shape (share of power per frequency), independent of
         loudness.
      2. "actual now" = mean shape over the most recent `recent_frac` of the
         recording (the current frequency state).
      3. Fit each bin's shape vs time over that recent portion and project it
         `horizon_sec` past the end of the data, renormalised to a shape.

    Returns (freqs, actual_now, predicted_future, meta) as PERCENT-of-total per
    bin. The projection lies BEYOND the data so it cannot be checked here - pair
    it with spectrum_backtest(), whose match score quantifies how trustworthy
    this kind of projection is. meta carries the MDF of now vs predicted.
    """
    freqs, specs, times = collect_session_spectra(segments, win_sec, step_sec, fs)
    meta = {"n_recent": 0}
    if specs.shape[0] < 3:
        ref = specs.mean(axis=0) if specs.shape[0] else np.zeros(freqs.size)
        ref = ref / (ref.sum() or 1.0)
        return freqs, _smooth_bins(ref) * 100.0, np.array([]), meta

    areas = specs.sum(axis=1, keepdims=True)
    areas[areas == 0] = 1.0
    shapes = specs / areas
    order = np.argsort(times)
    times, shapes = times[order], shapes[order]

    cut = times[0] + (1.0 - recent_frac) * (times[-1] - times[0])
    recent = times >= cut
    if recent.sum() < 3:
        recent = np.ones(times.size, dtype=bool)        # fall back to all data
    tr, hr = times[recent], shapes[recent]
    actual_now = hr.mean(axis=0)

    slope, intercept = np.polyfit(tr, hr, 1)            # per-bin (F,)
    t_future = tr[-1] + horizon_sec
    predicted = np.clip(slope * t_future + intercept, 0.0, None)
    predicted = predicted / (predicted.sum() or 1.0)

    def _mdf_of(shape):
        c = np.cumsum(shape)
        return float(freqs[np.searchsorted(c, c[-1] / 2.0)]) if c[-1] > 0 else 0.0

    meta = {"n_recent": int(recent.sum()), "horizon_s": float(horizon_sec),
            "recent_frac": recent_frac,
            "mdf_now": _mdf_of(actual_now), "mdf_pred": _mdf_of(predicted)}
    return (freqs, _smooth_bins(actual_now) * 100.0,
            _smooth_bins(predicted) * 100.0, meta)


def spectrum_backtest(segments, train_frac: float = 0.70,
                      win_sec: float = WIN_SEC, step_sec: float = STEP_SEC,
                      fs: int = FS, smooth_bins: int = 3):
    """Honest, verifiable prediction of the future frequency SHAPE.

    "Predict future visualisation of frequency given the patterns we find" - done
    as a proper train/test backtest, and on the spectral SHAPE (where the energy
    sits across frequency), not absolute amplitude. Amplitude swings with how
    hard the muscle fires and is not predictable on this pilot; the shape - the
    actual frequency signature - is far more stable and is what "visualisation of
    frequency" means.

      1. Collect the channel-averaged spectrum of every clean window, then
         NORMALISE each to unit area so it is a frequency distribution (the
         shape), independent of loudness.
      2. TRAIN on the first `train_frac` of the timeline: per frequency bin, fit
         the normalised value vs time (linear).
      3. PREDICT the shape at the mid-time of the held-out future, renormalise to
         unit area, and compare to the ACTUAL future shape.
      4. Report a match score and an error, both against a naive "future == the
         training-period shape" baseline, so the prediction is checkable.

    Returns (freqs, train_shape, predicted_shape, actual_shape, meta), all as
    PERCENT-of-total per bin (so the y-axis reads "share of power, %"). meta has
    match_pct (100 - L1 error, higher is better), baseline_match_pct, and the
    median-frequency of each shape. predicted/actual empty if too few windows.
    """
    freqs, specs, times = collect_session_spectra(segments, win_sec, step_sec, fs)
    meta = {"n_train": 0, "n_test": 0}
    if specs.shape[0] < 6:
        ref = specs.mean(axis=0) if specs.shape[0] else np.zeros(freqs.size)
        ref = ref / (ref.sum() or 1.0)
        return freqs, _smooth_bins(ref), np.array([]), np.array([]), meta

    # normalise every window to unit area -> spectral shape (distribution)
    areas = specs.sum(axis=1, keepdims=True)
    areas[areas == 0] = 1.0
    shapes = specs / areas

    order = np.argsort(times)
    times, shapes = times[order], shapes[order]
    split_t = times[0] + train_frac * (times[-1] - times[0])
    train = times <= split_t
    test = ~train
    if train.sum() < 3 or test.sum() < 1:
        ref = shapes.mean(axis=0)
        return freqs, _smooth_bins(ref), np.array([]), np.array([]), meta

    tt, hs = times[train], shapes[train]
    train_shape = hs.mean(axis=0)
    slope, intercept = np.polyfit(tt, hs, 1)             # per-bin (F,)
    t_pred = float(np.mean(times[test]))                 # mid of held-out future
    predicted = np.clip(slope * t_pred + intercept, 0.0, None)
    predicted = predicted / (predicted.sum() or 1.0)     # renormalise to a shape
    actual = shapes[test].mean(axis=0)

    def _mdf_of(shape):
        c = np.cumsum(shape)
        return float(freqs[np.searchsorted(c, c[-1] / 2.0)]) if c[-1] > 0 else 0.0

    # L1 distance between two unit-area distributions is in [0, 2]; turn the
    # model and the naive baseline into 0-100 "match" scores.
    err = float(np.sum(np.abs(predicted - actual)))
    base = float(np.sum(np.abs(train_shape - actual)))
    to_pct = 100.0
    meta = {"n_train": int(train.sum()), "n_test": int(test.sum()),
            "split_time_s": float(split_t), "train_frac": train_frac,
            "match_pct": 100.0 - 50.0 * err,           # 50*L1 maps [0,2]->[0,100]
            "baseline_match_pct": 100.0 - 50.0 * base,
            "mdf_train": _mdf_of(train_shape),
            "mdf_pred": _mdf_of(predicted),
            "mdf_actual": _mdf_of(actual)}
    return (freqs, _smooth_bins(train_shape) * to_pct,
            _smooth_bins(predicted) * to_pct,
            _smooth_bins(actual) * to_pct, meta)


def median_frequency(sig: np.ndarray, fs: int = FS) -> float:
    """Median (50% power) frequency of a 1-D signal."""
    x = sig - sig.mean()
    p = np.abs(rfft(x)) ** 2
    freqs = rfftfreq(x.size, 1.0 / fs)
    csum = np.cumsum(p)
    if csum[-1] <= 0:
        return 0.0
    return float(freqs[np.searchsorted(csum, csum[-1] / 2.0)])


def mdf_trend(seg: Segment, win_sec: float = WIN_SEC, step_sec: float = STEP_SEC):
    """Per-window mean MDF across channels -> (t_centers, mean_mdf, mdf[W, 8]).

    Gap-spanning windows are skipped, matching scan_segment.
    """
    win = max(2, int(round(win_sec * FS)))
    step = max(1, int(round(step_sec * FS)))
    t_centers, per_ch = [], []
    start = 0
    while start + win <= seg.n:
        if not _window_has_gap(seg, start, win):
            w = seg.data[start:start + win]
            t_centers.append(seg.t[start] + win_sec / 2.0)
            per_ch.append([median_frequency(w[:, c]) for c in range(N_CH)])
        start += step
    t_centers = np.array(t_centers)
    per_ch = np.array(per_ch) if per_ch else np.zeros((0, N_CH))
    mean_mdf = per_ch.mean(axis=1) if per_ch.size else np.array([])
    return t_centers, mean_mdf, per_ch


def forecast_linear(t: np.ndarray, y: np.ndarray, horizon_sec: float):
    """Illustrative linear extrapolation of y(t) with a +/-1 sigma band.

    Returns (t_future, y_future, lo, hi, slope). Deliberately simple: with only
    tens of seconds of unlabelled data a heavier model is not justified. The
    projected portion lies BEYOND the recorded clean data and must be labelled
    as illustrative wherever it is shown.
    """
    if t.size < 3:
        return np.array([]), np.array([]), np.array([]), np.array([]), 0.0
    a, b = np.polyfit(t, y, 1)                 # y ~ a*t + b
    resid_std = float(np.std(y - (a * t + b)))
    t_future = np.linspace(t[-1], t[-1] + horizon_sec, 40)
    y_future = a * t_future + b
    return t_future, y_future, y_future - resid_std, y_future + resid_std, float(a)


def forecast_regression(t: np.ndarray, y: np.ndarray, horizon_sec: float,
                        conf: float = 0.95, n_future: int = 80) -> dict:
    """Proper OLS linear forecast of y(t) with confidence + prediction bands.

    Built for the long clean segments (tens of seconds of MDF samples), where a
    real regression is justified. Returns a dict with, over both the observed
    span and the projected horizon beyond it:

        slope, intercept   : the fitted line y = slope*t + intercept (Hz/s, Hz)
        r2                 : coefficient of determination
        slope_ci           : (lo, hi) confidence interval on the slope
        p_value            : two-sided p for slope != 0
        t_fit, y_fit       : line over the observed span
        t_future, y_future : line over the projection (t[-1] .. t[-1]+horizon)
        ci_lo, ci_hi       : CONFIDENCE band for the mean response (t_future)
        pi_lo, pi_hi       : PREDICTION band for a new observation (t_future)
        resid_std          : residual standard deviation (Hz)

    The confidence band shows uncertainty in the trend line; the (wider)
    prediction band shows where an individual future MDF reading is expected.
    The projected portion lies BEYOND the recorded data.
    """
    out: dict = {"ok": False}
    n = t.size
    if n < 3:
        return out
    res = stats.linregress(t, y)
    a, b = float(res.slope), float(res.intercept)
    resid = y - (a * t + b)
    dof = n - 2
    s = float(np.sqrt(np.sum(resid ** 2) / dof)) if dof > 0 else 0.0
    xbar = float(np.mean(t))
    sxx = float(np.sum((t - xbar) ** 2))
    tcrit = float(stats.t.ppf(0.5 + conf / 2.0, dof)) if dof > 0 else 0.0

    t_fit = np.linspace(float(t[0]), float(t[-1]), n_future)
    y_fit = a * t_fit + b
    t_future = np.linspace(float(t[-1]), float(t[-1]) + horizon_sec, n_future)
    y_future = a * t_future + b

    # standard error of the mean response and of a new prediction at each point
    def _se(x, extra):
        return s * np.sqrt(extra + 1.0 / n + (x - xbar) ** 2 / sxx) if sxx > 0 else np.zeros_like(x)

    ci = tcrit * _se(t_future, 0.0)
    pi = tcrit * _se(t_future, 1.0)

    slope_ci = (a - tcrit * float(res.stderr), a + tcrit * float(res.stderr))
    out.update(
        ok=True, slope=a, intercept=b, r2=float(res.rvalue ** 2),
        slope_ci=slope_ci, p_value=float(res.pvalue), resid_std=s,
        t_fit=t_fit, y_fit=y_fit, t_future=t_future, y_future=y_future,
        ci_lo=y_future - ci, ci_hi=y_future + ci,
        pi_lo=y_future - pi, pi_hi=y_future + pi,
        conf=conf,
    )
    return out
