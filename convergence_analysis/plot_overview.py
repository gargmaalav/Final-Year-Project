"""
3-panel overview: time domain, frequency domain, predicted frequency.
=====================================================================

Generates out/overview.png — a single static figure the supervisor can see.

Panels:
  1. Time domain      (top, full width) — all 8 channels normalized, all clean
                       segments stitched with a visible gap between each.
  2. Frequency domain (bottom-left)    — per-channel mean amplitude spectrum
                       averaged across every clean 1-s window in the session.
                       Shown 0-30 Hz; unfiltered signal has negligible power
                       above ~10 Hz so the full 0-125 Hz view is mostly empty.
  3. Predicted freq.  (bottom-right)   — the spectral SHAPE (share of power per
                       frequency bin, %) measured over the most recent 30% of
                       the recording vs the same shape projected 120 s forward.
                       Dashed line = where the distribution is heading; shaded
                       area = the shift.

Run:  python plot_overview.py
      or    python run.py plot
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from core import (
    CHANNEL_COLORS, FS, N_CH, WIN_SEC,
    channel_spectrum, load_clean_segments, spectrum_forward,
)

DATA_FILE = "../data/OpenBCI-RAW-2026-03-21_15-37-11.txt"
OUT = Path("out") / "overview.png"

FORWARD_SEC = 120.0
RECENT_FRAC = 0.30
FREQ_XLIM = 30          # Hz — zoom in where the unfiltered signal has power
GAP_S = 2.0             # visual gap between segments in time domain panel
DOWNSAMPLE = 4          # plot every Nth sample in the time domain (speed)

BG, GRID, FG, MUTED = "#0d1117", "#2a3441", "#e6edf3", "#8b949e"
NOW_COL, PRED_COL = "#ffd93d", "#a78bfa"


def _style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.grid(True, color=GRID, alpha=0.5, lw=0.5)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(FG)


def _time_domain(ax, segments):
    t_offset = 0.0
    for i, seg in enumerate(segments):
        d = seg.data[::DOWNSAMPLE]
        t = np.arange(d.shape[0]) * (DOWNSAMPLE / FS)
        mu = d.mean(axis=0)
        sd = d.std(axis=0) + 1e-9
        z = (d - mu) / sd
        for c in range(N_CH):
            ax.plot(t_offset + t, z[:, c],
                    color=CHANNEL_COLORS[c], lw=0.6, alpha=0.75,
                    label=f"Ch {c}" if i == 0 else "")
        ax.axvspan(t_offset + seg.dur, t_offset + seg.dur + GAP_S,
                   color=BG, zorder=0)
        ax.axvline(t_offset, color=MUTED, lw=0.7, alpha=0.25, ls=":")
        ax.text(t_offset + 1, 4.2, f"Seg {seg.idx}", color=MUTED,
                fontsize=7, va="top")
        t_offset += seg.dur + GAP_S

    ax.set_ylim(-5, 5)
    ax.set_xlabel("Time within clean segments (s; gaps = dropout periods)",
                  fontsize=11)
    ax.set_ylabel("Normalized amplitude (z)", fontsize=11)
    ax.set_title(
        f"Time domain — 8 channels, {len(segments)} clean segments "
        f"(normalized per segment, unfiltered, FS={FS} Hz)",
        fontsize=12, pad=8)
    ax.legend(ncol=8, fontsize=7, loc="upper right",
              facecolor=BG, edgecolor=GRID, labelcolor=FG)
    ax.xaxis.set_major_locator(MaxNLocator(10))


def _freq_domain(ax, segments):
    win = max(2, int(round(WIN_SEC * FS)))
    step = max(1, int(round(0.1 * FS)))
    per_ch: list[list[np.ndarray]] = [[] for _ in range(N_CH)]

    for seg in segments:
        s = 0
        while s + win <= seg.n:
            if not seg.gap_mask[s:s + win].any():
                freqs, mags = channel_spectrum(seg.data[s:s + win], FS)
                for c in range(N_CH):
                    per_ch[c].append(mags[c])
            s += step

    for c in range(N_CH):
        if per_ch[c]:
            avg = np.mean(per_ch[c], axis=0)
            ax.plot(freqs, avg / 1000, color=CHANNEL_COLORS[c],
                    lw=1.3, alpha=0.85, label=f"Ch {c}")

    ax.set_xlim(0, FREQ_XLIM)
    ax.set_xlabel("Frequency (Hz)", fontsize=11)
    ax.set_ylabel("Mean FFT amplitude (×10³ µV)", fontsize=11)
    ax.set_title(
        f"Frequency domain — per-channel mean spectrum\n"
        f"(averaged over all clean 1-s windows, 0–{FREQ_XLIM} Hz shown)",
        fontsize=11, pad=8)
    ax.legend(ncol=4, fontsize=7, loc="upper right",
              facecolor=BG, edgecolor=GRID, labelcolor=FG)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(5))

    return freqs  # pass to caller so prediction panel can reuse


def _predicted_freq(ax, segments):
    ff, fwd_now, fwd_pred, meta = spectrum_forward(
        segments, recent_frac=RECENT_FRAC, horizon_sec=FORWARD_SEC)

    ax.plot(ff, fwd_now, color=NOW_COL, lw=2.0,
            label=f"measured now  (MDF {meta.get('mdf_now', 0):.0f} Hz)")
    if fwd_pred.size:
        ax.plot(ff, fwd_pred, color=PRED_COL, lw=2.0, ls="--",
                label=f"predicted +{FORWARD_SEC:.0f}s  "
                      f"(MDF {meta.get('mdf_pred', 0):.0f} Hz)")
        ax.fill_between(ff, fwd_now, fwd_pred,
                        color=PRED_COL, alpha=0.20, zorder=1)

    ymax = max(float(fwd_now.max()),
               float(fwd_pred.max()) if fwd_pred.size else 0.0) * 1.3 + 1e-9
    ax.set_xlim(0, FREQ_XLIM)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Frequency (Hz)", fontsize=11)
    ax.set_ylabel("Share of power (%)", fontsize=11)
    ax.set_title(
        f"Predicted frequency — spectral shape now vs +{FORWARD_SEC:.0f}s\n"
        f"(unfiltered; MDF reflects baseline drift, not fatigue marker)",
        fontsize=11, pad=8)
    ax.legend(fontsize=9, loc="upper right",
              facecolor=BG, edgecolor=GRID, labelcolor=FG)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.text(0.97, 0.55,
            f"based on {meta.get('n_recent', 0)} windows",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5, color=MUTED)


def run(data_file: str = DATA_FILE) -> Path:
    OUT.parent.mkdir(exist_ok=True)
    print(f"Loading {data_file} ...")
    segs = load_clean_segments(data_file)
    total_s = sum(s.dur for s in segs)
    print(f"  {len(segs)} clean segments, {total_s:.0f}s total")

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(15, 10), facecolor=BG)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1],
                          hspace=0.45, wspace=0.35,
                          left=0.07, right=0.97, top=0.91, bottom=0.08)
    ax_ts = fig.add_subplot(gs[0, :])
    ax_fr = fig.add_subplot(gs[1, 0])
    ax_pw = fig.add_subplot(gs[1, 1])

    for ax in (ax_ts, ax_fr, ax_pw):
        _style(ax)

    print("  plotting time domain ...")
    _time_domain(ax_ts, segs)

    print("  plotting frequency domain ...")
    _freq_domain(ax_fr, segs)

    print("  plotting predicted frequency ...")
    _predicted_freq(ax_pw, segs)

    fig.suptitle(
        f"EMG 8-channel overview  |  time domain / frequency domain / predicted frequency\n"
        f"OpenBCI-RAW-2026-03-21  |  FS={FS} Hz  |  "
        f"{len(segs)} clean segments, {total_s:.0f}s  |  unfiltered",
        color=FG, fontsize=13, y=0.975)

    fig.savefig(OUT, dpi=140, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {OUT}")
    return OUT


if __name__ == "__main__":
    run()
