"""Regenerate the chart-enhancement example stills referenced by ENHANCEMENTS.md.

Renders from the REAL Zenodo biceps data through render_window.py's exact loader
path (FS=250, bandpass) so the illustrations match the live chart. Outputs PNGs
into viz/. Run:  .venv/bin/python viz/make_examples.py

These are decision aids for the parked ideas (spectrogram, MDF slope), not the
deliverable itself - the PNGs are gitignored; this script is the source of truth.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import signal  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(REPO, "zenodo_biceps"), os.path.join(REPO, "convergence_analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)
import loader  # noqa: E402

DATA_ROOT = os.path.join(REPO, "zenodo_biceps", "sEMG_data")
OUT = os.path.join(REPO, "viz")
FS, WIN_SEC, STEP_SEC = 250, 4.0, 2.0
LABEL_COLOR = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}
LABEL_NAME = {0: "Fresh", 1: "Transition", 2: "Fatigued"}
BG = "#0d1117"; PANEL = "#161b22"; FG = "#e6edf3"; MUTED = "#8b949e"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.edgecolor": "#30363d",
    "font.family": "DejaVu Sans", "font.size": 11,
})


def load(subject, side="R"):
    seg = loader.load_biceps_segment(DATA_ROOT, subject, side, target_fs=FS, bandpass=True)
    fs = int(getattr(seg, "eff_fs", FS))
    lab_t, lab_v = loader.load_fatigue_labels(DATA_ROOT, subject, side)
    x = seg.data[:, 0].astype(float)
    mdf_t, mdf_v, _ = loader.mdf_trend(seg, fs=fs, win_sec=WIN_SEC, step_sec=STEP_SEC)

    def dom(tc):
        i = int(np.searchsorted(lab_t, tc))
        i = min(max(i, 0), len(lab_v) - 1)
        return int(lab_v[i])
    labels = np.array([dom(tc) for tc in mdf_t])
    return x, fs, mdf_t, mdf_v, labels


def spectrogram_fig(subject, t_ask, side="R"):
    x, fs, mdf_t, mdf_v, labels = load(subject, side)
    f, tt, Sxx = signal.spectrogram(x, fs=fs, nperseg=256, noverlap=192, window="hann")
    Sdb = 10 * np.log10(Sxx + 1e-12)
    fmask = f <= 150
    fig, ax = plt.subplots(figsize=(11, 5.0), dpi=150, constrained_layout=True)
    pcm = ax.pcolormesh(tt, f[fmask], Sdb[fmask], shading="gouraud", cmap="magma")
    ax.plot(mdf_t, mdf_v, color="#58a6ff", lw=2.2, label="Median frequency (MDF)")
    ax.axvline(t_ask, color="#b388ff", lw=2, ls="--")
    ha = "right" if t_ask > 0.8 * float(mdf_t[-1]) else "left"
    ax.text(t_ask + (-3 if ha == "right" else 3), 144, f"asked {t_ask:.0f}s",
            color="#d0b4ff", ha=ha, va="top", fontsize=9.5)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Frequency (Hz)")
    ax.set_title("spectral energy compresses downward as the muscle fatigues; "
                 "the MDF line tracks the median of each column",
                 color=MUTED, fontsize=9.5, loc="left", pad=6)
    fig.suptitle(f"Spectrogram + MDF overlay  -  Subject {subject} ({side} biceps)",
                 color=FG, fontsize=13.5, fontweight="bold", x=0.012, ha="left")
    cb = fig.colorbar(pcm, ax=ax, pad=0.01); cb.set_label("Power (dB)", color=FG)
    cb.ax.yaxis.set_tick_params(color=MUTED)
    ax.legend(loc="upper right", framealpha=0.3, facecolor=PANEL, edgecolor="#30363d",
              labelcolor=FG, fontsize=9.5)
    out = os.path.join(OUT, f"ex_spectrogram_s{subject}.png")
    fig.savefig(out); plt.close(fig)
    return out


def slope_fig(subject, t_ask, side="R"):
    x, fs, mdf_t, mdf_v, labels = load(subject, side)
    m, b = np.polyfit(mdf_t, mdf_v, 1)  # Hz per second
    slope_min = m * 60.0
    fit = m * mdf_t + b
    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=150, constrained_layout=True)
    for lab in (0, 1, 2):
        sel = labels == lab
        if sel.any():
            ax.scatter(mdf_t[sel], mdf_v[sel], c=LABEL_COLOR[lab], s=26,
                       label=LABEL_NAME[lab], zorder=3, edgecolors="none")
    ax.plot(mdf_t, fit, color=FG, lw=2, ls="--", zorder=4,
            label=f"fatigue trend: {slope_min:+.1f} Hz/min")
    ax.axvline(t_ask, color="#b388ff", lw=2, ls="--")
    ymin, ymax = ax.get_ylim()
    ha = "right" if t_ask > 0.8 * float(mdf_t[-1]) else "left"
    ax.text(t_ask + (-2 if ha == "right" else 2), ymax, f"asked {t_ask:.0f}s",
            color="#d0b4ff", ha=ha, va="top", fontsize=9.5)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Median frequency (Hz)")
    ax.set_title("a fitted decline line turns coloured dots into a quantified "
                 "fatigue rate (Hz/min) - the actual metric",
                 color=MUTED, fontsize=9.5, loc="left", pad=6)
    fig.suptitle(f"MDF fatigue trend + slope  -  Subject {subject} ({side} biceps)",
                 color=FG, fontsize=13.5, fontweight="bold", x=0.012, ha="left")
    ax.grid(True, color="#21262d", lw=0.6)
    ax.legend(loc="upper right", framealpha=0.3, facecolor=PANEL, edgecolor="#30363d",
              labelcolor=FG, fontsize=9.5)
    out = os.path.join(OUT, f"ex_slope_s{subject}.png")
    fig.savefig(out); plt.close(fig)
    return out, slope_min


if __name__ == "__main__":
    for s in (13, 2, 1):
        t_ask = 120 if s == 13 else 100
        try:
            sp = spectrogram_fig(s, t_ask)
            _, slp = slope_fig(s, t_ask)
            print(f"S{s}: slope {slp:+.2f} Hz/min | {os.path.basename(sp)}")
        except Exception as e:
            print(f"S{s}: FAILED {type(e).__name__}: {e}")
