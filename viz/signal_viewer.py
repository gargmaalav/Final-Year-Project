"""
Interactive EMG signal viewer for Zenodo 14182446 biceps data.

Single-subject mode (--subject N):
  - Top:    Raw filtered EMG coloured by fatigue label
  - Middle: Running MDF over time
  - Bottom: Live FFT spectrum of the current window

Multi-subject mode (default, no --subject):
  - 4x4 grid of MDF-over-time plots for all 13 subjects
  - Single time cursor scrubs / plays across all subjects simultaneously

Usage (from zenodo_biceps/):
    python signal_viewer.py --root sEMG_data               # all 13 subjects
    python signal_viewer.py --root sEMG_data --subject 1   # single subject
    python signal_viewer.py --root sEMG_data --subject 8 --side R
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider, Button

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loader
import core

LABEL_COLOR = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}
LABEL_NAME  = {0: "Fresh", 1: "Transition", 2: "Fatigued"}

WIN_SEC  = 4.0   # display / FFT window length
STEP_SEC = 0.5   # slider step


def mdf_series(x, fs, win, step):
    mdfs, ts = [], []
    start = 0
    while start + win <= len(x):
        w = x[start:start + win]
        mdfs.append(core.median_frequency(w, fs=fs))
        ts.append((start + win / 2) / fs)
        start += step
    return np.array(ts), np.array(mdfs)


def dominant_label(t_center, lab_t, lab_v, half=WIN_SEC / 2):
    mask = (lab_t >= t_center - half) & (lab_t <= t_center + half)
    if mask.sum() == 0:
        return int(lab_v[np.argmin(np.abs(lab_t - t_center))])
    vals, counts = np.unique(lab_v[mask], return_counts=True)
    return int(vals[np.argmax(counts)])


def build_viewer(root, subject, side):
    seg = loader.load_biceps_segment(root, subject, side, bandpass=True)
    fs  = int(getattr(seg, "eff_fs", loader.FS_NATIVE))
    lab_t, lab_v = loader.load_fatigue_labels(root, subject, side)

    x = seg.data[:, 0].astype(float)
    t = np.arange(len(x)) / fs

    win_samples  = int(WIN_SEC * fs)
    step_samples = int(STEP_SEC * fs)

    # pre-compute MDF + per-MDF-window label
    mdf_t, mdf_vals = mdf_series(x, fs, win_samples, step_samples)
    mdf_labels = np.array([dominant_label(tc, lab_t, lab_v) for tc in mdf_t])

    # ------------------------------------------------------------------ figure
    fig = plt.figure(figsize=(14, 8), facecolor="#111")
    fig.canvas.manager.set_window_title(
        f"EMG Signal Viewer — Subject {subject} ({side} biceps)"
    )

    ax_sig  = fig.add_axes([0.07, 0.62, 0.90, 0.32], facecolor="#1a1a2e")
    ax_mdf  = fig.add_axes([0.07, 0.35, 0.90, 0.22], facecolor="#1a1a2e")
    ax_fft  = fig.add_axes([0.07, 0.12, 0.90, 0.20], facecolor="#1a1a2e")
    ax_sl   = fig.add_axes([0.07, 0.05, 0.78, 0.04], facecolor="#333")
    ax_btn  = fig.add_axes([0.87, 0.04, 0.10, 0.06], facecolor="#333")

    for ax in [ax_sig, ax_mdf, ax_fft]:
        ax.tick_params(colors="#aaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    # ---------------------------------------------------------------- MDF axis (static coloured scatter)
    for lbl in [0, 1, 2]:
        mask = mdf_labels == lbl
        if mask.any():
            ax_mdf.scatter(mdf_t[mask], mdf_vals[mask], s=12,
                           color=LABEL_COLOR[lbl], label=LABEL_NAME[lbl],
                           alpha=0.8, zorder=2)
    ax_mdf.set_ylabel("MDF (Hz)", color="#ccc", fontsize=9)
    ax_mdf.set_xlabel("Time (s)", color="#ccc", fontsize=9)
    ax_mdf.set_title("Median Frequency (MDF) — fatigue marker",
                     color="#ddd", fontsize=10, pad=4)
    ax_mdf.legend(loc="upper right", fontsize=8,
                  facecolor="#222", labelcolor="white")
    ax_mdf.set_xlim(t[0], t[-1])
    ax_mdf.tick_params(colors="#aaa")
    win_line_mdf = ax_mdf.axvline(WIN_SEC / 2, color="white",
                                  linewidth=1.2, alpha=0.7, zorder=3)

    # ---------------------------------------------------------------- EMG signal axis (drawn per frame)
    ax_sig.set_ylabel("EMG (a.u., bandpass)", color="#ccc", fontsize=9)
    ax_sig.set_title(f"S{subject} {side} Biceps — Raw EMG (filtered 20-450 Hz)",
                     color="#ddd", fontsize=10, pad=4)
    ax_sig.set_xlim(0, WIN_SEC)
    ax_sig.set_ylim(x.min() * 1.1, x.max() * 1.1)
    sig_line, = ax_sig.plot([], [], color="#00d4ff", linewidth=0.6)
    sig_patch = ax_sig.axvspan(0, WIN_SEC, alpha=0.08, color="green")
    lbl_text  = ax_sig.text(0.99, 0.93, "", transform=ax_sig.transAxes,
                            ha="right", va="top", fontsize=13,
                            fontweight="bold", color="white")

    # ---------------------------------------------------------------- FFT axis
    freqs_disp = np.linspace(0, fs / 2, win_samples // 2 + 1)
    mask_band = freqs_disp <= 500
    ax_fft.set_xlim(0, 500)
    ax_fft.set_ylabel("Power (a.u.)", color="#ccc", fontsize=9)
    ax_fft.set_xlabel("Frequency (Hz)", color="#ccc", fontsize=9)
    ax_fft.set_title("FFT Spectrum of current window", color="#ddd",
                     fontsize=10, pad=4)
    fft_line, = ax_fft.plot([], [], color="#ff6b6b", linewidth=1.2)
    mdf_vline  = ax_fft.axvline(0, color="yellow", linewidth=1.5,
                                linestyle="--", label="MDF", alpha=0.9)
    ax_fft.legend(loc="upper right", fontsize=8,
                  facecolor="#222", labelcolor="white")

    # ---------------------------------------------------------------- title
    fig.text(0.5, 0.97,
             f"EMG Fatigue Progression — Subject {subject} ({side} Biceps Brachii)",
             ha="center", fontsize=13, color="white", fontweight="bold")

    legend_patches = [
        mpatches.Patch(color=LABEL_COLOR[k], label=LABEL_NAME[k])
        for k in [0, 1, 2]
    ]
    fig.legend(handles=legend_patches, loc="upper right",
               bbox_to_anchor=(0.99, 0.97), fontsize=9,
               facecolor="#222", labelcolor="white", framealpha=0.8)

    # ---------------------------------------------------------------- slider
    n_steps = max(1, (len(x) - win_samples) // step_samples)
    slider = Slider(ax_sl, "Time", 0, n_steps - 1, valinit=0, valstep=1,
                    color="#4c72b0")
    slider.label.set_color("#ccc")
    slider.valtext.set_color("#ccc")

    # ---------------------------------------------------------------- update fn
    def update(val=None):
        step_idx = int(slider.val)
        start    = step_idx * step_samples
        end      = start + win_samples
        end      = min(end, len(x))
        w        = x[start:end]
        tw       = np.arange(len(w)) / fs
        t_center = t[start] + WIN_SEC / 2

        # EMG
        sig_line.set_data(tw, w)
        lbl = dominant_label(t_center, lab_t, lab_v)
        sig_patch.set_facecolor(LABEL_COLOR[lbl])
        lbl_text.set_text(LABEL_NAME[lbl])
        lbl_text.set_color(LABEL_COLOR[lbl])

        # MDF cursor
        win_line_mdf.set_xdata([t_center, t_center])

        # FFT
        spec = np.abs(np.fft.rfft(w * np.hanning(len(w)))) ** 2
        spec = spec / (spec.max() + 1e-12)
        fft_freqs = np.linspace(0, fs / 2, len(spec))
        fft_line.set_data(fft_freqs, spec)
        ax_fft.set_ylim(0, 1.05)
        mdf_cur = core.median_frequency(w, fs=fs)
        mdf_vline.set_xdata([mdf_cur, mdf_cur])

        fig.canvas.draw_idle()

    slider.on_changed(update)
    update()

    # ---------------------------------------------------------------- play button
    state = {"playing": False, "timer": None}

    def on_play(event):
        state["playing"] = not state["playing"]
        btn_play.label.set_text("Pause" if state["playing"] else "Play")
        if state["playing"]:
            state["timer"] = fig.canvas.new_timer(interval=120)
            state["timer"].add_callback(advance)
            state["timer"].start()
        elif state["timer"]:
            state["timer"].stop()

    def advance():
        nxt = min(int(slider.val) + 1, n_steps - 1)
        slider.set_val(nxt)
        if nxt >= n_steps - 1:
            state["playing"] = False
            btn_play.label.set_text("Play")
            if state["timer"]:
                state["timer"].stop()

    btn_play = Button(ax_btn, "Play", color="#333", hovercolor="#555")
    btn_play.label.set_color("white")
    btn_play.on_clicked(on_play)

    plt.show()


def build_multi_viewer(root, side="R"):
    """3-panel layout (EMG / MDF / FFT) with all 13 subjects overlaid."""
    subjects = list(range(1, 14))
    # 13 visually distinct colours
    COLORS = [plt.cm.tab20(i / 13) for i in range(13)]

    print(f"Loading {len(subjects)} subjects ({side} biceps) ...", flush=True)
    all_data: dict = {}
    for s in subjects:
        try:
            seg      = loader.load_biceps_segment(root, s, side, bandpass=True)
            fs       = int(getattr(seg, "eff_fs", loader.FS_NATIVE))
            lab_t, lab_v = loader.load_fatigue_labels(root, s, side)
            x        = seg.data[:, 0].astype(float)
            t_full   = np.arange(len(x)) / fs
            amp      = float(np.percentile(np.abs(x), 99)) or 1.0
            win_s    = int(WIN_SEC  * fs)
            step_s   = int(STEP_SEC * fs)
            mdf_t, mdf_v = mdf_series(x, fs, win_s, step_s)
            # normalise raw signal to [-1, 1] so all 13 share one y-axis
            x_norm   = x / amp
            all_data[s] = dict(
                x=x_norm, fs=fs, lab_t=lab_t, lab_v=lab_v,
                t_max=float(t_full[-1]),
                mdf_t=mdf_t, mdf_v=mdf_v,
            )
            print(f"  S{s} ok ({t_full[-1]:.0f}s)", flush=True)
        except Exception as exc:
            print(f"  S{s} SKIP: {exc}", flush=True)

    if not all_data:
        print("No data loaded — check --root path.")
        return

    max_t = max(d["t_max"] for d in all_data.values())

    # ------------------------------------------------------------------ figure
    fig = plt.figure(figsize=(15, 9), facecolor="#111")
    fig.canvas.manager.set_window_title(
        f"EMG Overlay Viewer — All 13 Subjects ({side} Biceps)"
    )

    ax_sig = fig.add_axes([0.07, 0.62, 0.83, 0.30], facecolor="#1a1a2e")
    ax_mdf = fig.add_axes([0.07, 0.38, 0.83, 0.20], facecolor="#1a1a2e")
    ax_fft = fig.add_axes([0.07, 0.15, 0.83, 0.18], facecolor="#1a1a2e")
    ax_sl  = fig.add_axes([0.07, 0.06, 0.70, 0.03], facecolor="#333")
    ax_btn = fig.add_axes([0.80, 0.055, 0.10, 0.05], facecolor="#333")

    for ax in [ax_sig, ax_mdf, ax_fft]:
        ax.tick_params(colors="#aaa", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    # ---- MDF panel: full-recording scatter per subject (static) + cursor ----
    for i, s in enumerate(subjects):
        d = all_data[s]
        ax_mdf.plot(d["mdf_t"], d["mdf_v"],
                    color=COLORS[i], linewidth=0.8, alpha=0.75)

    ax_mdf.set_ylabel("MDF (Hz)", color="#ccc", fontsize=8)
    ax_mdf.set_xlabel("Time (s)", color="#ccc", fontsize=8)
    ax_mdf.set_title("Median Frequency over recording — all subjects",
                     color="#ddd", fontsize=9, pad=3)
    ax_mdf.set_xlim(0, max_t)
    win_line_mdf = ax_mdf.axvline(WIN_SEC / 2, color="white",
                                   linewidth=1.2, alpha=0.7, zorder=5)

    # ---- EMG panel: one normalised line per subject (live window) -----------
    ax_sig.set_xlim(0, WIN_SEC)
    ax_sig.set_ylim(-1.15, 1.15)
    ax_sig.set_ylabel("EMG (norm.)", color="#ccc", fontsize=8)
    ax_sig.set_title("Raw EMG — 4 s window (normalised per subject)",
                     color="#ddd", fontsize=9, pad=3)
    sig_lines = {}
    for i, s in enumerate(subjects):
        l, = ax_sig.plot([], [], color=COLORS[i], linewidth=0.5, alpha=0.75)
        sig_lines[s] = l

    # ---- FFT panel: one spectrum per subject (live window) ------------------
    ax_fft.set_xlim(0, 500)
    ax_fft.set_ylim(0, 1.05)
    ax_fft.set_ylabel("Power (norm.)", color="#ccc", fontsize=8)
    ax_fft.set_xlabel("Frequency (Hz)", color="#ccc", fontsize=8)
    ax_fft.set_title("FFT spectrum — current window",
                     color="#ddd", fontsize=9, pad=3)
    fft_lines = {}
    for i, s in enumerate(subjects):
        l, = ax_fft.plot([], [], color=COLORS[i], linewidth=0.8, alpha=0.65)
        fft_lines[s] = l

    # ---- title + subject legend (right of figure) --------------------------
    fig.text(0.5, 0.955,
             f"EMG Fatigue — All 13 Subjects ({side} Biceps Brachii)",
             ha="center", fontsize=13, color="white", fontweight="bold")

    legend_lines = [
        plt.Line2D([0], [0], color=COLORS[i], linewidth=1.5, label=f"S{s}")
        for i, s in enumerate(subjects)
    ]
    fig.legend(handles=legend_lines, loc="center right",
               bbox_to_anchor=(0.995, 0.55), fontsize=7, ncol=1,
               facecolor="#222", labelcolor="white", framealpha=0.8,
               title="Subject", title_fontsize=7)

    # ---- slider + play button ----------------------------------------------
    slider = Slider(ax_sl, "Time (s)", 0, max_t, valinit=0, valstep=STEP_SEC,
                    color="#4c72b0")
    slider.label.set_color("#ccc")
    slider.valtext.set_color("#ccc")

    def update(val=None):
        t_now = float(slider.val)
        t_center = t_now + WIN_SEC / 2

        for s in subjects:
            d     = all_data[s]
            fs    = d["fs"]
            x_arr = d["x"]
            win_s = int(WIN_SEC * fs)
            start = min(int(t_now * fs), max(0, len(x_arr) - win_s))
            end   = min(start + win_s, len(x_arr))
            w     = x_arr[start:end]
            tw    = np.arange(len(w)) / fs
            sig_lines[s].set_data(tw, w)

            spec  = np.abs(np.fft.rfft(w * np.hanning(len(w)))) ** 2
            spec  = spec / (spec.max() + 1e-12)
            freqs = np.linspace(0, fs / 2, len(spec))
            fft_lines[s].set_data(freqs, spec)

        win_line_mdf.set_xdata([t_center, t_center])
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update()

    # ---- play / pause ------------------------------------------------------
    state = {"playing": False, "timer": None}

    def on_play(event):
        state["playing"] = not state["playing"]
        btn_play.label.set_text("Pause" if state["playing"] else "Play")
        if state["playing"]:
            state["timer"] = fig.canvas.new_timer(interval=120)
            state["timer"].add_callback(advance)
            state["timer"].start()
        elif state["timer"]:
            state["timer"].stop()

    def advance():
        nxt = min(float(slider.val) + STEP_SEC, max_t)
        slider.set_val(nxt)
        if nxt >= max_t:
            state["playing"] = False
            btn_play.label.set_text("Play")
            if state["timer"]:
                state["timer"].stop()

    btn_play = Button(ax_btn, "Play", color="#333", hovercolor="#555")
    btn_play.label.set_color("white")
    btn_play.on_clicked(on_play)

    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",    required=True)
    ap.add_argument("--subject", type=int, default=None,
                    help="Single subject (1-13). Omit to view all 13.")
    ap.add_argument("--side",    choices=["R", "L"], default="R")
    args = ap.parse_args()

    if args.subject is None:
        build_multi_viewer(args.root, args.side)
    else:
        build_viewer(args.root, args.subject, args.side)


if __name__ == "__main__":
    main()
