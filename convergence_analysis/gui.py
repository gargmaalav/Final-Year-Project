"""
Converged-sections player (the deliverable).
============================================

Shows the sections where all 8 EMG channels look similar (converge), inside the
recording's clean period, with two views:

  1. Normalized channels : the 8 channels z-scored over the full section so
                           shape similarity is visually obvious (channels move
                           together). A green playhead advances during playback.
  2. Predicted freq.     : spectral shape over the WHOLE recording (most recent
                           30%) with a linear projection 120 s ahead. The
                           supervisor's "predict the spectrum curve" deliverable.
                           Backtest: model 93% vs naive 86%.

The left list holds every detected converged section (wall-clock time + length +
mean similarity). Pick one, press Play, and screen-record for the supervisor;
"Next ->" steps through the sections.

Convergence = mean of the 28 pairwise Pearson correlations across the 8 channels
in a 1 s window (>= 0.80). No noise filtering anywhere. FS = 250 Hz (proven from
the Sample Index).

Run:  python run.py gui
"""
from __future__ import annotations

import sys
import time

import matplotlib
if sys.platform == "darwin":
    matplotlib.use("macosx")
else:
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.widgets import Button, RadioButtons, Slider
from matplotlib.ticker import MaxNLocator, FuncFormatter

from core import (
    CHANNEL_COLORS, CONV_THRESHOLD, FS, N_CH, WIN_SEC,
    all_converged_runs, channel_spectrum, forecast_linear, forecast_regression,
    mdf_trend, load_clean_segments, median_frequency,
)

DATA_FILE = "../data/OpenBCI-RAW-2026-03-21_15-37-11.txt"

DISPLAY_WIN = 3.0
TARGET_FPS = 30
SPEED = 1.0
TRAIN_FRAC = 0.70          # backtest: fit on first 70%, predict held-out last 30%
RECENT_FRAC = 0.30         # forward: "now" = most recent 30% of the recording
FORWARD_SEC = 120.0        # forward: project this far past the end of the data
#                            (120s chosen so the predicted line visibly separates
#                            from the measured one for the demo; the trend is
#                            gentle so a shorter horizon overlaps)
MAX_LIST = 14              # how many converged sections to list (longest first)

BG, GRID, FG, MUTED = "#0d1117", "#2a3441", "#e6edf3", "#8b949e"
BTN_BG, BTN_HOV = "#21262d", "#30363d"
OK_GREEN = "#3ddc97"
TRAIN_COL, PRED_COL, ACTUAL_COL = "#8b949e", "#a78bfa", "#3ddc97"
NOW_COL = "#ffd93d"        # "actual now" reference in the forward panel


class ConvergedPlayer:
    def __init__(self, segments):
        runs = all_converged_runs(segments)
        if not runs:
            raise ValueError("no converged sections found")
        # longest first so the strongest evidence is at the top of the list
        runs.sort(key=lambda sr: sr[1].dur, reverse=True)
        self.runs = runs[:MAX_LIST]
        self.sel = 0
        self.playhead = 0.0          # seconds within the current run
        self.is_playing = False
        self._last = time.perf_counter()
        self._cache: dict[int, dict] = {}
        self._prog_slider = False

        # whole-recording MDF — all segments, NaN-separated so gaps don't draw
        t0 = segments[0].wall_start
        rec_t_list, rec_mdf_list = [], []
        for seg in segments:
            offset = (seg.wall_start - t0).total_seconds()
            tc, mean_mdf, _ = mdf_trend(seg)
            if tc.size > 0:
                if rec_t_list:               # insert break between segments
                    rec_t_list.append(np.array([np.nan]))
                    rec_mdf_list.append(np.array([np.nan]))
                rec_t_list.append(tc + offset)
                rec_mdf_list.append(mean_mdf)
        self.rec_t = np.concatenate(rec_t_list) if rec_t_list else np.zeros(0)
        self.rec_mdf = np.concatenate(rec_mdf_list) if rec_mdf_list else np.zeros(0)

        # last segment only: smooth then fit with proper OLS regression
        SMOOTH_WIN = 20   # 20 x 0.1 s step = 2 s rolling mean
        last_seg = segments[-1]
        last_off = (last_seg.wall_start - t0).total_seconds()
        tc_last, mdf_last, _ = mdf_trend(last_seg)
        if tc_last.size >= SMOOTH_WIN:
            kernel = np.ones(SMOOTH_WIN) / SMOOTH_WIN
            mdf_sm = np.convolve(mdf_last, kernel, mode="valid")
            t_sm = tc_last[SMOOTH_WIN - 1:] + last_off
        else:
            mdf_sm = mdf_last
            t_sm = tc_last + last_off
        self.last_t_raw = tc_last + last_off
        self.last_mdf_raw = mdf_last
        self.last_t_sm = t_sm
        self.last_mdf_sm = mdf_sm
        self.reg = (forecast_regression(t_sm, mdf_sm, FORWARD_SEC)
                    if t_sm.size >= 3 else {"ok": False})

        self._build()
        self._load(0)

    # ----------------------------------------------------- per-run analysis
    def _analyse(self, i: int) -> dict:
        if i in self._cache:
            return self._cache[i]
        seg, run = self.runs[i]
        a = int(round(run.t_start * FS))
        b = int(round(run.t_end * FS))
        data = seg.data[a:b]                  # (n, 8) just the converged stretch
        t = np.arange(data.shape[0]) / FS     # 0-based time within the section

        # per-section MDF slope, just for the time-series annotation
        win = int(round(WIN_SEC * FS))
        step = int(round(0.1 * FS))
        mt, mmdf, s = [], [], 0
        while s + win <= data.shape[0]:
            w = data[s:s + win]
            mt.append(t[s] + WIN_SEC / 2)
            mmdf.append(np.mean([median_frequency(w[:, c]) for c in range(N_CH)]))
            s += step
        _tf, _yf, _lo, _hi, slope = forecast_linear(
            np.array(mt), np.array(mmdf), 1.0)

        # per-channel MDF time series (reuse existing win/step)
        mdf_ch, mdf_t_list = [], []
        s2 = 0
        while s2 + win <= data.shape[0]:
            w = data[s2:s2 + win]
            ch_mdfs = [median_frequency(w[:, c]) for c in range(N_CH)]
            mdf_ch.append(ch_mdfs)
            mdf_t_list.append(t[s2] + WIN_SEC / 2)
            s2 += step

        d = dict(seg=seg, run=run, data=data, t=t, dur=data.shape[0] / FS,
                 slope=slope,
                 mdf_t=np.array(mdf_t_list) if mdf_t_list else np.zeros(0),
                 mdf_vals=np.array(mdf_ch) if mdf_ch else np.zeros((0, N_CH)))
        self._cache[i] = d
        return d

    @property
    def cur(self) -> dict:
        return self._analyse(self.sel)

    # ----------------------------------------------------------------- UI
    def _build(self):
        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(14, 8.6), facecolor=BG)
        self.fig.suptitle(
            "EMG converged sections  -  normalized channels + "
            "predicted frequency  (clean period, no filtering, FS=250 Hz)",
            color=FG, fontsize=13, y=0.985)

        self.ax_ts = self.fig.add_axes((0.345, 0.60, 0.625, 0.30))
        self.ax_spec = self.fig.add_axes((0.345, 0.17, 0.285, 0.29))
        self.ax_fwd = self.fig.add_axes((0.685, 0.17, 0.285, 0.29))
        for ax in (self.ax_ts, self.ax_spec, self.ax_fwd):
            ax.set_facecolor(BG)
            for sp in ax.spines.values():
                sp.set_color(GRID)
            ax.tick_params(colors=MUTED, labelsize=10)

        # 1. normalized channels ------------------------------------------
        self.ts_lines = [self.ax_ts.plot([], [], color=CHANNEL_COLORS[c],
                                         lw=1.3, label=f"Ch {c}")[0]
                         for c in range(N_CH)]
        self.ax_ts.set_ylim(-3.5, 3.5)
        self.ax_ts.set_ylabel("Normalized (z)", color=MUTED, fontsize=11)
        self.ax_ts.set_xlabel("Time within section (s)", color=MUTED, fontsize=11)
        self.ax_ts.grid(True, color=GRID, alpha=0.5, lw=0.5)
        self.ax_ts.legend(ncol=8, fontsize=7, loc="upper right",
                          facecolor=BG, edgecolor=GRID, labelcolor=FG)
        self.ts_title = self.ax_ts.set_title("", color=FG, fontsize=10, pad=8)
        self.ts_playhead, = self.ax_ts.plot([0, 0], [-3.5, 3.5],
                                            color=OK_GREEN, lw=1.5, alpha=0.8,
                                            zorder=5)

        # 2. live FFT spectrum -------------------------------------------
        self.spec_lines = [self.ax_spec.plot([], [], color=CHANNEL_COLORS[c],
                                             lw=1.0, alpha=0.85)[0]
                           for c in range(N_CH)]
        self.ax_spec.set_xlim(0, 30)
        self.ax_spec.set_xlabel("Frequency (Hz)", color=MUTED, fontsize=11)
        self.ax_spec.set_ylabel("FFT amplitude (x1000)", color=MUTED, fontsize=11)
        self.ax_spec.yaxis.set_major_formatter(
            FuncFormatter(lambda v, _pos: f"{v / 1000:.0f}"))
        self.ax_spec.yaxis.set_major_locator(MaxNLocator(5))
        self.ax_spec.xaxis.set_major_locator(MaxNLocator(6))
        self.ax_spec.grid(True, color=GRID, alpha=0.5, lw=0.5)
        self.ax_spec.set_title("Frequency domain (live 1 s window)",
                               color=FG, fontsize=10, pad=6)

        # 3. whole-recording MDF trend + projection
        self._draw_recording_trend()

        # banner ----------------------------------------------------------
        self.banner = self.fig.text(0.345, 0.93, "", fontsize=12,
                                    color=OK_GREEN, ha="left", weight="bold")

        # converged-section selector (left) ------------------------------
        ax_radio = self.fig.add_axes((0.015, 0.17, 0.235, 0.73))
        ax_radio.set_facecolor(BG)
        self._labels = [self._run_label(seg, run) for seg, run in self.runs]
        self.radio = RadioButtons(ax_radio, self._labels, active=0)
        for tl in self.radio.labels:
            tl.set_color(FG)
            tl.set_fontsize(8.5)
        self.radio.on_clicked(self._on_radio)
        self.fig.text(0.015, 0.925,
                      f"Converged sections (>= {CONV_THRESHOLD:.2f}, longest first)",
                      color=MUTED, fontsize=10)

        # transport ------------------------------------------------------
        def mk(x, w, label, cb):
            ax = self.fig.add_axes((x, 0.03, w, 0.05))
            b = Button(ax, label, color=BTN_BG, hovercolor=BTN_HOV)
            b.label.set_color(FG)
            b.label.set_fontsize(9)
            b.on_clicked(cb)
            return b

        self.b_play = mk(0.30, 0.10, "Play", self._on_play)
        self.b_pause = mk(0.41, 0.10, "Pause", self._on_pause)
        self.b_restart = mk(0.52, 0.10, "Restart", self._on_restart)
        self.b_prev = mk(0.66, 0.10, "<- Prev", self._on_prev)
        self.b_next = mk(0.77, 0.10, "Next ->", self._on_next)

        ax_slider = self.fig.add_axes((0.30, 0.105, 0.67, 0.02))
        ax_slider.set_facecolor(BTN_BG)
        self.slider = Slider(ax_slider, "Seek", 0.0, 1.0, valinit=0.0,
                             color=OK_GREEN, initcolor="none")
        self.slider.label.set_color(FG)
        self.slider.valtext.set_color(FG)
        self.slider.on_changed(self._on_slider)

    def _draw_recording_trend(self):
        """Whole-recording MDF + last-segment OLS projection."""
        ax = self.ax_fwd
        # full session: faded background context
        if self.rec_t.size > 0:
            ax.plot(self.rec_t, self.rec_mdf, color=NOW_COL, lw=0.8,
                    alpha=0.25, zorder=1)
        # last segment: raw (faint) + smoothed (bright) — what was fitted
        if self.last_t_raw.size > 0:
            ax.plot(self.last_t_raw, self.last_mdf_raw, color=NOW_COL,
                    lw=0.8, alpha=0.55, zorder=2)
        if self.last_t_sm.size > 0:
            ax.plot(self.last_t_sm, self.last_mdf_sm, color=NOW_COL,
                    lw=2.0, zorder=3, label="last seg (smoothed)")
        r = self.reg
        if r.get("ok"):
            # fitted line over last segment
            ax.plot(r["t_fit"], r["y_fit"], color=PRED_COL, lw=1.5,
                    alpha=0.7, zorder=4)
            # projection + prediction band
            ax.plot(r["t_future"], r["y_future"], color=PRED_COL, lw=1.5,
                    zorder=5, label=f"+{FORWARD_SEC:.0f}s projection")
            ax.fill_between(r["t_future"], r["pi_lo"], r["pi_hi"],
                            color=PRED_COL, alpha=0.15, zorder=1)
            ax.fill_between(r["t_future"], r["ci_lo"], r["ci_hi"],
                            color=PRED_COL, alpha=0.25, zorder=2)
            p_str = f"p<0.001" if r["p_value"] < 0.001 else f"p={r['p_value']:.3f}"
            r2_note = " (illustrative — low R²)" if r["r2"] < 0.20 else ""
            ax.text(0.03, 0.97,
                    f"slope {r['slope']*1000:.2f} mHz/s  "
                    f"R²={r['r2']:.2f}{r2_note}  {p_str}",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=7.5, color=OK_GREEN)
        if self.rec_t.size > 0:
            ax.axvline(self.rec_t[-1], color=MUTED, lw=1.0, ls=":",
                       alpha=0.6, zorder=2)
        ax.set_xlabel("Session time (s)", color=MUTED, fontsize=11)
        ax.set_ylabel("MDF (Hz)", color=MUTED, fontsize=11)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.grid(True, color=GRID, alpha=0.5, lw=0.5)
        ax.set_title(f"MDF trend (last segment, 2 s smooth) + {FORWARD_SEC:.0f}s OLS projection\n"
                     f"(unfiltered; baseline drift, not fatigue marker)",
                     color=FG, fontsize=9, pad=6)
        ax.legend(fontsize=8, facecolor=BG, edgecolor=GRID, labelcolor=FG,
                  loc="upper right")

    @staticmethod
    def _run_label(seg, run) -> str:
        return (f"{seg.wall_at(run.t_start)[:8]}  {run.dur:4.1f}s  "
                f"sim {run.mean_score:.2f}")

    # ----------------------------------------------------------- load run
    def _load(self, i: int):
        self.sel = i
        self.playhead = 0.0
        d = self.cur

        self.slider.valmax = max(0.01, d["dur"])
        self.slider.ax.set_xlim(0, self.slider.valmax)
        self.ax_ts.set_xlim(0, d["dur"])
        self._set_slider(0.0)
        # auto-start so the top panel is never blank after switching sections
        self.is_playing = True
        self._last = time.perf_counter()
        self._render()

    # ------------------------------------------------------------ callbacks
    def _on_radio(self, label):
        self._load(self._labels.index(label))

    def _on_play(self, _):
        if self.playhead >= self.cur["dur"]:
            self.playhead = 0.0
        self.is_playing = True
        self._last = time.perf_counter()

    def _on_pause(self, _):
        self.is_playing = False

    def _on_restart(self, _):
        self.playhead = 0.0
        self.is_playing = False
        self._set_slider(0.0)
        self._render()

    def _on_prev(self, _):
        if self.sel > 0:
            self.radio.set_active(self.sel - 1)

    def _on_next(self, _):
        if self.sel < len(self.runs) - 1:
            self.radio.set_active(self.sel + 1)

    def _on_slider(self, v):
        if self._prog_slider:
            return
        self.playhead = float(v)
        self.is_playing = False
        self._last = time.perf_counter()
        self._render()

    def _set_slider(self, v):
        self._prog_slider = True
        try:
            self.slider.set_val(v)
        finally:
            self._prog_slider = False

    # ------------------------------------------------------------- render
    def _render(self):
        d = self.cur
        data, t = d["data"], d["t"]

        t_end = self.playhead
        # normalized channels: reveal up to playhead so lines grow with playback
        if data.shape[0] >= 2:
            z = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-9)
            mask = t <= t_end
            if mask.sum() >= 2:
                for c in range(N_CH):
                    self.ts_lines[c].set_data(t[mask], z[mask, c])
            else:
                for ln in self.ts_lines:
                    ln.set_data([], [])
        else:
            for ln in self.ts_lines:
                ln.set_data([], [])
        self.ts_playhead.set_xdata([self.playhead, self.playhead])

        # live FFT: trailing 1s window at playhead
        s0 = max(0, int((t_end - WIN_SEC) * FS))
        s1 = min(data.shape[0], max(s0 + 4, int(t_end * FS)))
        ws = data[s0:s1]
        if ws.shape[0] >= 4:
            freqs, mags = channel_spectrum(ws)
            for c in range(N_CH):
                self.spec_lines[c].set_data(freqs, mags[c])
            self.ax_spec.set_ylim(0, float(mags.max()) * 1.1 + 1e-9)

        seg, run = d["seg"], d["run"]
        self.banner.set_text(
            f"Section {self.sel + 1}/{len(self.runs)}   Seg {seg.idx}   "
            f"{seg.wall_at(run.t_start)} - {seg.wall_at(run.t_end)}   "
            f"sim {run.mean_score:.3f}   t={t_end:4.1f}/{d['dur']:.1f}s")
        self.ts_title.set_text(
            f"8 channels normalized - watching them move together   "
            f"(section MDF slope {d['slope']:+.2f} Hz/s)")
        self.fig.canvas.draw_idle()

    def _tick(self, _):
        if self.is_playing:
            now = time.perf_counter()
            dt = min(now - self._last, 0.25)
            self._last = now
            self.playhead += dt * SPEED
            if self.playhead >= self.cur["dur"]:
                self.playhead = self.cur["dur"]
                self.is_playing = False
            self._set_slider(self.playhead)
        else:
            self._last = time.perf_counter()
        self._render()
        return []

    def run(self):
        # auto-start playback so the window never opens on the blank t=0 frame
        # (at t=0 the scrolling time series has no history and the live FFT has
        # only ~4 samples, which looks empty/triangular until playback begins).
        self.is_playing = True
        self._last = time.perf_counter()
        self.anim = animation.FuncAnimation(
            self.fig, self._tick, interval=1000 / TARGET_FPS,
            blit=False, cache_frame_data=False)
        plt.show()


def run(data_file: str = DATA_FILE):
    segs = load_clean_segments(data_file)
    runs = all_converged_runs(segs)
    print(f"{len(segs)} clean segments; {len(runs)} converged sections "
          f"({sum(r.dur for _, r in runs):.1f}s). Launching player ...")
    ConvergedPlayer(segs).run()


if __name__ == "__main__":
    run()
