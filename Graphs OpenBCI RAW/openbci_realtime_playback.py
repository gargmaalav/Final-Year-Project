"""
OpenBCI 8-channel EXG real-time playback visualisation
======================================================

Animates the recorded OpenBCI .txt file as if the data were arriving live.
A scrolling 5-second window shows all 8 channels overlaid using the raw
voltage values (no filtering, no mean-centering).

Controls:
    Play     - start/resume the animation
    Pause    - pause at the current position
    Restart  - jump back to the start of the recording
    Seek bar - drag to jump to any point in the recording (works while
               playing or paused)

Usage:
    python openbci_realtime_playback.py /path/to/OpenBCI-RAW-2026-03-21_15-37-11.txt

Or edit SRC below and run with no arguments.

Requires: numpy, pandas, matplotlib
"""
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SRC = "OpenBCI-RAW-2026-03-21_15-37-11.txt"   # default; override via CLI arg

WINDOW_SECONDS    = 5.0     # how many seconds of history to show on screen
TARGET_FPS        = 60      # max refresh rate (matplotlib may deliver less)
PLAYBACK_SPEED    = 1.0     # 1.0 = real-time, 2.0 = double speed, 0.5 = half

# Where the playhead should start when the script opens. Either a number of
# seconds from the beginning of the file (e.g. 30.0) or a pandas Timestamp
# matching the wall-clock time in the recording. Leave as None to start at 0.
INITIAL_TIME = None
# Examples:
#   INITIAL_TIME = 12.5
#   INITIAL_TIME = pd.Timestamp("2026-03-21 15:42:28")

# Optional: restrict playback to a sub-window of the file. Set both to None
# to play the entire recording.
START_TIME = None   # e.g. pd.Timestamp("2026-03-21 15:42:25")
END_TIME   = None   # e.g. pd.Timestamp("2026-03-21 15:43:15")

# Dark theme colours
CHANNEL_COLORS = [
    "#f5a623",  # Ch 0  orange
    "#4ec9ec",  # Ch 1  cyan
    "#3ddc97",  # Ch 2  green
    "#a78bfa",  # Ch 3  purple
    "#ff6b9d",  # Ch 4  pink
    "#ffd93d",  # Ch 5  yellow
    "#5eead4",  # Ch 6  teal
    "#fb7185",  # Ch 7  coral
]
BG       = "#0d1117"
GRID     = "#2a3441"
FG       = "#e6edf3"
MUTED    = "#8b949e"
BTN_BG   = "#21262d"
BTN_HOV  = "#30363d"


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_openbci(path: str) -> pd.DataFrame:
    """Load the OpenBCI raw .txt export into a tidy DataFrame."""
    df = pd.read_csv(path, comment="%", skipinitialspace=True, engine="python")
    df.columns = [c.strip() for c in df.columns]

    df["t_epoch"] = pd.to_numeric(df["Timestamp"], errors="coerce")
    df["t_fmt"]   = pd.to_datetime(df["Timestamp (Formatted)"], errors="coerce")
    df = df.dropna(subset=["t_epoch", "t_fmt"]).reset_index(drop=True)

    for c in [f"EXG Channel {i}" for i in range(8)]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Player class - holds state and drives the animation
# ---------------------------------------------------------------------------
class RealtimePlayer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.exg_cols = [f"EXG Channel {i}" for i in range(8)]

        # Convert timestamps to seconds-from-start for easy arithmetic
        self.t = self.df["t_epoch"].to_numpy()
        self.t = self.t - self.t[0]   # 0 = first sample of playback

        self.values = {c: self.df[c].to_numpy() for c in self.exg_cols}

        # State
        self.playback_time = 0.0    # current playhead in seconds-since-start
        self.is_playing    = False
        self.dt            = PLAYBACK_SPEED / TARGET_FPS  # seconds per frame
        self._last_frame_time = time.perf_counter()   # for wall-clock timing

        # Y-axis range based on the global min/max so it doesn't jump around
        all_vals = np.concatenate(list(self.values.values()))
        self.y_min = np.nanmin(all_vals)
        self.y_max = np.nanmax(all_vals)
        margin = 0.05 * (self.y_max - self.y_min)
        self.y_min -= margin
        self.y_max += margin

        self._build_figure()

    # ------------------------------------------------------------------ UI
    def _build_figure(self):
        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(13, 7.5), facecolor=BG)

        # Layout from top to bottom:
        #   main plot (0.26 .. 0.94)
        #   status text strip (around 0.20)
        #   seek slider (0.13 .. 0.16)
        #   play/pause/restart buttons (0.03 .. 0.09)
        self.ax = self.fig.add_axes((0.07, 0.26, 0.90, 0.68))
        self.ax.set_facecolor(BG)
        self.ax.set_xlim(0, WINDOW_SECONDS)
        self.ax.set_ylim(self.y_min, self.y_max)
        self.ax.set_xlabel("Time within rolling window (s)",
                           fontsize=10, color=MUTED)
        self.ax.set_ylabel("Amplitude (µV)", fontsize=10, color=MUTED)
        self.ax.grid(True, color=GRID, alpha=0.5, linewidth=0.5)
        self.ax.tick_params(colors=MUTED, labelsize=9)
        for spine in self.ax.spines.values():
            spine.set_color(GRID)

        # One Line2D per channel, will be updated on each frame
        self.lines = []
        for i, c in enumerate(self.exg_cols):
            (ln,) = self.ax.plot([], [],
                                 color=CHANNEL_COLORS[i],
                                 linewidth=1.2,
                                 label=f"Ch {i}")
            self.lines.append(ln)

        self.ax.legend(ncol=8, fontsize=9, loc="upper right",
                       facecolor=BG, edgecolor=GRID, labelcolor=FG)

        # Title shows wall-clock position of the playhead
        self.title = self.ax.set_title(
            "OpenBCI playback — 00:00.00", fontsize=13, color=FG, pad=10)

        # Status text just below the plot
        self.status_text = self.fig.text(
            0.07, 0.20, "Paused — press Play to begin or drag the slider to seek",
            fontsize=10, color=MUTED, ha="left")

        # Seek slider: drag to jump to any time in the recording
        ax_slider = self.fig.add_axes((0.10, 0.13, 0.85, 0.025))
        ax_slider.set_facecolor(BTN_BG)
        self.slider = Slider(
            ax_slider, "Seek", 0.0, float(self.t[-1]),
            valinit=0.0, valstep=0.01,
            color="#4ec9ec", initcolor="none",
        )
        self.slider.label.set_color(FG)
        self.slider.label.set_fontsize(10)
        self.slider.valtext.set_color(FG)
        self.slider.valtext.set_fontsize(9)
        # Show the slider value as MM:SS instead of raw seconds
        self.slider.valtext.set_text(self._format_time(0.0))
        # Suppress the per-update redraw flicker
        self._slider_user_dragging = False
        self.slider.on_changed(self._on_slider_change)

        # Play/Pause/Restart buttons along the bottom
        ax_play    = self.fig.add_axes((0.30, 0.03, 0.10, 0.06))
        ax_pause   = self.fig.add_axes((0.45, 0.03, 0.10, 0.06))
        ax_restart = self.fig.add_axes((0.60, 0.03, 0.10, 0.06))

        self.btn_play    = Button(ax_play, "Play",
                                  color=BTN_BG, hovercolor=BTN_HOV)
        self.btn_pause   = Button(ax_pause, "Pause",
                                  color=BTN_BG, hovercolor=BTN_HOV)
        self.btn_restart = Button(ax_restart, "Restart",
                                  color=BTN_BG, hovercolor=BTN_HOV)

        for btn in (self.btn_play, self.btn_pause, self.btn_restart):
            btn.label.set_color(FG)
            btn.label.set_fontsize(10)

        self.btn_play.on_clicked(self._on_play)
        self.btn_pause.on_clicked(self._on_pause)
        self.btn_restart.on_clicked(self._on_restart)

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _format_time(seconds: float) -> str:
        mm = int(seconds // 60)
        ss = seconds - mm * 60
        return f"{mm:02d}:{ss:05.2f}"

    def _sync_slider_to_playhead(self):
        """Update slider position without firing its callback (avoids feedback)."""
        self._programmatic_slider_update = True
        try:
            self.slider.set_val(self.playback_time)
        finally:
            self._programmatic_slider_update = False
        self.slider.valtext.set_text(self._format_time(self.playback_time))

    # ------------------------------------------------------------- callbacks
    def _on_play(self, _event):
        if self.playback_time >= self.t[-1]:
            self.playback_time = 0.0   # auto-rewind if at end
        self.is_playing = True
        self._last_frame_time = time.perf_counter()   # reset clock anchor
        self.status_text.set_text("Playing")

    def _on_pause(self, _event):
        self.is_playing = False
        self.status_text.set_text("Paused")

    def _on_restart(self, _event):
        self.playback_time = 0.0
        self.status_text.set_text("Restarted — press Play")
        self.is_playing = False
        self._sync_slider_to_playhead()
        self._render_window()  # immediately repaint at the start

    def _on_slider_change(self, value: float):
        """User dragged the seek bar — jump the playhead there."""
        if getattr(self, "_programmatic_slider_update", False):
            return   # ignore changes we triggered ourselves
        self.playback_time = float(value)
        self._last_frame_time = time.perf_counter()   # reset clock anchor
        self.slider.valtext.set_text(self._format_time(self.playback_time))
        if not self.is_playing:
            self.status_text.set_text(
                f"Seeked to {self._format_time(self.playback_time)} (paused)")
        self._render_window()

    # ---------------------------------------------------------- frame update
    def _update(self, _frame):
        if self.is_playing:
            # Advance the playhead by the actual wall-clock time elapsed since
            # the last frame, scaled by PLAYBACK_SPEED. This makes playback
            # speed independent of rendering speed — if a frame takes 50 ms
            # instead of 16 ms, the playhead still advances by 50 ms so the
            # recording plays at real time instead of falling behind.
            now = time.perf_counter()
            elapsed = now - self._last_frame_time
            self._last_frame_time = now
            # Clamp to avoid huge jumps after long pauses or seeks
            elapsed = min(elapsed, 0.25)

            self.playback_time += elapsed * PLAYBACK_SPEED
            if self.playback_time >= self.t[-1]:
                self.playback_time = self.t[-1]
                self.is_playing = False
                self.status_text.set_text("End of recording")
            # Keep the slider thumb in sync as playback advances
            self._sync_slider_to_playhead()
        else:
            # Keep the clock anchor fresh while paused so resuming Play doesn't
            # cause the playhead to leap forward by the whole pause duration
            self._last_frame_time = time.perf_counter()
        self._render_window()
        return [*self.lines, self.title, self.status_text]

    def _render_window(self):
        """Slice the data to [playhead - WINDOW, playhead] and refresh lines."""
        t_end   = self.playback_time
        t_start = max(0.0, t_end - WINDOW_SECONDS)

        # searchsorted is O(log n) — much faster than a boolean mask, which
        # matters at 60 fps with a 5s window of ~1000+ samples per channel
        i_start = np.searchsorted(self.t, t_start, side="left")
        i_end   = np.searchsorted(self.t, t_end,   side="right")

        if i_end - i_start == 0:
            for ln in self.lines:
                ln.set_data([], [])
        else:
            t_rel = self.t[i_start:i_end] - t_start
            for i, c in enumerate(self.exg_cols):
                self.lines[i].set_data(t_rel, self.values[c][i_start:i_end])

        # Title shows playhead in MM:SS.cc relative to start of recording
        mm  = int(self.playback_time // 60)
        ss  = self.playback_time - mm * 60
        self.title.set_text(
            f"OpenBCI playback — {mm:02d}:{ss:05.2f}  "
            f"(speed {PLAYBACK_SPEED}×, window {WINDOW_SECONDS:.0f}s)")

    # ---------------------------------------------------------------- run
    def run(self):
        # interval is in milliseconds
        interval_ms = 1000.0 / TARGET_FPS
        self.anim = animation.FuncAnimation(
            self.fig,
            self._update,
            interval=interval_ms,
            blit=False,            # blit=True is faster but breaks the title
            cache_frame_data=False,
        )
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    src = sys.argv[1] if len(sys.argv) > 1 else SRC
    if not Path(src).exists():
        sys.exit(f"Input file not found: {src}")

    print(f"Loading {src} ...")
    df = load_openbci(src)
    print(f"  loaded {len(df):,} rows")

    if START_TIME is not None:
        df = df[df["t_fmt"] >= START_TIME]
    if END_TIME is not None:
        df = df[df["t_fmt"] < END_TIME]
    df = df.reset_index(drop=True)
    print(f"  using {len(df):,} rows for playback")
    print(f"  duration: {df['t_epoch'].iloc[-1] - df['t_epoch'].iloc[0]:.1f} s")

    player = RealtimePlayer(df)

    # Apply the requested initial playhead position, if any
    if INITIAL_TIME is not None:
        if isinstance(INITIAL_TIME, pd.Timestamp):
            # Translate the wall-clock timestamp into seconds-since-start-of-file
            first_ts = df["t_fmt"].iloc[0]
            target_s = (INITIAL_TIME - first_ts).total_seconds()
        else:
            target_s = float(INITIAL_TIME)
        target_s = max(0.0, min(target_s, float(player.t[-1])))
        player.playback_time = target_s
        player._sync_slider_to_playhead()
        player._render_window()
        player.status_text.set_text(
            f"Ready at {player._format_time(target_s)} — press Play")
        print(f"  starting playhead at {player._format_time(target_s)}")

    player.run()


if __name__ == "__main__":
    main()