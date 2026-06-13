"""
OpenBCI 8-channel EXG per-second visualisation - RAW VERSION
============================================================

DEMONSTRATION SCRIPT: this version plots the data as it appears in the
text file, with NO mean-centering and NO rail-value filtering. It exists
purely to show why those two steps are needed in the main pipeline.

Compare the output of this script with openbci_per_second.py to see the
difference. In this script:
  - All 8 channels are plotted at their absolute voltage
  - Samples at +/-187,500 microvolts (electrode saturation) are kept
  - Ch 0-3 will sit near -150,000 microvolts (their DC offset)
  - Ch 4-5 will sit near +180,000 microvolts (their DC offset)
  - Ch 6-7 will appear as wild swings between the rails
  - Any small muscle signal is invisible because the +/-300,000 microvolt
    baseline difference dominates the y-axis

Usage:
    python openbci_per_second_raw.py /path/to/OpenBCI-RAW-2026-03-21_15-37-11.txt

Or edit SRC below and run with no arguments.

Saves:
    - openbci_per_second_RAW_combined.png   (all 7 panels in one figure)
    - openbci_sec_RAW_01.png ... openbci_sec_RAW_07.png   (individual panels)
And displays the combined figure in an interactive window.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SRC = "OpenBCI-RAW-2026-03-21_15-37-11.txt"   # default; override via CLI arg
OUT_DIR = Path(".")

WIN_START = pd.Timestamp("2026-03-21 15:42:28")
WIN_END   = pd.Timestamp("2026-03-21 15:42:35")

# Dark theme colours matching the user's reference screenshot
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
BG = "#0d1117"
GRID = "#2a3441"
FG = "#e6edf3"
MUTED = "#8b949e"


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_openbci(path: str) -> pd.DataFrame:
    """Load an OpenBCI raw .txt export into a tidy DataFrame."""
    df = pd.read_csv(path, comment="%", skipinitialspace=True, engine="python")
    df.columns = [c.strip() for c in df.columns]

    df["t_epoch"] = pd.to_numeric(df["Timestamp"], errors="coerce")
    df["t_fmt"]   = pd.to_datetime(df["Timestamp (Formatted)"], errors="coerce")
    df = df.dropna(subset=["t_epoch", "t_fmt"]).reset_index(drop=True)

    for c in [f"EXG Channel {i}" for i in range(8)]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def slice_window(df: pd.DataFrame,
                 start: pd.Timestamp,
                 end: pd.Timestamp) -> pd.DataFrame:
    """Restrict to [start, end). No rail filter applied."""
    return df[(df["t_fmt"] >= start) & (df["t_fmt"] < end)].copy()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_one_second_raw(ax, chunk: pd.DataFrame, title: str) -> None:
    """Draw all 8 channels for one 1-second chunk - RAW VALUES (no centering)."""
    exg = [f"EXG Channel {i}" for i in range(8)]

    t = chunk["t_epoch"].to_numpy()
    if len(t) == 0:
        ax.set_facecolor(BG)
        ax.text(0.5, 0.5, "no samples", ha="center", va="center",
                color=MUTED, transform=ax.transAxes)
        ax.set_title(title, fontsize=11, color=FG, pad=8)
        return
    t_ms = (t - t[0]) * 1000.0   # ms within the panel

    for i, c in enumerate(exg):
        y = chunk[c].to_numpy(dtype=float)
        # NO MEAN-CENTERING - plot the raw values exactly as in the file
        ax.plot(t_ms, y, "o-",
                color=CHANNEL_COLORS[i],
                linewidth=1.2, markersize=2.5,
                alpha=0.9, label=f"Ch {i}")

    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, alpha=0.6, linewidth=0.5)
    ax.set_title(title, fontsize=11, color=FG, pad=8)
    ax.set_xlabel("Time within window (ms)", fontsize=9, color=MUTED)
    ax.set_ylabel("Amplitude, RAW (µV)", fontsize=9, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)

    # Annotate the saturation rails so the audience can spot them
    ax.axhline(y=187500, color="#ff4444", linestyle=":", linewidth=0.7,
               alpha=0.5, label="_nolegend_")
    ax.axhline(y=-187500, color="#ff4444", linestyle=":", linewidth=0.7,
               alpha=0.5, label="_nolegend_")


def build_combined_figure(per_sec_chunks):
    """4x2 grid (7 used, 1 hidden) with all 1-second panels."""
    plt.style.use("dark_background")
    fig, axes = plt.subplots(4, 2, figsize=(14, 14), facecolor=BG)
    axes = axes.flatten()

    fig.suptitle(
        f"OpenBCI 8-channel EXG  ·  {WIN_START:%H:%M:%S} – {WIN_END:%H:%M:%S}"
        f"  ·  RAW (no mean-centering, no rail filter)",
        fontsize=14, color=FG, y=0.995,
    )

    for i, (label, chunk) in enumerate(per_sec_chunks):
        plot_one_second_raw(axes[i], chunk, label)
        if i == 0:
            axes[i].legend(ncol=4, fontsize=8, loc="upper right",
                           facecolor=BG, edgecolor=GRID, labelcolor=FG)

    for j in range(len(per_sec_chunks), len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def build_single_figure(label: str, chunk: pd.DataFrame):
    """One standalone 1-second panel as its own figure."""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
    plot_one_second_raw(ax, chunk, label)
    ax.legend(ncol=4, fontsize=8, loc="upper right",
              facecolor=BG, edgecolor=GRID, labelcolor=FG)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    src = sys.argv[1] if len(sys.argv) > 1 else SRC
    if not Path(src).exists():
        sys.exit(f"Input file not found: {src}")

    print(f"Loading {src} ...")
    df = load_openbci(src)
    print(f"  total rows: {len(df):,}")

    w = slice_window(df, WIN_START, WIN_END)
    print(f"  rows in {WIN_START.time()}–{WIN_END.time()} (no rail filter): {len(w):,}")

    # Bin into 1-second chunks
    w["sec_idx"] = ((w["t_fmt"] - WIN_START).dt.total_seconds()).astype(int)

    per_sec_chunks = []
    for s in sorted(w["sec_idx"].unique()):
        chunk = w[w["sec_idx"] == s]
        if len(chunk) < 1:
            continue
        start = WIN_START + pd.Timedelta(seconds=int(s))
        end   = WIN_START + pd.Timedelta(seconds=int(s) + 1)
        label = (f"Sec {s+1}/7   {start:%H:%M:%S} → {end:%H:%M:%S}"
                 f"   (n={len(chunk)}, RAW)")
        per_sec_chunks.append((label, chunk))
        print(f"    {label}")

    # Combined figure
    combined = build_combined_figure(per_sec_chunks)
    combined_path = OUT_DIR / "openbci_per_second_RAW_combined.png"
    combined.savefig(combined_path, dpi=140, facecolor=BG)
    print(f"\nWrote {combined_path}")

    # Individual figures
    for i, (label, chunk) in enumerate(per_sec_chunks, start=1):
        fig = build_single_figure(label, chunk)
        p = OUT_DIR / f"openbci_sec_RAW_{i:02d}.png"
        fig.savefig(p, dpi=140, facecolor=BG)
        plt.close(fig)
        print(f"Wrote {p}")

    # Show interactively
    plt.show()


if __name__ == "__main__":
    main()