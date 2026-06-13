"""
Frequency forecast over the long clean segments.
================================================

The supervisor's meeting note: "plot the cleaned period dynamically (time series
+ frequency), then predict future visualisation of frequency given the patterns
we find." This module is the "predict" half, done properly.

For each LONG clean segment (>= MIN_FORECAST_SEC) it:
  1. tracks the median frequency (MDF, mean across the 8 channels) over the
     whole segment - tens of seconds of samples, not a 3-7s converged snippet;
  2. fits an ordinary-least-squares trend with R^2, a slope confidence interval
     and a slope significance test;
  3. projects the trend FORECAST_SEC beyond the data with both a 95% confidence
     band (uncertainty in the trend) and a wider 95% prediction band (where an
     individual future reading is expected).

Honest caveat (documented, by design): no filter is applied, so this MDF is the
raw-signal median (~0.7-4 Hz, dominated by baseline drift), not the muscle-band
fatigue frequency from the slides. The forecast machinery is correct; what it
forecasts is the unfiltered MDF. Add a 20-120 Hz bandpass in mdf_trend to
forecast the muscle-band marker instead.

Outputs (out/):
    forecast_<seg>.png        per-segment trend + forecast figure
    frequency_forecast.csv     slope / R^2 / projected value per segment
    forecast_summary.txt       human-readable summary

Run:  python run.py forecast
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core import (
    FS, forecast_regression, load_clean_segments, mdf_trend,
)

DATA_FILE = "../data/OpenBCI-RAW-2026-03-21_15-37-11.txt"
OUT = Path("out")
MIN_FORECAST_SEC = 30.0     # only segments with enough data to forecast properly
FORECAST_SEC = 20.0         # how far beyond the data to project

BG, GRID, FG, MUTED = "#0d1117", "#2a3441", "#e6edf3", "#8b949e"
DATA_COL, FIT_COL, PROJ_COL = "#ffd93d", "#4ec9ec", "#a78bfa"


def _plot(seg, mt, mmdf, fit: dict) -> Path:
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=BG)
    ax.set_facecolor(BG)

    # observed MDF
    ax.plot(mt, mmdf, color=DATA_COL, lw=1.0, alpha=0.5, zorder=2)
    ax.scatter(mt, mmdf, s=10, color=DATA_COL, alpha=0.8, zorder=3,
               label="observed MDF (mean of 8 ch)")
    # fitted trend over the data
    ax.plot(fit["t_fit"], fit["y_fit"], color=FIT_COL, lw=2.0, zorder=4,
            label=f"OLS trend  ({fit['slope']:+.3f} Hz/s, R²={fit['r2']:.2f})")
    # projection beyond the data
    ax.plot(fit["t_future"], fit["y_future"], color=PROJ_COL, lw=2.0, ls="--",
            zorder=4, label=f"forecast (+{FORECAST_SEC:.0f}s)")
    ax.fill_between(fit["t_future"], fit["pi_lo"], fit["pi_hi"],
                    color=PROJ_COL, alpha=0.12,
                    label=f"{fit['conf']*100:.0f}% prediction band")
    ax.fill_between(fit["t_future"], fit["ci_lo"], fit["ci_hi"],
                    color=PROJ_COL, alpha=0.30,
                    label=f"{fit['conf']*100:.0f}% confidence band")
    ax.axvline(mt[-1], color=FG, lw=1.0, alpha=0.6, zorder=1)
    ax.text(mt[-1], ax.get_ylim()[1], "  end of data ->", color=MUTED,
            fontsize=9, va="top")

    ax.set_title(
        f"Frequency forecast - Seg {seg.idx}  "
        f"{seg.wall_at(0)} (+{seg.dur:.0f}s clean)   "
        f"raw-signal MDF, unfiltered",
        color=FG, fontsize=13, pad=10)
    ax.set_xlabel("Seconds within segment", color=MUTED, fontsize=12)
    ax.set_ylabel("Median frequency (Hz)", color=MUTED, fontsize=12)
    ax.grid(True, color=GRID, alpha=0.5, lw=0.5)
    ax.tick_params(colors=MUTED, labelsize=11)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.legend(fontsize=9, facecolor=BG, edgecolor=GRID, labelcolor=FG,
              loc="upper left")
    fig.tight_layout()
    p = OUT / f"forecast_seg{seg.idx}.png"
    fig.savefig(p, dpi=140, facecolor=BG)
    plt.close(fig)
    return p


def run(data_file: str = DATA_FILE) -> dict:
    OUT.mkdir(exist_ok=True)
    print(f"Loading {data_file} ...")
    segments = load_clean_segments(data_file)
    longs = [s for s in segments if s.dur >= MIN_FORECAST_SEC]
    print(f"  {len(segments)} clean segments; {len(longs)} long enough "
          f"(>= {MIN_FORECAST_SEC:.0f}s) to forecast\n")

    rows = []
    for seg in longs:
        mt, mmdf, _ = mdf_trend(seg)
        fit = forecast_regression(mt, mmdf, FORECAST_SEC)
        if not fit.get("ok"):
            print(f"  Seg {seg.idx}: too few windows, skipped")
            continue
        png = _plot(seg, mt, mmdf, fit)
        end_val = float(fit["y_future"][-1])
        pi_lo, pi_hi = float(fit["pi_lo"][-1]), float(fit["pi_hi"][-1])
        rows.append({
            "segment": seg.idx,
            "dur_s": round(seg.dur, 1),
            "mdf_mean_hz": round(float(np.mean(mmdf)), 3),
            "slope_hz_per_s": round(fit["slope"], 4),
            "slope_ci_lo": round(fit["slope_ci"][0], 4),
            "slope_ci_hi": round(fit["slope_ci"][1], 4),
            "r2": round(fit["r2"], 3),
            "p_value": round(fit["p_value"], 4),
            f"forecast_+{FORECAST_SEC:.0f}s_hz": round(end_val, 3),
            "pred_lo": round(pi_lo, 3),
            "pred_hi": round(pi_hi, 3),
        })
        trend = ("rising" if fit["slope"] > 0 else "falling"
                 if fit["slope"] < 0 else "flat")
        sig = "significant" if fit["p_value"] < 0.05 else "not significant"
        print(f"  Seg {seg.idx} ({seg.dur:5.1f}s): MDF {trend} "
              f"{fit['slope']:+.3f} Hz/s (R²={fit['r2']:.2f}, {sig}) "
              f"-> {png.name}")

    # ---- CSV -----------------------------------------------------------
    csv_path = OUT / "frequency_forecast.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {csv_path}")

    # ---- summary -------------------------------------------------------
    summary = OUT / "forecast_summary.txt"
    with open(summary, "w") as f:
        f.write("FREQUENCY FORECAST - LONG CLEAN SEGMENTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Source     : {data_file}\n")
        f.write(f"Method     : OLS linear regression of MDF(t) per segment,\n")
        f.write(f"             95% confidence + prediction bands, projected "
                f"+{FORECAST_SEC:.0f}s.\n")
        f.write(f"Segments   : {len(longs)} long ones (>= {MIN_FORECAST_SEC:.0f}s)"
                f" of {len(segments)} clean.\n")
        f.write(f"Sample rate: {FS} Hz (Sample-Index-proven).\n")
        f.write("Caveat     : UNFILTERED -> MDF is the raw-signal median (~0.7-4\n")
        f.write("             Hz, baseline drift), not muscle-band fatigue freq.\n")
        f.write("             Add a 20-120 Hz bandpass in mdf_trend for the\n")
        f.write("             muscle-band marker. The regression itself is sound.\n")
        f.write("=" * 60 + "\n\n")
        for r in rows:
            f.write(f"  Seg {r['segment']} ({r['dur_s']}s): "
                    f"slope {r['slope_hz_per_s']:+.3f} Hz/s "
                    f"[{r['slope_ci_lo']:+.3f}, {r['slope_ci_hi']:+.3f}], "
                    f"R²={r['r2']}, p={r['p_value']}\n")
            f.write(f"        forecast +{FORECAST_SEC:.0f}s: "
                    f"{r[f'forecast_+{FORECAST_SEC:.0f}s_hz']} Hz "
                    f"(95% PI {r['pred_lo']}..{r['pred_hi']})\n")
    print(f"  summary {summary}")
    return {"segments_forecast": len(rows)}


if __name__ == "__main__":
    run()
