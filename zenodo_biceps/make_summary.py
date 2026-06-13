"""
Generate live summary outputs for the Zenodo biceps pipeline.

This intentionally recomputes the metrics from the extracted dataset rather than
using hard-coded tables. Outputs:
  - out/pipeline_summary[_250hz].csv
  - out/pipeline_summary[_250hz].json
  - out/team_summary[_250hz].png
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loader  # noqa: E402
import core    # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def summarize_subject(root: str, subject: int, side: str,
                      target_fs: int | None, win: float,
                      step: float) -> dict:
    seg = loader.load_biceps_segment(
        root, subject, side, target_fs=target_fs, bandpass=True)
    fs = int(getattr(seg, "eff_fs", loader.FS_NATIVE))
    duration = float(seg.data.shape[0] / fs)

    t_mdf, mean_mdf, _ = loader.mdf_trend(seg, fs=fs, win_sec=win,
                                          step_sec=step)
    if t_mdf.size >= 3:
        slope_hz_per_min = float(np.polyfit(t_mdf, mean_mdf, 1)[0] * 60.0)
        mdf_start = float(mean_mdf[0])
        mdf_end = float(mean_mdf[-1])
    else:
        slope_hz_per_min = float("nan")
        mdf_start = float("nan")
        mdf_end = float("nan")

    _, _, _, _, meta = core.spectrum_backtest(
        [seg], train_frac=0.70, win_sec=win, step_sec=step, fs=fs)

    return {
        "subject": subject,
        "side": side,
        "fs": fs,
        "duration_s": duration,
        "mdf_start_hz": mdf_start,
        "mdf_end_hz": mdf_end,
        "mdf_slope_hz_per_min": slope_hz_per_min,
        "forecast_match_pct": float(meta.get("match_pct", float("nan"))),
        "baseline_match_pct": float(meta.get("baseline_match_pct", float("nan"))),
        "beats_baseline": bool(
            meta.get("match_pct", -np.inf) > meta.get("baseline_match_pct", np.inf)),
        "mdf_train_hz": float(meta.get("mdf_train", float("nan"))),
        "mdf_pred_hz": float(meta.get("mdf_pred", float("nan"))),
        "mdf_actual_hz": float(meta.get("mdf_actual", float("nan"))),
    }


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(rows: list[dict], path: str, title: str) -> None:
    subjects = np.array([r["subject"] for r in rows])
    mdf_start = np.array([r["mdf_start_hz"] for r in rows])
    mdf_end = np.array([r["mdf_end_hz"] for r in rows])
    slope = np.array([r["mdf_slope_hz_per_min"] for r in rows])
    forecast = np.array([r["forecast_match_pct"] for r in rows])
    baseline = np.array([r["baseline_match_pct"] for r in rows])
    durations = np.array([r["duration_s"] for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(subjects, mdf_start, "o-", color="#f5a623",
            label="MDF start", lw=1.6, ms=6)
    ax.plot(subjects, mdf_end, "s--", color="#e05c5c",
            label="MDF end", lw=1.6, ms=6)
    ax.set_title("Median frequency: start vs end")
    ax.set_xlabel("Subject")
    ax.set_ylabel("MDF (Hz)")
    ax.set_xticks(subjects)
    ax.grid(alpha=.3)
    ax.legend()

    ax = axes[1]
    colors = ["#e05c5c" if v < 0 else "#aaaaaa" for v in slope]
    ax.bar(subjects, slope, color=colors, edgecolor="none")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("MDF slope (negative = fatigue)")
    ax.set_xlabel("Subject")
    ax.set_ylabel("Hz/min")
    ax.set_xticks(subjects)
    ax.grid(alpha=.3, axis="y")
    for s, d, v in zip(subjects, durations, slope):
        if d < 60:
            ax.annotate(f"S{s}\n({d:.0f}s)", xy=(s, v),
                        xytext=(s, v - 1.8), ha="center",
                        fontsize=7, color="gray")

    ax = axes[2]
    ax.bar(subjects - 0.2, forecast, 0.35, color="#4ec9ec",
           label="forecast match%", edgecolor="none")
    ax.bar(subjects + 0.2, baseline, 0.35, color="#aaaaaa",
           label="naive baseline%", edgecolor="none")
    ax.set_title("Frequency forecast vs naive baseline")
    ax.set_xlabel("Subject")
    ax.set_ylabel("Spectral match (%)")
    ax.set_ylim(max(0, np.nanmin([forecast, baseline]) - 10), 100)
    ax.set_xticks(subjects)
    ax.grid(alpha=.3, axis="y")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="sEMG_data")
    ap.add_argument("--side", choices=["R", "L"], default="R")
    ap.add_argument("--target-fs", type=int, default=None)
    ap.add_argument("--win", type=float, default=4.0)
    ap.add_argument("--step", type=float, default=2.0)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    rows = []
    for subject in range(1, 14):
        row = summarize_subject(args.root, subject, args.side,
                                args.target_fs, args.win, args.step)
        rows.append(row)
        beats = "YES" if row["beats_baseline"] else "-"
        print(
            f"S{subject:<2} dur={row['duration_s']:>6.1f}s "
            f"MDF {row['mdf_start_hz']:>6.1f}->{row['mdf_end_hz']:<6.1f} "
            f"slope={row['mdf_slope_hz_per_min']:>+6.2f} Hz/min "
            f"forecast={row['forecast_match_pct']:>5.1f}% "
            f"baseline={row['baseline_match_pct']:>5.1f}% beats={beats}"
        )

    fs = args.target_fs or loader.FS_NATIVE
    tag = f"_{fs}hz" if args.target_fs is not None else ""
    csv_path = os.path.join(OUT, f"pipeline_summary{tag}.csv")
    json_path = os.path.join(OUT, f"pipeline_summary{tag}.json")
    fig_path = os.path.join(OUT, f"team_summary{tag}.png")

    write_csv(csv_path, rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    save_figure(rows, fig_path,
                f"Zenodo biceps sEMG - {args.side} biceps, {fs} Hz")

    n_neg = sum(1 for r in rows if r["mdf_slope_hz_per_min"] < 0)
    n_beats = sum(1 for r in rows if r["beats_baseline"])
    print(f"\n{n_neg}/13 show negative slope (fatigue). "
          f"Forecast beats baseline in {n_beats}/13 subjects.")
    print(f"saved {csv_path}")
    print(f"saved {json_path}")
    print(f"saved {fig_path}")


if __name__ == "__main__":
    main()
