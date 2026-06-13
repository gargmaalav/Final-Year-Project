"""
Headless converged-section detector.
====================================

Finds the sections where all 8 EMG channels look similar (converge) inside the
clean period, and writes durable evidence that needs no screen:

    out/converged_sections.csv    ranked table (segment, wall time, length, sim)
    out/convergence_summary.txt   human-readable summary
    out/section_<rank>_*.png      normalized 8-channel snapshot of each top run

A "converged section" is a contiguous stretch where the mean of the 28 pairwise
Pearson correlations across the 8 channels stays >= CONV_THRESHOLD (0.80) for at
least MIN_RUN_SEC. No noise filtering; FS = 250 Hz (proven from the Sample
Index). See core.py for the full rationale.

Run:  python run.py detect
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # no display needed for the headless detector
import matplotlib.pyplot as plt
import numpy as np

from core import (
    CHANNEL_COLORS, CONV_THRESHOLD, FS, MIN_RUN_SEC, N_CH,
    ConvergedRun, Segment, all_converged_runs, converged_runs,
    load_clean_segments, scan_segment,
)

DATA_FILE = "../data/OpenBCI-RAW-2026-03-21_15-37-11.txt"
OUT = Path("out")
TOP_GLOBAL = 6                   # how many top sections get a PNG snapshot


def _snapshot(seg: Segment, run: ConvergedRun, rank: int) -> Path:
    """Save a normalized 8-channel view of one converged section."""
    i0 = max(0, int(round(run.t_start * FS)))
    i1 = min(seg.n, int(round(run.t_end * FS)))
    w = seg.data[i0:i1]
    t = seg.t[i0:i1]
    z = (w - w.mean(axis=0)) / (w.std(axis=0) + 1e-9)   # per-channel z-score

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 5), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    for c in range(N_CH):
        ax.plot(t, z[:, c], color=CHANNEL_COLORS[c], lw=1.2, alpha=0.9,
                label=f"Ch {c}")
    ax.set_title(
        f"Converged section #{rank}  -  Seg {seg.idx}  "
        f"{seg.wall_at(run.t_start)} - {seg.wall_at(run.t_end)}  "
        f"({run.dur:.1f}s, mean sim {run.mean_score:.3f})",
        color="#e6edf3", fontsize=12, pad=10,
    )
    ax.set_xlabel("Time within segment (s)", color="#8b949e")
    ax.set_ylabel("Normalized amplitude (z)", color="#8b949e")
    ax.grid(True, color="#2a3441", alpha=0.6, lw=0.5)
    ax.legend(ncol=8, fontsize=8, loc="upper right",
              facecolor="#0d1117", edgecolor="#2a3441", labelcolor="#e6edf3")
    ax.tick_params(colors="#8b949e")
    fig.tight_layout()
    p = OUT / f"section_{rank:02d}_seg{seg.idx}_{run.dur:.0f}s.png"
    fig.savefig(p, dpi=140, facecolor="#0d1117")
    plt.close(fig)
    return p


def run(data_file: str = DATA_FILE) -> dict:
    OUT.mkdir(exist_ok=True)
    print(f"Loading {data_file} ...")
    segments = load_clean_segments(data_file)
    total = sum(s.dur for s in segments)
    print(f"  {len(segments)} clean segments, {total:.1f}s total "
          f"(FS={FS} Hz, no filtering)\n")

    runs = all_converged_runs(segments)          # [(seg, ConvergedRun), ...]

    def _union_dur(seg_runs):
        intervals = sorted((r.t_start, r.t_end) for r in seg_runs)
        if not intervals:
            return 0.0
        merged = [list(intervals[0])]
        for s, e in intervals[1:]:
            if s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        return sum(e - s for s, e in merged)

    from itertools import groupby
    conv_total = sum(
        _union_dur([r for _, r in grp])
        for _, grp in groupby(runs, key=lambda sr: sr[0].idx)
    )

    # per-segment console line
    for seg in segments:
        rr = converged_runs(seg)
        t, corr, _ = scan_segment(seg)
        peak = corr.max() if corr.size else float("nan")
        print(f"  {seg.label()}  | peak sim {peak:.3f} | "
              f"{len(rr)} converged section(s), {_union_dur(rr):.1f}s")

    # ---- ranked CSV (longest section first) -----------------------------
    ranked = sorted(runs, key=lambda sr: sr[1].dur, reverse=True)
    csv_path = OUT / "converged_sections.csv"
    with open(csv_path, "w", newline="") as f:
        cols = ["rank", "segment", "wall_start", "wall_end", "duration_s",
                "mean_sim", "peak_sim"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for rank, (seg, r) in enumerate(ranked, start=1):
            w.writerow({
                "rank": rank, "segment": seg.idx,
                "wall_start": seg.wall_at(r.t_start),
                "wall_end": seg.wall_at(r.t_end),
                "duration_s": round(r.dur, 2),
                "mean_sim": round(r.mean_score, 4),
                "peak_sim": round(r.peak_score, 4),
            })
    print(f"\nWrote {csv_path}  ({len(ranked)} converged sections, "
          f"{conv_total:.1f}s)")

    # ---- PNG snapshots of the top sections ------------------------------
    for rank, (seg, r) in enumerate(ranked[:TOP_GLOBAL], start=1):
        print(f"  snapshot {_snapshot(seg, r, rank)}")

    # ---- text summary ---------------------------------------------------
    summary = OUT / "convergence_summary.txt"
    with open(summary, "w") as f:
        f.write("8-CHANNEL EMG CONVERGENCE - DETECTED SECTIONS\n")
        f.write("=" * 62 + "\n")
        f.write(f"Source      : {data_file}\n")
        f.write(f"Sample rate : {FS} Hz, proven from the Cyton Sample Index wrap\n")
        f.write("              rate (534 wraps x 256 / 547.5s = 250 Hz). The file\n")
        f.write("              header says 1000 Hz and the old pipeline trusted it,\n")
        f.write("              which made every MDF/MNF value 4x too high.\n")
        f.write("Filtering   : NONE. Convergence = mean of the 28 pairwise Pearson\n")
        f.write("              correlations across 8 channels in a 1s window;\n")
        f.write("              Pearson removes the DC offset (not filtering).\n")
        f.write(f"Definition  : a section is contiguous time with mean sim >= "
                f"{CONV_THRESHOLD:.2f}\n")
        f.write(f"              lasting >= {MIN_RUN_SEC:.0f}s.\n")
        f.write(f"Clean period: {len(segments)} segments, {total:.1f}s. "
                f"Converged: {conv_total:.1f}s ({100*conv_total/total:.0f}%)"
                f" (union of overlapping sections).\n")
        f.write("Caveat      : with no filter, 70-99% of power is < 5 Hz (baseline\n")
        f.write("              drift), so the channels converge mainly on slow\n")
        f.write("              wander and raw MDF reads ~0.7-4 Hz. Add a 20-120 Hz\n")
        f.write("              bandpass if muscle-band convergence is wanted.\n")
        f.write("=" * 62 + "\n\n")
        f.write("TOP CONVERGED SECTIONS (longest first):\n\n")
        for rank, (seg, r) in enumerate(ranked[:TOP_GLOBAL], start=1):
            f.write(f"  #{rank}  Seg {seg.idx}  "
                    f"{seg.wall_at(r.t_start)} - {seg.wall_at(r.t_end)}  "
                    f"{r.dur:5.1f}s  mean sim {r.mean_score:.3f}\n")
        f.write("\nPER-SEGMENT:\n\n")
        for seg in segments:
            rr = converged_runs(seg)
            f.write(f"  Seg {seg.idx} ({seg.dur:5.1f}s): {len(rr)} section(s), "
                    f"{sum(x.dur for x in rr):5.1f}s converged\n")
    print(f"  summary {summary}")

    return {"segments": len(segments), "sections": len(ranked),
            "converged_s": conv_total}


if __name__ == "__main__":
    run()
