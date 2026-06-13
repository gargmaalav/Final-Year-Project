"""
Runner: biceps fatigue + frequency-curve forecast on the Zenodo sEMG set.
========================================================================

Drop-in replacement for the broken OpenBCI capture. Loads one biceps trial,
computes the MDF-decline trend (the fatigue signature) and runs the proven
`core.spectrum_backtest` / `core.spectrum_forward` frequency-curve forecast
on the biceps channel, all at the data's true 1259 Hz.

Usage:
    python run_biceps.py --root /path/to/sEMG_data --subject 5 --side R
    python run_biceps.py --root /path/to/sEMG_data --subject 5 --side R --target-fs 250
        (the --target-fs form downsamples to mimic the OpenBCI rig.)

Output: zenodo_biceps/out/S{n}_{side}_biceps.png + printed match scores.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loader  # noqa: E402
import core    # noqa: E402  (made importable by loader)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="path to sEMG_data/")
    ap.add_argument("--subject", type=int, default=5)
    ap.add_argument("--side", choices=["R", "L"], default="R")
    ap.add_argument("--target-fs", type=int, default=None,
                    help="downsample to this rate (e.g. 250 to mimic OpenBCI)")
    ap.add_argument("--win", type=float, default=4.0, help="window sec (protocol=4)")
    ap.add_argument("--step", type=float, default=2.0, help="step sec (protocol=2)")
    args = ap.parse_args()

    seg = loader.load_biceps_segment(
        args.root, args.subject, args.side,
        target_fs=args.target_fs, bandpass=True)
    fs = int(getattr(seg, "eff_fs", loader.FS_NATIVE))
    muscle = loader.TRIAL_TO_MUSCLE[loader.BICEPS_TRIAL[args.side]]
    print(f"loaded {muscle}  subject {args.subject}  "
          f"fs={fs}Hz  dur={seg.data.shape[0]/fs:.1f}s  n={seg.data.shape[0]}")

    # 1) fatigue signature: MDF decline over the recording
    t_mdf, mean_mdf, _ = loader.mdf_trend(seg, fs=fs, win_sec=args.win, step_sec=args.step)
    if t_mdf.size >= 3:
        slope = np.polyfit(t_mdf, mean_mdf, 1)[0]
        print(f"MDF: start~{mean_mdf[0]:.1f}Hz  end~{mean_mdf[-1]:.1f}Hz  "
              f"slope={slope*60:+.2f} Hz/min  (fatigue => negative)")
        # sanity gate the advisor asked for: fresh biceps MDF ~ 80-120 Hz
        if not (40 <= mean_mdf[0] <= 200):
            print(f"  WARNING: first MDF {mean_mdf[0]:.1f}Hz outside 40-200Hz "
                  f"-- suspect an fs scaling bug, trace before trusting.")

    # 2) frequency-curve forecast (the supervisor's deliverable), at the true fs
    bt = core.spectrum_backtest(
        [seg], train_frac=0.70, win_sec=args.win, step_sec=args.step, fs=fs)
    freqs, train_sh, pred_sh, actual_sh, meta = bt
    if meta.get("n_test"):
        print(f"forecast backtest: match={meta['match_pct']:.1f}%  "
              f"baseline={meta['baseline_match_pct']:.1f}%  "
              f"MDF train/pred/actual = "
              f"{meta['mdf_train']:.1f}/{meta['mdf_pred']:.1f}/{meta['mdf_actual']:.1f} Hz")

    # fatigue ground-truth markers, if the label tree is present
    trans_t, fatig_t = loader.fatigue_onsets(
        *loader.load_fatigue_labels(args.root, args.subject, args.side))

    # ---- figure ----
    os.makedirs(OUT, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(t_mdf, mean_mdf, color="#f5a623", lw=1.6)
    ax[0].set_title(f"{muscle}  S{args.subject}  MDF trend ({fs}Hz)")
    ax[0].set_xlabel("s"); ax[0].set_ylabel("MDF (Hz)"); ax[0].grid(alpha=.3)
    for mt, lab, col in ((trans_t, "non-fatigue end", "k"),
                         (fatig_t, "transition end", "r")):
        if mt is not None:
            ax[0].axvline(mt, ls="--", color=col, lw=1, label=lab)
    if trans_t is not None or fatig_t is not None:
        ax[0].legend(fontsize=8)

    if actual_sh.size:
        ax[1].plot(freqs, train_sh, label="train shape", color="#888")
        ax[1].plot(freqs, pred_sh, label="predicted", color="#4ec9ec", lw=1.8)
        ax[1].plot(freqs, actual_sh, label="actual future", color="#3ddc97", lw=1.8)
        ax[1].set_xlim(0, min(fs / 2, 250))
    ax[1].set_title("frequency-shape forecast (held-out future)")
    ax[1].set_xlabel("Hz"); ax[1].set_ylabel("share of power (%)")
    ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)

    fig.tight_layout()
    png = os.path.join(OUT, f"S{args.subject}_{args.side}_biceps.png")
    fig.savefig(png, dpi=120)
    print(f"saved {png}")


if __name__ == "__main__":
    main()
