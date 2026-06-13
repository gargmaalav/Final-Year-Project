"""F1 discriminator: is "all 8 channels correlated" real signal or common-mode artifact?
Compares PSD of a 'valid' (high-corr) segment vs an 'invalid' segment.
Power piled <5-10 Hz across all channels = drift/motion (artifact).
Broadband structured content = plausibly real neuromuscular signal.
"""
import glob, os
import numpy as np
import pandas as pd
from scipy.signal import welch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CH = [f"EXG Channel {i} Cleaned uV" for i in range(8)]
VALID_DIR = "classification_segments_output/valid"
INVALID_DIR = "classification_segments_output/invalid"

def load(path):
    df = pd.read_csv(path)
    t = df["t_rel_s"].to_numpy()
    x = df[CH].to_numpy()  # samples x 8
    # effective fs from average rate over the segment (t_rel_s is quantized,
    # so per-sample diff is unreliable; n/duration is the true mean rate)
    fs = (len(t) - 1) / (t[-1] - t[0])
    return t, x, fs

def lowfrac(f, p, cutoff):
    return p[f <= cutoff].sum() / p.sum()

def analyse(path, label):
    t, x, fs = load(path)
    dur = t[-1] - t[0]
    # pairwise corr across 8 channels
    c = np.corrcoef(x.T)
    iu = np.triu_indices(8, 1)
    mean_corr = c[iu].mean()
    print(f"\n=== {label}: {os.path.basename(path)} ===")
    print(f"  duration={dur:.1f}s  n={len(t)}  effective_fs={fs:.1f}Hz  mean_pairwise_corr={mean_corr:.3f}")
    nperseg = min(512, len(t))
    psd_each = []
    for i in range(8):
        f, p = welch(x[:, i] - x[:, i].mean(), fs=fs, nperseg=nperseg)
        psd_each.append(p)
        print(f"  ch{i}: <5Hz={lowfrac(f,p,5)*100:5.1f}%  <10Hz={lowfrac(f,p,10)*100:5.1f}%  <20Hz={lowfrac(f,p,20)*100:5.1f}%  peak@{f[np.argmax(p)]:.1f}Hz")
    psd_mean = np.mean(psd_each, axis=0)
    print(f"  MEAN over 8ch: <5Hz={lowfrac(f,psd_mean,5)*100:.1f}%  <10Hz={lowfrac(f,psd_mean,10)*100:.1f}%  <20Hz={lowfrac(f,psd_mean,20)*100:.1f}%")
    return t, x, fs, f, psd_each, mean_corr, label

valid_files = sorted(glob.glob(os.path.join(VALID_DIR, "*.csv")))
invalid_files = sorted(glob.glob(os.path.join(INVALID_DIR, "*.csv")))
# pick highest-corr valid (most suspect) and a non-railed invalid if possible
results = []
results.append(analyse(valid_files[0], "VALID"))
# choose an invalid segment; pick the smallest (least likely all-railed) for a fair look
inv = min(invalid_files, key=lambda p: os.path.getsize(p))
results.append(analyse(inv, "INVALID"))

# Plot: waveforms (left) + PSD log (right), valid top / invalid bottom
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for r, (t, x, fs, f, psd_each, mc, lab) in enumerate(results):
    tt = t - t[0]
    n = min(len(tt), int(fs * 4))  # first 4s
    for i in range(8):
        axes[r, 0].plot(tt[:n], x[:n, i], lw=0.6)
    axes[r, 0].set_title(f"{lab} waveform (first 4s)  mean_corr={mc:.2f}")
    axes[r, 0].set_xlabel("s"); axes[r, 0].set_ylabel("uV")
    for i in range(8):
        axes[r, 1].semilogy(f, psd_each[i], lw=0.7)
    axes[r, 1].set_title(f"{lab} PSD (8 channels)")
    axes[r, 1].set_xlabel("Hz"); axes[r, 1].set_ylabel("PSD"); axes[r, 1].set_xlim(0, min(fs/2, 100))
fig.tight_layout()
out = "f1_spectral_check.png"
fig.savefig(out, dpi=110)
print(f"\nSaved {out}")
