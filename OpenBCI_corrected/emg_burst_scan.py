"""Where is the real bicep-flexion EMG, and did the pipeline label it valid?
Scans the whole recording for EMG-band (20-100 Hz) power over time and checks
whether the high-EMG moments fall inside the pipeline's 'valid' segments."""
import numpy as np, pandas as pd
from scipy.signal import welch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

CH = [f"EXG Channel {i}" for i in range(8)]
df = pd.read_csv("cleaned_output/openbci_cleaned_full.csv",
                 usecols=["t_rel_s"] + CH + ["all_8_channels_valid"])
t = df["t_rel_s"].to_numpy()
x = df[CH].to_numpy()
fs = (len(t) - 1) / (t[-1] - t[0])
print(f"rows={len(t)}  dur={t[-1]-t[0]:.1f}s  effective_fs={fs:.1f}Hz")

WIN = 1.0  # s
step = int(fs * WIN)
centers, low_p, emg_p = [], [], []
for s in range(0, len(x) - step, step):
    seg = x[s:s+step]
    if not np.all(np.isfinite(seg)):
        continue
    lo, eg = [], []
    for c in range(8):
        f, p = welch(seg[:, c] - seg[:, c].mean(), fs=fs, nperseg=min(step, 256))
        lo.append(p[(f >= 0) & (f < 5)].sum())
        eg.append(p[(f >= 20) & (f <= 100)].sum())
    centers.append(t[s] + WIN/2); low_p.append(np.mean(lo)); emg_p.append(np.mean(eg))
centers, low_p, emg_p = map(np.array, (centers, low_p, emg_p))

# valid segment intervals
summ = pd.read_csv("classification_segments_output/combined_label_summary.csv")
valid_iv = summ[summ.class_name == "valid_pattern"][["start_s", "end_s"]].to_numpy()

def in_valid(tc):
    return any(a <= tc <= b for a, b in valid_iv)
mask_valid = np.array([in_valid(tc) for tc in centers])

print(f"\nEMG-band (20-100Hz) power, mean over 1s windows:")
print(f"  inside 'valid' segments: {emg_p[mask_valid].mean():.3g}  (n={mask_valid.sum()})")
print(f"  everywhere else:         {emg_p[~mask_valid].mean():.3g}  (n={(~mask_valid).sum()})")
print(f"\nEMG/low power ratio (higher = more EMG-like):")
ratio = emg_p / (low_p + 1e-9)
print(f"  inside 'valid': {np.median(ratio[mask_valid]):.4f} median")
print(f"  elsewhere:      {np.median(ratio[~mask_valid]):.4f} median")

# top EMG-band-power moments
order = np.argsort(emg_p)[::-1][:10]
print("\nTop 10 raw EMG-band-power 1s windows (mostly railing - check amplitudes):")
for i in order:
    print(f"  t={centers[i]:6.1f}s  emg_power={emg_p[i]:.3g}  in_valid={in_valid(centers[i])}")

# --- decisive check: is there real EMG in the genuinely clean (non-railed) data? ---
# A railed window fakes broadband power; only non-railed windows can show real EMG.
# Exclude the 50 Hz mains line (NZ powerline sits mid-EMG-band).
RAIL = 187500.0
print(f"\nSamples with >=1 channel pegged at the rail: "
      f"{np.mean(np.any(np.abs(x) > 0.9*RAIL, axis=1))*100:.1f}%")
clean_emg, clean_low, clean_mains, clean_t = [], [], [], []
for s in range(0, len(x) - step, step):
    seg = x[s:s+step]
    if np.mean(np.any(np.abs(seg) > 0.9*RAIL, axis=1)) >= 0.05:
        continue   # not clean
    eg, lo, mn = [], [], []
    for c in range(8):
        f, p = welch(seg[:, c] - seg[:, c].mean(), fs=fs, nperseg=min(step, 256))
        lo.append(p[(f >= 0) & (f < 5)].sum())
        eg.append(p[((f >= 20) & (f < 45)) | ((f >= 55) & (f <= 100))].sum())
        mn.append(p[(f >= 48) & (f <= 52)].sum())
    clean_t.append(t[s] + WIN/2); clean_emg.append(np.mean(eg))
    clean_low.append(np.mean(lo)); clean_mains.append(np.mean(mn))
clean_emg, clean_low, clean_mains, clean_t = map(np.array, (clean_emg, clean_low, clean_mains, clean_t))
cratio = clean_emg / (clean_low + 1e-9)
print(f"\nCLEAN windows (rail<5%): n={len(clean_emg)} of {len(centers)}")
print(f"  median EMG/low ratio = {np.median(cratio):.3f}  (drift dominates if <<1)")
print("  Top clean windows by mains-excluded EMG-band power:")
for i in np.argsort(clean_emg)[::-1][:6]:
    print(f"    t={clean_t[i]:6.1f}s  emg={clean_emg[i]:.3g}  low={clean_low[i]:.3g}  "
          f"mains50Hz={clean_mains[i]:.3g}  emg/low={cratio[i]:.3f}")

fig, ax = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
ax[0].semilogy(centers, emg_p, lw=0.8, label="EMG band 20-100Hz")
ax[0].semilogy(centers, low_p, lw=0.8, alpha=0.6, label="low band <5Hz")
for a, b in valid_iv:
    ax[0].axvspan(a, b, color="green", alpha=0.25)
ax[0].legend(); ax[0].set_ylabel("mean band power"); ax[0].set_title("Band power over recording (green = 'valid' segments)")
ax[1].plot(centers, ratio, lw=0.8, color="purple")
for a, b in valid_iv:
    ax[1].axvspan(a, b, color="green", alpha=0.25)
ax[1].set_ylabel("EMG/low ratio"); ax[1].set_xlabel("time (s)")
fig.tight_layout(); fig.savefig("emg_burst_scan.png", dpi=110)
print("\nSaved emg_burst_scan.png")
