"""Make a supervisor-facing 'the amp is railed' figure."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

CH = [f"EXG Channel {i}" for i in range(8)]
df = pd.read_csv("cleaned_output/openbci_cleaned_full.csv", usecols=["t_rel_s"] + CH)
t = df["t_rel_s"].to_numpy(); x = df[CH].to_numpy()
RAIL = 187500.0
fs = (len(t) - 1) / (t[-1] - t[0])

# per-1s window: fraction of samples with >=1 channel pegged, and max |amplitude|
step = int(fs)
wt, railfrac, maxamp = [], [], []
for s in range(0, len(x) - step, step):
    seg = x[s:s+step]
    wt.append(t[s] + 0.5)
    railfrac.append(np.mean(np.any(np.abs(seg) > 0.9*RAIL, axis=1)))
    maxamp.append(np.max(np.abs(seg)))
wt, railfrac, maxamp = map(np.array, (wt, railfrac, maxamp))
overall = np.mean(np.any(np.abs(x) > 0.9*RAIL, axis=1)) * 100

fig, ax = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True)

ax[0].plot(wt, maxamp / 1000, lw=0.7, color="#1f3b73")
ax[0].axhline(RAIL/1000, color="crimson", ls="--", lw=1.3, label=f"hardware rail (+/-{RAIL/1000:.1f} mV)")
ax[0].axhline(-0)  # baseline
ax[0].set_ylabel("max |amplitude|\nper second (mV)")
ax[0].set_title(f"OpenBCI recording: the amplifier is saturated for {overall:.0f}% of the session",
                fontsize=13, fontweight="bold")
ax[0].legend(loc="upper right", fontsize=9)
ax[0].margins(x=0)

ax[1].fill_between(wt, railfrac*100, color="crimson", alpha=0.55, step="mid")
ax[1].axhline(overall, color="black", ls=":", lw=1.1,
              label=f"overall: {overall:.1f}% of samples railed")
ax[1].set_ylabel("% of samples\nat the rail")
ax[1].set_xlabel("time (s)")
ax[1].set_ylim(0, 100)
ax[1].legend(loc="upper right", fontsize=9)
ax[1].margins(x=0)

fig.tight_layout()
fig.savefig("railing_figure.png", dpi=130)
print(f"overall railed {overall:.1f}%  -> railing_figure.png")
