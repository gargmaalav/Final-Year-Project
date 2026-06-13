"""
Generate a 3-panel summary figure across all 13 subjects for team presentation.
Panels: MDF start/end, slope, forecast vs baseline match%.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Data from the batch run
subjects = list(range(1, 14))
mdf_start = [70.8, 66.3, 66.5, 79.5, 103.8, 108.8, 87.8, 105.3, 73.3, 73.0, 84.3, 55.5, 73.3]
mdf_end   = [54.5, 47.8, 64.5, 55.3,  82.8, 106.0,  96.5,  88.8, 68.0, 50.5, 78.3, 60.0, 56.3]
slope     = [-8.63, -10.34, +0.57, -6.05, -4.55, -10.30, +0.13, -2.62, -0.86, -5.34, -2.45, -1.29, -3.15]
forecast  = [89.3, 88.9, 91.2, 90.8, 90.3, 72.8, 92.1, 91.5, 90.4, 88.4, 91.4, 92.0, 90.1]
baseline  = [88.4, 87.3, 94.4, 89.4, 93.4, 82.5, 95.3, 95.2, 93.7, 87.8, 95.9, 94.6, 93.6]
dur_s     = [178.9, 177.2, 283.7, 255.8, 191.1, 24.5, 324.9, 348.6, 260.3, 257.0, 491.0, 343.6, 218.1]

xs = np.array(subjects)
ORANGE = "#f5a623"
TEAL   = "#4ec9ec"
GREEN  = "#3ddc97"
RED    = "#e05c5c"
GRAY   = "#aaaaaa"

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Zenodo biceps sEMG — all 13 subjects, R biceps, 1259 Hz", fontsize=13, fontweight="bold")

# Panel 1: MDF start vs end
ax = axes[0]
ax.plot(xs, mdf_start, "o-", color=ORANGE, label="MDF start", lw=1.6, ms=6)
ax.plot(xs, mdf_end,   "s--", color=RED,    label="MDF end",   lw=1.6, ms=6)
ax.set_title("Median Frequency: start vs end")
ax.set_xlabel("Subject"); ax.set_ylabel("MDF (Hz)")
ax.set_xticks(xs); ax.grid(alpha=.3); ax.legend()

# Panel 2: MDF slope (fatigue => negative)
ax = axes[1]
colors = [RED if s < 0 else GRAY for s in slope]
bars = ax.bar(xs, slope, color=colors, edgecolor="none")
ax.axhline(0, color="k", lw=0.8)
ax.set_title("MDF slope (negative = fatigue)")
ax.set_xlabel("Subject"); ax.set_ylabel("Hz / min")
ax.set_xticks(xs); ax.grid(alpha=.3, axis="y")
# flag S6 (outlier - only 24.5s)
ax.annotate("S6\n(24s)", xy=(6, slope[5]), xytext=(6, slope[5] - 1.8),
            ha="center", fontsize=7, color="gray")

# Panel 3: forecast match vs naive baseline
ax = axes[2]
ax.bar(xs - 0.2, forecast,  0.35, color=TEAL,  label="forecast match%",  edgecolor="none")
ax.bar(xs + 0.2, baseline,  0.35, color=GRAY,  label="naive baseline%",   edgecolor="none")
ax.set_title("Frequency forecast vs naive baseline")
ax.set_xlabel("Subject"); ax.set_ylabel("Spectral match (%)")
ax.set_ylim(60, 100); ax.set_xticks(xs); ax.grid(alpha=.3, axis="y"); ax.legend()

fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "team_summary.png")
fig.savefig(out, dpi=130)
print(f"saved {out}")

# Print a quick table for the team
print("\n--- Summary table ---")
print(f"{'S':>3} {'dur':>7} {'MDF start':>9} {'MDF end':>7} {'slope':>7} {'forecast':>9} {'baseline':>9} {'beats?':>7}")
for i, s in enumerate(subjects):
    beats = "YES" if forecast[i] > baseline[i] else "-"
    print(f"{s:>3} {dur_s[i]:>7.1f}s {mdf_start[i]:>9.1f} {mdf_end[i]:>7.1f} "
          f"{slope[i]:>+7.2f} {forecast[i]:>9.1f}% {baseline[i]:>9.1f}% {beats:>7}")
n_neg = sum(1 for s in slope if s < 0)
n_beats = sum(1 for i in range(13) if forecast[i] > baseline[i])
print(f"\n{n_neg}/13 show negative slope (fatigue). Forecast beats baseline in {n_beats}/13 subjects.")
