"""
forecast_fatigue(seg, fs, horizon_sec) -> future MDF trend projection
========================================================================

(Named fatigue_forecast.py, not forecast.py, to avoid shadowing/being
shadowed by convergence_analysis/forecast.py -- both live on sys.path at
once once loader.py pulls convergence_analysis in.)

Reuses convergence_analysis.core.forecast_regression's OLS trend forecast
(slope, R^2, significance, confidence + prediction bands), applied to
loader.mdf_trend()'s bandpassed muscle-band MDF -- the same MDF this project
already computes for classification (models/classify.py) and for the chart
(viz/render_window.py). Works for any Segment: the 13 Zenodo subjects or a
user-uploaded recording (see models/classify.py's classify_upload()).

Not a new model -- forecast_regression already existed, built by Rayyan for
this exact "predict future frequency" ask, but only ever run previously on
unfiltered legacy OpenBCI data (convergence_analysis/FINDINGS.md documents
that MDF there tracked baseline drift, not muscle fatigue, since no bandpass
was applied). Applying the same regression to correctly-filtered muscle-band
MDF is the fix, not a rewrite.
"""
from __future__ import annotations

import os
import sys

import numpy as np

# A straight-line MDF trend stops being meaningful long before this, but the
# cap is what stops "will I be tired in an hour?" returning a negative
# frequency. 180 s is roughly the longest projection the observed trends
# support without the prediction band swallowing the whole plausible range.
MAX_HORIZON_SEC = 180.0

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO_ROOT, "zenodo_biceps"),
           os.path.join(_REPO_ROOT, "convergence_analysis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import loader  # noqa: E402  mdf_trend
import core    # noqa: E402  forecast_regression


def _trend_summary(fit: dict, horizon_sec: float) -> str:
    slope_per_min = fit["slope"] * 60.0
    direction = ("rising" if fit["slope"] > 0.001 else
                "falling" if fit["slope"] < -0.001 else "flat")
    sig = ("significant" if fit["p_value"] < 0.05
           else "not statistically significant")
    end_val = float(fit["y_future"][-1])
    lo, hi = float(fit["pi_lo"][-1]), float(fit["pi_hi"][-1])
    # An insignificant slope means the trend is indistinguishable from flat --
    # projecting a number off it would present noise as a prediction.
    if fit["p_value"] >= 0.05:
        return (
            f"Median frequency (the fatigue marker) shows no statistically "
            f"significant trend over this recording "
            f"(slope {slope_per_min:+.2f} Hz/min, R^2={fit['r2']:.2f}, "
            f"p={fit['p_value']:.3f}), so no reliable projection can be made.")
    return (
        f"Median frequency (the fatigue marker) is {direction} at "
        f"{slope_per_min:+.2f} Hz/min ({sig}, R^2={fit['r2']:.2f}, "
        f"p={fit['p_value']:.3f}). Projected to be about {end_val:.1f} Hz "
        f"in {horizon_sec:.0f}s (95% range {lo:.1f}-{hi:.1f} Hz)."
    )


def forecast_fatigue(seg, fs: int, horizon_sec: float = 20.0,
                     win_sec: float = 4.0, step_sec: float = 2.0,
                     t_end: float | None = None) -> dict:
    """Project the MDF trend `horizon_sec` forward from `t_end`.

    t_end: fit the trend using only MDF history up to this time, and project
        forward from it. Defaults to the end of the recording. Pass the window
        the user actually asked about, otherwise the "forecast" is fitted on
        data from AFTER that moment and projected from the recording's end --
        two different points in time reported as if they were one.

    The horizon is capped at MAX_HORIZON_SEC and the projection is floored at
    0 Hz. This is an OLS straight line, so left unbounded it happily predicts
    negative median frequency (-106 Hz at a 1 hour horizon, measured). MDF is
    a spectral quantity and cannot go below zero; extrapolating a linear
    fatigue trend for many minutes is not physiologically meaningful anyway.

    Returns core.forecast_regression()'s dict (slope, r2, p_value, y_future,
    ci_lo/hi, pi_lo/hi, t_fit, t_future, ...) plus a "summary" plain-language
    sentence and "horizon_sec"/"clipped" fields, or {"ok": False} if there
    isn't enough MDF history (fewer than 3 windows) to fit a trend.
    """
    horizon_sec = float(max(0.0, min(horizon_sec, MAX_HORIZON_SEC)))

    t_centers, mean_mdf, _ = loader.mdf_trend(seg, fs=fs, win_sec=win_sec,
                                              step_sec=step_sec)
    if t_end is not None and t_centers.size:
        keep = t_centers <= float(t_end)
        t_centers, mean_mdf = t_centers[keep], mean_mdf[keep]

    fit = core.forecast_regression(t_centers, mean_mdf, horizon_sec)
    if not fit.get("ok"):
        return {"ok": False}

    # MDF is a frequency: clamp the line and both bands at the physical floor
    clipped = bool(np.min(fit["y_future"]) < 0.0)
    for key in ("y_future", "ci_lo", "ci_hi", "pi_lo", "pi_hi"):
        fit[key] = np.maximum(fit[key], 0.0)

    fit["horizon_sec"] = horizon_sec
    fit["clipped_at_zero"] = clipped
    fit["summary"] = _trend_summary(fit, horizon_sec)
    return fit


if __name__ == "__main__":
    # smoke test -- needs the Zenodo dataset present.
    import loader as _loader
    seg = _loader.load_biceps_segment(
        os.path.join(_REPO_ROOT, "zenodo_biceps", "sEMG_data"), 13, "R",
        target_fs=250, bandpass=True)
    fs = int(getattr(seg, "eff_fs", 250))
    result = forecast_fatigue(seg, fs, horizon_sec=20.0)
    print(result.get("summary", result))
