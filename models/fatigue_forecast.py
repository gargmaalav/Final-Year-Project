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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import forecast_lstm  # noqa: E402  predict_delta

# Measured leave-one-subject-out MAE of the LSTM forecaster, in Hz, at the
# horizons it was validated on (models/FORECAST_VALIDATION.md). Used as the
# prediction band, because an empirical out-of-subject error is a far better
# statement of uncertainty than an OLS band computed from residuals about a
# line that does not actually extrapolate.
_LSTM_MAE_HZ = {10.0: 2.91, 30.0: 3.32, 60.0: 4.08}
# MAE -> sd assuming roughly normal errors, then a 95% interval.
_MAE_TO_95PC = 1.2533 * 1.96


def _lstm_mae(h: np.ndarray) -> np.ndarray:
    """Expected absolute error at each horizon, linearly extrapolated past 60 s
    so uncertainty keeps growing where the model was never validated."""
    xs = np.array(sorted(_LSTM_MAE_HZ))
    ys = np.array([_LSTM_MAE_HZ[x] for x in xs])
    slope = (ys[-1] - ys[0]) / (xs[-1] - xs[0])
    return np.where(h <= xs[-1], np.interp(h, xs, ys),
                    ys[-1] + slope * (h - xs[-1]))


def _trend_summary(fit: dict, horizon_sec: float) -> str:
    """Describe the observed trend, then the projection.

    The two come from different places and the sentence keeps them separate:
    the slope/R^2/p describe what the MDF actually did (OLS, valid), while the
    projected number comes from the trained forecaster where one is loaded.
    """
    slope_per_min = fit["slope"] * 60.0
    direction = ("rising" if fit["slope"] > 0.001 else
                "falling" if fit["slope"] < -0.001 else "flat")
    sig = ("significant" if fit["p_value"] < 0.05
           else "not statistically significant")
    end_val = float(fit["y_future"][-1])
    lo, hi = float(fit["pi_lo"][-1]), float(fit["pi_hi"][-1])

    observed = (
        f"Median frequency (the fatigue marker) is {direction} at "
        f"{slope_per_min:+.2f} Hz/min over this recording ({sig}, "
        f"R^2={fit['r2']:.2f}, p={fit['p_value']:.3f}).")

    if fit.get("method") == "lstm":
        change = end_val - float(fit["lstm"]["mdf_now"])
        return (
            f"{observed} A sequence model trained on the other subjects "
            f"projects about {end_val:.1f} Hz in {horizon_sec:.0f}s "
            f"({change:+.1f} Hz from now, 95% range {lo:.1f}-{hi:.1f} Hz).")

    # No trained forecaster available, so this falls back to extrapolating the
    # OLS line -- which is only defensible when the slope is real, and even
    # then it is measurably worse than assuming no change beyond ~10 s.
    if fit["p_value"] >= 0.05:
        return (f"{observed} No reliable projection can be made from a trend "
                "indistinguishable from flat.")
    return (
        f"{observed} Extrapolating that line gives about {end_val:.1f} Hz "
        f"in {horizon_sec:.0f}s (95% range {lo:.1f}-{hi:.1f} Hz), but a "
        "straight-line projection is unreliable beyond about 10s.")


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

    # The OLS slope/R^2/p above are a valid DESCRIPTION of the observed trend
    # and are kept. Its extrapolation is not: measured leave-one-subject-out,
    # projecting that line is 28% worse than assuming no change at a 30 s
    # horizon and 51% worse at 60 s (both p < 0.01). So where the trained
    # forecaster is available, it replaces the projected values only.
    fit["method"] = "ols"
    pred = forecast_lstm.predict_delta(seg, fs, t_end=t_end)
    if pred is not None:
        t_now, mdf_now = pred["t_now"], pred["mdf_now"]
        # anchor delta = 0 at h = 0; np.interp holds the last trained horizon
        # flat beyond 60 s, which is deliberate -- nothing beat "no further
        # change" at long range, so the model should not invent one.
        hs = np.array([0.0] + list(pred["horizons_sec"]))
        ds = np.array([0.0] + list(pred["delta_hz"]))
        t_future = np.linspace(t_now, t_now + horizon_sec, fit["t_future"].size)
        h = t_future - t_now
        y_future = mdf_now + np.interp(h, hs, ds)
        # Both bands are the measured out-of-subject error, not an OLS residual
        # band: the inner one is the typical (mean absolute) error, the outer
        # one the 95% interval implied by it.
        mae = _lstm_mae(h)
        fit["t_future"] = t_future
        fit["y_future"] = y_future
        fit["ci_lo"], fit["ci_hi"] = y_future - mae, y_future + mae
        fit["pi_lo"] = y_future - mae * _MAE_TO_95PC
        fit["pi_hi"] = y_future + mae * _MAE_TO_95PC
        fit["method"] = "lstm"
        fit["lstm"] = pred

    # MDF is a frequency: clamp the line and both bands at the physical floor
    clipped = bool(np.min(fit["y_future"]) < 0.0)
    for key in ("y_future", "ci_lo", "ci_hi", "pi_lo", "pi_hi"):
        fit[key] = np.maximum(fit[key], 0.0)

    # the actual measured MDF, so a chart can show the data the trend line was
    # fitted to and the point the forecast is anchored at, rather than only two
    # lines that now come from two different models and need not meet
    fit["t_observed"] = t_centers
    fit["y_observed"] = mean_mdf

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
