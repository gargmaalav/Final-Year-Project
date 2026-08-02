"""Tests for the classify()/classify_upload() contract and its guards.

Run:
    python models/test_classify.py          # needs the dataset + fatigue_model.pt
    python models/test_classify.py --fast   # skips the dataset-backed tests

These exist because the self-calibrated upload path shipped without any test
scoring it against the labelled subjects, and it was measurably wrong: a 15 s
baseline dropped accuracy from 84% to 75% and flipped 30% of window labels
while keeping ~93% confidence. Anything that changes normalisation should have
to get past test_upload_tracks_calibrated_path().
"""
from __future__ import annotations

import dataclasses
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO_ROOT, "zenodo_biceps"))

import classify as C           # noqa: E402
import fatigue_forecast as F   # noqa: E402
import forecast_lstm as FL     # noqa: E402
import loader                  # noqa: E402

DATA_ROOT = os.path.join(REPO_ROOT, "zenodo_biceps", "sEMG_data")
_failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        _failures.append(name)


def _synthetic_segment(duration_sec: float, fs: int = 250, seed: int = 0):
    """Band-limited noise shaped like EMG, no dataset required."""
    rng = np.random.default_rng(seed)
    n = int(duration_sec * fs)
    x = rng.standard_normal(n) * 1e-4
    return loader.to_segment(np.arange(n) / fs, x, fs=fs, target_fs=None,
                             bandpass=True)


# ---------------------------------------------------------------------------
# guards -- these do not need the dataset
# ---------------------------------------------------------------------------
def test_short_recording_is_refused():
    """A clip shorter than FRESH_SEC * MIN_BASELINE_FRACTION cannot produce a
    trustworthy baseline: it would be normalised against itself and always read
    'average'. It must raise, not answer."""
    bundle, _ = C._load()
    cfg, feats = bundle["config"], bundle["base_feats"]
    for duration in (8.0, 20.0, 60.0):
        seg = _synthetic_segment(duration)
        try:
            C.compute_fresh_baseline(seg, 250, cfg, feats)
            check(f"{duration:.0f}s clip refused", False, "it returned a baseline")
        except ValueError:
            check(f"{duration:.0f}s clip refused", True)


def test_long_recording_is_accepted():
    bundle, _ = C._load()
    cfg, feats = bundle["config"], bundle["base_feats"]
    seg = _synthetic_segment(400.0)
    bl = C.compute_fresh_baseline(seg, 250, cfg, feats)
    check("400s clip accepted", bl["n_windows"] >= C.MIN_FRESH_WINDOWS,
          f"{bl['n_windows']} baseline windows")
    check("baseline reports provenance", bl["source"] == "computed-from-recording")


def test_upload_result_flags_calibration():
    seg = _synthetic_segment(400.0)
    result, _ = C.classify_upload(seg, 250, 100.0)
    check("upload result carries contract keys",
          {"mdf_hz", "fatigue_label", "confidence"} <= set(result))
    check("upload result flags self-calibration",
          result.get("calibration", {}).get("kind") == "self-calibrated")


def test_forecast_never_goes_negative():
    """An unbounded OLS line predicted -106 Hz at a 1 hour horizon."""
    seg = _synthetic_segment(400.0)
    for horizon in (60.0, 1800.0, 7200.0):
        fit = F.forecast_fatigue(seg, 250, horizon_sec=horizon)
        if not fit.get("ok"):
            continue
        lo = min(float(np.min(fit["y_future"])), float(np.min(fit["pi_lo"])))
        check(f"horizon {horizon:.0f}s stays >= 0 Hz", lo >= 0.0, f"min {lo:.1f} Hz")
    fit = F.forecast_fatigue(seg, 250, horizon_sec=7200.0)
    check("horizon is capped", fit.get("horizon_sec") == F.MAX_HORIZON_SEC,
          f"{fit.get('horizon_sec')}s")


# ---------------------------------------------------------------------------
# dataset-backed -- the ones that actually score the model
# ---------------------------------------------------------------------------
def test_classify_contract():
    result = C.classify(13, 60.0, "R")
    check("classify returns the contract keys",
          {"mdf_hz", "fatigue_label", "confidence"} <= set(result))
    check("fatigue_label is a valid class", result["fatigue_label"] in (0, 1))
    check("confidence is a probability", 0.0 <= result["confidence"] <= 1.0)
    check("mdf is in the muscle band", 10.0 < result["mdf_hz"] < 125.0,
          f"{result['mdf_hz']:.1f} Hz")


def test_forecast_is_anchored_to_t_start():
    """Fitting on the whole recording and projecting from its end, while
    reporting a classification for t=20s, describes two different moments."""
    seg = loader.load_biceps_segment(DATA_ROOT, 13, "R", target_fs=250, bandpass=True)
    fs = int(getattr(seg, "eff_fs", 250))
    early = F.forecast_fatigue(seg, fs, horizon_sec=20.0, t_end=60.0)
    late = F.forecast_fatigue(seg, fs, horizon_sec=20.0)
    check("anchored forecast starts at the queried time",
          abs(float(early["t_future"][0]) - 60.0) <= 4.0,
          f"starts at {float(early['t_future'][0]):.0f}s")
    check("unanchored forecast starts at the recording end",
          float(late["t_future"][0]) > float(early["t_future"][0]) + 60.0)


def test_forecaster_cannot_see_the_future():
    """The single most damaging bug a forecaster can have.

    Asking for a forecast at t=90s must give the same answer whether or not
    the recording happens to continue past 90s. If truncating the input
    changes the prediction, the model is reading future windows and every
    accuracy number it produces is fiction.
    """
    if FL.load_deployed() is None:
        check("forecaster is causal", True, "no model trained, skipped")
        return
    seg = loader.load_biceps_segment(DATA_ROOT, 13, "R", target_fs=250, bandpass=True)
    fs = int(getattr(seg, "eff_fs", 250))
    t_end = 90.0

    full = FL.predict_delta(seg, fs, t_end=t_end)
    # Slice the Segment directly rather than rebuilding it through
    # to_segment(): re-interpolating onto a fresh grid can shift the last
    # window by a sample and would make this test fail on an artifact of its
    # own setup rather than on real leakage.
    n = int(round((t_end + 2.0) * fs))
    trunc = dataclasses.replace(seg, data=seg.data[:n], t=seg.t[:n],
                                gap_mask=seg.gap_mask[:n])
    cut = FL.predict_delta(trunc, fs, t_end=t_end)

    check("forecaster runs on truncated input", full is not None and cut is not None)
    if full is None or cut is None:
        return
    gap = float(np.max(np.abs(np.array(full["delta_hz"]) - np.array(cut["delta_hz"]))))
    check("forecast is unchanged by removing the future", gap < 0.01,
          f"max difference {gap:.4f} Hz")


def test_forecast_uses_the_trained_model():
    seg = loader.load_biceps_segment(DATA_ROOT, 13, "R", target_fs=250, bandpass=True)
    fs = int(getattr(seg, "eff_fs", 250))
    fit = F.forecast_fatigue(seg, fs, horizon_sec=30.0, t_end=120.0)
    expected = "lstm" if FL.load_deployed() is not None else "ols"
    check("forecast reports which method produced it",
          fit.get("method") == expected, f"method={fit.get('method')}")
    if fit.get("method") != "lstm":
        return
    # A 30 s projection that moves MDF by more than the whole observed spread
    # is not a forecast, it is an extrapolation blowing up.
    move = abs(float(fit["y_future"][-1]) - float(fit["lstm"]["mdf_now"]))
    check("30s projection stays physically plausible", move < 15.0,
          f"{move:.1f} Hz of change")


def test_upload_tracks_calibrated_path():
    """The self-calibrated path must stay close to the calibrated one.

    This is the regression guard for the baseline window. Measured on all 13
    labelled subjects: 15 s -> 70% label agreement, 60 s -> 90%. If someone
    shortens FRESH_SEC again, this fails.
    """
    bundle, model = C._load()
    cfg, feats = bundle["config"], bundle["base_feats"]
    agreements = []
    for subject in (11, 12, 13):
        seg = loader.load_biceps_segment(DATA_ROOT, subject, "R",
                                         target_fs=cfg["target_fs"], bandpass=True)
        fs = int(getattr(seg, "eff_fs", 250))
        bl = bundle["baselines"].get(subject) or bundle["baselines"].get(str(subject))
        mu_s, sd_s = np.array(bl["mu"], float), np.array(bl["sd"], float)
        fresh = C.compute_fresh_baseline(seg, fs, cfg, feats)
        mu_f, sd_f = np.array(fresh["mu"], float), np.array(fresh["sd"], float)

        same = total = 0
        for t_start in np.arange(10.0, float(seg.t[-1]) - 5.0, 10.0):
            a = C._classify_window(seg, fs, float(t_start), mu_s, sd_s, cfg, feats, model)
            b = C._classify_window(seg, fs, float(t_start), mu_f, sd_f, cfg, feats, model)
            same += a["fatigue_label"] == b["fatigue_label"]
            total += 1
        agreements.append(same / max(total, 1))
        print(f"        subject {subject}: {same}/{total} windows agree")

    mean_agreement = float(np.mean(agreements))
    check("self-calibrated agrees with calibrated >= 80% of windows",
          mean_agreement >= 0.80, f"{mean_agreement * 100:.1f}%")


def main() -> int:
    fast = "--fast" in sys.argv
    print("guards (no dataset needed):")
    test_short_recording_is_refused()
    test_long_recording_is_accepted()
    test_upload_result_flags_calibration()
    test_forecast_never_goes_negative()

    if fast:
        print("\nskipping dataset-backed tests (--fast)")
    elif not os.path.isdir(DATA_ROOT):
        print(f"\nskipping dataset-backed tests: {DATA_ROOT} not found")
    else:
        print("\ndataset-backed:")
        test_classify_contract()
        test_forecast_is_anchored_to_t_start()
        test_forecaster_cannot_see_the_future()
        test_forecast_uses_the_trained_model()
        test_upload_tracks_calibrated_path()

    print()
    if _failures:
        print(f"{len(_failures)} FAILED: {', '.join(_failures)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
