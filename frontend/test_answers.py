"""Regression set for how answers are built.

    python frontend/test_answers.py

test_understanding.py checks that a question is understood. This checks the
other half: that the facts handed to the model say the right thing. No model,
no dataset, no network -- every case constructs a result dict by hand and
inspects the text produced from it, so it runs instantly and deterministically.

These exist because the wrong answers this project has actually shipped were
not phrasing slips. They were facts that were correct in isolation and wrong
in combination: a comparison that handed over two raw hertz values with no
shared reference, and an overview that described a falling median frequency as
falling fatigue. Both passed every test there was at the time, because the
tests only covered question parsing.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import prompt          # noqa: E402


def _result(mdf, fresh, label=1, conf=0.9, duration=200.0, t_start=180.0):
    return {"mdf_hz": mdf, "fresh_mdf": fresh, "fatigue_label": label,
            "confidence": conf, "duration": duration, "t_start": t_start,
            "drop_percent": (fresh - mdf) / fresh * 100.0 if fresh else None}


def _summary(mdf_start, mdf_end, onset, self_calibrated=False, subject=13):
    return {
        "subject": None if self_calibrated else subject,
        "side": None if self_calibrated else "R",
        "name": "the uploaded recording" if self_calibrated else None,
        "self_calibrated": self_calibrated,
        "duration": 200.0, "step": 2.5, "n_readings": 80,
        "mdf_start": mdf_start, "mdf_end": mdf_end,
        "mdf_drop": mdf_start - mdf_end,
        "mdf_min": min(mdf_start, mdf_end), "mdf_max": max(mdf_start, mdf_end),
        "fraction_fatigued": 0.6, "onset": onset, "readings": [],
    }


def _onset(found=True, t_start=67.0, **kw):
    base = {"found": found, "t_start": t_start, "confidence": 0.9,
            "mdf_hz": 70.0, "sustain": 2, "step": 2.5, "typical_error": 6.0,
            "error_measured": True, "self_calibrated": False,
            "baseline_sec": None, "fraction_fatigued": 0.6,
            "fatigued_from_start": False, "inside_baseline": False}
    base.update(kw)
    return base


CHECKS = []


def check(name):
    def wrap(fn):
        CHECKS.append((name, fn))
        return fn
    return wrap


# --- the comparison verdict -------------------------------------------------
# Given two rows of similar-looking numbers the 3B model reported "8% and 4%"
# for a real 21% and 4%, inventing one of them. The determination is therefore
# computed here, and these pin it.

@check("the bigger drop is named as more fatigued")
def _():
    facts = prompt.compare_facts({
        "kind": "subjects", "subjects": [5, 9], "side": "R", "t_start": None,
        "fraction": 0.9, "durations": {5: 191.0, 9: 200.0}, "clamped": [],
        "short": [],
        "results": {5: _result(70.0, 73.3), 9: _result(45.0, 62.0)}})
    verdict = [f for f in facts if f.startswith("CONCLUSION")]
    assert len(verdict) == 1, facts
    # subject 9 fell 27%, subject 5 fell 4%
    assert "subject 9 is further" in verdict[0], verdict[0]
    assert verdict[0].index("subject 9") < verdict[0].index("subject 5"), verdict[0]


@check("a near-tie is called a tie, not a winner")
def _():
    facts = prompt.compare_facts({
        "kind": "subjects", "subjects": [5, 9], "side": "R", "t_start": None,
        "fraction": 0.9, "durations": {5: 191.0, 9: 200.0}, "clamped": [],
        "short": [],
        "results": {5: _result(70.0, 73.3), 9: _result(59.5, 62.0)}})
    verdict = [f for f in facts if f.startswith("CONCLUSION")][0]
    assert "similar" in verdict, verdict
    assert "is further from" not in verdict, "named a winner inside the noise"


@check("an upload needs a wider gap before a winner is called")
def _():
    # 21% vs 24% -- decisive between two measured baselines, not decisive when
    # one of them is assumed from the recording's own opening seconds
    comparison = {
        "kind": "upload_vs_subject", "subject": 1, "side": "R", "fraction": 0.9,
        "upload": _result(59.0, 74.7), "subject_result": _result(53.8, 70.8),
        "short": False}
    verdict = [f for f in prompt.compare_facts(comparison)
               if f.startswith("CONCLUSION")][0]
    assert "similar" in verdict, verdict


@check("the assumed baseline is only ever attributed to the upload")
def _():
    # The model stated this backwards, crediting the dataset subject with the
    # assumed baseline. The caveat must name the upload as the assumed one.
    facts = prompt.compare_facts({
        "kind": "upload_vs_subject", "subject": 5, "side": "R", "fraction": 0.9,
        "upload": _result(59.0, 74.7), "subject_result": _result(70.0, 73.3),
        "short": False})
    caveat = [f for f in facts if f.startswith("CAVEAT")][0].lower()
    assert "your recording's" in caveat and "assumed" in caveat, caveat
    assert caveat.index("your recording") < caveat.index("subject 5"), caveat
    assert "not assumed" in caveat, caveat


# --- direction of change ----------------------------------------------------
# "median frequency fell" was read as "fatigue fell" -- the exact inversion.

@check("a falling median frequency is stated as rising fatigue")
def _():
    facts = prompt.overview_facts(_summary(75.0, 57.0, _onset()))
    change = [f for f in facts if "FELL" in f][0]
    assert "fatigue INCREASED" in change, change
    assert "Never describe this as a fall or reduction in fatigue" in change


@check("a rising median frequency is not dressed up as fatigue")
def _():
    facts = prompt.overview_facts(_summary(57.0, 75.0, _onset()))
    change = [f for f in facts if "ROSE" in f][0]
    assert "does not show the usual fatigue trend" in change, change


# --- onset honesty ----------------------------------------------------------

@check("a dataset onset quotes the measured error, not the scan step")
def _():
    facts = prompt.onset_facts(_summary(75.0, 57.0, _onset()))
    text = " ".join(facts)
    assert "within about 6s" in text, text
    assert "2s of the labelled" not in text, "quoted the scan step as accuracy"


@check("an uploaded onset quotes no error figure at all")
def _():
    facts = prompt.onset_facts(_summary(
        75.0, 57.0, _onset(error_measured=False, self_calibrated=True,
                           baseline_sec=60.0),
        self_calibrated=True))
    text = " ".join(facts)
    assert "do not quote any error figure" in text, text
    assert "within about" not in text, "quoted a dataset error for an upload"
    assert "assuming the person started unfatigued" in text, text


@check("an onset inside the baseline window is flagged as untrustworthy")
def _():
    facts = prompt.onset_facts(_summary(
        75.0, 57.0, _onset(t_start=30.0, error_measured=False,
                           self_calibrated=True, baseline_sec=60.0,
                           inside_baseline=True),
        self_calibrated=True))
    warning = [f for f in facts if f.startswith("WARNING")]
    assert warning and "should not be trusted" in warning[0], facts


@check("fatigued-from-the-start is reported as that, not as a time")
def _():
    facts = prompt.onset_facts(_summary(
        75.0, 57.0, _onset(found=False, fatigued_from_start=True)))
    text = " ".join(facts)
    assert "no onset to report" in text, text
    assert "first appears and holds" not in text, text


@check("an upload is named by its file, a subject by number and arm")
def _():
    assert prompt._whose(_summary(75.0, 57.0, _onset())) == "subject 13, right arm"
    assert prompt._whose(_summary(75.0, 57.0, _onset(), self_calibrated=True)) \
        == "the uploaded recording"


# --- the offline fallback ---------------------------------------------------
# When Ollama is unreachable the facts are shown directly, so the instructions
# aimed at the model must not be shown with them.

@check("model-facing instructions are stripped from the offline fallback")
def _():
    facts = prompt.compare_facts({
        "kind": "upload_vs_subject", "subject": 5, "side": "R", "fraction": 0.9,
        "upload": _result(59.0, 74.7), "subject_result": _result(70.0, 73.3),
        "short": False})
    shown = " ".join(prompt.readable_facts(facts))
    for leak in ("Do not reverse this", "Never say the dataset",
                 "CONCLUSION --", "CAVEAT --", "IMPORTANT:",
                 "Address them as", "NOTE (phrasing"):
        assert leak not in shown, f"leaked {leak!r}:\n{shown}"
    # ...while the measurements themselves survive
    assert "down 21%" in shown and "down 5%" in shown, shown
    assert "is further from their own fresh level" in shown, shown


@check("the fallback keeps the direction-of-change warning readable")
def _():
    shown = " ".join(prompt.readable_facts(
        prompt.overview_facts(_summary(75.0, 57.0, _onset()))))
    assert "FELL by 18.0 Hz" in shown, shown
    assert "Never describe" not in shown, shown


def main() -> int:
    passed, failures = 0, []
    for name, fn in CHECKS:
        try:
            fn()
            passed += 1
        except AssertionError as e:
            failures.append(f"  {name}\n     {e}")
        except Exception as e:
            failures.append(f"  {name}\n     {type(e).__name__}: {e}")

    if failures:
        print("FAILURES:")
        print("\n".join(failures))
    print(f"\n{passed} passed, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
