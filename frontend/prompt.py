"""
build_prompt(features, user_query) -> str
==========================================

The team integration contract (README.md):
  Produced by: Maalav.
  Build the LLM system prompt from pipeline features + the user's question.
  Constrain the LLM to only use the provided feature values -- no
  hallucinated numbers.

`features` is the dict returned by models/classify.py's classify():
  {"mdf_hz": float, "fatigue_label": int, "confidence": float,
   "fatigue_state": str}

`forecast` is optional: models/fatigue_forecast.py's forecast_fatigue() dict (must
have "ok": True and a "summary" string) if a trend projection is available
for this reading. Kept as an optional trailing arg so the contract's two-arg
signature (features, user_query) still works unchanged for callers that
don't have a forecast.
"""
from __future__ import annotations


def describe_window(window: dict | None) -> str:
    """Plain-English name for the window that was actually classified."""
    if not window:
        return ""
    if window.get("source") == "upload":
        return f"the uploaded recording at {window['t_start']:.0f}s"
    side = "right" if window.get("side", "R").upper() == "R" else "left"
    return (f"subject {window['subject']}, {side} arm, "
            f"at {window['t_start']:.0f}s")


def build_facts_prompt(heading: str, facts: list[str], user_query: str,
                       instruction: str) -> str:
    """Grounding prompt for the whole-recording and cross-subject answers.

    Same contract as build_prompt(): the numbers are computed in Python and
    handed over as text, and the LLM's only job is to phrase them. It is told
    not to introduce values, because the failure this architecture exists to
    prevent is a confident answer containing a number nobody measured.
    """
    body = "\n".join(f"- {f}" for f in facts)
    return (
        "You are an assistant reporting results from a lab EMG muscle-fatigue "
        "analysis. These are factual engineering measurements, not medical "
        "diagnoses, so it is safe and expected to state them plainly. Ground "
        "your answer ONLY in the values below. Do not introduce any number "
        "that is not listed, and do not re-round the ones that are.\n\n"
        f"{heading}\n{body}\n\n"
        f"User question: {user_query}\n\n"
        f"{instruction}"
    )


def onset_facts(summary: dict) -> list[str]:
    onset = summary["onset"]
    facts = [
        f"subject {summary['subject']}, {'right' if summary['side'] == 'R' else 'left'} arm",
        f"recording length: {summary['duration']:.0f}s",
        f"scanned every {summary['step']:.0f}s ({summary['n_readings']} readings)",
        f"fatigued for {summary['fraction_fatigued'] * 100:.0f}% of the recording",
    ]
    if onset["found"]:
        facts.append(
            f"fatigue first appears and holds at {onset['t_start']:.0f}s "
            f"(confidence {onset['confidence'] * 100:.0f}%), requiring "
            f"{onset['sustain']} consecutive fatigued readings to count")
        facts.append(
            f"because the scan step is {onset['step']:.0f}s, that time is "
            f"accurate to about ±{onset['step']:.0f}s, not to the second")
    else:
        facts.append("fatigue never appears and holds for consecutive readings "
                     "anywhere in this recording")
    return facts


def overview_facts(summary: dict) -> list[str]:
    side = "right" if summary["side"] == "R" else "left"
    # Stated as "fell by X" / "rose by X" rather than a signed number.
    # mdf_drop is start-minus-end, so a *positive* value means the frequency
    # went DOWN -- handing "+10.2 Hz" to the model got it phrased as an
    # increase, which is the opposite of what was measured and is a fatigue
    # claim, not a cosmetic slip.
    drop = summary["mdf_drop"]
    if drop is None:
        change = "not enough readings to state a change"
    elif drop > 0:
        change = (f"median frequency FELL by {drop:.1f} Hz from start to end "
                  "-- falling is the direction that indicates fatigue")
    elif drop < 0:
        change = (f"median frequency ROSE by {-drop:.1f} Hz from start to end "
                  "-- it did not fall, so this recording does not show the "
                  "usual fatigue trend")
    else:
        change = "median frequency did not change from start to end"

    facts = [
        f"subject {summary['subject']}, {side} arm, {summary['duration']:.0f}s long",
        f"scanned every {summary['step']:.0f}s ({summary['n_readings']} readings)",
        f"median frequency at the start: {summary['mdf_start']:.1f} Hz",
        f"median frequency at the end: {summary['mdf_end']:.1f} Hz",
        change,
        f"classified as fatigued in {summary['fraction_fatigued'] * 100:.0f}% of readings",
    ]
    onset = summary["onset"]
    if onset["found"]:
        facts.append(f"fatigue sets in around {onset['t_start']:.0f}s "
                     f"(±{onset['step']:.0f}s)")
    return facts


def compare_facts(comparison: dict) -> list[str]:
    facts = []
    if comparison["kind"] == "sides":
        facts.append(f"subject {comparison['subject']}, both arms compared at "
                     f"{comparison['t_start']:.0f}s")
        for side, r in comparison["results"].items():
            name = "right" if side == "R" else "left"
            state = "fatigued" if r["fatigue_label"] in (1, 2) else "not fatigued"
            facts.append(f"{name} arm: {state}, median frequency "
                         f"{r['mdf_hz']:.1f} Hz, confidence "
                         f"{r['confidence'] * 100:.0f}%")
        return facts

    if comparison.get("fraction") is not None:
        facts.append(
            f"each subject read at {comparison['fraction'] * 100:.0f}% of the way "
            "through their own recording -- the recordings differ in length "
            "(these are efforts to exhaustion), so this compares them at a "
            "comparable point in each person's effort rather than at the same "
            "absolute second")
    else:
        facts.append(f"all subjects read at {comparison['t_start']:.0f}s")

    side = "right" if comparison["side"] == "R" else "left"
    facts.append(f"{side} arm")
    for s, r in comparison["results"].items():
        state = "fatigued" if r["fatigue_label"] in (1, 2) else "not fatigued"
        facts.append(f"subject {s} (recording {r['duration']:.0f}s, read at "
                     f"{r['t_start']:.0f}s): {state}, median frequency "
                     f"{r['mdf_hz']:.1f} Hz, confidence {r['confidence'] * 100:.0f}%")
    if comparison.get("clamped"):
        facts.append("recording shorter than the time asked about, so the last "
                     "window was used for subject(s): "
                     + ", ".join(str(s) for s in comparison["clamped"]))
    if comparison.get("short"):
        facts.append("too short to show a fatigue arc at all, treat with "
                     "caution: subject(s) "
                     + ", ".join(str(s) for s in comparison["short"]))
    return facts


def ranking_facts(ranking: dict) -> list[str]:
    side = "right" if ranking["side"] == "R" else "left"
    facts = [
        f"{side} arm, {len(ranking['ranked'])} subjects ranked",
        f"ranked by the fall in median frequency between "
        f"{ranking['early_fraction'] * 100:.0f}% and "
        f"{ranking['late_fraction'] * 100:.0f}% of each subject's own recording "
        "-- a bigger fall means more fatigue developed over the effort",
        "median frequency drop is used rather than the classifier's "
        "confidence, because confidence is not a severity score",
    ]
    for rank, s in enumerate(ranking["ranked"], start=1):
        r = ranking["results"][s]
        state = "fatigued" if r["fatigue_label"] in (1, 2) else "not fatigued"
        # "fell by" / "rose by", never a signed "drop", for the same reason as
        # overview_facts(): a signed number labelled "drop" gets phrased as an
        # increase about half the time.
        drop = r["mdf_drop"]
        moved = (f"fell by {drop:.1f} Hz" if drop > 0 else
                 f"ROSE by {-drop:.1f} Hz" if drop < 0 else "did not change")
        facts.append(
            f"{rank}. subject {s}: median frequency {r['mdf_early']:.1f} -> "
            f"{r['mdf_late']:.1f} Hz ({moved}), {state} at the late reading")
    if ranking.get("excluded"):
        facts.append("excluded as too short to show a fatigue arc: subject(s) "
                     + ", ".join(str(s) for s in ranking["excluded"]))
    return facts


def build_prompt(features: dict, user_query: str, forecast: dict | None = None,
                 window: dict | None = None) -> str:
    """Build the grounding prompt.

    `window` names the reading's provenance (which subject/time/side, or that
    it came from an upload) and is stated in the answer on purpose. Without it
    a follow-up like "and at 90 seconds?" produces an answer that never says
    which window it used, so a wrong resolution is invisible to the reader.
    """
    forecast_line = (
        f"- fatigue trend: {forecast['summary']}\n" if forecast and forecast.get("ok") else ""
    )
    where = describe_window(window)
    where_line = f"- reading taken from: {where}\n" if where else ""

    calib = features.get("calibration")
    calib_line = (
        f"- calibration caveat: {calib['note']}; this is less reliable than a "
        "reading for one of the 13 dataset subjects\n" if calib else ""
    )
    say_where = (
        f"Begin by stating that this reading is for {where}. " if where else ""
    )
    say_calib = (
        "Also mention in one short clause that the baseline was estimated from "
        "the uploaded recording itself, so the result is less reliable. "
        if calib else ""
    )

    return (
        "You are an assistant reporting the result of a lab sensor reading "
        "(muscle fatigue from an EMG sensor) -- this is a factual "
        "engineering measurement, not a medical diagnosis, so it is safe "
        "and expected to state it plainly. Ground your answer ONLY in the "
        "measured values below -- do not invent, adjust, or round "
        "differently any number that isn't listed here.\n\n"
        f"Measured result:\n"
        f"{where_line}"
        f"- fatigue state: {features['fatigue_state']}\n"
        f"- median frequency: {features['mdf_hz']:.1f} Hz\n"
        f"- model confidence: {features['confidence'] * 100:.1f}%\n"
        f"{calib_line}"
        f"{forecast_line}\n"
        f"User question: {user_query}\n\n"
        f"Answer in 1-4 sentences. {say_where}State the fatigue state and "
        f"confidence plainly. {say_calib}Mention the trend only if it's "
        "relevant to the question or a forecast is provided above. If the "
        "question also asks for sport or training suggestions, ignore that "
        "part here -- it is answered separately; just report the measured "
        "fatigue result."
    )
