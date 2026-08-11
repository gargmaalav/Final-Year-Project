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


def build_followup_prompt(previous_question: str, previous_answer: str,
                          facts: list[str], user_query: str) -> str:
    """Re-explain the last answer, from the same measured numbers.

    Every other call is built from scratch with no conversation history, which
    is right for the grounded answers -- but it means "why?" had nothing to
    refer to. Rather than threading history through every call (and giving the
    model a chance to quote stale figures as if freshly measured), the previous
    exchange is handed over explicitly, with the same facts that produced it,
    and nothing new is measured.
    """
    body = "\n".join(f"- {f}" for f in facts) if facts else "- (none recorded)"
    return (
        "You are re-explaining an answer you already gave about a lab EMG "
        "muscle-fatigue measurement. The reader wants the SAME result "
        "explained differently -- not a new measurement.\n\n"
        f"What they originally asked:\n{previous_question}\n\n"
        f"What you answered:\n{previous_answer}\n\n"
        f"The measured values that answer came from:\n{body}\n\n"
        f"They have now said: {user_query}\n\n"
        "Answer in 2-4 sentences, in plainer language than before, for a "
        "reader who is not a signal-processing specialist. Explain the "
        "reasoning behind the result -- what was measured and why it means "
        "what it means. Use ONLY the values listed above; do not introduce a "
        "number that is not there, and do not change the conclusion. If they "
        "asked why, explain the causal chain: a fatiguing muscle's fibres "
        "conduct more slowly, which shifts the signal's power to lower "
        "frequencies, which is what the median frequency measures."
    )


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
        # The honest figure is the measured error against ground truth, not
        # the scan step -- the step is precision and is roughly four times
        # tighter than the accuracy, so quoting it would overstate the answer.
        facts.append(
            f"checked against the dataset's own labels, this onset estimate is "
            f"typically within about {onset.get('typical_error', 20):.0f}s of "
            "the labelled transition, so give it as an approximate time and do "
            "not imply it is accurate to the second")
    elif onset.get("fatigued_from_start"):
        facts.append(
            "this recording is already classified as fatigued at its very "
            "first reading, so there is no onset to report within it -- either "
            "the transition happened before the recording starts or the early "
            "readings are wrong. Say that plainly rather than giving a time")
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
        # Spelled out because the model read "a fall in median frequency" as
        # "a fall in fatigue" -- the exact inversion of what it means.
        change = (f"median frequency FELL by {drop:.1f} Hz from start to end. "
                  "A falling median frequency means fatigue INCREASED -- the "
                  "muscle got more fatigued, not less. Never describe this as "
                  "a fall or reduction in fatigue")
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
        # the measured error against ground truth, not the scan step -- the
        # same overclaim onset_facts() had
        # "within 12s of" became "12s after" when phrased -- which asserts a
        # direction the measurement does not claim (the error runs both ways)
        facts.append(
            f"fatigue sets in around {onset['t_start']:.0f}s. Call this "
            "approximate; do not state how far off it is or in which "
            "direction")
    elif onset.get("fatigued_from_start"):
        facts.append("already classified as fatigued at the very first "
                     "reading, so no onset time can be given for it")
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

    # Absolute median frequency is not comparable between people: fresh
    # subjects in this dataset span 59-81 Hz, so whoever happens to sit higher
    # is not thereby less fatigued. Handing over raw Hz alone got exactly that
    # wrong conclusion ("5 is at 70 Hz, 9 at 61 Hz, so 5 is more fatigued"), so
    # the drop from each person's own fresh level is given and named as the
    # thing to compare.
    facts.append(
        "IMPORTANT: compare the subjects on how far each has dropped from "
        "their OWN fresh level, not on their raw hertz values. A higher or "
        "lower median frequency than someone else means nothing on its own, "
        "because everyone starts at a different level")

    side = "right" if comparison["side"] == "R" else "left"
    facts.append(f"{side} arm")
    for s, r in comparison["results"].items():
        state = "fatigued" if r["fatigue_label"] in (1, 2) else "not fatigued"
        drop = r.get("drop_percent")
        drop_txt = ""
        if drop is not None:
            drop_txt = (f", down {drop:.0f}% from their own fresh level of "
                        f"{r['fresh_mdf']:.1f} Hz" if drop >= 0 else
                        f", UP {abs(drop):.0f}% on their own fresh level of "
                        f"{r['fresh_mdf']:.1f} Hz (not the fatigue direction)")
        facts.append(f"subject {s} (recording {r['duration']:.0f}s, read at "
                     f"{r['t_start']:.0f}s): {state}, median frequency "
                     f"{r['mdf_hz']:.1f} Hz{drop_txt}, confidence "
                     f"{r['confidence'] * 100:.0f}%")
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
                 window: dict | None = None, reading: dict | None = None) -> str:
    """Build the grounding prompt.

    `window` names the reading's provenance (which subject/time/side, or that
    it came from an upload) and is stated in the answer on purpose. Without it
    a follow-up like "and at 90 seconds?" produces an answer that never says
    which window it used, so a wrong resolution is invisible to the reader.

    `reading` is interpret.py's plain-language view of the same numbers
    ({"lines": [...], "technical": str}). Optional, so the contract's original
    signature still works; when absent the prompt falls back to stating the
    raw values as it always did.
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

    # The plain-language lines lead. Raw Hz and confidence are still handed
    # over, but last and labelled as secondary: a reader who is not on this
    # project cannot do anything with "51.2 Hz", because median frequency has
    # no absolute meaning across people. What they can use is how far the
    # person has moved from their own fresh state.
    plain = reading.get("lines") if reading else None
    plain_block = ("\n".join(f"- {line}" for line in plain) + "\n") if plain else (
        f"- fatigue state: {features['fatigue_state']}\n"
        f"- median frequency: {features['mdf_hz']:.1f} Hz\n"
        f"- model confidence: {features['confidence'] * 100:.1f}%\n")
    technical = reading.get("technical") if reading else None
    technical_block = (f"\nRaw figures, secondary -- include at most one of "
                       f"these and only if the question asked for numbers:\n"
                       f"- {technical}\n") if technical else ""

    return (
        "You are an assistant reporting the result of a lab sensor reading "
        "(muscle fatigue from an EMG sensor) -- this is a factual "
        "engineering measurement, not a medical diagnosis, so it is safe "
        "and expected to state it plainly. Ground your answer ONLY in the "
        "measured values below -- do not invent, adjust, or round "
        "differently any number that isn't listed here.\n\n"
        "The reader is NOT a signal-processing specialist. Lead with what the "
        "result means for the person, in ordinary words. Do not open with "
        "hertz or with the phrase 'median frequency'.\n\n"
        f"Measured result:\n"
        f"{where_line}"
        f"{plain_block}"
        f"{calib_line}"
        f"{forecast_line}"
        f"{technical_block}\n"
        f"User question: {user_query}\n\n"
        f"Answer in 2-4 sentences of plain English. {say_where}Say whether "
        "the muscle is fatigued, how far through the effort the reading is, "
        "and how far the signal has moved from that person's own fresh level. "
        f"{say_calib}Mention the trend only if it's relevant to the question "
        "or a forecast is provided above. If the question also asks for sport "
        "or training suggestions, ignore that part here -- it is answered "
        "separately; just report the measured fatigue result.\n\n"
        "Three rules, each added after the model broke it:\n"
        "1. Report the fatigue verdict as given. Do NOT argue with it, "
        "soften it, or conclude the person is fine, performing well, or "
        "unaffected when the verdict says fatigued. Where a note says the "
        "verdict and the median-frequency marker disagree, report that as a "
        "caveat ON the verdict -- the verdict still stands.\n"
        "2. Never call the confidence figure certainty. Do not write 'the "
        "model is 100% certain' or 'definitely'. It is the model's own "
        "confidence in its call, and it is not a measure of correctness.\n"
        "3. Use the percentages exactly as written. Do not convert them into "
        "fractions or approximations -- 89% must not become 'three-quarters'."
    )
