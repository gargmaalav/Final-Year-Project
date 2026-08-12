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

import re

# Labels that mark a fact as carrying an instruction to the model. The fact
# itself is still readable once the label is gone.
_FACT_LABELS = (
    "CONCLUSION -- state this as the answer: ",
    "CAVEAT -- state it exactly this way round: ",
    "IMPORTANT: ",
    "WARNING: ",
)

# Prefix for a "fact" that is purely an instruction about how to phrase the
# others -- it states nothing measured, so the offline fallback drops it whole
# rather than trying to salvage a sentence from it.
MODEL_ONLY = "NOTE (phrasing, not for the reader): "

# Sentences addressed to the model rather than to the reader. They exist
# because the 3B model broke each of these rules at least once, but showing
# them to a person -- which is what the offline fallback did -- reads as the
# system talking to itself.
_DIRECTIVE = re.compile(
    r"^\s*(do not|don't|never|say (?:this|so|that)|state (?:this|that|it|the)|"
    r"give this|call this|treat |mention |use only)", re.IGNORECASE)


def readable_facts(facts: list[str]) -> list[str]:
    """The measured facts with the model-facing instructions taken out.

    Used when the LLM is unavailable and the raw numbers are shown directly.
    Without this the fallback printed lines like "Do not reverse this, and do
    not recompute it from the hertz values" straight to the reader.
    """
    out = []
    for fact in facts or []:
        if fact.startswith(MODEL_ONLY):
            continue
        for label in _FACT_LABELS:
            if fact.startswith(label):
                fact = fact[len(label):]
                break
        kept = [s for s in re.split(r"(?<=[.!?])\s+", fact.strip())
                if s and not _DIRECTIVE.match(s)]
        text = " ".join(kept).strip(" ,;-")
        if text:
            out.append(text)
    return out


def describe_window(window: dict | None) -> str:
    """Plain-English name for the window that was actually classified."""
    if not window:
        return ""
    if window.get("source") == "upload":
        # a whole-recording answer (onset, summary) names no single moment
        if window.get("t_start") is None:
            return f"the uploaded recording ({window.get('name', 'your file')})"
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


def _whose(summary: dict) -> str:
    """Names the recording a whole-recording answer is about.

    An upload has no subject number and no side -- it is one file someone
    attached -- so the dataset phrasing cannot just be reused with a blank in
    it.
    """
    if summary.get("subject") is None:
        return summary.get("name") or "the uploaded recording"
    side = "right" if summary.get("side") == "R" else "left"
    return f"subject {summary['subject']}, {side} arm"


# An uploaded file is measured against a baseline taken from its own opening
# seconds, on the assumption the person started fresh. That assumption is
# unverifiable here, and the accuracy figures quoted for the dataset subjects
# were measured with stored baselines, so they do not carry over.
def _self_calibration_facts(summary: dict) -> list[str]:
    if not summary.get("self_calibrated"):
        return []
    return [
        "this recording has no stored calibration, so its fresh reference was "
        "computed from its own opening seconds, assuming the person started "
        "unfatigued -- if they did not, every reading here is shifted",
        MODEL_ONLY + "state that caveat in one short clause. Do not quote an "
        "accuracy or error figure for this recording: none has been measured "
        "for self-calibrated files",
    ]


def onset_facts(summary: dict) -> list[str]:
    onset = summary["onset"]
    facts = [
        _whose(summary),
        f"recording length: {summary['duration']:.0f}s",
        # The scan step is deliberately NOT given here. This answer states an
        # error bar, and handed a step of 2.5 s alongside it the model did
        # arithmetic on it -- inventing "a +/-12.5-second margin of error" in
        # the same paragraph as the measured "about 6 seconds", contradicting
        # itself with a figure nobody computed. The step is an implementation
        # detail; the accuracy is measured and stated on its own below.
        f"{summary['n_readings']} readings taken across the recording",
        f"fatigued for {summary['fraction_fatigued'] * 100:.0f}% of the recording",
    ]
    if onset["found"]:
        facts.append(
            f"fatigue first appears and holds at {onset['t_start']:.0f}s "
            f"(confidence {onset['confidence'] * 100:.0f}%), requiring "
            f"{onset['sustain']} consecutive fatigued readings to count")
        facts.append(
            MODEL_ONLY + "state at most ONE margin of error, and only the one "
            "given below. Do not calculate an error figure of your own from "
            "the number of readings or from anything else here")
        if onset.get("error_measured", True):
            # The honest figure is the measured error against ground truth, not
            # the scan step -- the step is precision and is roughly four times
            # tighter than the accuracy, so quoting it would overstate the answer.
            facts.append(
                f"checked against the dataset's own labels, this onset estimate is "
                f"typically within about {onset.get('typical_error', 20):.0f}s of "
                "the labelled transition, so give it as an approximate time and do "
                "not imply it is accurate to the second")
        else:
            facts.append(
                "give this as an approximate time only. How close it is to the "
                "true transition has been measured for the dataset subjects but "
                "NOT for uploaded recordings, so do not quote any error figure")
        if onset.get("inside_baseline"):
            # The reference was built from the opening seconds; an onset inside
            # that span was detected against a stretch that already contained
            # the fatigue it was meant to detect.
            facts.append(
                f"WARNING: this onset falls inside the first "
                f"{onset['baseline_sec']:.0f}s, which is the very stretch used "
                "as the fresh reference. That means the comparison is against a "
                "period that already contained fatigue, so the time is "
                "unreliable -- say clearly that it should not be trusted and "
                "that a recording starting from genuine rest is needed")
    elif onset.get("fatigued_from_start"):
        facts.append(
            "this recording is already classified as fatigued at its very "
            "first reading, so there is no onset to report within it -- either "
            "the transition happened before the recording starts or the early "
            "readings are wrong. Say that plainly rather than giving a time")
    else:
        facts.append("fatigue never appears and holds for consecutive readings "
                     "anywhere in this recording")
    return facts + _self_calibration_facts(summary)


def overview_facts(summary: dict) -> list[str]:
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
        f"{_whose(summary)}, {summary['duration']:.0f}s long",
        # step omitted for the same reason as onset_facts: this answer also
        # quotes an approximate onset time, and a step figure sitting beside it
        # gets turned into a fabricated margin of error
        f"{summary['n_readings']} readings taken across the recording",
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
    return facts + _self_calibration_facts(summary)


def _drop_phrase(r: dict) -> str:
    """"down 12% from their own fresh level of 70.1 Hz", or the rise case."""
    drop = r.get("drop_percent")
    if drop is None:
        return ""
    if drop >= 0:
        return (f", down {drop:.0f}% from their own fresh level of "
                f"{r['fresh_mdf']:.1f} Hz")
    return (f", UP {abs(drop):.0f}% on their own fresh level of "
            f"{r['fresh_mdf']:.1f} Hz (not the fatigue direction)")


def _verdict(entries: list[tuple[str, dict]], close_within: float = 2.0) -> list[str]:
    """Who is more fatigued, decided in Python and handed over as the answer.

    Asked to work this out from two rows of similar-looking numbers,
    llama3.2:3b got it wrong in a way that is not a phrasing slip: given 21%
    and 4%, it reported "8% and 4%" -- inventing one of the two values -- and
    then attached the wrong caveat to the wrong party. It is the same failure
    the ranking answer had. A comparison between measured values is arithmetic,
    so it is done here, and the model is left only to phrase a conclusion it
    has been given.
    """
    scored = [(name, r["drop_percent"]) for name, r in entries
              if r.get("drop_percent") is not None]
    if len(scored) < 2:
        return []
    scored.sort(key=lambda e: -e[1])
    (top, top_drop), (bottom, bottom_drop) = scored[0], scored[-1]

    if abs(top_drop - bottom_drop) < close_within:
        # Inside the noise the classifier itself carries; naming a winner here
        # would be reporting a difference the measurement cannot support.
        return [f"CONCLUSION -- state this as the answer: {top} and {bottom} "
                f"are close ({top_drop:.0f}% and {bottom_drop:.0f}% below "
                "their own fresh levels). That gap is too small to call one "
                "more fatigued than the other, so say they are similar"]
    return [f"CONCLUSION -- state this as the answer: {top} is further from "
            f"their own fresh level than {bottom} ({top_drop:.0f}% below "
            f"versus {bottom_drop:.0f}% below), so {top} is the more "
            "fatigued of the two. Do not reverse this, and do not recompute "
            "it from the hertz values"]


def compare_facts(comparison: dict) -> list[str]:
    facts = []
    if comparison["kind"] == "upload_vs_subject":
        up, s = comparison["upload"], comparison["subject_result"]
        side = "right" if comparison["side"] == "R" else "left"
        subject = comparison["subject"]
        # "your recording", not "the uploaded recording": the file belongs to
        # the person asking, and they ask in the first person ("how do I
        # compare to subject 1?"). Given the detached label the model lost
        # track of whose numbers were whose and answered that it did not have
        # the user's results -- while they were listed directly above.
        facts = [
            f"your recording (the file you uploaded) compared against subject "
            f"{subject} ({side} arm)",
            MODEL_ONLY + "'your recording' means the person asking this "
            "question. Address them as 'you' and 'your'",
            f"both read at {comparison['fraction'] * 100:.0f}% of the way "
            "through their own recording, because the two recordings are "
            "different lengths",
            "IMPORTANT: compare them on how far each has dropped from their "
            "OWN fresh level, never on their raw hertz values -- everyone "
            "starts at a different level, so a higher or lower reading than "
            "someone else means nothing by itself",
        ]
        entries = [("your recording", up), (f"subject {subject}", s)]
        for label, r in entries:
            state = "fatigued" if r["fatigue_label"] in (1, 2) else "not fatigued"
            facts.append(
                f"{label} (recording {r['duration']:.0f}s, read at "
                f"{r['t_start']:.0f}s): {state}, median frequency "
                f"{r['mdf_hz']:.1f} Hz{_drop_phrase(r)}, confidence "
                f"{r['confidence'] * 100:.0f}%")
        # A wider "too close to call" band than the dataset comparison uses.
        # There, both fresh levels are measured. Here one of them is an
        # assumption about the file's opening seconds, and a few percentage
        # points of difference cannot outrank that uncertainty.
        facts += _verdict(entries, close_within=5.0)
        # The asymmetry is the whole caveat, and it runs in ONE direction. The
        # model stated it backwards -- crediting the dataset subject with the
        # assumed baseline -- so it is worded to name only the upload.
        facts.append(
            "CAVEAT -- state it exactly this way round: your recording's "
            "fresh level was assumed from its own opening seconds, so if you "
            "did not start rested, your drop is understated. Subject "
            f"{subject}'s fresh level is not assumed; it comes from the "
            "labelled dataset. Never say the dataset subject's baseline was "
            "assumed or estimated")
        if comparison.get("short"):
            facts.append("the uploaded recording is also too short to show a "
                         "full fatigue arc, so treat it with extra caution")
        return facts

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
        facts.append(f"subject {s} (recording {r['duration']:.0f}s, read at "
                     f"{r['t_start']:.0f}s): {state}, median frequency "
                     f"{r['mdf_hz']:.1f} Hz{_drop_phrase(r)}, confidence "
                     f"{r['confidence'] * 100:.0f}%")
    facts += _verdict([(f"subject {s}", r)
                       for s, r in comparison["results"].items()])
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
        f"ranked by how far each subject's median frequency has fallen from "
        f"their OWN fresh level, read at "
        f"{ranking['late_fraction'] * 100:.0f}% of the way through their own "
        "recording -- a bigger fall means more fatigue developed",
        "the fall is a percentage of each person's own fresh level, not a "
        "drop in hertz: everyone starts at a different level, so the same "
        "number of hertz does not mean the same thing for two people",
        "median frequency is used rather than the classifier's confidence, "
        "because confidence is not a severity score",
    ]
    for rank, s in enumerate(ranking["ranked"], start=1):
        r = ranking["results"][s]
        state = "fatigued" if r["fatigue_label"] in (1, 2) else "not fatigued"
        # "fell by" / "rose by", never a signed number, for the same reason as
        # overview_facts(): a signed value labelled "drop" gets phrased as an
        # increase about half the time.
        pct = r["drop_percent"]
        moved = (f"fell to {pct:.0f}% below it" if pct >= 1 else
                 f"ROSE to {abs(pct):.0f}% above it" if pct <= -1 else
                 "did not change")
        facts.append(
            f"{rank}. subject {s}: fresh level {r['fresh_mdf']:.1f} Hz -> "
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
    # interpret.prose_notes(): the reading with every number stripped out. See
    # its docstring -- each figure handed to the model came back misattributed
    # somewhere eventually, and the numbers are all rendered in Python anyway.
    # Falls back to the full lines for callers that don't build prose notes.
    plain = (reading.get("prose") or reading.get("lines")) if reading else None
    if plain:
        plain = [line for line in plain
                 if "certainty in the fatigued" not in line]
    plain_block = ("\n".join(f"- {line}" for line in plain) + "\n") if plain else (
        f"- fatigue state: {features['fatigue_state']}\n"
        f"- median frequency: {features['mdf_hz']:.1f} Hz\n"
        f"- model confidence: {features['confidence'] * 100:.1f}%\n")
    # The job depends on the verdict. Asked "what does this mean in practical
    # terms" about a NOT-fatigued reading, the model went looking for fatigue
    # to talk about and wrote "their muscles are starting to feel fatigued"
    # directly under a rendered line saying they were not -- a question whose
    # natural answer contradicts the finding. For a not-fatigued reading it is
    # asked to explain why the change is too small to count instead, which is
    # a question the finding actually answers.
    if features.get("fatigue_label") in (1, 2):
        task = ("Write 1-3 short sentences that go after it: what this "
                "reading means for the person in practical terms. Your first "
                "sentence must NOT be about whether they are fatigued -- that "
                f"is already written. {say_calib}")
    else:
        task = ("Write 1-3 short sentences that go after it, explaining why a "
                "change this small does not count as fatigue, and roughly "
                "what would have to change for it to count. Do NOT say they "
                "are starting to fatigue, beginning to tire, that fatigue is "
                "setting in or developing, or that they may be affected -- "
                "the measurement says they are not fatigued and your text "
                f"must not undercut that. {say_calib}")

    # The raw figures (z-score, confidence) are NOT given to the model at all.
    # They used to travel as "secondary figures, include at most one", and the
    # model duly worked them into the prose -- "with a z-score of -1.2 and
    # confidence level of 100.0%" -- which is jargon to the reader this answer
    # is written for, and reintroduced the confidence-as-certainty problem
    # after it had been removed from the plain lines. They are rendered
    # directly under the answer instead, where a technical reader can find
    # them and a non-technical one can ignore them.
    technical_block = ""

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
        # The verdict and the two hertz figures are already rendered above the
        # model's text by the caller. Asking it to restate them is what kept
        # inverting them, so it is told they are written and its job starts
        # after them.
        f"IMPORTANT: the finding itself -- whether they are fatigued, how far "
        "through the effort this is, and the two hertz figures -- has ALREADY "
        "been written out and shown to the reader directly above your text. "
        "Do NOT restate it, do not re-open with it, and above all do not "
        "contradict it.\n\n"
        f"{task}"
        "Write in words only. Do NOT put any number, hertz value, percentage "
        "or z-score in your text -- every figure is already shown to the "
        "reader above and below it, and you have not been given any, so any "
        "you write will be wrong.\n"
        "Do NOT say what will happen next, that fatigue will continue, or "
        "that it will get worse, unless a fatigue trend is given above. "
        "Nothing here measures the future. If the question also asks for "
        "sport or training suggestions, ignore that part -- it is answered "
        "separately.\n\n"
        "Four rules, each added after the model broke it:\n"
        "1. Report the fatigue verdict as given, in BOTH directions. If the "
        "measurement says NOT fatigued, your answer must say they are not "
        "fatigued -- never drop the 'not'. If it says fatigued, do not argue "
        "with it, soften it, or conclude the person is fine, performing well, "
        "or unaffected. Where a note says the verdict and the "
        "median-frequency marker disagree, report that as a caveat ON the "
        "verdict -- the verdict still stands.\n"
        "2. Keep each number with the label it was given. The value marked "
        "'now' is the current reading and the value marked 'when fresh' is "
        "the baseline; do not swap them, and do not work out a difference "
        "between them that was not given to you.\n"
        "3. Never call the confidence figure certainty. Do not write 'the "
        "model is 100% certain' or 'definitely'. It is the model's own "
        "confidence in its call, and it is not a measure of correctness.\n"
        "4. Use the percentages exactly as written. Do not convert them into "
        "fractions or approximations -- 89% must not become 'three-quarters'."
    )
