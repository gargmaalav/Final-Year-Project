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

import intent

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


_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def strip_unfactual_numbers(prose: str,
                            facts: list[str]) -> tuple[str, list[str]]:
    """Drop sentences quoting a number that was never measured.

    Returns (cleaned prose, the removed sentences).

    interpret.strip_invented_numbers() does this for a single-window reading,
    where the model is handed no figures at all and so ANY digit it writes is
    fabricated. The whole-recording answers are the other case: they are given
    figures, and the model is meant to quote them. What it must not do is
    produce a number that is not among them.

    Observed on subject 12, whose recording is fatigued at its very first
    reading and therefore has no onset time and no accuracy attached to one.
    Given "138 readings taken across the recording" and nothing else numeric,
    the model wrote "the accuracy of this measurement is +/-1 reading ... we
    are 100% certain" -- an error bar and a certainty claim, neither measured,
    both derived from a count that means nothing of the sort.

    A prompt rule was already carrying this on the onset path and it is the
    kind that half-holds: it fixed the case it was written for and the same
    invention reappeared on the branch it did not cover. Checking the output
    against the facts holds on every branch, including ones not written yet.
    """
    if not prose:
        return prose, []
    allowed = {float(n) for n in _NUMBER.findall(" ".join(facts or []))}

    def _invents(sentence: str) -> bool:
        return any(not any(abs(value - a) < 0.05 for a in allowed)
                   for value in (float(n) for n in _NUMBER.findall(sentence)))

    kept, removed = [], []
    for para in re.split(r"\n\s*\n", prose.strip()):
        sentences = re.split(r"(?<=[.!?])\s+", para.strip())
        good = [s for s in sentences if not _invents(s)]
        removed += [s for s in sentences if _invents(s)]
        if good:
            kept.append(" ".join(good))
    return "\n\n".join(kept).strip(), removed


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


# Markers that the recording being re-explained moved the WRONG way for
# fatigue -- median frequency up, not down. Every one of these is rendered in
# Python (overview_facts, _drop_phrase, interpret's lines), so matching on
# them is matching on our own text, not on the model's.
_ROSE_MARKERS = (
    "median frequency ROSE",
    "does not show the usual fatigue trend",
    "above their own fresh level",
    "above their fresh level",
    "not the fatigue direction",
    "not the direction fatigue moves it",
    "ROSE to",
)


def _signal_rose(facts: list[str], previous_answer: str) -> bool:
    """True when the answer being re-explained had a RISING median frequency."""
    hay = " ".join(list(facts or []) + [previous_answer or ""])
    return any(m.lower() in hay.lower() for m in _ROSE_MARKERS)


# The NOT-fatigued verdict, as this codebase renders it -- interpret's
# verdict_sentence, plain_lines and prose_notes, plus the analysis paths.
# Matching our own rendered text, never the model's.
_NOT_FATIGUED_MARKERS = (
    "not showing signs of fatigue",
    "NOT showing signs of fatigue",
    "is NOT fatigued",
    "not fatigued",
    "no signs of fatigue",
    "not enough to count as fatigue",
)


def _reads_not_fatigued(facts: list[str], previous_answer: str) -> bool:
    """True when the answer being re-explained concluded NOT fatigued."""
    hay = " ".join(list(facts or []) + [previous_answer or ""]).lower()
    return any(m.lower() in hay for m in _NOT_FATIGUED_MARKERS)


# What each kind of follow-up is actually asking for, and what a useful answer
# to it contains. One generic "explain the reasoning" instruction covered all
# four and the model answered every one of them identically -- by paraphrasing
# the answer sitting in its prompt. Given the previous answer as context and no
# statement of what to ADD, restating it is the likeliest completion, so each
# kind now names the material it is supposed to reach for instead.
_FOLLOWUP_TASK = {
    intent.WHY: (
        "They are asking WHY the result came out that way. Explain the "
        "reasoning: which measurement was taken, what it was compared "
        "against, and why that comparison leads to this conclusion rather "
        "than the opposite one."),
    intent.MEANING: (
        "They are asking what this result MEANS for them -- what to take away "
        "from it, not the finding repeated. Cover, in this order: what the "
        "reading says about the muscle's state right now; how far it is from "
        "the point where the answer would change, using the measured values "
        "above; and what would be worth watching from here. Every one of "
        "those is something your previous answer did NOT say."),
    intent.SIMPLER: (
        "They did not follow the previous answer, so say the same thing again "
        "in ordinary words. Use an everyday comparison for what the "
        "measurement tracks. Do not use the words median, frequency, hertz, "
        "baseline, standard deviation or classifier at all."),
    intent.MORE: (
        "They want the detail the previous answer left out. Go to the "
        "measured values above that the previous answer did not mention and "
        "explain what those add -- how settled the result is, where in the "
        "effort it sits, how far from the person's normal range it falls."),
}

# Bolted onto every follow-up, whichever kind it is. The failure this is aimed
# at is not the model being wrong -- follow-ups are factually fine -- it is the
# model spending its whole answer re-asserting a verdict the reader has just
# read and is asking about, which leaves them with nothing they did not have
# before. A negative instruction alone did not hold, so it is paired with
# interpret.strip_verdict_echo / drop_repeated_sentences downstream.
_FOLLOWUP_NEVER_RESTATE = (
    "Your previous answer is printed on screen directly above your reply, and "
    "they have read it. NEVER open by restating the finding, and never write "
    "a sentence that says the same thing as one of the sentences above. Every "
    "sentence you write must add something that answer did not contain. Do "
    "not change the conclusion either -- add to it.")


def _addressing(who: str | None) -> str:
    """Who the answer is about, and in which person to write about them.

    A dataset subject is a third party and an uploaded recording is the
    reader's own. With neither stated, a follow-up about subject 11 came back
    as "your muscle signal is still relatively healthy ... how close YOU are to
    fatigue", which hands the reader someone else's measurement as their own.
    The reading prompt names the subject in every line it is given; the
    follow-up prompt is built from prose and had nothing to anchor to.
    """
    if not who:
        return ""
    if who.lower().startswith(("subject", "the subject")):
        return (f" The measurement is of {who}, who is NOT the person reading "
                f"this. Write about them in the third person -- \"they\", "
                f"\"their\" -- and never \"you\" or \"your\".")
    return (f" The measurement is of {who}, which belongs to the person "
            "reading this, so \"you\" and \"your\" are correct.")


def build_followup_prompt(previous_question: str, previous_answer: str,
                          facts: list[str], user_query: str,
                          kind: str | None = None,
                          who: str | None = None) -> str:
    """Re-explain the last answer, from the same measured numbers.

    Every other call is built from scratch with no conversation history, which
    is right for the grounded answers -- but it means "why?" had nothing to
    refer to. Rather than threading history through every call (and giving the
    model a chance to quote stale figures as if freshly measured), the previous
    exchange is handed over explicitly, with the same facts that produced it,
    and nothing new is measured.

    `kind` is one of intent.WHY / MEANING / SIMPLER / MORE -- what sort of
    re-explanation was asked for. It decides which task is set, because the
    answer to "why?" and the answer to "so what does this mean?" are different
    answers and a prompt aimed at both hits neither.
    """
    task = _FOLLOWUP_TASK.get(kind or intent.MEANING,
                              _FOLLOWUP_TASK[intent.MEANING])
    body = "\n".join(f"- {f}" for f in facts) if facts else "- (none recorded)"
    # The causal chain is generic physics and was pasted in unconditionally.
    # Asked to re-explain subject 7 -- whose median frequency ROSE, which the
    # answer above it said plainly -- the model recited the chain and closed
    # with "which shifts the signal's power to lower frequencies, exactly what
    # happened here". It did not happen here. A fluent, physically-correct
    # explanation of something the measurement contradicts is the failure this
    # whole architecture exists to prevent, so which chain gets asked for is
    # decided here rather than left to the model to notice.
    #
    # The chain is asked for only when the reader asked WHY. Pasted onto a
    # "what does this mean?" it filled the whole 2-4 sentence budget with
    # textbook physics and never reached the reader's actual question; on a
    # "simpler terms" it reintroduced the exact vocabulary that request is
    # asking to be rid of. The rising-signal WARNING is not optional in the
    # same way and is kept for every kind.
    rose = _signal_rose(facts, previous_answer)
    chain = ""
    if kind == intent.WHY:
        # The chain describes what fatigue DOES, so under a not-fatigued
        # verdict it has to be framed as what did not happen. Stated flat, the
        # model welded it to the conclusion and answered "why is subject 11 not
        # fatigued" with "because their signal power has shifted to lower
        # frequencies" -- reciting the mechanism of fatigue as the reason for
        # its absence. Observed live, and exactly the fluent self-contradiction
        # the _signal_rose branch below already exists to stop.
        if _reads_not_fatigued(facts, previous_answer):
            chain = (
                # The qualifier goes FIRST. Asked for it last, the model spent
                # its whole length on the mechanism and the answer was cut
                # before reaching "but not here" -- leaving a fluent
                # explanation of fatigue standing as the reason for its
                # absence, which is the failure this branch exists to stop.
                #
                # Lower case deliberately. Written "WOULD shift", the model
                # copied the shouting into the answer -- "a fatiguing muscle's
                # fibres WOULD cause the signal to shift". "Causal chain" is
                # avoided for the same reason: it came back verbatim as "the
                # causal chain can be explained as follows".
                " Open by saying the signal has not moved far enough for this "
                "to read as fatigue. Only then say what fatigue would have "
                "done, in the conditional: a fatiguing muscle's fibres conduct "
                "more slowly, which would shift the signal's power to lower "
                "frequencies, which is what the median frequency measures. "
                "Never write that the signal HAS shifted to lower frequencies "
                "here, and never give the fatigue mechanism as the reason the "
                "answer is NOT fatigue.")
        else:
            chain = (
                " Explain the causal chain: a fatiguing muscle's fibres "
                "conduct more slowly, which shifts the signal's power to "
                "lower frequencies, which is what the median frequency "
                "measures.")
    if rose:
        chain += (
            " You MUST say that this recording does NOT follow the usual "
            "fatigue pattern -- its median frequency went UP, not down, which "
            "is not the direction fatigue moves it. Never write that the "
            "frequency fell here, and never say the usual pattern is what "
            "happened here.")

    # Fixed instructions first, the per-question material last. Ollama can only
    # reuse a cached prompt prefix up to the first byte that differs, so a
    # variable block placed above a static one costs seconds with nothing in
    # the output to show for it -- see the note on build_prompt().
    return (
        "You are re-explaining an answer you already gave about a lab EMG "
        "muscle-fatigue measurement. The reader wants the SAME result "
        "explained further -- not a new measurement.\n\n"
        "Answer in 2-4 sentences, in plain everyday words, for a reader who "
        "is not a signal-processing specialist. Use ONLY the measured values "
        "listed below; do not introduce a number that is not there. Do not "
        "quote any figure in hertz (Hz) and never say one hertz figure is "
        "above or below another -- those are printed underneath your answer "
        "already. Percentages and times are fine."
        + _addressing(who) + "\n\n"
        + _FOLLOWUP_NEVER_RESTATE + "\n\n"
        + task + chain + "\n\n"
        f"What they originally asked:\n{previous_question}\n\n"
        f"What you answered:\n{previous_answer}\n\n"
        f"The measured values that answer came from:\n{body}\n\n"
        f"They have now said: {user_query}"
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
        # build_prompt() has carried this since the single-window answers were
        # written and this path never got it, so the two read like different
        # products: "Subject 13 is not showing signs of fatigue, 60s into a
        # 218s effort" against "The EMG muscle-fatigue analysis for subject 7
        # shows a unique development... The median frequency starts at 72.2 Hz".
        "The reader is NOT a signal-processing specialist. Lead with what the "
        "result means for the person, in ordinary words, and put the figures "
        "after that rather than in front of it. Do not open with a hertz "
        "value, with the phrase 'median frequency', or with 'the EMG "
        "analysis' -- say what happened to the muscle first. Where you can, "
        "call it the muscle signal rather than naming the measurement.\n\n"
        # Asking for plainer words immediately bought the mistake recommend.py
        # already guards against: "the muscle signal shows a decrease in
        # ACTIVITY over time". A falling median frequency is a shift in the
        # signal's frequency content, not the muscle doing less, and that
        # reading is wrong in a way a non-specialist has no way to catch.
        "Plainer wording must not change what was measured. A rise or fall "
        "here is a shift in the signal's frequency, NOT the muscle being more "
        "or less active. Never describe it as more or less muscle activity, "
        "effort, strength, engagement, contraction or energy.\n"
        # "The majority of readings (43%) were classified as fatigued" -- the
        # figure was quoted correctly and then described as its opposite.
        "Do not characterise a percentage in words that disagree with it: "
        "under half is not 'most' or 'the majority', and a small share is not "
        "'much of'. If you are unsure, give the figure and no adjective.\n\n"
        f"{heading}\n{body}\n\n"
        f"User question: {user_query}\n\n"
        f"{instruction}\n\n"
        # build_prompt() carries this rule for the single-window answers and
        # this path did not, so "we are 100% certain that fatigue occurred"
        # went out under an onset the classifier was not certain of at all.
        "Never present anything here as certainty. Do not write that you are "
        "certain, sure, or confident to a percentage: a confidence figure is "
        "the model's own confidence in its call, not a measure of how correct "
        "it is. Do not invent a margin of error, an accuracy, or a precision "
        "that is not stated above."
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


# When no onset was found there is no time to attach an accuracy to, so the
# honest number of error figures is zero. The `found` branch caps them at one;
# without this the no-onset branches carried no such note at all, and the model
# duly invented "the accuracy of this measurement is +/-1 reading" out of the
# readings count, alongside "we are 100% certain".
_NO_ERROR_FIGURE = (
    MODEL_ONLY + "there is no onset time in this answer, so there is no margin "
    "of error to give. Do not state one, and do not derive an accuracy, a "
    "precision or a certainty figure from the number of readings, the "
    "recording length, or anything else here")


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
        facts.append(_NO_ERROR_FIGURE)
    else:
        facts.append("fatigue never appears and holds for consecutive readings "
                     "anywhere in this recording")
        facts.append(_NO_ERROR_FIGURE)
    return facts + _self_calibration_facts(summary)


def render_onset_answer(summary: dict) -> str | None:
    """The onset answer for a recording that has no onset, written here.

    Returns None when there IS an onset time, which is a real measurement with
    an accuracy attached and reads better as prose.

    When there is not, the whole answer is one fixed determination -- "already
    fatigued at the first reading, so there is no onset in this recording", or
    "fatigue never appears and holds anywhere in it". There is no number to
    phrase and no judgement to make, and handing that to the model produced, on
    the two recordings it applies to:

      - "the accuracy of this measurement is +/-1 reading ... we are 100%
        certain that fatigue occurred" -- an error bar and a certainty claim
        invented out of a readings count (subject 12)
      - "Fatigue set in at reading 10 ... fatigue never appeared during the
        recording" -- a self-contradiction in consecutive sentences, on a
        recording where it never set in at all (subject 6)

    Screening the numbers stopped the first and could not stop the second: 10
    was a real measured value, used to assert something nobody measured. This
    is the same trade the ranking and comparison answers already take -- a
    determination that is settled in Python is rendered in Python.
    """
    onset = summary["onset"]
    if onset["found"]:
        return None

    who = _whose(summary).capitalize()
    length = f"{summary['duration']:.0f}s"
    fraction = summary["fraction_fatigued"] * 100

    if onset.get("fatigued_from_start"):
        return (
            f"**{who}** — there's no onset to report for this recording.\n\n"
            f"It is already classified as fatigued at its very first reading, "
            f"so the change happened before the recording starts, or the early "
            f"readings are wrong. Either way there is no moment inside these "
            f"{length} where fatigue begins, and no time I can give you for it."
            f"\n\nAcross the whole recording, {fraction:.0f}% of readings came "
            f"out fatigued.")
    return (
        f"**{who}** — fatigue never sets in anywhere in this recording.\n\n"
        f"No point in the {length} shows fatigue appearing and then holding "
        f"for consecutive readings, which is what it takes to count as an "
        f"onset rather than a one-off noisy window. {fraction:.0f}% of "
        f"readings were classified as fatigued.")


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
                 window: dict | None = None, reading: dict | None = None,
                 chart_shown: bool = False) -> str:
    """Build the grounding prompt.

    `window` names the reading's provenance (which subject/time/side, or that
    it came from an upload) and is stated in the answer on purpose. Without it
    a follow-up like "and at 90 seconds?" produces an answer that never says
    which window it used, so a wrong resolution is invisible to the reader.

    `reading` is interpret.py's plain-language view of the same numbers
    ({"lines": [...], "technical": str}). Optional, so the contract's original
    signature still works; when absent the prompt falls back to stating the
    raw values as it always did.

    `chart_shown` says whether the app actually produced a chart for this
    turn (a reading normally does; an error or a plain analysis answer
    doesn't). Left unstated, a question that mentions "graph" got answered
    with "I don't have the capability to display graphs" even when a chart
    was sitting right there in a collapsed expander -- the model has no way
    to know the app's own behaviour unless it's told.
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
    # The old wording described a collapsed "Show the signal" expander and
    # invited the model to point at it -- that was the Streamlit UI, and it
    # cost a whole sentence ("You can expand the chart below to see the exact
    # values") on every single answer. The chart is now already on screen in
    # the figures panel beside the answer, so there is nothing to expand and
    # nothing worth spending a sentence telling the reader to go and look at.
    chart_line = (
        "A chart of this reading is already visible in the figures panel "
        "beside your answer -- the reader can see it without doing anything. "
        "Do NOT tell them to open, expand, or look at the chart, and do not "
        "spend a sentence on it. You cannot see it yourself, so do not "
        "describe values in it beyond what's given above.\n"
        if chart_shown else
        "No chart is available for this reading. If the question asked to "
        "see a graph/chart/plot, say so in one short clause -- do NOT say "
        "you are unable to display charts at all, since the app normally "
        "can; this particular reading just doesn't have one.\n"
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
        task = ("Write ONE or TWO short sentences that go after it: what this "
                "reading means for the person in practical terms. Your first "
                "sentence must NOT be about whether they are fatigued -- that "
                f"is already written. {say_calib}")
    else:
        # The "what would have to change for it to count" half used to be
        # asked for here, and it reliably produced a second, longer sentence
        # of the same boilerplate every time ("for this reading to be
        # considered indicative of fatigue, there would need to be a more
        # significant deviation from their normal fresh range"). It says
        # nothing the first sentence didn't, so it is no longer asked for.
        task = ("Write ONE or TWO short sentences that go after it, saying "
                "in plain words why a change this small does not count as "
                "fatigue. Do NOT say they "
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
        # Left to itself the model wrote lab-report English -- "This reading
        # falls within their normal fresh range, indicating that the muscle
        # signal is only a little below its own level" -- which is longer and
        # harder to read than the plain version and says no more. Naming the
        # specific words it reaches for works better than asking for "plain
        # language" in the abstract, which it reads as a style note and
        # ignores.
        # A twelve-item banned-words list was tried first and llama3.2:3b
        # walked straight through it ("indicating a substantial decrease in
        # performance capacity"). A 3B model holds two or three prohibitions,
        # not twelve, so the list is short and the length limit is enforced in
        # code instead of asked for -- see interpret.trim_sentences().
        "Write the way you would say it out loud to the person who did the "
        "exercise: short sentences, everyday words. Never write 'indicating', "
        "'indicative of', or 'deviation'. If one sentence covers it, stop at "
        "one.\n\n"
        # ORDER MATTERS FOR SPEED, not just for sense. Ollama reuses the KV
        # cache for whatever prefix a prompt shares with the previous one, and
        # on this CPU-only box prompt processing is the single biggest cost in
        # a turn -- ~15 s of a ~26 s answer, three times what generating the
        # text costs. Measured: an identical 572-token prefix evaluated in
        # 13.71 s the first time and 0.30 s the second; the same text placed
        # AFTER the variable part took the full 13.21 s again.
        #
        # So every fixed instruction is emitted first and the per-question
        # material -- the measured values and the user's question -- goes last.
        # Nothing below was reworded to achieve this; the blocks were only
        # reordered. Keep it that way: moving a variable block up in front of
        # a static one silently costs seconds per answer and nothing about the
        # output changes to show it.
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
        "\n\n"
        # Everything above is identical for every reading and is therefore a
        # cache hit after the first question of a session. Everything below
        # changes per question, so it is the only part actually evaluated.
        # `task` and `chart_line` sit here rather than higher up because both
        # have two variants -- they cost a little to re-evaluate when the
        # variant flips, and nothing when it does not.
        f"{task}"
        f"{chart_line}\n"
        f"Measured result:\n"
        f"{where_line}"
        f"{plain_block}"
        f"{calib_line}"
        f"{forecast_line}"
        f"{technical_block}\n"
        f"User question: {user_query}"
    )
