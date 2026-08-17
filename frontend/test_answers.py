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

import intent          # noqa: E402
import interpret       # noqa: E402
import prompt          # noqa: E402
import recommend       # noqa: E402


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


@check("the scan step is withheld from answers that state an error bar")
def _():
    # Given the step, the model multiplied it and reported "a +/-12.5-second
    # margin of error" beside the measured "about 6 seconds" -- two error bars
    # in one answer, the invented one wrong.
    for facts in (prompt.onset_facts(_summary(75.0, 57.0, _onset())),
                  prompt.overview_facts(_summary(75.0, 57.0, _onset()))):
        text = " ".join(facts)
        assert "2.5s" not in text and "every 2.5" not in text, text
        assert "80 readings" in text, text


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


# A margin of error was forbidden only on the branch that HAS an onset. On the
# two branches that do not, the model built one out of the readings count.

@check("a no-onset answer forbids any error figure at all")
def _():
    for onset in (_onset(found=False, fatigued_from_start=True),
                  _onset(found=False)):
        facts = prompt.onset_facts(_summary(75.0, 57.0, onset))
        note = [f for f in facts if "no margin" in f]
        assert note, f"no error-figure guard on this branch: {facts}"
        assert "number of readings" in note[0], note[0]


@check("a recording with no onset is rendered, not phrased")
def _():
    # subject 12: fatigued at its first reading. Asked to phrase this, the
    # model invented "+/-1 reading" and "100% certain"; screened for numbers,
    # it answered without saying anything at all.
    text = prompt.render_onset_answer(_summary(
        75.0, 57.0, _onset(found=False, fatigued_from_start=True), subject=12))
    assert text, "no rendered answer for a fatigued-from-start recording"
    assert "no onset to report" in text, text
    assert "very first reading" in text, text

    # subject 6: fatigue never appears and holds. The model answered "fatigue
    # set in at reading 10" and "fatigue never appeared" in consecutive
    # sentences -- 10 was a real measured value asserting an unmeasured claim,
    # which is why screening numbers could not catch it.
    text = prompt.render_onset_answer(_summary(
        75.0, 57.0, _onset(found=False), subject=6))
    assert text and "never sets in" in text, text
    assert "set in at" not in text, text


@check("an onset that was found is still phrased by the model")
def _():
    assert prompt.render_onset_answer(_summary(75.0, 57.0, _onset())) is None


# --- numbers on the whole-recording answers ---------------------------------
# These are GIVEN figures and are meant to quote them, so they cannot be
# scrubbed of digits the way a single reading is. Only unmeasured ones go.

@check("a number that was never measured is stripped from a facts answer")
def _():
    facts = ["recording length: 344s", "138 readings taken across the recording"]
    prose = ("This recording runs 344s. The accuracy of this measurement is "
             "±1 reading, so we are 100% certain.")
    cleaned, invented = prompt.strip_unfactual_numbers(prose, facts)
    assert cleaned == "This recording runs 344s.", cleaned
    assert len(invented) == 1, invented


@check("measured numbers survive the strip untouched")
def _():
    facts = ["recording length: 218s", "88 readings taken across the recording",
             "fatigue first appears and holds at 83s (confidence 90%)",
             "typically within about 6s of the labelled transition"]
    prose = ("Fatigue first appears at about 83s, with 90% confidence. That "
             "comes from 88 readings across the 218s recording, and is "
             "typically within about 6s of the labelled transition.")
    cleaned, invented = prompt.strip_unfactual_numbers(prose, facts)
    assert cleaned == prose, cleaned
    assert not invented, invented


@check("word-only prose is never stripped from a facts answer")
def _():
    prose = ("Fatigue set in partway through the effort and held from there "
             "on. Treat the time as approximate.")
    cleaned, invented = prompt.strip_unfactual_numbers(prose, ["recording: 218s"])
    assert cleaned == prose, cleaned
    assert not invented, invented


@check("the facts prompt forbids certainty and invented error bars")
def _():
    built = prompt.build_facts_prompt("Measured results:", ["a fact"],
                                      "when did fatigue set in?", "Answer it.")
    assert "Never present anything here as certainty" in built, built
    assert "Do not invent a margin of error" in built, built


@check("the whole-recording answers are told to lead in plain words")
def _():
    # build_prompt has carried this since the readings were written and this
    # path never got it, so the two read like different products: "Subject 13
    # is not showing signs of fatigue, 60s into a 218s effort" against "The
    # EMG muscle-fatigue analysis for subject 7 shows a unique development".
    built = prompt.build_facts_prompt("Measured results:", ["a fact"],
                                      "summarise subject 7", "Answer it.")
    assert "NOT a signal-processing specialist" in built, built
    assert "Do not open with a hertz value" in built, built


@check("plainer wording may not turn a frequency shift into muscle activity")
def _():
    # asking for plainer words immediately bought "the muscle signal shows a
    # decrease in ACTIVITY over time" -- wrong, and invisible to the reader
    # this answer is written for
    built = prompt.build_facts_prompt("Measured results:", ["a fact"],
                                      "summarise subject 13", "Answer it.")
    assert "NOT the muscle being more or less active" in built, built
    for banned in ("effort", "strength", "engagement", "contraction", "energy"):
        assert banned in built, f"{banned} missing from the banned list"


@check("a percentage may not be described as its opposite")
def _():
    # "The majority of readings (43%) were classified as fatigued"
    built = prompt.build_facts_prompt("Measured results:", ["a fact"],
                                      "summarise subject 7", "Answer it.")
    assert "under half is not 'most'" in built, built


@check("an upload is named by its file, a subject by number and arm")
def _():
    assert prompt._whose(_summary(75.0, 57.0, _onset())) == "subject 13, right arm"
    assert prompt._whose(_summary(75.0, 57.0, _onset(), self_calibrated=True)) \
        == "the uploaded recording"


# --- the verdict ------------------------------------------------------------
# The worst failure found in live testing: handed "Subject 13 is not showing
# signs of fatigue", the model answered "subject 13's right arm is fatigued".
# It dropped the "not", inverting the result.

def _reading(mdf, fresh, sd, label, t=65.0, duration=218.0):
    return interpret.describe_reading(
        {"mdf_hz": mdf, "fatigue_label": label, "confidence": 0.98},
        {"fresh_mdf": fresh, "sd_mdf": sd}, t, duration)


@check("a not-fatigued verdict is stated on its own line and stressed")
def _():
    lines = interpret.plain_lines(_reading(61.8, 62.7, 3.8, 0), "Subject 13")
    assert "NOT fatigued" in lines[0], lines
    assert "not showing signs of fatigue" in lines[0], lines
    # the position must not share the line -- burying the negation mid-sentence
    # is what let it get lost
    assert "seconds into" not in lines[0], lines[0]


@check("a fatigued verdict is stated on its own line too")
def _():
    lines = interpret.plain_lines(_reading(58.2, 62.7, 3.8, 1), "Subject 13")
    assert "IS showing signs of fatigue" in lines[0], lines
    assert "NOT" not in lines[0], lines[0]


@check("a small drop under a not-fatigued verdict is not glossed as fatigue")
def _():
    # "a falling signal is what muscle fatigue looks like", printed under a
    # NOT-fatigued verdict, argues against the verdict it is attached to
    lines = interpret.plain_lines(_reading(61.8, 62.7, 3.8, 0), "Subject 13")
    drop = [l for l in lines if "below their own fresh level" in l][0]
    assert "not enough to count as fatigue" in drop, drop
    assert "what muscle fatigue looks like" not in drop, drop


@check("current and fresh values are labelled so they cannot be swapped")
def _():
    # the model reported "decreased by 1.7 Hz from its initial fresh level of
    # 61.8 Hz" -- 61.8 was the current value, and 1.7 Hz was invented
    lines = interpret.plain_lines(_reading(61.8, 62.7, 3.8, 0), "Subject 13")
    drop = [l for l in lines if "below their own fresh level" in l][0]
    assert "now 61.8 Hz" in drop, drop
    assert "62.7 Hz when fresh" in drop, drop


@check("the rendered verdict sentence states the finding in both directions")
def _():
    not_fat = interpret.verdict_sentence(_reading(61.8, 62.7, 3.8, 0), "Subject 13")
    assert "is not showing signs of fatigue" in not_fat, not_fat
    fat = interpret.verdict_sentence(_reading(58.2, 62.7, 3.8, 1), "Subject 13")
    assert "is showing signs of fatigue" in fat, fat
    assert "not showing" not in fat, fat


@check("the rendered verdict states the size of the change without hertz")
def _():
    # The verdict used to quote the raw pair ("62.7 Hz fresh -> 61.8 Hz now")
    # as well as the percentage. Both numbers appear again in the provenance
    # line immediately below it, and hertz is the least meaningful form of
    # them for the reader this sentence is written for, so the headline keeps
    # the percentage only. The fresh/current pair is still rendered -- and
    # still guarded against being swapped -- by interpret.plain_lines(), see
    # "current and fresh values are labelled so they cannot be swapped".
    s = interpret.verdict_sentence(_reading(61.8, 62.7, 3.8, 0), "Subject 13")
    assert "Hz" not in s, s
    assert "1% below their own fresh level" in s, s


@check("an unchanged signal is not reported as '0% below'")
def _():
    # subject 7 sits 0.1 Hz off its fresh level; "0% below their own fresh
    # level" reads as a broken measurement rather than as a result
    s = interpret.verdict_sentence(_reading(77.0, 77.1, 2.5, 1), "Subject 7")
    assert "essentially unchanged" in s, s
    assert "0% below" not in s and "0% above" not in s, s


@check("a duplicate opening restatement of the verdict is dropped")
def _():
    # observed in all five sampled readings, under a rendered verdict line
    # that already said it
    prose = ("Subject 13 is showing no signs of fatigue at this point.\n\n"
             "This reading falls within their normal fresh range, so they can "
             "keep going at this intensity.")
    out = interpret.strip_verdict_echo(prose)
    assert out.startswith("This reading falls"), out
    assert "showing no signs" not in out, out


@check("prose that adds detail is never stripped, even if it mentions fatigue")
def _():
    prose = ("Subject 13 is showing signs of fatigue, and the drop has been "
             "steady since the 80 second mark rather than sudden, which is "
             "the usual pattern for a sustained effort like this one.\n\n"
             "They are close to the end of the recording.")
    assert interpret.strip_verdict_echo(prose) == prose


@check("a single-paragraph answer is never stripped to nothing")
def _():
    prose = "Subject 13 is showing signs of fatigue."
    assert interpret.strip_verdict_echo(prose) == prose


@check("a trailing restatement is dropped as well as a leading one")
def _():
    # the not-fatigued answers restated the verdict at BOTH ends, sandwiching
    # the sentences that actually said something
    prose = ("Subject 13 is showing no signs of fatigue.\n\n"
             "This small deviation does not indicate fatigue because it falls "
             "within the expected variation during an effort.\n\n"
             "The subject is not fatigued at 65 seconds.")
    out = interpret.strip_verdict_echo(prose)
    assert out.startswith("This small deviation"), out
    assert out.endswith("during an effort."), out


@check("a paragraph opening with the subject's name is a restatement")
def _():
    # observed live, duplicating the rendered verdict directly above it
    prose = ("Subject 13 is showing a muscle signal slightly below their "
             "fresh level, which puts this reading within their normal "
             "range.\n\n"
             "For it to count as fatigue the signal would have to fall much "
             "further.")
    out = interpret.strip_verdict_echo(prose, "Subject 13")
    assert out.startswith("For it to count"), out
    # ...and without knowing the name there is nothing safe to match on
    assert interpret.strip_verdict_echo(prose) == prose


@check("content-shaped openings survive even with the name known")
def _():
    # widening the pattern by wording would have deleted this -- a genuinely
    # useful answer that opens the same way the restatements describe things
    prose = ("This reading falls within their normal fresh range, indicating "
             "the signal has barely moved.\n\n"
             "They could keep going at this intensity.")
    assert interpret.strip_verdict_echo(prose, "Subject 13") == prose


@check("an upload's answers are de-duplicated too")
def _():
    prose = ("This recording is showing a signal below its fresh level.\n\n"
             "The drop is large enough to count as fatigue.")
    out = interpret.strip_verdict_echo(prose, "This recording")
    assert out.startswith("The drop is large"), out


@check("stripping echoes never empties an answer")
def _():
    prose = ("Subject 13 is not fatigued.\n\nThey show no signs of fatigue.")
    out = interpret.strip_verdict_echo(prose)
    assert out.strip(), "stripped the answer to nothing"


# --- follow-ups that only say the previous answer again ---------------------
# Asked "so what does this mean?" under a reading, the model returned three
# paragraphs and all three restated the verdict the reader had just read. The
# prompt forbids it; this is what happens when the prompt does not hold.

@check("a follow-up sentence copied from the previous answer is dropped")
def _():
    previous = ("Subject 11 is not showing signs of fatigue, 60s in. Their "
                "muscle signal is 5% below their own fresh level.")
    prose = ("Subject 11 is not showing signs of fatigue, 60s in.\n\n"
             "It would take roughly a fifth off their fresh level before this "
             "would read as fatigue.")
    out = interpret.drop_repeated_sentences(prose, previous)
    assert "60s in" not in out, out
    assert "roughly a fifth" in out, out


@check("a paraphrase of the previous answer is dropped too, not just a copy")
def _():
    previous = "Subject 11 is not showing signs of fatigue at 60 seconds."
    prose = ("Subject 11 is not fatigued at 60 seconds.\n\n"
             "The reading sits comfortably inside the spread their rested "
             "recordings covered, so there is room before it would change.")
    out = interpret.drop_repeated_sentences(prose, previous)
    assert "not fatigued at 60 seconds" not in out, out
    assert "comfortably inside" in out, out


@check("a follow-up that adds something survives untouched")
def _():
    previous = ("Subject 11 is not showing signs of fatigue, 60s in. Their "
                "muscle signal is 5% below their own fresh level.")
    prose = ("The reading sits inside the range their rested recordings "
             "covered, so nothing here is outside normal variation. It would "
             "need to fall roughly four times further before the answer "
             "changed. Worth checking again later in the effort.")
    assert interpret.drop_repeated_sentences(prose, previous) == prose


@check("dropping repeats never empties an answer")
def _():
    previous = "Subject 11 is not showing signs of fatigue at 60 seconds."
    prose = "Subject 11 is not showing signs of fatigue at 60 seconds."
    out = interpret.drop_repeated_sentences(prose, previous)
    assert out.strip(), "stripped the follow-up to nothing"


@check("a short sentence is left for the trimmer rather than judged on nothing")
def _():
    # three content words, all of them in the previous answer -- scored on so
    # little that anything would round to a repeat
    previous = "Subject 11 is not showing signs of fatigue at 60 seconds."
    prose = "Fatigue is gradual. It builds as the muscle's fibres slow down."
    out = interpret.drop_repeated_sentences(prose, previous)
    assert "Fatigue is gradual" in out, out


@check("confidence and the raw figures are withheld from the prose")
def _():
    # Given the confidence the model called it certainty -- "extremely sure",
    # "100% certain". Given the technical line it wrote "with a z-score of
    # -1.2 and confidence level of 100.0%" into an answer meant for a
    # non-specialist. Neither reaches it now; both are rendered under the
    # answer instead.
    described = _reading(58.2, 62.7, 3.8, 1)
    lines = interpret.plain_lines(described, "Subject 13")
    assert any("certainty in the fatigued" in l for l in lines), lines
    built = prompt.build_prompt(
        {"mdf_hz": 58.2, "fatigue_label": 1, "confidence": 0.98,
         "fatigue_state": "fatigue"},
        "is subject 13 fatigued?", None, None,
        {"lines": lines, "technical": interpret.technical_line(described)})
    assert "certainty in the fatigued" not in built, built
    assert "z = " not in built and "confidence 98" not in built, built


@check("the prose notes contain no numbers at all")
def _():
    # every figure given to the model came back misattributed somewhere: the
    # current reading reported as the baseline, "1.2 standard deviations"
    # quoted as proof a reading was *within* a range it was below, a z-score
    # pasted in as a "Caveat"
    import re as _re
    for mdf, fresh, label in ((61.8, 62.7, 0), (58.2, 62.7, 1),
                              (77.0, 77.1, 1), (56.5, 70.8, 1)):
        notes = interpret.prose_notes(_reading(mdf, fresh, 3.8, label),
                                      "Subject 13")
        joined = " ".join(notes)
        digits = _re.sub(r"Subject 13|eight", "", joined)
        assert not _re.search(r"\d", digits), f"number leaked: {joined}"


@check("fabricated measurements are stripped from the prose")
def _():
    # observed verbatim: the real values were 58.2 and 62.7 Hz, and the model
    # was given neither
    prose = ("This reading indicates that their muscle signal has dropped "
             "below their normal fresh level. The current reading of 12 Hz is "
             "lower than the subject's fresh level of 15 Hz.")
    cleaned, invented = interpret.strip_invented_numbers(prose)
    assert "12 Hz" not in cleaned and "15 Hz" not in cleaned, cleaned
    assert cleaned.startswith("This reading indicates"), cleaned
    assert len(invented) == 1, invented


@check("percentages, z-scores and bare numbers are stripped too")
def _():
    for bad in ("It has fallen by 7% from their fresh level.",
                "The reading sits at 1.2 standard deviations.",
                "Their signal is now 58.2 below where it started.",
                # a timestamp lifted from the provenance line and reported as
                # a measurement -- observed on an uploaded recording
                "The person's current reading is 200s, which is outside "
                "their normal fresh range."):
        cleaned, invented = interpret.strip_invented_numbers(
            "Their muscle signal is below its fresh level. " + bad)
        assert invented, bad
        assert cleaned == "Their muscle signal is below its fresh level.", cleaned


@check("the subject's own name may contain digits without being stripped")
def _():
    prose = "Subject 13 has moved well outside their usual range."
    cleaned, invented = interpret.strip_invented_numbers(prose, "Subject 13")
    assert cleaned == prose, cleaned
    assert not invented, invented
    # ...but a real number in the same sentence still goes
    prose2 = "Subject 13 has fallen to 51 below their usual range."
    cleaned2, invented2 = interpret.strip_invented_numbers(prose2, "Subject 13")
    assert not cleaned2 and invented2, (cleaned2, invented2)


@check("word-only reasoning is never stripped")
def _():
    prose = ("This drop is small enough to sit inside their normal range, so "
             "it does not count as fatigue.\n\nFor it to count, the signal "
             "would have to fall considerably further.")
    cleaned, invented = interpret.strip_invented_numbers(prose)
    assert cleaned == prose, cleaned
    assert not invented, invented


@check("the prose notes still carry the verdict, size and any conflict")
def _():
    notes = interpret.prose_notes(_reading(56.5, 70.8, 3.8, 1), "Subject 1")
    joined = " ".join(notes)
    assert "SHOWING signs of fatigue" in joined, joined
    assert "far below their own fresh level" in joined, joined

    notes = interpret.prose_notes(_reading(61.8, 62.7, 3.8, 0), "Subject 13")
    joined = " ".join(notes)
    assert "NOT showing signs of fatigue" in joined, joined
    assert "a little below their own fresh level" in joined, joined

    notes = interpret.prose_notes(_reading(77.0, 77.1, 2.5, 1), "Subject 7")
    assert any("the two disagree here" in n for n in notes), notes


@check("build_prompt prefers the numberless notes over the full lines")
def _():
    described = _reading(58.2, 62.7, 3.8, 1)
    built = prompt.build_prompt(
        {"mdf_hz": 58.2, "fatigue_label": 1, "confidence": 0.98,
         "fatigue_state": "fatigue"},
        "is subject 13 fatigued?", None, None,
        {"lines": interpret.plain_lines(described, "Subject 13"),
         "prose": interpret.prose_notes(described, "Subject 13"),
         "technical": interpret.technical_line(described)})
    assert "58.2" not in built and "62.7" not in built, built
    assert "standard deviations" not in built, built


@check("a not-fatigued reading is not asked a question that invites fatigue")
def _():
    # "what does this mean in practical terms" about a not-fatigued reading
    # produced "their muscles are starting to feel fatigued", under a rendered
    # line saying they were not
    not_fat = prompt.build_prompt(
        {"mdf_hz": 61.8, "fatigue_label": 0, "confidence": 0.98,
         "fatigue_state": "non-fatigue"}, "is subject 13 fatigued?")
    assert "does not count as fatigue" in not_fat, not_fat
    assert "starting to fatigue" in not_fat, "missing the anti-hedge rule"

    fat = prompt.build_prompt(
        {"mdf_hz": 58.2, "fatigue_label": 1, "confidence": 0.98,
         "fatigue_state": "fatigue"}, "is subject 13 fatigued?")
    assert "practical terms" in fat, fat
    assert "does not count as fatigue" not in fat, fat


@check("the prose is told not to predict the future without a forecast")
def _():
    # "The trend in fatigue suggests that it will likely continue to worsen"
    # -- with nothing in the answer measuring the future
    built = prompt.build_prompt(
        {"mdf_hz": 56.5, "fatigue_label": 1, "confidence": 1.0,
         "fatigue_state": "fatigue"},
        "is subject 1 fatigued?", None, None, None)
    assert "Nothing here measures the future" in built, built


@check("a fatigued verdict inside the normal range is flagged as contested")
def _():
    # subject 7 at z = -0.04: fatigued, but the marker has not moved
    r = _reading(77.0, 77.1, 2.5, 1)
    assert r["conflict"], r
    lines = interpret.plain_lines(r, "Subject 7")
    assert any("the two disagree here" in l for l in lines), lines


@check("a clearly fatigued reading is not flagged as contested")
def _():
    r = _reading(50.0, 62.7, 3.8, 1)      # z well below -1
    assert not r["conflict"], r


# --- the recommendation box -------------------------------------------------
# It was the one answer still asking the model to state the verdict. Asked to
# restate it "exactly as the facts above state it", it opened with "they are
# showing signs of fatigue" under a rendered line saying they were NOT.

@check("the recommendation no longer asks the model for the verdict")
def _():
    built = recommend.build_recommendation_prompt(
        {"mdf_hz": 61.8, "fatigue_label": 0, "confidence": 0.98,
         "fatigue_state": "non-fatigue"}, None, "what sport would suit them?")
    assert "ALREADY been written out" in built, built
    assert "Do NOT restate it" in built, built
    assert "Start by saying whether it is showing signs of fatigue" not in built


@check("the recommendation is handed the reading without its numbers")
def _():
    described = _reading(58.2, 62.7, 3.8, 1)
    built = recommend.build_recommendation_prompt(
        {"mdf_hz": 58.2, "fatigue_label": 1, "confidence": 0.98,
         "fatigue_state": "fatigue"}, None, "what gym plan should they follow?",
        None, {"lines": interpret.plain_lines(described, "Subject 13"),
               "prose": interpret.prose_notes(described, "Subject 13"),
               "technical": interpret.technical_line(described)})
    assert "58.2" not in built and "62.7" not in built, built


@check("hertz and percentages are stripped from a suggestion")
def _():
    text = ("Focus on active recovery this week. Their signal is 12% below "
            "their fresh level of 70.8 Hz, so scale back.")
    out = recommend.strip_measurements(text)
    assert out == "Focus on active recovery this week.", out


@check("ordinary training numbers survive the strip")
def _():
    # sets, reps and rest days are the substance of the suggestion -- only
    # measurements are invented here
    text = ("Try 3 sets of 10 repetitions, twice a week, with 48 hours "
            "between sessions.")
    assert recommend.strip_measurements(text) == text


@check("the disclaimer still survives a stripped suggestion")
def _():
    assert "not medical" in recommend.ensure_disclaimer(
        recommend.strip_measurements("Rest today.")).lower()


# --- the follow-up ("why?") -------------------------------------------------
# The causal chain (fatigue -> slower conduction -> lower frequency) was pasted
# into every follow-up. Re-explaining subject 7, whose median frequency ROSE,
# the model recited it and closed "...shifts the signal's power to lower
# frequencies, exactly what happened here" -- directly contradicting the answer
# it was re-explaining.

@check("a rising signal is detected from the facts or the answer it produced")
def _():
    rose = prompt.overview_facts(_summary(57.0, 75.0, _onset()))
    assert prompt._signal_rose(rose, ""), rose
    assert prompt._signal_rose([], "Subject 7 is 5% above their own fresh level")
    fell = prompt.overview_facts(_summary(75.0, 57.0, _onset()))
    assert not prompt._signal_rose(fell, ""), fell
    assert not prompt._signal_rose([], "Subject 13 is 3% below their fresh level")


@check("a follow-up about a rising signal must not claim the usual pattern")
def _():
    for kind in (intent.WHY, intent.MEANING, intent.SIMPLER, intent.MORE):
        built = prompt.build_followup_prompt(
            "summarise subject 7", "median frequency ROSE by 4.8 Hz",
            prompt.overview_facts(_summary(57.0, 75.0, _onset())), "why?", kind)
        # the warning is not conditional on which follow-up was asked: any of
        # the four can be answered with "and that is why the frequency fell"
        assert "does NOT follow the usual fatigue pattern" in built, (kind, built)
        assert "went UP, not down" in built, (kind, built)
        assert "never say the usual pattern is what happened here" in built.lower()


@check("a follow-up about a falling signal still gets the plain causal chain")
def _():
    built = prompt.build_followup_prompt(
        "summarise subject 13", "median frequency FELL by 18.0 Hz",
        prompt.overview_facts(_summary(75.0, 57.0, _onset())), "why?",
        intent.WHY)
    assert "shifts the signal's power to lower frequencies" in built, built
    assert "does NOT follow the usual fatigue pattern" not in built, built


# The causal chain answers "why?". Pasted onto the other three follow-ups it
# spent the whole 2-4 sentence budget on textbook physics and never reached
# what was asked -- and on "in simpler terms" it reintroduced the exact
# vocabulary that request exists to get rid of.

@check("only a why-follow-up is asked for the causal chain")
def _():
    facts = prompt.overview_facts(_summary(75.0, 57.0, _onset()))
    for kind in (intent.MEANING, intent.SIMPLER, intent.MORE):
        built = prompt.build_followup_prompt("summarise subject 13",
                                             "median frequency FELL", facts,
                                             "so what does this mean?", kind)
        assert "shifts the signal's power" not in built, (kind, built)


@check("every follow-up is told not to restate the answer it is explaining")
def _():
    for kind in (intent.WHY, intent.MEANING, intent.SIMPLER, intent.MORE):
        built = prompt.build_followup_prompt("summarise subject 13", "answered",
                                             [], "why?", kind)
        assert "NEVER open by restating the finding" in built, (kind, built)
        assert "must add something that answer did not contain" in built, kind


@check("the four follow-ups are set four different tasks")
def _():
    tasks = {prompt._FOLLOWUP_TASK[k] for k in
             (intent.WHY, intent.MEANING, intent.SIMPLER, intent.MORE)}
    assert len(tasks) == 4, tasks


# Ollama reuses a cached prompt prefix only up to the first byte that differs,
# so a block that changes per question placed ABOVE a fixed one costs seconds
# with nothing in the output to show for it. The follow-up prompt used to put
# the previous exchange in the middle and the instructions after it.

# Asked "why?" under a NOT-fatigued reading, the model answered "because their
# muscle signal power has shifted towards lower frequencies" -- reciting the
# mechanism of fatigue as the reason for its absence. Observed live. The chain
# describes what fatigue does, so under a not-fatigued verdict it has to be
# asked for in the conditional.

@check("a not-fatigued verdict is recognised from the answer being re-explained")
def _():
    assert prompt._reads_not_fatigued(
        [], "**Subject 11 is not showing signs of fatigue**, 60s in.")
    assert prompt._reads_not_fatigued(
        ["Subject 11 is NOT fatigued at this point"], "")
    assert not prompt._reads_not_fatigued(
        [], "**Subject 13 is showing signs of fatigue**, 200s in.")


@check("why under a not-fatigued verdict asks for the chain in the conditional")
def _():
    built = prompt.build_followup_prompt(
        "is subject 11 fatigued at 60s?",
        "**Subject 11 is not showing signs of fatigue**, 60s in.",
        ["Subject 11 is NOT fatigued at this point"], "why?", intent.WHY)
    assert "would shift the signal's power" in built, built
    assert "never give the fatigue mechanism as the reason" in built.lower()
    # the qualifier must come BEFORE the mechanism, or the length cap cuts it
    assert "not moved far enough" in built, built
    assert built.index("not moved far enough") < built.index("would shift")


@check("why under a fatigued verdict still gets the chain stated flat")
def _():
    built = prompt.build_followup_prompt(
        "is subject 13 fatigued at 200s?",
        "**Subject 13 is showing signs of fatigue**, 200s in.",
        ["Subject 13 IS showing signs of fatigue"], "why?", intent.WHY)
    assert "which shifts the signal's power to lower frequencies" in built, built
    assert "would shift" not in built, built


# A dataset subject is a third party. With nobody named, a follow-up about
# subject 11 came back as "your muscle signal ... how close you are to
# fatigue", handing the reader someone else's measurement as their own.

@check("a follow-up about a dataset subject is told to stay in the third person")
def _():
    built = prompt.build_followup_prompt("q", "a", [], "so what does this mean?",
                                         intent.MEANING, "Subject 11")
    assert "NOT the person reading this" in built, built
    assert "never \"you\" or \"your\"" in built, built


@check("a follow-up about the reader's own upload keeps the second person")
def _():
    built = prompt.build_followup_prompt("q", "a", [], "so what does this mean?",
                                         intent.MEANING, "This recording")
    assert "\"you\" and \"your\" are correct" in built, built
    assert "NOT the person reading this" not in built, built


@check("the follow-up prompt keeps its fixed instructions ahead of the data")
def _():
    built = prompt.build_followup_prompt("summarise subject 13", "ANSWER-HERE",
                                         ["FACT-HERE"], "QUERY-HERE",
                                         intent.WHY)
    assert built.index("NEVER open by restating") < built.index("ANSWER-HERE")
    assert built.index("ANSWER-HERE") < built.index("FACT-HERE") < built.index("QUERY-HERE")


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


# --- conversation state -----------------------------------------------------
# "+ New chat" cleared two session keys by hand and missed four. A fresh chat
# inherited the previous one's last answer, so "why?" as the very first
# message produced a confident explanation of a conversation the reader had
# never had, quoting figures from it. app.py imports streamlit, so this is
# checked against the source rather than by running it.

# Session keys that survive a chat change on purpose. Anything NOT listed here
# must appear in _CONVERSATION_KEYS, which is what forces the choice to be
# made deliberately.
_KEEPS_ACROSS_CHATS = {
    "chat",           # replaced by the caller, not reset
    "model",          # a UI preference
    "sample_rate",    # a UI preference
    "athlete_note",   # a UI preference
    "uploads",        # a cache keyed by file, unreachable once last_upload goes
    "confirm_clear",  # transient UI state for one button
    "draft_nonce",    # bumped rather than cleared, to force the box to redraw
}


@check("every conversation-scoped session key is cleared on a new chat")
def _():
    import re as _re
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "app.py"), encoding="utf-8").read()
    initialised = set(_re.findall(r'if "(\w+)" not in st\.session_state', src))
    block = _re.search(r"_CONVERSATION_KEYS = \{(.*?)\n\}", src, _re.S)
    assert block, "could not find _CONVERSATION_KEYS in app.py"
    reset = set(_re.findall(r'"(\w+)":', block.group(1)))

    missed = initialised - reset - _KEEPS_ACROSS_CHATS
    assert not missed, (f"session key(s) {sorted(missed)} are never cleared "
                        "when the chat changes -- add them to "
                        "_CONVERSATION_KEYS, or to _KEEPS_ACROSS_CHATS here "
                        "if they genuinely should survive")
    stale = reset - initialised
    assert not stale, f"_CONVERSATION_KEYS lists unknown key(s) {sorted(stale)}"


@check("both chat-switch paths reset through the shared helper")
def _():
    import re as _re
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "app.py"), encoding="utf-8").read()
    # The original bug was clearing keys inline, one at a time. First-time
    # initialisers look identical, so only flag an assignment that is NOT
    # guarded by its own "if not in st.session_state" check.
    lines = src.splitlines()
    inline = []
    for i, line in enumerate(lines):
        m = _re.match(r"\s*st\.session_state\.(last_\w+) = None\s*$", line)
        if not m:
            continue
        guard = f'if "{m.group(1)}" not in st.session_state:'
        preceding = [l for l in lines[max(0, i - 6):i]
                     if l.strip() and not l.strip().startswith("#")]
        if not preceding or guard not in preceding[-1]:
            inline.append(line.strip())
    assert not inline, (f"{inline} cleared inline; use _reset_conversation() "
                        "so no key is forgotten")
    assert src.count("_reset_conversation()") >= 4, \
        "expected the helper at its definition and every chat-switch path"


@check("llama3.2:1b's inverted opening verdict is stripped")
def _():
    # 1b opens EVERY reading with "The person is not fatigued." -- including
    # for subjects the classifier called fatigued. It writes it as its own
    # paragraph, which is what lets strip_verdict_echo drop it; the rendered
    # verdict above the prose is the one the reader sees, and it is built in
    # Python from the label, so it cannot be inverted.
    out = interpret.strip_verdict_echo(
        "The person is not fatigued.\n\nThis reading puts them clearly below "
        "their normal fresh range.", "Subject 4")
    assert not out.startswith("The person is not fatigued"), out
    assert "clearly below" in out, out


@check("an inline echo survives, and that is deliberate")
def _():
    # The stripper works on paragraphs and never removes the last one, so it
    # cannot empty an answer or delete reasoning. The cost is that a verdict
    # echo written inline, in the same paragraph as real content, is kept.
    # Recorded here so the limit is known rather than assumed away: if a model
    # starts inlining inverted verdicts, this is the test that has to change.
    inline = "The person is not fatigued. This reading means the muscle is tired."
    assert interpret.strip_verdict_echo(inline, "Subject 4") == inline


@check("lab-report words the model keeps using are swapped for plain ones")
def _():
    # observed live under a prompt that explicitly forbade the word
    out = interpret.plain_words(
        "Their signal dropped, indicating they are no longer at peak.")
    assert "indicating" not in out, out
    assert "which means they are no longer at peak" in out, out
    # capitalisation is carried over, so a swap can open a sentence
    assert interpret.plain_words("Indicating a drop.").startswith("Showing"), \
        interpret.plain_words("Indicating a drop.")
    # ...but after a verb "indicating" is a participle, not a connective, and
    # "which means" there produced "would start which means fatigue" live
    after_verb = interpret.plain_words(
        "There is time before the model would start indicating fatigue.")
    assert "start showing fatigue" in after_verb, after_verb
    assert "which means" not in after_verb, after_verb
    # the longer phrase wins over the shorter key inside it
    assert "a sign of" in interpret.plain_words("This is indicative of fatigue."), \
        interpret.plain_words("This is indicative of fatigue.")


@check("the model's prose is cut to two sentences however long it runs")
def _():
    # asked for "ONE or TWO short sentences", llama3.2:3b wrote four
    long = ("This reading falls within their normal fresh range. A change "
            "this small does not count as fatigue. For it to count there "
            "would need to be more. And a fourth sentence.")
    out = interpret.trim_sentences(long, 2)
    assert out.endswith("does not count as fatigue."), out
    assert "fourth" not in out, out


@check("an answer cut off mid-sentence loses the unfinished part")
def _():
    # what hitting llm.NUM_PREDICT looks like: generation stops mid-word
    cut = ("This small change does not count as fatigue. Their muscle signal "
           "is still with")
    out = interpret.trim_sentences(cut, 2)
    assert out == "This small change does not count as fatigue.", out
    # a complete second sentence is of course kept
    whole = ("This small change does not count as fatigue. Their muscle "
             "signal is still within range.")
    assert interpret.trim_sentences(whole, 2) == whole


@check("a cut-off sentence in its own paragraph is dropped too")
def _():
    # The model writes the truncated sentence as a separate paragraph as often
    # as inline. Judging "did anything complete survive?" per paragraph kept
    # this fragment, because that paragraph alone had nothing complete in it.
    cut = ("Muscles tire.\n\nAs the body's energy stores are depleted, "
           "fatigue sets in, causing a decline in")
    assert interpret.trim_sentences(cut, 2) == "Muscles tire.", \
        interpret.trim_sentences(cut, 2)


@check("a lone unfinished sentence is kept rather than emptying the answer")
def _():
    # nothing else survived, so showing the fragment beats showing nothing --
    # the rendered verdict above it still carries the finding either way
    only = "Their muscle signal is still with"
    assert interpret.trim_sentences(only, 2) == only


@check("trimming does not split a decimal or flatten paragraphs and lists")
def _():
    # "66.8 Hz" must not read as a sentence boundary
    assert interpret.trim_sentences("It is 66.8 Hz now. That is fine.", 2) == \
        "It is 66.8 Hz now. That is fine."
    # each paragraph gets its own budget -- the caller stacks the rendered
    # verdict and the recommendation as separate paragraphs
    two = interpret.trim_sentences("A one. A two. A three.\n\nB one. B two.", 2)
    assert two == "A one. A two.\n\nB one. B two.", two
    # a bullet list is one list, not N sentences to cut in half
    bullets = "- first\n- second\n- third"
    assert interpret.trim_sentences(bullets, 2) == bullets


# The per-paragraph budget puts no ceiling on an answer whose paragraphs are
# ALL the model's, which is the follow-up case: asked for 2-4 sentences it
# returned three paragraphs of three, every one inside the per-paragraph limit.

# Observed live: both figures real and correctly attributed, the relation
# between them backwards. 66.8 is below 70.0, not above it.

@check("a follow-up placing one hertz figure above another loses that sentence")
def _():
    prose = ("Their signal has eased off a little. Their current signal is "
             "66.8 Hz, which is still above the fresh level of 70.0 Hz. It is "
             "worth another reading later on.")
    out = interpret.drop_hertz_comparisons(prose)
    assert "above the fresh level" not in out, out
    assert "eased off a little" in out and "another reading later" in out, out
    # the decimal point inside "70.0 Hz" must not end the sentence for the
    # match -- written [^.!?]* it stopped there and let this straight through
    threshold = ("It is worth watching. We need the point at which their "
                 "signal would drop below 70.0 Hz. That is the fresh level.")
    out = interpret.drop_hertz_comparisons(threshold)
    assert "drop below 70.0 Hz" not in out, out
    assert "worth watching" in out and "fresh level" in out, out


@check("percentages, times and a lone hertz figure are left alone")
def _():
    for keep in ("Their signal is 5% below their own fresh level.",
                 "This reading was taken 60 seconds in.",
                 "The median frequency here is 66.8 Hz.",
                 "It is lower than it was when they started."):
        assert interpret.drop_hertz_comparisons(keep) == keep, keep


@check("dropping hertz comparisons never empties an answer")
def _():
    only = "It is 66.8 Hz, above the fresh level of 70.0 Hz."
    assert interpret.drop_hertz_comparisons(only).strip(), "stripped to nothing"


@check("every follow-up is told to leave the hertz figures out")
def _():
    built = prompt.build_followup_prompt("q", "a", [], "why?", intent.WHY)
    assert "never say one hertz figure is above or below another" in built, built


@check("a total budget caps the whole answer, not just each paragraph")
def _():
    prose = "A one. A two. A three.\n\nB one. B two.\n\nC one."
    out = interpret.trim_sentences(prose, 3, total=4)
    assert out == "A one. A two. A three.\n\nB one.", out


@check("the total budget drops whole paragraphs, never part of one")
def _():
    prose = "A one. A two.\n\nB one. B two."
    assert interpret.trim_sentences(prose, 3, total=2) == "A one. A two."


@check("no total budget leaves the per-paragraph behaviour untouched")
def _():
    prose = "A one. A two. A three.\n\nB one. B two."
    assert (interpret.trim_sentences(prose, 3)
            == interpret.trim_sentences(prose, 3, total=None) == prose)


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
