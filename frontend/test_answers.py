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

import interpret       # noqa: E402
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


@check("the rendered verdict never swaps the fresh and current values")
def _():
    s = interpret.verdict_sentence(_reading(61.8, 62.7, 3.8, 0), "Subject 13")
    assert "62.7 Hz fresh" in s and "61.8 Hz now" in s, s
    assert s.index("62.7") < s.index("61.8"), "fresh must precede current"


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


@check("percentages and z-scores are stripped too")
def _():
    for bad in ("It has fallen by 7% from their fresh level.",
                "The reading sits at 1.2 standard deviations.",
                "Their signal is now 58.2 below where it started."):
        cleaned, invented = interpret.strip_invented_numbers(
            "Their muscle signal is below its fresh level. " + bad)
        assert invented, bad
        assert cleaned == "Their muscle signal is below its fresh level.", cleaned


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
