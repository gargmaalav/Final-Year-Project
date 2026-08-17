"""
Regression set for question understanding.

    python frontend/test_understanding.py

Checks routing and slot resolution ONLY -- no model, no dataset, and the LLM
fallback switched off, so it runs in about a second on any machine and gives
the same answer every time. The point is that "the chatbot handles loose
phrasing" becomes a number that either holds or drops when someone edits a
regex, rather than a claim resting on whoever last tried it by hand. Every
case here must therefore pass on the deterministic patterns alone.

The live section at the end is the exception: it runs only when Ollama is
reachable, and checks the opposite property -- that when a real 3B model is
asked a follow-up with no subject in it and invents one, grounding still
throws the invented window away.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import extract          # noqa: E402
import intent           # noqa: E402
import recommend        # noqa: E402

SUBJECTS = list(range(1, 14))
DURATION = 200.0        # stand-in recording length for the anchor cases

# (question, expected intent, expected subject, expected t_start)
# t_start None means "should ask rather than guess"
ROUTING = [
    # --- a misspelt or shortened subject word ----------------------------
    # One typo used to lose the subject entirely, and a question that plainly
    # named a subject and a time came back as "which subject did you mean?".
    ("is subjet 5 fatigued at 60 seconds",           intent.READING, 5, 60.0),
    ("is subjcet 5 fatigued at 60 seconds",          intent.READING, 5, 60.0),
    ("is sub 5 fatigued at 60 seconds",              intent.READING, 5, 60.0),
    ("particpant 3 at 60 seconds",                   intent.READING, 3, 60.0),
    # --- open-ended about one subject: offer the menu, don't demand a time
    ("hows subject 5 doing",                         intent.MENU, 5, None),
    ("how's subject 5 doing?",                       intent.MENU, 5, None),
    ("whats up with subject 5",                      intent.MENU, 5, None),
    ("sub 5",                                        intent.MENU, 5, None),

    # --- plain readings, the original supported shape --------------------
    ("Is subject 13 fatigued at 60 seconds?",        intent.READING, 13, 60.0),
    ("subject 13 at 60s right side",                 intent.READING, 13, 60.0),
    ("s13 at 60s",                                   intent.READING, 13, 60.0),
    ("participant 4, 120 seconds",                   intent.READING, 4, 120.0),

    # --- loose phrasing that used to fail --------------------------------
    ("is subject thirteen fatigued at forty five seconds",
                                                     intent.READING, 13, 45.0),
    ("subject 13 at 1:30",                           intent.READING, 13, 90.0),
    ("subject 13 at 1 minute 30",                    intent.READING, 13, 90.0),
    ("subject 13 two minutes in",                    intent.READING, 13, 120.0),
    ("how about subject 5 near the end",             intent.READING, 5, 200.0),
    ("subject 5 towards the end",                    intent.READING, 5, 200.0),
    ("subject 6 at the start",                       intent.READING, 6, 0.0),
    ("subject 7 halfway through",                    intent.READING, 7, 100.0),
    ("subject 9 in the last 30 seconds",             intent.READING, 9, 170.0),
    ("at 60 seconds for subject 13",                 intent.READING, 13, 60.0),
    ("subject 2 half a minute in",                   intent.READING, 2, 30.0),
    # decimals must belong to the number: "60.5" once matched as "60", failed
    # on the ".", then re-matched later and read the window as 5 seconds
    ("subject 13 at 60.5 seconds",                   intent.READING, 13, 60.5),
    ("subject 4 at 12.25 seconds",                   intent.READING, 4, 12.25),
    ("subject 13 at 1.5 minutes",                    intent.READING, 13, 90.0),

    # --- new question types ----------------------------------------------
    ("when did subject 13 start getting fatigued?",  intent.ONSET, 13, None),
    ("at what point does subject 4 fatigue",         intent.ONSET, 4, None),
    ("summarise subject 2",                          intent.OVERVIEW, 2, None),
    ("how did subject 7 change over the whole recording?",
                                                     intent.OVERVIEW, 7, None),
    ("compare subject 5 and subject 9",              intent.COMPARE, 5, None),
    ("compare subject 5 and 9",                      intent.COMPARE, 5, None),
    ("which subject is most fatigued?",              intent.COMPARE, None, None),
    ("who fatigues the fastest",                     intent.COMPARE, None, None),
    # superlatives phrased with the verb first -- these fell through to a
    # single-window reading until the compare patterns were widened
    ("which subject fatigued the most?",             intent.COMPARE, None, None),
    ("which subject dropped the most",               intent.COMPARE, None, None),
    ("which participant fatigued fastest",           intent.COMPARE, None, None),
    ("what does median frequency mean?",             intent.EXPLAIN, None, None),
    ("what is MDF",                                  intent.EXPLAIN, None, None),
    ("explain the forecast",                         intent.EXPLAIN, None, None),
    ("what data do you have?",                       intent.CATALOGUE, None, None),
    ("which subjects are available",                 intent.CATALOGUE, None, None),

    # --- must NOT be misrouted -------------------------------------------
    # a window question that happens to start with "what is"
    ("what is subject 13's fatigue at 60 seconds",   intent.READING, 13, 60.0),
    # a recommendation request is a reading plus the recommendation block
    ("what should I do to train better",             intent.READING, None, None),
    ("recommend a training plan",                    intent.READING, None, None),
]

# A subject named with no question attached: offer the menu rather than
# guessing which of a dozen possible readings was wanted.
VAGUE = [
    "tell me about subject 13",
    "subject 13",
    "subject 5?",
    "info on subject 7",
    "show me subject 2",
]

BOTH_SIDES = [
    "compare the left and right arm for subject 4",
    "is subject 4 more fatigued in the left or right arm",
    "both arms for subject 9",
    "which arm is worse for subject 3",
]

# Follow-ups: (previous params, question, expected subject, expected t_start)
FOLLOW_UPS = [
    ({"subject": 13, "t_start": 60.0, "side": "R"}, "and at 90 seconds?", 13, 90.0),
    ({"subject": 13, "t_start": 60.0, "side": "R"}, "what about the left arm?", 13, 60.0),
    ({"subject": 13, "t_start": 60.0, "side": "R"}, "how about subject 5?", 5, 60.0),
]

# The invention failure this architecture exists to prevent: with no previous
# turn to carry from, a follow-up must ask, never fill in a plausible window.
MUST_ASK = [
    "what about the left arm?",
    "and at 90 seconds?",          # no subject anywhere
    "is it fatigued?",
]

# After a file is uploaded, the next question arrives with no file attached.
# It must still be answered about that recording, unless it names a subject or
# asks something the dataset owns.
STAYS_ON_UPLOAD = [
    ("when did I start fatiguing?", True),
    ("summarise my recording", True),
    ("am I fatigued at 100 seconds", True),
    ("how did it go overall", True),
    ("what about near the end", True),
    ("recommend a training plan", True),
    # names a subject, but the upload is the other half of the comparison
    ("compare me to subject 5", True),
    ("how do I compare to subject 9", True),
    # back to the dataset
    ("is subject 13 fatigued at 60 seconds", False),
    ("compare subject 5 and 9", False),
    ("which arm is worse for subject 4", False),
    ("what does median frequency mean?", False),
    ("what data do you have?", False),
    ("why?", False),
    ("explain that", False),
]

# An explicit reference to the uploaded file must win over whatever the
# conversation was last about. "summarise my recording", asked straight after
# a dataset question, was answered with a dataset subject's numbers.
NAMES_OWN_RECORDING = [
    ("summarise my recording", True),
    ("when did I start fatiguing?", True),
    ("how do I compare to subject 1", True),
    ("am I fatigued yet", True),
    ("what does my file show", True),
    ("summarise the uploaded recording", True),
    ("summarise subject 7", False),
    ("when did subject 4 start fatiguing?", False),
    ("compare subject 5 and 9", False),
    ("what does median frequency mean?", False),
]

# The sport/training block is gated on keywords, and several of those words
# have a second, machine-learning meaning. "What training data was used" is a
# question about the project; answering it with a diet plan is the kind of
# thing that makes the tool look unserious in a demo.
RECOMMEND = [
    ("what sport would suit me", True),
    ("recommend a training plan", True),
    ("any diet suggestions for subject 4", True),
    ("what gym work should I do", True),
    ("what training data was used?", False),
    ("how was the model trained", False),
    ("which subjects were in the training set", False),
    ("what was it trained on", False),
    ("is subject 13 fatigued at 60 seconds", False),
    ("when did subject 4 start fatiguing", False),
]

# Out-of-range input must produce a stated correction, not a silent fallback
# to the previous turn's window.
# Forecast horizons. "over the next minute" names a span with no digit in it,
# so it fell past the digit-based reader and took the 20 s default instead --
# a question that plainly said a minute was answered with a third of one.
# None means "no forecast asked for": every question getting an unrequested
# projection chart is the failure the default=None case exists to prevent.
HORIZONS = [
    ("will subject 2 get more tired over the next minute?", 60.0),
    ("what will it look like in the next minute?", 60.0),
    ("what about over the next 40 seconds?", 40.0),
    ("in the next 2 minutes", 120.0),
    # no definite span named, so the default stands rather than a guess
    ("will they tire over the next few minutes?", 20.0),
    ("will subject 2 get more tired?", 20.0),
    ("is subject 13 fatigued at 60 seconds?", None),
    ("summarise subject 7", None),
]

MUST_FLAG = [
    ("what about subject 20?", "subject 20"),
    ("subject 4 at 9999 seconds", "200s"),
    # a leading minus used to be dropped, so "-5 seconds" read as 5 seconds
    ("subject 4 at -5 seconds", "negative"),
]


def main() -> int:
    passed = failed = 0
    failures = []

    for question, want_kind, want_subject, want_t in ROUTING:
        got = intent.route(question)
        ok = got.kind == want_kind
        if ok and want_subject is not None:
            ok = bool(got.subjects) and got.subjects[0] == want_subject
        if ok and want_t is not None:
            resolved = extract.resolve_query(question, None, duration=DURATION,
                                             subjects=SUBJECTS, use_llm=False)
            ok = resolved.ok and abs(resolved.params["t_start"] - want_t) < 0.01
        if ok:
            passed += 1
        else:
            failed += 1
            failures.append(f"  {question!r}\n     wanted {want_kind}"
                            f"/subject={want_subject}/t={want_t}, got "
                            f"{got.kind}/subjects={got.subjects}")

    for question in VAGUE:
        got = intent.route(question)
        if got.kind == intent.MENU and got.subjects:
            passed += 1
        else:
            failed += 1
            failures.append(f"  {question!r}\n     wanted the subject menu, got "
                            f"{got.kind}")

    for question in BOTH_SIDES:
        got = intent.route(question)
        if got.kind == intent.COMPARE and got.both_sides:
            passed += 1
        else:
            failed += 1
            failures.append(f"  {question!r}\n     wanted a both-sides compare, "
                            f"got {got.kind} both_sides={got.both_sides}")

    for previous, question, want_subject, want_t in FOLLOW_UPS:
        resolved = extract.resolve_query(question, previous, duration=DURATION,
                                         subjects=SUBJECTS, use_llm=False)
        if (resolved.ok and resolved.params["subject"] == want_subject
                and abs(resolved.params["t_start"] - want_t) < 0.01):
            passed += 1
        else:
            failed += 1
            failures.append(f"  follow-up {question!r} after {previous}\n"
                            f"     wanted subject={want_subject} t={want_t}, "
                            f"got {resolved.params or resolved.ask}")

    for question in MUST_ASK:
        resolved = extract.resolve_query(question, None, duration=DURATION,
                                         subjects=SUBJECTS, use_llm=False)
        if not resolved.ok and resolved.ask:
            passed += 1
        else:
            failed += 1
            failures.append(f"  {question!r} invented {resolved.params} instead "
                            "of asking")

    for question, want in STAYS_ON_UPLOAD:
        if intent.stays_on_upload(question) == want:
            passed += 1
        else:
            failed += 1
            failures.append(f"  {question!r} after an upload should have gone "
                            f"to the {'upload' if want else 'dataset'}")

    for question, want in NAMES_OWN_RECORDING:
        if intent.names_own_recording(question) == want:
            passed += 1
        else:
            failed += 1
            failures.append(f"  {question!r} should {'' if want else 'not '}"
                            "have been read as naming the uploaded recording")

    for question, want in RECOMMEND:
        if recommend.wants_recommendation(question) == want:
            passed += 1
        else:
            failed += 1
            failures.append(f"  {question!r} should {'' if want else 'not '}"
                            "have triggered the sport/training block")

    # A message asking several things was answered with one and said nothing
    # about the rest. Precision matters more than recall here: a false extra
    # offers the reader a question they never asked.
    COMPOUND = [
        ("subject 5 and also how does that compare to 9 and when did it start",
         [intent.COMPARE]),
        ("summarise subject 7 and when did fatigue start", [intent.OVERVIEW]),
        ("what does median frequency mean and when did subject 5 start "
         "fatiguing", [intent.EXPLAIN]),
        # one question that happens to match two patterns is still one question
        ("which subject fatigued the most overall?", []),
        ("is subject 13 fatigued at 60 seconds?", []),
        ("compare subject 5 and 9", []),
        ("summarise subject 7", []),
        ("when did subject 13 start fatiguing?", []),
    ]
    for question, want_kinds in COMPOUND:
        got = [e.kind for e in intent.extra_requests(question)]
        if got == want_kinds:
            passed += 1
        else:
            failed += 1
            failures.append(f"  {question!r} should report extra requests "
                            f"{want_kinds}, reported {got}")

    # The fuzzy subject matcher must not fire on the ordinary words that sit
    # in front of a number. Every one of these would be a wrong subject, not
    # merely a missed one, which is the more damaging failure.
    for question in ("at 60 seconds", "in the next 2 minutes",
                     "compare 5 and 9", "between 5 and 9",
                     "what about 5 seconds", "around 30 secs",
                     "suggest 3 exercises", "minutes 5"):
        got = extract.subjects_in_text(question)
        if not got:
            passed += 1
        else:
            failed += 1
            failures.append(f"  {question!r} read a subject {got} out of an "
                            "ordinary word before a number")

    for question, want in HORIZONS:
        got = extract.extract_horizon_seconds(question, default=None)
        if got == want:
            passed += 1
        else:
            failed += 1
            failures.append(f"  {question!r} should give horizon {want}, "
                            f"gave {got}")

    # An unknown subject named the range in the problem AND again in the
    # question that followed it: "I don't have subject 99 -- the dataset has
    # subjects 1-13. Which subject did you mean? I have subjects 1-13."
    for question in ("is subject 99 fatigued at 60 seconds?",
                     "is subject 99 fatigued?"):
        resolved = extract.resolve_query(question, None, duration=DURATION,
                                         subjects=SUBJECTS, use_llm=False)
        whole = " ".join(resolved.problems + [resolved.ask or ""])
        if whole.count("subjects 1-13") == 1:
            passed += 1
        else:
            failed += 1
            failures.append(f"  {question!r} states the subject range "
                            f"{whole.count('subjects 1-13')} times: {whole!r}")

    for question, expect_in_message in MUST_FLAG:
        resolved = extract.resolve_query(question, {"subject": 13, "t_start": 60.0,
                                                    "side": "R"},
                                         duration=DURATION, subjects=SUBJECTS,
                                         use_llm=False)
        message = " ".join(resolved.problems)
        if expect_in_message in message:
            passed += 1
        else:
            failed += 1
            failures.append(f"  {question!r} should have said {expect_in_message!r}, "
                            f"said {message!r}")

    # Live check: the deterministic set above proves the patterns work. This
    # proves the other half -- that a real 3B model inventing a window on a
    # subject-less follow-up still cannot get that window through.
    #
    # Opt-in, because it is genuinely slow: measured at ~5 minutes per call on
    # a laptop with a cold llama3.2:3b, so the whole section took 20 minutes.
    # A test nobody will wait for is a test nobody runs.
    live_note = "skipped (pass --live to run)"
    reachable = False
    if "--live" in sys.argv:
        try:
            from llm import chat
            chat([{"role": "user", "content": "reply with the single word ok"}],
                 timeout=20)
            reachable = True
        except Exception as e:
            live_note = f"skipped (Ollama not answering: {e})"

    if reachable:
        live_pass = live_fail = 0
        for question in MUST_ASK:
            resolved = extract.resolve_query(question, None, duration=DURATION,
                                             subjects=SUBJECTS, use_llm=True)
            if not resolved.ok:
                live_pass += 1
            else:
                live_fail += 1
                failures.append(f"  LIVE: {question!r} let the model's invented "
                                f"window {resolved.params} through")
        passed += live_pass
        failed += live_fail
        live_note = f"{live_pass}/{live_pass + live_fail} held against a live model"

    if failures:
        print("FAILURES:")
        print("\n".join(failures))
    print(f"\n{passed} passed, {failed} failed")
    print(f"grounding vs. live LLM invention: {live_note}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
