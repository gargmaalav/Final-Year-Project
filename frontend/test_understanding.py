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

SUBJECTS = list(range(1, 14))
DURATION = 200.0        # stand-in recording length for the anchor cases

# (question, expected intent, expected subject, expected t_start)
# t_start None means "should ask rather than guess"
ROUTING = [
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

# Out-of-range input must produce a stated correction, not a silent fallback
# to the previous turn's window.
MUST_FLAG = [
    ("what about subject 20?", "subject 20"),
    ("subject 4 at 9999 seconds", "200s"),
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
