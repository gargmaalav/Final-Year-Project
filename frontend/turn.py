"""
Turn-handling engine, extracted from app.py (Maalav/Aryan's Streamlit frontend).
=================================================================================

Same pipeline, same modules (analysis/extract/intent/interpret/prompt/recommend/
classify/fatigue_forecast/render_window), just driven over HTTP by
models/serve.py's POST /turn instead of rendered by Streamlit -- see
docs/decisions.md for why: the group agreed to drop the Streamlit frontend and
use viz/chatbot_ui.html as the one UI, but keep this backend logic rather than
rebuild it, since none of it actually depends on Streamlit (`st.session_state`
was the only piece of glue, replaced below with an explicit `session` dict).

handle_turn(session, user_text, uploaded_file) is the single entry point,
equivalent to app.py's `_handle_turn` minus the chat-message bookkeeping (the
JS UI keeps its own chat list in localStorage, so there is no server-side
chat history to append to here).

`session` is a plain dict, one per browser chat (see new_session()), held in
models/serve.py's in-memory SESSIONS dict keyed by the session_id the UI
already assigns each chat. This is a local, single-process demo server (same
deployment model serve.py already had), so process memory is an acceptable
place for it -- no auth, no multi-tenancy, no persistence needed beyond the
process lifetime.
"""
from __future__ import annotations

import concurrent.futures
import functools
import os
import re
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO_ROOT, "models"),
           os.path.join(_REPO_ROOT, "viz"),
           os.path.join(_REPO_ROOT, "zenodo_biceps")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from classify import (classify, classify_upload,      # noqa: E402
                      available_subjects, subject_reference, upload_reference)
from fatigue_forecast import forecast_fatigue         # noqa: E402
from render_window import render_window               # noqa: E402
import loader as data_loader                           # noqa: E402

import analysis                                  # noqa: E402
import charts                                    # noqa: E402
import extract                                   # noqa: E402
import intent as intent_router                   # noqa: E402
import interpret                                 # noqa: E402
from llm import LLMError, chat, MODEL as llm_default_model  # noqa: E402
from prompt import (build_facts_prompt, build_followup_prompt,  # noqa: E402
                    build_prompt,
                    compare_facts, describe_window, onset_facts,
                    overview_facts, readable_facts,
                    render_onset_answer, strip_unfactual_numbers)
from recommend import (build_recommendation_prompt,  # noqa: E402
                       ensure_disclaimer, strip_measurements,
                       wants_recommendation)
from upload import UploadError, load_uploaded_segment, parse_uploaded_csv  # noqa: E402

# matches models/serve.py's _STATE mapping, kept in sync manually since this
# calls classify()/classify_upload() directly instead of going through
# serve.py's /classify HTTP route
_FATIGUE_STATE = {0: "non-fatigue", 1: "fatigue", 2: "fatigue"}
DATA_ROOT = os.path.join(_REPO_ROOT, "zenodo_biceps", "sEMG_data")

_READING_CHART_CAPTION = (
    "Top: the raw EMG signal at this moment. Middle: median frequency over "
    "time, colour-coded by fatigue stage -- the trend the reading is based "
    "on. Bottom: the frequency spectrum of this specific window.")
_UPLOAD_CHART_CAPTION = (
    "Top: your uploaded signal. Bottom: median frequency over time -- this "
    "trend is what the fatigue reading above is based on.")
_FORECAST_CHART_CAPTION = (
    "Grey dots are the measured median frequency so far. The dashed line "
    "projects the trend forward; the shaded bands show the typical and 95% "
    "uncertainty range around that projection.")

# Taken from llm.py rather than repeated: this was a second hardcoded
# "llama3.2:3b" that every new session used, so changing the model in llm.py
# alone had no effect on the app at all.
DEFAULT_MODEL = llm_default_model


def new_session() -> dict:
    """Fresh per-chat state -- what st.session_state held per browser session."""
    return {
        "model": DEFAULT_MODEL,
        "sample_rate": None,     # blank until the user states it
        "athlete_note": "",
        "uploads": {},           # key -> {"seg", "fs", "baseline", "name", "scan"}
        "last_upload": None,     # uploads key most recently attached
        "last_source": None,     # "upload" or "dataset"
        "last_turn_context": None,
        "last_params": None,     # last resolved {subject, t_start, side}
        "last_answer": None,     # {question, answer, facts} -- what "why?" refers to
        # the last answer asked which subject was meant, so a bare "5" on this
        # turn names one rather than being a stray number
        "awaiting_subject": False,
    }


# ---------------------------------------------------------------------------
# pipeline helpers -- module-level caches (cross-session, like st.cache_data
# was: keyed only by args, not by which chat asked)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=8)
def _load_subject_segment(subject: int, side: str):
    seg = data_loader.load_biceps_segment(DATA_ROOT, subject, side,
                                          target_fs=250, bandpass=True)
    return seg, int(getattr(seg, "eff_fs", 250))


@functools.lru_cache(maxsize=1)
def _subjects() -> list[int]:
    try:
        return available_subjects()
    except Exception:
        return list(range(1, 14))


# Rebuilding the figure costs ~2.2 s even when the underlying segment is
# already cached -- the time is Plotly assembling ~130 animation frames and
# serialising them, not reading the data. Asking the same question twice (or
# re-opening a chat) paid it again every time. Keyed on the three arguments
# this module actually passes; render_window's `model_pred` is unhashable and
# no longer drawn, which is why the cache lives here rather than on it.
#
# maxsize is small on purpose: each entry is a ~3.3 MB HTML string.
@functools.lru_cache(maxsize=8)
def _cached_chart(subject: int, t_start: float, side: str) -> str:
    return render_window(subject, t_start, side)


# Loads the recording's samples alongside classify(), which does its own
# loading of the same file -- run in sequence they cost ~1.2 s + ~1.1 s, and
# started together the turn waits only for the longer. One worker: this is a
# local demo server answering one question at a time, and any thread doing CPU
# work here is competing with the Ollama process that generates the answer.
_IO_POOL = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="segment")


def render_chart_ref(session: dict, ref: dict) -> str | None:
    """Draw the figure a `chart_ref` points at, on demand.

    Called by serve.py's GET /chart when the reader clicks "Show graph" --
    the only place a reading chart is built. Returns None rather than raising
    so a chart failure stays a missing picture, not a failed request.
    """
    if not ref:
        return None
    if ref.get("source") == "upload":
        cache = session["uploads"].get(session["last_upload"])
        return _upload_chart(cache) if cache else None
    try:
        return _cached_chart(int(ref["subject"]), float(ref["t_start"]),
                             str(ref["side"]))
    except Exception:
        return None


@functools.lru_cache(maxsize=8)
def _cached_scan(subject: int, side: str) -> dict:
    seg, fs = _load_subject_segment(subject, side)
    return analysis.summarise(analysis.scan_recording(subject, side, seg=seg, fs=fs))


@functools.lru_cache(maxsize=2)
def _cached_ranking(side: str) -> dict:
    return analysis.rank_subjects(_subjects(), side=side)


@functools.lru_cache(maxsize=8)
def _cached_compare_subjects(subjects: tuple[int, ...], t_start: float | None,
                             side: str) -> dict:
    return analysis.compare_subjects(list(subjects), t_start=t_start, side=side)


@functools.lru_cache(maxsize=8)
def _cached_compare_sides(subject: int, t_start: float | None) -> dict:
    return analysis.compare_sides(subject, t_start=t_start)


def _plain_reading(result: dict, reference: dict | None, t_start: float | None,
                   duration: float | None, who: str) -> dict | None:
    """interpret.py's plain-language view, or None if it can't be built.

    Never allowed to break an answer: if anything here fails the prompt falls
    back to stating the raw values, which is what it did before.
    """
    try:
        described = interpret.describe_reading(result, reference, t_start, duration)
        return {"lines": interpret.plain_lines(described, who),
                "prose": interpret.prose_notes(described, who),
                "verdict": interpret.verdict_sentence(described, who),
                "technical": interpret.technical_line(described),
                "who": who}
    except Exception:
        return None


def _duration_of(subject: int, side: str) -> float | None:
    """Recording length, needed to resolve "near the end" and to reject a time
    past the end of the data."""
    try:
        seg, _ = _load_subject_segment(subject, side)
        return float(seg.t[-1]) if seg.t.size else None
    except Exception:
        return None


def _subject_for(intent, previous: dict | None) -> int | None:
    """The subject an analysis question is about: named this turn, else the
    one already under discussion."""
    if intent.subjects:
        return intent.subjects[0]
    return (previous or {}).get("subject")


def _needs_subject_msg() -> str:
    subs = _subjects()
    return (f"Which subject did you mean? I have subjects "
            f"{subs[0]}-{subs[-1]}.")


def _catalogue_text(lead: str | None = None) -> dict:
    """What the dataset holds, with the openers offered as buttons.

    `lead` prefixes an acknowledgement when this is being shown because the
    question could not be understood, rather than because it was asked for.
    """
    subs = _subjects()
    text = (
        f"I have surface EMG recordings for {len(subs)} subjects "
        f"(numbered {subs[0]}-{subs[-1]}), each with a right and a left biceps "
        "recording. They're efforts held to exhaustion, so the recordings vary "
        "a lot in length -- from about 25 seconds to about 8 minutes.\n\n"
        "Pick one to try it, or attach your own EMG recording as a CSV with "
        "the + button.")
    suggestions = [
        ("A reading at a moment", "is subject 13 fatigued at 60 seconds?"),
        ("In plain terms", "how about subject 5 near the end?"),
        ("When fatigue set in", "when did subject 13 start fatiguing?"),
        ("A whole recording", "summarise subject 7"),
        ("Two subjects", "compare subject 5 and 9"),
        ("Left vs right", "which arm is worse for subject 4?"),
        ("Across everyone", "which subject fatigued the most?"),
        ("A forecast", "will subject 2 get more tired over the next minute?"),
        ("A definition", "what does median frequency mean?"),
    ]
    return {"content": (f"{lead}\n\n{text}" if lead else text),
            "suggestions": suggestions, "awaiting_subject": True}


def _subject_menu(subject: int) -> dict:
    """What can be asked about one subject."""
    duration = _duration_of(subject, "R")
    length = (f"Their right-arm recording is {duration:.0f} seconds long"
              if duration else "I have a right and a left arm recording for them")
    other = 9 if subject != 9 else 5
    text = (
        f"**Subject {subject}** — what would you like to know?\n\n"
        f"{length}, and it's an effort held until exhaustion, so they start "
        "fresh and fatigue as it goes.\n\n"
        "Pick one below to fill it in — you can edit it before sending — or "
        "just ask in your own words.")
    suggestions = [
        ("Fatigued at a moment?", f"is subject {subject} fatigued at 60 seconds?"),
        ("How about near the end?", f"subject {subject} near the end"),
        ("When did fatigue set in?", f"when did subject {subject} start fatiguing?"),
        ("How did the whole effort go?", f"summarise subject {subject}"),
        ("Which arm held up better?", f"which arm is worse for subject {subject}?"),
        ("What happens next?",
         f"will subject {subject} get more tired over the next minute?"),
        ("How do they compare?", f"compare subject {subject} and {other}"),
    ]
    return {"content": text, "suggestions": suggestions}


# Percentage points. Below this two subjects are not meaningfully apart --
# the same threshold the two-subject comparison uses to decline a winner.
RANK_TIE_PERCENT = 2.0


def _ranking_answer(ranking: dict) -> dict:
    """The league table, rendered directly rather than phrased by the LLM."""
    results, ranked = ranking["results"], ranking["ranked"]
    if not ranked:
        return {"content": "I don't have enough usable recordings to rank."}

    side = "right" if ranking["side"] == "R" else "left"
    lines = [
        f"Ranked by how far each subject's median frequency has fallen from "
        f"**their own fresh level**, read "
        f"{ranking['late_fraction'] * 100:.0f}% of the way through their own "
        f"recording ({side} arm). The recordings differ in length, so reading "
        "them at the same fraction of each person's effort is fairer than at "
        "the same absolute second — and the fall is given as a percentage "
        "because everyone starts at a different level, so the same drop in "
        "hertz does not mean the same thing for two people.\n",
    ]
    for rank, s in enumerate(ranked, start=1):
        r = results[s]
        pct = r["drop_percent"]
        moved = (f"**{pct:.0f}% below** their fresh level" if pct >= 1 else
                 f"**{abs(pct):.0f}% above** their fresh level" if pct <= -1 else
                 "**level with** their fresh level")
        state = "" if r["fatigue_label"] in (1, 2) else ", not fatigued at the late reading"
        lines.append(f"{rank}. **Subject {s}** — {moved} "
                     f"({r['fresh_mdf']:.1f} → {r['mdf_late']:.1f} Hz){state}")

    top = ranked[0]
    top_pct = results[top]["drop_percent"]
    tied = [s for s in ranked
            if abs(results[s]["drop_percent"] - top_pct) < RANK_TIE_PERCENT]
    if len(tied) > 1:
        names = ", ".join(f"**subject {s}**" for s in tied[:-1])
        lines.append(f"\n{names} and **subject {tied[-1]}** are level at the "
                     f"top — {top_pct:.0f}% against "
                     f"{results[tied[-1]]['drop_percent']:.0f}% is too small a "
                     "gap to separate them.")
    else:
        lines.append(f"\n**Subject {top}** fatigued the most, "
                     f"{top_pct:.0f}% below their own fresh level.")
    if ranking.get("excluded"):
        excluded = ", ".join(str(s) for s in ranking["excluded"])
        lines.append(f"Excluded as too short to show a fatigue arc: "
                     f"subject {excluded}.")
    return {"content": "\n".join(lines)}


def _comparison_answer(comparison: dict) -> dict:
    """A comparison rendered directly, not phrased by the LLM."""
    if comparison["kind"] == "sides":
        results = comparison["results"]
        lines = [f"**Subject {comparison['subject']}**, both arms at "
                 f"{comparison['t_start']:.0f}s "
                 f"(the length of the shorter recording).\n"]
        for side in ("R", "L"):
            r = results.get(side)
            if not r:
                continue
            state = ("**fatigued**" if r["fatigue_label"] in (1, 2)
                     else "not fatigued")
            lines.append(f"- **{'Right' if side == 'R' else 'Left'} arm** — "
                         f"{state}, median frequency {r['mdf_hz']:.1f} Hz "
                         f"(model confidence {r['confidence'] * 100:.0f}%)")
        left, right = results.get("L"), results.get("R")
        if left and right:
            gap = abs(left["mdf_hz"] - right["mdf_hz"])
            if gap < 1.0:
                lines.append("\nThe two arms are level — that difference is "
                             "too small to read anything into.")
            else:
                lower = "left" if left["mdf_hz"] < right["mdf_hz"] else "right"
                lines.append(
                    f"\nThe **{lower} arm** is further along, by {gap:.1f} Hz. "
                    "Both arms are measured against this subject's own stored "
                    "baseline, so the two figures can be compared directly.")
        return {"content": "\n".join(lines)}

    if comparison["kind"] == "upload_vs_subject":
        subject = comparison["subject"]
        entries = [("Your recording", "your recording", "your",
                    comparison["upload"]),
                   (f"Subject {subject}", f"subject {subject}", "their",
                    comparison["subject_result"])]
        close_within = 5.0
        head = (f"**Your recording vs. subject {subject}** "
                f"({'right' if comparison['side'] == 'R' else 'left'} arm), "
                f"both read {comparison['fraction'] * 100:.0f}% of the way "
                "through their own recording — the two are different lengths, "
                "so the same absolute second would not be a fair comparison.\n")
        tail = [
            f"\n_Your recording has no stored calibration, so its fresh level "
            f"was taken from its own first seconds, assuming you started "
            f"rested. If you did not, your drop is understated. Subject "
            f"{subject}'s fresh level comes from the labelled dataset._"]
        if comparison.get("short"):
            tail.append("_Your recording is also short enough that it may not "
                        "show a full fatigue arc._")
    else:
        results = comparison["results"]
        entries = [(f"Subject {s}", f"subject {s}", "their", r)
                   for s, r in results.items()]
        close_within = 2.0
        if comparison.get("fraction") is not None:
            head = (f"Read at {comparison['fraction'] * 100:.0f}% of the way "
                    "through each subject's own recording — these are efforts "
                    "to exhaustion and the recordings differ in length, so "
                    "that is a fairer comparison than the same absolute "
                    "second.\n")
        else:
            head = f"All read at {comparison['t_start']:.0f}s.\n"
        tail = []
        if comparison.get("clamped"):
            tail.append("_Shorter than the time asked about, so the last "
                        "window was used: subject "
                        + ", ".join(str(s) for s in comparison["clamped"]) + "._")
        if comparison.get("short"):
            tail.append("_Too short to show a fatigue arc, treat with caution: "
                        "subject "
                        + ", ".join(str(s) for s in comparison["short"]) + "._")

    lines = [head]
    for title, _mid, possessive, r in entries:
        state = "**fatigued**" if r["fatigue_label"] in (1, 2) else "not fatigued"
        drop = r.get("drop_percent")
        if drop is None:
            moved = "no fresh level available to measure against"
        elif drop >= 0:
            moved = (f"**{drop:.0f}% below** {possessive} own fresh level "
                     f"({r['fresh_mdf']:.1f} → {r['mdf_hz']:.1f} Hz)")
        else:
            moved = (f"**{abs(drop):.0f}% above** {possessive} own fresh level "
                     f"({r['fresh_mdf']:.1f} → {r['mdf_hz']:.1f} Hz), which is "
                     "not the direction fatigue moves it")
        lines.append(f"- **{title}** — {state}, {moved}")

    scored = sorted(((t, m, p, r["drop_percent"]) for t, m, p, r in entries
                     if r.get("drop_percent") is not None),
                    key=lambda e: -e[3])
    if len(scored) >= 2:
        (top, top_mid, top_poss, top_drop) = scored[0]
        (_bottom, bottom_mid, _bp, bottom_drop) = scored[-1]
        if abs(top_drop - bottom_drop) < close_within:
            lines.append(f"\n{top_mid.capitalize()} and {bottom_mid} are "
                         f"level — {top_drop:.0f}% against {bottom_drop:.0f}% "
                         "is too small a gap to call one more fatigued than "
                         "the other.")
        else:
            lines.append(f"\n**{top}** is the more fatigued of the two — "
                         f"{top_drop:.0f}% below {top_poss} own fresh level "
                         f"against {bottom_drop:.0f}%.")
        lines.append("\nThey're compared on how far each has fallen from "
                     "their *own* fresh level, not on the raw hertz: everyone "
                     "starts somewhere different, so one person's reading "
                     "being higher than another's means nothing by itself.")

    return {"content": "\n".join(lines + tail)}


def _analysis_turn(user_text: str, intent, previous: dict | None) -> dict:
    """Handle every question that isn't a single-window reading."""
    kind = intent.kind

    if kind == intent_router.CATALOGUE:
        return _catalogue_text()

    if kind == intent_router.MENU:
        subject = intent.subjects[0]
        if subject not in _subjects():
            subs = _subjects()
            return {"content": f"I don't have subject {subject} -- the dataset "
                               f"has subjects {subs[0]}-{subs[-1]}."}
        return _subject_menu(subject)

    if kind == intent_router.EXPLAIN:
        text = analysis.define(intent.term)
        return {"content": text} if text else _catalogue_text()

    if kind in (intent_router.ONSET, intent_router.OVERVIEW):
        subject = _subject_for(intent, previous)
        if subject is None:
            return {"content": _needs_subject_msg()}
        if subject not in _subjects():
            subs = _subjects()
            return {"content": f"I don't have subject {subject} -- the dataset "
                               f"has subjects {subs[0]}-{subs[-1]}."}
        side = extract.side_from_text(user_text) or (previous or {}).get("side") or "R"
        summary = _cached_scan(subject, side)
        window = {"subject": subject, "side": side, "source": "dataset",
                  "kind": kind}
        # a recording with no onset is answered directly -- see
        # prompt.render_onset_answer for what the model did with it instead
        if kind == intent_router.ONSET:
            rendered = render_onset_answer(summary)
            if rendered:
                return {"content": rendered, "facts": onset_facts(summary),
                        "window": window, "user_text": user_text}
        facts = (onset_facts(summary) if kind == intent_router.ONSET
                 else overview_facts(summary))
        instruction = (
            "Answer in ONE or TWO short sentences, in plain everyday words. "
            "Say when fatigue set in and give the ±accuracy once -- do not "
            "imply a more precise moment than was measured."
            if kind == intent_router.ONSET else
            "Answer in TWO or THREE short sentences, in plain everyday words. "
            "Say whether they tired over the recording and roughly when. "
            "Quote each figure at most once, and do not explain a number by "
            "restating the same number.")
        return {"prompt": build_facts_prompt(
                    "Measured results:", facts, user_text, instruction),
                "facts": facts, "window": window, "user_text": user_text}

    if intent.both_sides:
        subject = _subject_for(intent, previous)
        if subject is None:
            return {"content": _needs_subject_msg()}
        if subject not in _subjects():
            subs = _subjects()
            return {"content": f"I don't have subject {subject} -- the dataset "
                               f"has subjects {subs[0]}-{subs[-1]}."}
        comparison = _cached_compare_sides(subject, None)
    elif len(intent.subjects) >= 2:
        subjects = [s for s in intent.subjects if s in _subjects()]
        if len(subjects) < 2:
            return {"content": "I need two subjects I actually have to compare. "
                               + _needs_subject_msg()}
        side = extract.side_from_text(user_text) or "R"
        t_start = extract.t_start_from_text(user_text, None)
        comparison = _cached_compare_subjects(tuple(subjects), t_start, side)
    else:
        side = extract.side_from_text(user_text) or "R"
        return _ranking_answer(_cached_ranking(side))

    return {**_comparison_answer(comparison), "facts": compare_facts(comparison),
            "window": None, "user_text": user_text}


def _followup_turn(session: dict, user_text: str) -> dict:
    """"why?" / "explain that" -- re-explain the last answer, measure nothing new."""
    last = session["last_answer"]
    if not last or not last.get("answer"):
        return {"content": (
            "There's nothing to explain yet — ask me about a subject first, "
            "then say \"why?\" and I'll unpack that answer.")}
    facts = last.get("facts") or []
    return {"prompt": build_followup_prompt(
                last.get("question") or "(their previous question)",
                last["answer"], facts, user_text),
            "facts": facts,
            # A follow-up re-explains an answer that has already been shown, so
            # the figures IN that answer are legitimate to restate -- they were
            # rendered from measurements, not invented. Screened against the
            # facts alone, every number in a follow-up was stripped and the
            # answers came back as dangling fragments ("Instead, the muscle
            # showed...", with the sentence it contrasted against removed).
            "grounded_in": [last["answer"]],
            "window": None, "user_text": user_text}


def _dataset_turn(session: dict, user_text: str, previous: dict | None) -> dict:
    intent = intent_router.route(user_text)

    if intent.kind == intent_router.FOLLOWUP:
        return _followup_turn(session, user_text)

    if (intent.kind == intent_router.MENU and previous
            and previous.get("t_start") is not None):
        intent = intent_router.Intent(kind=intent_router.READING,
                                      subjects=intent.subjects)

    if intent.kind != intent_router.READING:
        return _analysis_turn(user_text, intent, previous)

    subjects = _subjects()
    provisional = intent.subjects[0] if intent.subjects else (previous or {}).get("subject")
    side_hint = extract.side_from_text(user_text) or (previous or {}).get("side") or "R"
    duration = _duration_of(provisional, side_hint) if provisional in subjects else None

    resolved = extract.resolve_query(user_text, previous, duration=duration,
                                     subjects=subjects)

    if (not resolved.ok and duration is not None
            and extract.extract_horizon_seconds(user_text, default=None) is not None):
        resolved = extract.resolve_query(
            user_text, {**(previous or {}), "subject": provisional,
                        "t_start": duration, "side": side_hint},
            duration=duration, subjects=subjects)
        if resolved.ok:
            # The window handed in above is synthetic -- it exists only to
            # supply the end of the recording as a default time -- so the time
            # was defaulted, not carried over from anything the reader said.
            # Left in, the provenance line claimed both at once: "(time
            # carried over from your previous question) · no time given, so
            # this reads the end of the recording", on the first message of a
            # brand-new chat. Provenance is the one line the reader is meant
            # to be able to trust.
            resolved.params["carried_over"] = [
                c for c in (resolved.params.get("carried_over") or [])
                if c != "time"]
            resolved.problems.append(
                "no time given, so this reads the end of the recording")

    if not resolved.ok:
        # Nothing in the message and nothing in the conversation to work from
        # -- "tell me everything", "i dont know what to ask", or plain noise.
        # Answering that with "which subject and which point in the
        # recording?" asks someone who has just arrived to name a subject and
        # a timestamp before anything has told them either exists, and offers
        # no buttons to get there. The catalogue says what is here and hands
        # over the openers as chips, which is what that person needs.
        if (provisional is None
                and extract.t_start_from_text(user_text, None) is None):
            return _catalogue_text(
                "I'm not sure what you're asking about yet."
                if (user_text or "").strip() else None)
        message = " ".join(resolved.problems + [resolved.ask])
        return {"content": message}

    params = resolved.params
    subject, t_start, side = params["subject"], params["t_start"], params["side"]
    notes = resolved.problems
    window = {"subject": subject, "t_start": t_start, "side": side,
              "source": "dataset", "carried_over": params.get("carried_over", []),
              "notes": notes}
    # The chart is NOT drawn here. It used to be, on every single reading, and
    # it was measurably not free even after being moved onto a background
    # thread: a turn with a cold chart took 10.21 s against 8.27 s without one,
    # and the model call inside it slowed from 7.25 s to 7.84 s. Plotly figure
    # building is CPU work and it competes with the Ollama process generating
    # the answer, on a box with no GPU. Most answers were also never looked at
    # as a chart, so that was ~2 s and a 3.3 MB payload spent per question on
    # something usually unwanted.
    #
    # What travels instead is a reference: enough to draw the figure later, if
    # the reader asks. serve.py's GET /chart redeems it. See _chart_ref below.
    seg_future = _IO_POOL.submit(_load_subject_segment, subject, side)

    try:
        result = classify(subject, t_start, side)
    except KeyError:
        seg_future.cancel()
        return {"content": f"Subject {subject} has no stored fresh-baseline "
                           "calibration, so I can't classify their fatigue yet."}
    except Exception as e:
        seg_future.cancel()
        return {"content": f"Couldn't classify that window: {e}"}
    result["fatigue_state"] = _FATIGUE_STATE.get(
        result["fatigue_label"], str(result["fatigue_label"]))

    seg, fs = seg_future.result()
    forecast, forecast_chart_html = _forecast(seg, fs, user_text, t_start)
    reading = _plain_reading(result, subject_reference(subject), t_start,
                             float(seg.t[-1]) if seg.t.size else None,
                             f"Subject {subject}")

    return {"features": result, "reading": reading,
            "chart_ref": {"source": "dataset", "subject": subject,
                          "t_start": t_start, "side": side},
            "chart_caption": _READING_CHART_CAPTION,
            "forecast": forecast, "forecast_chart_html": forecast_chart_html,
            "forecast_chart_caption": _FORECAST_CHART_CAPTION if forecast_chart_html else None,
            "user_text": user_text, "window": window}


def _upload_key(f) -> str:
    return f"{f.name}:{f.size}"


def _upload_turn(session: dict, user_text: str, uploaded_file) -> dict:
    key = _upload_key(uploaded_file)
    cache = session["uploads"].get(key)
    if cache is None:
        try:
            t, x, fs_native = parse_uploaded_csv(uploaded_file, session["sample_rate"])
            seg = load_uploaded_segment(t, x, fs_native)
        except UploadError as e:
            return {"content": f"Couldn't read that file: {e}"}
        except Exception as e:      # malformed CSVs must not crash the server
            return {"content": f"Couldn't read that file ({type(e).__name__}: {e})"}
        fs = int(getattr(seg, "eff_fs", 250))
        cache = {"seg": seg, "fs": fs, "baseline": None, "name": uploaded_file.name,
                 "scan": None}
        session["uploads"][key] = cache

    session["last_upload"] = key
    return _upload_question(user_text, cache)


def _upload_question(user_text: str, cache: dict) -> dict:
    """Any question about an already-loaded uploaded recording."""
    intent = intent_router.route(user_text or "")
    if intent.kind in (intent_router.ONSET, intent_router.OVERVIEW):
        return _upload_analysis(user_text, cache, intent.kind)
    if intent.kind == intent_router.COMPARE:
        return _upload_compare(user_text, cache, intent)
    return _upload_reading(user_text, cache)


def _upload_compare(user_text: str, cache: dict, intent) -> dict:
    """The uploaded recording against one named dataset subject."""
    subjects = [s for s in intent.subjects if s in _subjects()]
    if intent.both_sides or len(subjects) != 1:
        return {"content": (
            "For an uploaded recording I can only compare it against one "
            "dataset subject at a time — try \"compare this to subject 5\". "
            "There's no second arm in the file to compare against, and I "
            "won't rank it inside the dataset table: those subjects have "
            "measured baselines and this file's is assumed from its own "
            "opening seconds, so they aren't like for like.")}

    side = extract.side_from_text(user_text) or "R"
    try:
        comparison, baseline = analysis.compare_upload_to_subject(
            cache["seg"], cache["fs"], subjects[0],
            baseline=cache["baseline"], side=side)
    except ValueError as e:
        return {"content": f"I can't compare that file reliably: {e}"}
    except Exception as e:
        return {"content": f"Couldn't run that comparison ({type(e).__name__}: {e})"}
    cache["baseline"] = baseline

    return {**_comparison_answer(comparison),
            "facts": compare_facts(comparison), "user_text": user_text,
            "chart_ref": {"source": "upload"},
            "window": {"source": "upload", "name": cache["name"],
                       "kind": "compare"}}


def _upload_analysis(user_text: str, cache: dict, kind: str) -> dict:
    """Onset and whole-recording summary for an uploaded file."""
    if cache.get("scan") is None:
        try:
            scan, baseline = analysis.scan_upload(
                cache["seg"], cache["fs"], baseline=cache["baseline"],
                name=f"the uploaded recording ({cache['name']})")
        except ValueError as e:
            return {"content": f"I can't scan that file reliably: {e}"}
        except Exception as e:
            return {"content": f"Couldn't scan that recording ({type(e).__name__}: {e})"}
        cache["baseline"] = baseline
        cache["scan"] = analysis.summarise(scan)

    summary = cache["scan"]
    window = {"source": "upload", "name": cache["name"], "kind": kind}
    # rendered rather than phrased when there is no onset, exactly as for a
    # dataset subject
    if kind == intent_router.ONSET:
        rendered = render_onset_answer(summary)
        if rendered:
            return {"content": rendered, "facts": onset_facts(summary),
                    "user_text": user_text, "chart_ref": {"source": "upload"},
                    "window": window}
    facts = (onset_facts(summary) if kind == intent_router.ONSET
             else overview_facts(summary))
    instruction = (
        "Answer in ONE or TWO short sentences, in plain everyday words. Say "
        "when fatigue set in and state plainly that it is approximate."
        if kind == intent_router.ONSET else
        "Answer in TWO or THREE short sentences, in plain everyday words. "
        "Say whether they tired over the recording and roughly when. Quote "
        "each figure at most once, and do not explain a number by restating "
        "the same number.")
    return {"prompt": build_facts_prompt("Measured results:", facts, user_text,
                                         instruction),
            "facts": facts, "user_text": user_text,
            "chart_ref": {"source": "upload"}, "window": window}


def _upload_chart(cache: dict):
    try:
        mdf_t, mdf_v, _ = data_loader.mdf_trend(cache["seg"], fs=cache["fs"])
        return charts.raw_and_mdf_figure(cache["seg"], mdf_t, mdf_v,
                                         title=f"Uploaded: {cache['name']}")
    except Exception:
        return None


def _upload_reading(user_text: str, cache: dict) -> dict:
    uploaded_name = cache["name"]
    seg, fs = cache["seg"], cache["fs"]
    duration = float(seg.t[-1]) if seg.t.size else 0.0
    t_start = extract.extract_t_start_seconds(user_text, duration)
    if t_start is None:
        t_start = duration
    elif t_start > duration:
        t_start = duration

    try:
        result, baseline = classify_upload(seg, fs, t_start, baseline=cache["baseline"])
    except ValueError as e:
        return {"content": f"I can't give you a reliable reading for that file: {e}"}
    cache["baseline"] = baseline
    result["fatigue_state"] = _FATIGUE_STATE.get(
        result["fatigue_label"], str(result["fatigue_label"]))

    notes = []
    forecast, forecast_chart_html = _forecast(seg, fs, user_text, t_start)
    try:
        reference = upload_reference(cache["baseline"])
    except Exception:
        reference = None
    reading = _plain_reading(result, reference, t_start, duration,
                             "This recording")

    return {"features": result, "reading": reading,
            "chart_ref": {"source": "upload"},
            "chart_caption": _UPLOAD_CHART_CAPTION,
            "forecast": forecast, "forecast_chart_html": forecast_chart_html,
            "forecast_chart_caption": _FORECAST_CHART_CAPTION if forecast_chart_html else None,
            "user_text": user_text,
            "window": {"t_start": t_start, "source": "upload",
                       "name": uploaded_name, "notes": notes}}


def _forecast(seg, fs: int, user_text: str, t_start: float | None = None):
    """Forecast only when the question actually asks about the future."""
    horizon = extract.extract_horizon_seconds(user_text, default=None)
    if horizon is None:
        return None, None
    try:
        forecast = forecast_fatigue(seg, fs, horizon_sec=horizon, t_end=t_start)
    except Exception:
        return None, None
    if not forecast.get("ok"):
        return forecast, None
    try:
        return forecast, charts.forecast_figure(forecast)
    except Exception:
        return forecast, None


def _finalize(session: dict, turn: dict) -> dict:
    # Only the reading path starts a background chart, and it is joined at the
    # very end of that path -- after the model call it was launched to overlap.
    # Every other path resolves it here, immediately, so those turns behave
    # exactly as they did before.
    if "prompt" in turn:
        try:
            content = chat([{"role": "user", "content": turn["prompt"]}],
                           model=session["model"])
            # These answers are given figures and are meant to quote them, so
            # they cannot be scrubbed of numbers the way a reading is. What
            # they must not contain is a number nobody measured -- see
            # strip_unfactual_numbers(). If that empties the answer, the
            # measured facts are shown instead, exactly as when the model is
            # unreachable: the measurement is not lost, only its wording.
            cleaned, _invented = strip_unfactual_numbers(
                content, (turn.get("facts") or []) + (turn.get("grounded_in") or []))
            # A follow-up on a definition has no measured facts to fall back
            # on, so the bullet list can be empty too -- never let both be and
            # ship a blank answer.
            shown = "\n".join(f"- {line}"
                              for line in readable_facts(turn.get("facts") or []))
            content = cleaned or shown or (
                "I couldn't put that into words without quoting figures that "
                "were never measured, so I've left it out rather than guess. "
                "Ask again and I'll have another go.")
            # Same plain-language and length treatment the reading answers get
            # further down. Three sentences rather than two: an overview has a
            # start, an end and an onset to cover, and the numbers here ARE
            # the answer (they were measured), so only the wording is trimmed.
            content = interpret.trim_sentences(interpret.plain_words(content), 3)
        except LLMError as e:
            lines = readable_facts(turn.get("facts") or [])
            content = ("\n".join(f"- {line}" for line in lines)
                       + f"\n\n_Couldn't phrase this in prose: {e}_")
        return {"content": content, "chart_ref": turn.get("chart_ref"),
                "chart_caption": turn.get("chart_caption"),
                "forecast_chart_html": None, "forecast_chart_caption": None,
                "recommendation": None, "facts": turn.get("facts"),
                "user_text": turn.get("user_text"), "window": turn.get("window")}

    if "features" not in turn:
        return {"content": turn["content"],
                "chart_ref": turn.get("chart_ref"),
                "chart_caption": turn.get("chart_caption"),
                "forecast_chart_html": None, "forecast_chart_caption": None,
                "recommendation": None,
                "facts": turn.get("facts"), "window": turn.get("window"),
                "user_text": turn.get("user_text"),
                "suggestions": turn.get("suggestions"),
                "awaiting_subject": turn.get("awaiting_subject")}

    features, forecast, user_text = turn["features"], turn.get("forecast"), turn["user_text"]
    window = turn.get("window")
    # Whether a chart is available to the reader, not whether one is drawn:
    # it is drawn only if they ask (GET /chart). Knowable exactly now, which
    # the old background-render version could not manage -- it had to guess.
    chart_shown = turn.get("chart_ref") is not None

    reading = turn.get("reading")
    verdict = (reading or {}).get("verdict")
    try:
        prose = chat([{"role": "user",
                       "content": build_prompt(features, user_text, forecast,
                                               window, reading, chart_shown)}],
                     model=session["model"])
        if verdict:
            who = (reading or {}).get("who")
            cleaned, invented = interpret.strip_invented_numbers(
                interpret.strip_verdict_echo(prose, who), who)
            # Trim last, after the echo and the invented figures are gone:
            # cutting to two sentences first would spend the budget on a
            # duplicate opening restatement and drop the real explanation.
            cleaned = interpret.trim_sentences(interpret.plain_words(cleaned), 2)
            content = f"{verdict}\n\n{cleaned}" if cleaned else verdict
        else:
            content = prose
    except LLMError as e:
        if verdict:
            content = (f"{verdict}\n\n_Couldn't add the plain-English notes: "
                       f"{e}_")
        elif reading and reading.get("lines"):
            content = ("\n".join(f"- {line}" for line in reading["lines"])
                       + f"\n\n_{reading['technical']}_"
                       + f"\n\n_Couldn't phrase this in prose: {e}_")
        else:
            content = (f"{features['fatigue_state']} "
                      f"(median frequency {features['mdf_hz']:.1f} Hz, "
                      f"confidence {features['confidence'] * 100:.1f}%). "
                      f"_Couldn't phrase this in prose: {e}_")

    recommendation = None
    if wants_recommendation(user_text):
        try:
            suggestion = chat(
                [{"role": "user", "content": build_recommendation_prompt(
                    features, forecast, user_text,
                    session["athlete_note"] or None, turn.get("reading"))}],
                model=session["model"])
            # The verdict leads here for the same reason it leads the main
            # answer: asked to restate it, the model opened the box with
            # "they are showing signs of fatigue" under a reading that said
            # they were not. It is rendered, and any Hz or percentage in the
            # suggestion is invented by construction -- see strip_measurements.
            suggestion = strip_measurements(suggestion)
            recommendation = ensure_disclaimer(
                f"{verdict}\n\n{suggestion}" if verdict and suggestion
                else (suggestion or verdict))
        except LLMError as e:
            recommendation = f"[Recommendation unavailable: {e}]"

    # Both model calls are done; collect the chart that has been rendering
    # alongside them. By now it is almost always already finished.
    return {"content": content, "chart_ref": turn.get("chart_ref"),
            "chart_caption": turn.get("chart_caption"),
            "forecast_chart_html": turn.get("forecast_chart_html"),
            "forecast_chart_caption": turn.get("forecast_chart_caption"),
            "recommendation": recommendation,
            "provenance": _provenance(window, features, reading),
            "features": features, "reading": turn.get("reading"),
            "forecast": forecast, "user_text": user_text, "window": window}


def _provenance(window: dict | None, features: dict,
                reading: dict | None = None) -> str | None:
    """One line naming exactly what was measured, shown under every answer."""
    if not window:
        return None
    parts = [f"Reading: {describe_window(window)}"]
    carried = window.get("carried_over") or []
    if carried:
        parts.append(f"({' and '.join(carried)} carried over from your previous question)")
    for note in window.get("notes") or []:
        parts.append(f"· {note}")
    if features.get("calibration"):
        cal = features["calibration"]
        parts.append(f"· self-calibrated from the first {cal['fresh_sec']:.0f}s "
                    f"({cal['n_windows']} baseline windows) — less reliable than "
                    "a stored calibration")
    technical = (reading or {}).get("technical")
    if technical:
        parts.append(f"· {technical}")
    return " ".join(parts)


def _followup_upload(session: dict, user_text: str) -> dict | None:
    """The upload this question is still about, or None for the dataset."""
    cache = session["uploads"].get(session["last_upload"])
    if cache is None:
        return None
    if not intent_router.stays_on_upload(user_text):
        return None
    if intent_router.names_own_recording(user_text):
        return cache
    return cache if session["last_source"] == "upload" else None


# A reply that is nothing but a number, sent straight after being asked which
# subject was meant. "5" is the obvious answer to "Which subject did you mean?"
# and it used to be read as a stray number, so the same question came back.
_BARE_NUMBER_RE = re.compile(r"^\s*[#-]?\s*(\d{1,2})\s*[?.!]*\s*$")


def _name_the_subject(session: dict, user_text: str) -> str:
    """"5" -> "subject 5", but only when a subject was just asked for."""
    if not session.get("awaiting_subject"):
        return user_text
    m = _BARE_NUMBER_RE.match(user_text or "")
    if not m or int(m.group(1)) not in _subjects():
        return user_text
    return f"subject {int(m.group(1))}"


def _note_unanswered(user_text: str, final: dict) -> None:
    """Offer the parts of a compound question this answer did not cover.

    Routing picks one intent, so a message asking three things was answered
    with one and said nothing about the other two. Rather than trying to
    answer them all in one reply -- which would produce a wall of text and
    several separate charts -- the rest are named and offered as chips, which
    is the same staged-suggestion path the menus already use.
    """
    extras = intent_router.extra_requests(user_text or "")
    if not extras:
        return
    subs = _subjects()
    chips = []
    for found in extras:
        named = [s for s in found.subjects if s in subs]
        first = named[0] if named else None
        if found.kind == intent_router.ONSET and first:
            chips.append(("When fatigue set in",
                          f"when did subject {first} start fatiguing?"))
        elif found.kind == intent_router.OVERVIEW and first:
            chips.append(("The whole recording", f"summarise subject {first}"))
        elif found.kind == intent_router.COMPARE:
            if found.both_sides and first:
                chips.append(("Left vs right",
                              f"which arm is worse for subject {first}?"))
            elif len(named) >= 2:
                chips.append(("How they compare",
                              f"compare subject {named[0]} and {named[1]}"))
        elif found.kind == intent_router.EXPLAIN and found.term:
            chips.append((f"What {found.term} means",
                          f"what does {found.term} mean?"))
    if not chips:
        return
    final["content"] = (final.get("content") or "") + (
        "\n\n_You asked more than one thing there, and this answers one part "
        "of it. Here's the rest:_")
    final["suggestions"] = (final.get("suggestions") or []) + chips


def warm_up() -> None:
    """Load the model and prime Ollama's prompt cache before anyone asks.

    Two separate costs land on the first question of a session and nowhere
    else. Ollama loads the model off disk (~6.7 s), and it evaluates the whole
    prompt from scratch (~15-19 s) because there is no previous prompt to share
    a prefix with. Every question after that reuses both and pays ~2 s.

    Sending one throwaway prompt at startup moves both onto the server's boot,
    where nobody is waiting. The prompt is built by the real build_prompt() so
    its long fixed preamble is byte-identical to what the first real question
    will send -- a hand-written approximation would share no prefix and prime
    nothing. num_predict=1 because only the prompt side needs warming; the
    answer is discarded.

    Never raises: Ollama may not be running, and a demo server that refuses to
    start because a warm-up failed is worse than a slow first answer.
    """
    try:
        features = {"mdf_hz": 60.0, "fatigue_label": 1, "confidence": 0.9,
                    "fatigue_state": "fatigue"}
        primer = build_prompt(features, "warm up", None,
                              {"subject": 1, "t_start": 0.0, "side": "R",
                               "source": "dataset"}, None, True)
        chat([{"role": "user", "content": primer}], num_predict=1, timeout=180)
    except Exception:
        pass


def handle_turn(session: dict, user_text: str, uploaded_file=None) -> dict:
    """The single entry point: one user turn in, one finalized answer out.

    Equivalent to app.py's `_handle_turn`, minus the chat-message bookkeeping
    (history.save_chat / st.session_state.chat) -- the JS UI keeps its own
    per-chat message list in localStorage, so there's no server-side chat
    transcript to append to here, only the resolved-conversation state
    needed for follow-ups ("why?", "and at 90 seconds?").
    """
    # What the reader typed is what the transcript keeps; what the pipeline
    # sees is the resolved form, so `display_text` is captured before the
    # rewrite rather than after it -- otherwise a bare "5" would be recorded
    # as the question "subject 5", which is not what they asked.
    display_text = user_text or (f"[uploaded {uploaded_file.name}]" if uploaded_file else "")
    user_text = _name_the_subject(session, user_text)

    if uploaded_file is not None:
        turn = _upload_turn(session, user_text, uploaded_file)
    else:
        cache = _followup_upload(session, user_text)
        turn = (_upload_question(user_text, cache) if cache
                else _dataset_turn(session, user_text, session["last_params"]))
    final = _finalize(session, turn)
    if uploaded_file is None:
        _note_unanswered(user_text, final)

    session["last_turn_context"] = final if "features" in final else None
    # Whether a bare number on the NEXT turn names a subject. The catalogue
    # says so outright; any answer that asks "which subject" is the same
    # situation, so it is read off the text rather than flagged at each of
    # the several places that ask.
    session["awaiting_subject"] = bool(
        final.get("awaiting_subject")
        or "which subject" in (final.get("content") or "").lower())
    if final.get("content") and not intent_router.is_followup(user_text or ""):
        session["last_answer"] = {
            "question": display_text,
            "answer": final["content"],
            "facts": (final.get("facts")
                      or (final.get("reading") or {}).get("lines")),
        }
    window = final.get("window")
    if window and window.get("source") in ("upload", "dataset"):
        session["last_source"] = window["source"]
    if window and window.get("source") == "dataset":
        prev = session["last_params"] or {}
        session["last_params"] = {
            "subject": window.get("subject", prev.get("subject")),
            "t_start": window.get("t_start", prev.get("t_start")),
            "side": window.get("side", prev.get("side", "R"))}
    return final
