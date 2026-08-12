"""
Maalav's chatbot frontend.
==========================

Run:
    streamlit run frontend/app.py

Wires the pipeline directly in Python -- no LLM tool-calling involved for the
fatigue numbers themselves:

    user question -> extract.parse_query() / regex   -> classify()/classify_upload()
                                                       -> forecast_fatigue()
                  -> prompt.build_prompt() / recommend.build_recommendation_prompt()
                  -> LLM phrases the final answer

classify(), classify_upload() and forecast_fatigue() are called as plain
Python functions, not via models/serve.py's HTTP bridge, so the fatigue
numbers can never be skipped, mis-called, or hallucinated by the LLM -- it
only ever phrases an answer (and, for sport/plan questions, adds its own
general knowledge on top) from numbers it's already been handed.
"""
from __future__ import annotations

import os
import sys

import streamlit as st

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
import history                                   # noqa: E402
import intent as intent_router                   # noqa: E402
import interpret                                 # noqa: E402
from llm import LLMError, chat, list_models      # noqa: E402
from prompt import (build_facts_prompt, build_followup_prompt,  # noqa: E402
                    build_prompt,
                    compare_facts, describe_window, onset_facts,
                    overview_facts, ranking_facts, readable_facts)
from recommend import (build_recommendation_prompt,  # noqa: E402
                       ensure_disclaimer, wants_recommendation)
from upload import UploadError, load_uploaded_segment, parse_uploaded_csv  # noqa: E402

# matches models/serve.py's _STATE mapping, kept in sync manually since this
# app calls classify()/classify_upload() directly instead of going through
# serve.py's /classify HTTP route
_FATIGUE_STATE = {0: "non-fatigue", 1: "fatigue", 2: "fatigue"}
DATA_ROOT = os.path.join(_REPO_ROOT, "zenodo_biceps", "sEMG_data")

st.set_page_config(page_title="EMG Fatigue Chatbot", layout="wide")


# ---------------------------------------------------------------------------
# session state
# ---------------------------------------------------------------------------
if "chat" not in st.session_state:
    st.session_state.chat = history.new_chat()
if "model" not in st.session_state:
    st.session_state.model = "llama3.2:3b"
if "sample_rate" not in st.session_state:
    st.session_state.sample_rate = None   # blank until the user states it
if "athlete_note" not in st.session_state:
    st.session_state.athlete_note = ""
if "uploads" not in st.session_state:
    st.session_state.uploads = {}          # key -> {"seg", "fs", "baseline", ...}
if "last_upload" not in st.session_state:
    # the uploads key of the file most recently attached, so questions on
    # later turns can still be about it -- Streamlit only hands the file over
    # on the turn it is actually attached
    st.session_state.last_upload = None
if "last_source" not in st.session_state:
    st.session_state.last_source = None    # "upload" or "dataset"
if "confirm_clear" not in st.session_state:
    st.session_state.confirm_clear = False
if "last_turn_context" not in st.session_state:
    st.session_state.last_turn_context = None
if "last_params" not in st.session_state:
    st.session_state.last_params = None   # last resolved {subject, t_start, side}
if "last_answer" not in st.session_state:
    # {question, answer, facts} for the previous turn, whatever kind it was --
    # this is what "why?" and "explain that" are asking about. Kept separately
    # from last_turn_context, which only ever holds single-window readings.
    st.session_state.last_answer = None
if "draft" not in st.session_state:
    st.session_state.draft = None         # suggestion staged for review, not sent
if "draft_nonce" not in st.session_state:
    st.session_state.draft_nonce = 0      # forces the draft box to take a new value


# ---------------------------------------------------------------------------
# pipeline helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=8)
def _load_subject_segment(subject: int, side: str):
    """Cached segment load.

    Streamlit re-runs this whole script on every interaction, and one turn
    otherwise re-read and re-resampled the same multi-MB CSV three times
    (classify() internally, the forecast, and render_window()) -- about 3.4 s
    of duplicate I/O per question.
    """
    seg = data_loader.load_biceps_segment(DATA_ROOT, subject, side,
                                          target_fs=250, bandpass=True)
    return seg, int(getattr(seg, "eff_fs", 250))


@st.cache_data(show_spinner=False, max_entries=1)
def _subjects() -> list[int]:
    try:
        return available_subjects()
    except Exception:
        return list(range(1, 14))


@st.cache_data(show_spinner=False, max_entries=8)
def _cached_scan(subject: int, side: str) -> dict:
    seg, fs = _load_subject_segment(subject, side)
    return analysis.summarise(analysis.scan_recording(subject, side, seg=seg, fs=fs))


@st.cache_data(show_spinner=False, max_entries=2)
def _cached_ranking(side: str) -> dict:
    return analysis.rank_subjects(_subjects(), side=side)


@st.cache_data(show_spinner=False, max_entries=8)
def _cached_compare_subjects(subjects: tuple[int, ...], t_start: float | None,
                             side: str) -> dict:
    return analysis.compare_subjects(list(subjects), t_start=t_start, side=side)


@st.cache_data(show_spinner=False, max_entries=8)
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
                "technical": interpret.technical_line(described)}
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


def _catalogue_text() -> dict:
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
    return {"content": text, "suggestions": suggestions}


def _subject_menu(subject: int) -> dict:
    """What can be asked about one subject.

    Shown when a subject is named but nothing specific is asked. Answering
    that with one window's numbers guesses at the question; this states the
    couple of facts that are cheap to get, then offers the real options.

    The options come back as (label, prompt) pairs rather than baked into the
    markdown, so the UI can offer them as buttons that fill the draft box.
    """
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


def _ranking_answer(ranking: dict) -> dict:
    """The league table, rendered directly rather than phrased by the LLM.

    An ordered ranking is data, not prose. Asked to restate one, llama3.2:3b
    reliably mangled it -- it named "1, 4, 10" as the top three when the real
    order was 1, 11, 2, and quoted drops of 20.0, 11.8, 15.0, which is not
    even descending. Explicitly instructing it to preserve the order did not
    fix that. Rendering it here makes the answer both correct and instant,
    which is the same trade the catalogue and definitions already take.
    """
    results, ranked = ranking["results"], ranking["ranked"]
    if not ranked:
        return {"content": "I don't have enough usable recordings to rank."}

    side = "right" if ranking["side"] == "R" else "left"
    lines = [
        f"Ranked by how far each subject's median frequency fell between "
        f"{ranking['early_fraction'] * 100:.0f}% and "
        f"{ranking['late_fraction'] * 100:.0f}% of their own recording "
        f"({side} arm). A bigger fall means more fatigue developed over the "
        "effort — the recordings differ in length, so comparing them at the "
        "same fraction of each person's effort is fairer than at the same "
        "absolute second.\n",
    ]
    for rank, s in enumerate(ranked, start=1):
        r = results[s]
        drop = r["mdf_drop"]
        moved = (f"fell {drop:.1f} Hz" if drop > 0 else
                 f"**rose** {-drop:.1f} Hz" if drop < 0 else "did not change")
        state = "" if r["fatigue_label"] in (1, 2) else ", not fatigued at the late reading"
        lines.append(f"{rank}. **Subject {s}** — {moved} "
                     f"({r['mdf_early']:.1f} → {r['mdf_late']:.1f} Hz){state}")

    top = ranked[0]
    lines.append(f"\n**Subject {top}** fatigued the most, with a "
                 f"{results[top]['mdf_drop']:.1f} Hz fall.")
    if ranking.get("excluded"):
        excluded = ", ".join(str(s) for s in ranking["excluded"])
        lines.append(f"Excluded as too short to show a fatigue arc: "
                     f"subject {excluded}.")
    return {"content": "\n".join(lines)}


def _comparison_answer(comparison: dict) -> dict:
    """A comparison rendered directly, not phrased by the LLM.

    Comparisons have now gone wrong three separate ways in live testing, and
    none of them was a phrasing slip:

      1. handed two raw hertz values, it concluded the subject with the HIGHER
         median frequency was more fatigued -- backwards
      2. handed two drop percentages (21% and 4%), it reported "8% and 4%",
         inventing one of the two numbers
      3. handed a pre-computed "these two are too close to call", it ignored
         that and declared a winner from the percentages anyway

    Fixing 1 and 2 in the prompt worked. 3 did not, and it is the same thing
    that made the ranking answer deterministic: a 3B model will not decline to
    draw a conclusion when the numbers to draw it from are in front of it.
    Deciding in Python and rendering the result makes the answer correct and
    instant. compare_facts() is still built, so a following "why?" has the
    measured values to work from.
    """
    if comparison["kind"] == "sides":
        results = comparison["results"]
        # Both arms are scored against the same stored per-subject baseline,
        # so here -- and only here -- the raw values are directly comparable.
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

    # (title, mid-sentence form, possessive, result) -- the uploaded recording
    # belongs to the person asking, so it is addressed in the second person
    # while a dataset subject is a third party
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
        facts = (onset_facts(summary) if kind == intent_router.ONSET
                 else overview_facts(summary))
        instruction = (
            "Answer in 2-4 sentences. Say when fatigue set in and how sure the "
            "reading is, and state the ±accuracy that comes from the scan step "
            "-- do not imply a more precise moment than was measured."
            if kind == intent_router.ONSET else
            "Answer in 3-5 sentences. Describe how the recording develops from "
            "start to finish, quoting the median frequency at each end and how "
            "much of the recording was classified as fatigued.")
        return {"prompt": build_facts_prompt(
                    "Measured results:", facts, user_text, instruction),
                "facts": facts,
                "window": {"subject": subject, "side": side, "source": "dataset",
                           "kind": kind},
                "user_text": user_text}

    # comparisons -- rendered directly, see _comparison_answer for why. The
    # facts still travel so a following "why?" has the measured values.
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
        # a superlative over the whole field ("which subject fatigued most?")
        side = extract.side_from_text(user_text) or "R"
        return _ranking_answer(_cached_ranking(side))

    return {**_comparison_answer(comparison), "facts": compare_facts(comparison),
            "window": None, "user_text": user_text}


def _followup_turn(user_text: str) -> dict:
    """"why?" / "explain that" -- re-explain the last answer, measure nothing new."""
    last = st.session_state.last_answer
    if not last or not last.get("answer"):
        return {"content": (
            "There's nothing to explain yet — ask me about a subject first, "
            "then say \"why?\" and I'll unpack that answer.")}
    return {"prompt": build_followup_prompt(
                last.get("question") or "(their previous question)",
                last["answer"], last.get("facts") or [], user_text),
            "window": None, "user_text": user_text}


def _dataset_turn(user_text: str, previous: dict | None) -> dict:
    intent = intent_router.route(user_text)

    if intent.kind == intent_router.FOLLOWUP:
        return _followup_turn(user_text)

    # Mid-conversation, "what about subject 5?" is a follow-up asking for the
    # same reading on someone else, not a request to start over -- so the menu
    # is only offered when there is no previous window to carry a time from.
    if (intent.kind == intent_router.MENU and previous
            and previous.get("t_start") is not None):
        intent = intent_router.Intent(kind=intent_router.READING,
                                      subjects=intent.subjects)

    if intent.kind != intent_router.READING:
        return _analysis_turn(user_text, intent, previous)

    subjects = _subjects()
    # resolve the subject first so the recording's length is known, which is
    # what lets "near the end" and "halfway" mean anything
    provisional = intent.subjects[0] if intent.subjects else (previous or {}).get("subject")
    side_hint = extract.side_from_text(user_text) or (previous or {}).get("side") or "R"
    duration = _duration_of(provisional, side_hint) if provisional in subjects else None

    resolved = extract.resolve_query(user_text, previous, duration=duration,
                                     subjects=subjects)

    # "will subject 2 get more tired over the next minute?" states a horizon
    # but no start. Asking "which point in the recording?" back is obtuse --
    # the question plainly means "from where they are now", which is how the
    # upload path has always read a timeless question. Defaulted, not guessed:
    # the provenance line under the answer says the time was not given.
    if (not resolved.ok and duration is not None
            and extract.extract_horizon_seconds(user_text, default=None) is not None):
        resolved = extract.resolve_query(
            user_text, {**(previous or {}), "subject": provisional,
                        "t_start": duration, "side": side_hint},
            duration=duration, subjects=subjects)
        if resolved.ok:
            resolved.problems.append(
                "no time given, so this reads the end of the recording")

    if not resolved.ok:
        message = " ".join(resolved.problems + [resolved.ask])
        return {"content": message}

    params = resolved.params
    subject, t_start, side = params["subject"], params["t_start"], params["side"]
    notes = resolved.problems
    window = {"subject": subject, "t_start": t_start, "side": side,
              "source": "dataset", "carried_over": params.get("carried_over", []),
              "notes": notes}
    try:
        result = classify(subject, t_start, side)
    except KeyError:
        return {"content": f"Subject {subject} has no stored fresh-baseline "
                           "calibration, so I can't classify their fatigue yet."}
    except Exception as e:
        return {"content": f"Couldn't classify that window: {e}"}
    result["fatigue_state"] = _FATIGUE_STATE.get(
        result["fatigue_label"], str(result["fatigue_label"]))

    chart_html = None
    try:
        chart_html = render_window(subject, t_start, side)
    except Exception:
        pass  # chart is additive; never block the text answer

    seg, fs = _load_subject_segment(subject, side)
    forecast, forecast_chart_html = _forecast(seg, fs, user_text, t_start)
    reading = _plain_reading(result, subject_reference(subject), t_start,
                             float(seg.t[-1]) if seg.t.size else None,
                             f"Subject {subject}")

    return {"features": result, "reading": reading,
            "chart_html": chart_html, "forecast": forecast,
            "forecast_chart_html": forecast_chart_html, "user_text": user_text,
            "window": window}


def _upload_key(f) -> str:
    return f"{f.name}:{f.size}"


def _upload_turn(user_text: str, uploaded_file) -> dict:
    key = _upload_key(uploaded_file)
    cache = st.session_state.uploads.get(key)
    if cache is None:
        try:
            t, x, fs_native = parse_uploaded_csv(uploaded_file, st.session_state.sample_rate)
            seg = load_uploaded_segment(t, x, fs_native)
        except UploadError as e:
            return {"content": f"Couldn't read that file: {e}"}
        except Exception as e:      # malformed CSVs must not crash the app
            return {"content": f"Couldn't read that file ({type(e).__name__}: {e})"}
        fs = int(getattr(seg, "eff_fs", 250))
        cache = {"seg": seg, "fs": fs, "baseline": None, "name": uploaded_file.name,
                 "scan": None}
        st.session_state.uploads[key] = cache

    # Remembered so the NEXT question can still be about this file. Streamlit
    # only hands the file over on the turn it is attached, so without this
    # "when did I start fatiguing?" straight after an upload fell through to
    # the dataset path and asked which subject was meant.
    st.session_state.last_upload = key
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
    """The uploaded recording against one named dataset subject.

    Only that shape. An upload is a single channel, so there is no second arm
    to put it against, and ranking it inside the dataset league table would
    put an assumed baseline alongside twelve measured ones as if they were
    equivalent.
    """
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
            "chart_html": _upload_chart(cache),
            "window": {"source": "upload", "name": cache["name"],
                       "kind": "compare"}}


def _upload_analysis(user_text: str, cache: dict, kind: str) -> dict:
    """Onset and whole-recording summary for an uploaded file.

    Cached on the upload itself rather than through st.cache_data: the segment
    is a live object in session state, not a hashable cache key, and the scan
    is the expensive part of both answers.
    """
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
    facts = (onset_facts(summary) if kind == intent_router.ONSET
             else overview_facts(summary))
    instruction = (
        "Answer in 2-4 sentences. Say when fatigue set in and how sure the "
        "reading is, and state plainly that it is approximate."
        if kind == intent_router.ONSET else
        "Answer in 3-5 sentences. Describe how the recording develops from "
        "start to finish, quoting the median frequency at each end and how "
        "much of the recording was classified as fatigued.")
    return {"prompt": build_facts_prompt("Measured results:", facts, user_text,
                                         instruction),
            "facts": facts, "user_text": user_text,
            "chart_html": _upload_chart(cache),
            "window": {"source": "upload", "name": cache["name"], "kind": kind}}


def _upload_chart(cache: dict):
    try:
        mdf_t, mdf_v, _ = data_loader.mdf_trend(cache["seg"], fs=cache["fs"])
        return charts.raw_and_mdf_figure(cache["seg"], mdf_t, mdf_v,
                                         title=f"Uploaded: {cache['name']}")
    except Exception:
        return None      # chart is additive; never block the text answer


def _upload_reading(user_text: str, cache: dict) -> dict:
    uploaded_name = cache["name"]
    seg, fs = cache["seg"], cache["fs"]
    duration = float(seg.t[-1]) if seg.t.size else 0.0
    # duration is passed so "near the end" / "halfway" resolve on an upload
    # exactly as they do for a dataset subject
    t_start = extract.extract_t_start_seconds(user_text, duration)
    if t_start is None:
        t_start = duration  # "how fatigued am I" with no time -> right now
    elif t_start > duration:
        t_start = duration

    try:
        result, baseline = classify_upload(seg, fs, t_start, baseline=cache["baseline"])
    except ValueError as e:
        # the calibration guards live here -- their messages are user-facing
        return {"content": f"I can't give you a reliable reading for that file: {e}"}
    cache["baseline"] = baseline
    result["fatigue_state"] = _FATIGUE_STATE.get(
        result["fatigue_label"], str(result["fatigue_label"]))

    forecast, forecast_chart_html = _forecast(seg, fs, user_text, t_start)
    try:
        reference = upload_reference(cache["baseline"])
    except Exception:
        reference = None
    reading = _plain_reading(result, reference, t_start, duration,
                             "This recording")

    return {"features": result, "reading": reading,
            "chart_html": _upload_chart(cache), "forecast": forecast,
            "forecast_chart_html": forecast_chart_html, "user_text": user_text,
            "window": {"t_start": t_start, "source": "upload",
                       "name": uploaded_name}}


def _forecast(seg, fs: int, user_text: str, t_start: float | None = None):
    """Forecast only when the question actually asks about the future.

    It used to run on every turn (the horizon defaulted to 20 s), so a plain
    "is subject 13 fatigued?" got an unrequested projection chart. It is also
    anchored at `t_start` now, so the trend is fitted on history up to the
    moment being asked about rather than the whole recording.
    """
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
        return forecast, None   # chart is additive; keep the text answer


def _finalize(turn: dict) -> dict:
    # a pre-built analysis prompt (onset / overview / comparison / ranking):
    # the numbers are already computed, the LLM only phrases them
    if "prompt" in turn:
        try:
            content = chat([{"role": "user", "content": turn["prompt"]}],
                           model=st.session_state.model)
        except LLMError as e:
            # Everything above was measured in Python, so the answer is not
            # lost when the model is -- only its wording is. State the numbers.
            lines = readable_facts(turn.get("facts") or [])
            content = ("\n".join(f"- {line}" for line in lines)
                       + f"\n\n_Couldn't phrase this in prose: {e}_")
        return {"content": content, "chart_html": turn.get("chart_html"),
                "forecast_chart_html": None, "recommendation": None,
                "facts": turn.get("facts"),
                "user_text": turn.get("user_text"), "window": turn.get("window")}

    if "features" not in turn:
        # Already-final text: the catalogue, a definition, the ranking table,
        # a comparison, or an error. It still carries its chart, its measured
        # facts (so a following "why?" has them) and its window (so the next
        # turn knows whether the conversation is on an upload or the dataset).
        return {"content": turn["content"],
                "chart_html": turn.get("chart_html"),
                "forecast_chart_html": None, "recommendation": None,
                "facts": turn.get("facts"), "window": turn.get("window"),
                "user_text": turn.get("user_text"),
                "suggestions": turn.get("suggestions")}

    features, forecast, user_text = turn["features"], turn.get("forecast"), turn["user_text"]
    window = turn.get("window")

    # The finding leads, written in Python so it cannot be inverted, and the
    # model's prose follows it. Handed the verdict to phrase, llama3.2:3b
    # inverted a "not fatigued" into "is fatigued" and twice reported the
    # current reading as the fresh baseline -- with prompt rules in place
    # forbidding both. Rendering the sentence is the only thing that held.
    reading = turn.get("reading")
    verdict = (reading or {}).get("verdict")
    try:
        prose = chat([{"role": "user",
                       "content": build_prompt(features, user_text, forecast,
                                               window, reading)}],
                     model=st.session_state.model)
        if verdict:
            # The model is given no figures for a reading, so anything
            # numeric it produces is fabricated -- drop those sentences
            # before the reader ever sees them.
            cleaned, invented = interpret.strip_invented_numbers(
                interpret.strip_verdict_echo(prose))
            content = f"{verdict}\n\n{cleaned}" if cleaned else verdict
        else:
            content = prose
    except LLMError as e:
        # Ollama slow, missing, or not installed on the server yet. The
        # measurement is unaffected -- only the prose around it is lost.
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
            recommendation = ensure_disclaimer(chat(
                [{"role": "user", "content": build_recommendation_prompt(
                    features, forecast, user_text,
                    st.session_state.athlete_note or None, turn.get("reading"))}],
                model=st.session_state.model))
        except LLMError as e:
            recommendation = f"[Recommendation unavailable: {e}]"

    return {"content": content, "chart_html": turn.get("chart_html"),
            "forecast_chart_html": turn.get("forecast_chart_html"),
            "recommendation": recommendation,
            "provenance": _provenance(window, features, reading),
            "features": features, "reading": turn.get("reading"),
            "forecast": forecast, "user_text": user_text, "window": window}


def _provenance(window: dict | None, features: dict,
                reading: dict | None = None) -> str | None:
    """One line naming exactly what was measured, shown under every answer.

    Without it the reader cannot tell which window the model actually scored,
    so a mis-resolved follow-up looks identical to a correct one. It also
    carries the raw figures, which are no longer given to the model at all --
    it wove them into the prose as "a z-score of -1.2 and confidence level of
    100.0%", which is jargon in the middle of an answer written for someone
    who is not a specialist. Here they are out of the way but still checkable.
    """
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


def _followup_upload(user_text: str) -> dict | None:
    """The upload this question is still about, or None for the dataset."""
    if st.session_state.last_source != "upload":
        return None
    cache = st.session_state.uploads.get(st.session_state.last_upload)
    if cache is None:
        return None
    return cache if intent_router.stays_on_upload(user_text) else None


def _handle_turn(user_text: str, uploaded_file) -> None:
    chat_obj = st.session_state.chat
    display_text = user_text or (f"[uploaded {uploaded_file.name}]" if uploaded_file else "")
    chat_obj["messages"].append({"role": "user", "content": display_text or "(empty message)"})

    with st.spinner("Working on it..."):
        if uploaded_file is not None:
            turn = _upload_turn(user_text, uploaded_file)
        else:
            cache = _followup_upload(user_text)
            turn = (_upload_question(user_text, cache) if cache
                    else _dataset_turn(user_text, st.session_state.last_params))
        final = _finalize(turn)

    chat_obj["messages"].append({"role": "assistant", **final})
    st.session_state.last_turn_context = final if "features" in final else None
    # Kept for every kind of answer, not just readings, so "why?" works after an
    # onset or comparison too. A follow-up is not itself something to follow up
    # on, so it does not overwrite the answer it was explaining.
    if final.get("content") and not intent_router.is_followup(user_text or ""):
        st.session_state.last_answer = {
            "question": display_text,
            "answer": final["content"],
            "facts": (final.get("facts")
                      or (final.get("reading") or {}).get("lines")),
        }
    # remember the resolved window so the next turn can say "and at 90 seconds?".
    # Analysis turns (onset/overview) name a subject but no single time, so the
    # previous time is kept rather than dropped -- otherwise asking "summarise
    # subject 7" mid-conversation would strand the follow-up.
    window = final.get("window")
    # Which thread the conversation is on now. Only a turn that actually read a
    # recording moves it -- a definition or a "why?" leaves it where it was.
    if window and window.get("source") in ("upload", "dataset"):
        st.session_state.last_source = window["source"]
    if window and window.get("source") == "dataset":
        prev = st.session_state.last_params or {}
        st.session_state.last_params = {
            "subject": window.get("subject", prev.get("subject")),
            "t_start": window.get("t_start", prev.get("t_start")),
            "side": window.get("side", prev.get("side", "R"))}
    history.save_chat(chat_obj)


def _regenerate() -> None:
    ctx = st.session_state.last_turn_context
    if not ctx or "features" not in ctx:
        return
    chat_obj = st.session_state.chat
    if chat_obj["messages"] and chat_obj["messages"][-1]["role"] == "assistant":
        chat_obj["messages"].pop()

    turn = {"features": ctx["features"], "forecast": ctx.get("forecast"),
           "reading": ctx.get("reading"),
           "user_text": ctx["user_text"], "chart_html": ctx.get("chart_html"),
           "forecast_chart_html": ctx.get("forecast_chart_html"),
           "window": ctx.get("window")}
    final = _finalize(turn)
    chat_obj["messages"].append({"role": "assistant", **final})
    st.session_state.last_turn_context = final if "features" in final else None
    history.save_chat(chat_obj)


# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("EMG Fatigue Chatbot")

    if st.button("+ New chat", use_container_width=True):
        st.session_state.chat = history.new_chat()
        st.session_state.last_turn_context = None
        st.session_state.last_params = None
        st.rerun()

    st.caption("Chats")
    for c in history.list_chats():
        is_active = c["id"] == st.session_state.chat["id"]
        col_open, col_del = st.columns([5, 1])
        label = c.get("title") or "New chat"
        if col_open.button(("• " if is_active else "") + label,
                           key=f"open_{c['id']}", use_container_width=True):
            st.session_state.chat = c
            st.session_state.last_turn_context = None
            st.session_state.last_params = None
            st.rerun()
        if col_del.button("🗑", key=f"del_{c['id']}"):
            history.delete_chat(c["id"])
            if is_active:
                st.session_state.chat = history.new_chat()
            st.rerun()
        st.caption(history.relative_time(c.get("updated_at", "")))

    st.divider()
    model_options = list_models()
    model_index = (model_options.index(st.session_state.model)
                  if st.session_state.model in model_options else 0)
    st.session_state.model = st.selectbox("Model", options=model_options, index=model_index)
    # left blank on purpose: a non-empty default silently assumed every
    # single-column upload was 1000 Hz, which quietly rescales the whole
    # recording and makes the "I need a sample rate" error unreachable
    st.session_state.sample_rate = st.number_input(
        "Sample rate for single-column uploads (Hz)",
        min_value=1.0, value=st.session_state.sample_rate, step=1.0,
        placeholder="required for 1-column files",
        help="Two-column files (time_s, signal) infer this automatically.")
    st.session_state.athlete_note = st.text_input(
        "Sport/goal (optional, for recommendations)",
        value=st.session_state.athlete_note,
        placeholder="e.g. competitive rock climbing")

    st.divider()
    if not st.session_state.confirm_clear:
        if st.button("Clear all history"):
            st.session_state.confirm_clear = True
            st.rerun()
    else:
        st.warning("Delete every saved chat? This can't be undone.")
        c1, c2 = st.columns(2)
        if c1.button("Yes, delete"):
            for c in history.list_chats():
                history.delete_chat(c["id"])
            st.session_state.chat = history.new_chat()
            st.session_state.confirm_clear = False
            st.rerun()
        if c2.button("Cancel"):
            st.session_state.confirm_clear = False
            st.rerun()

    st.divider()
    st.caption("Educational final-year project demo -- not medical, "
              "professional coaching, or nutrition advice.")


# ---------------------------------------------------------------------------
# main chat area
# ---------------------------------------------------------------------------
st.title("EMG Fatigue Chatbot")

def _stage(prompt_text: str) -> None:
    """Put a suggestion in the draft box rather than sending it.

    st.chat_input has no `value` parameter in Streamlit 1.60, so a suggestion
    cannot be typed into the real chat box. Staging it in an editable field
    gives the same thing that actually matters: the wording is filled in for
    you, and nothing is sent until you have read it and pressed send.
    """
    st.session_state.draft = prompt_text
    st.session_state.draft_nonce += 1


for i, msg in enumerate(st.session_state.chat["messages"]):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("provenance"):
            st.caption(msg["provenance"])
        # Collapsed, not removed. The signal viewer is three stacked technical
        # panels (raw EMG, an MDF scatter, an FFT spectrum) with a scrubber --
        # genuinely useful to us and to a supervisor, and unreadable to the
        # non-technical reader this chatbot is for. Open by default it also
        # pushed the actual answer off the top of the screen. It stays one
        # click away.
        if msg.get("chart_html"):
            with st.expander("Show the signal (technical view)"):
                st.iframe(msg["chart_html"], height=600)
        if msg.get("forecast_chart_html"):
            with st.expander("Show the forecast chart"):
                st.iframe(msg["forecast_chart_html"], height=420)
        if msg.get("recommendation"):
            st.info(msg["recommendation"])
        suggestions = msg.get("suggestions") or []
        if suggestions:
            for row_start in range(0, len(suggestions), 3):
                row = suggestions[row_start:row_start + 3]
                for col, (label, prompt_text) in zip(st.columns(len(row)), row):
                    col.button(label, key=f"sug_{i}_{row_start}_{label}",
                               use_container_width=True,
                               on_click=_stage, args=(prompt_text,))

messages = st.session_state.chat["messages"]
if (messages and messages[-1]["role"] == "assistant" and st.session_state.last_turn_context):
    if st.button("🔄 Regenerate"):
        _regenerate()
        st.rerun()

# The staged suggestion: filled in, editable, and not sent until you say so.
if st.session_state.draft is not None:
    with st.container(border=True):
        st.caption("Check or edit this, then send it:")
        edited = st.text_input(
            "Staged question", value=st.session_state.draft,
            key=f"draft_box_{st.session_state.draft_nonce}",
            label_visibility="collapsed")
        col_send, col_cancel, _ = st.columns([1, 1, 4])
        if col_send.button("Send", type="primary", use_container_width=True):
            st.session_state.draft = None
            _handle_turn(edited, None)
            st.rerun()
        if col_cancel.button("Cancel", use_container_width=True):
            st.session_state.draft = None
            st.rerun()

chat_value = st.chat_input(
    "Ask about a subject, e.g. \"Is subject 13 fatigued at 60 seconds on the "
    "right side?\", or attach your own EMG recording (+ button).",
    accept_file=True, file_type=["csv"])

if chat_value is not None:
    st.session_state.draft = None
    _handle_turn(chat_value.text, chat_value.files[0] if chat_value.files else None)
    st.rerun()
