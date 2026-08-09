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
                      available_subjects)
from fatigue_forecast import forecast_fatigue         # noqa: E402
from render_window import render_window               # noqa: E402
import loader as data_loader                           # noqa: E402

import analysis                                  # noqa: E402
import charts                                    # noqa: E402
import extract                                   # noqa: E402
import history                                   # noqa: E402
import intent as intent_router                   # noqa: E402
from llm import LLMError, chat, list_models      # noqa: E402
from prompt import (build_facts_prompt, build_prompt,  # noqa: E402
                    compare_facts, describe_window, onset_facts,
                    overview_facts, ranking_facts)
from recommend import build_recommendation_prompt, wants_recommendation  # noqa: E402
from upload import UploadError, load_uploaded_segment, parse_uploaded_csv  # noqa: E402

# matches models/serve.py's _STATE mapping, kept in sync manually since this
# app calls classify()/classify_upload() directly instead of going through
# serve.py's /classify HTTP route
_FATIGUE_STATE = {0: "non-fatigue", 1: "fatigue", 2: "fatigue"}
DATA_ROOT = os.path.join(_REPO_ROOT, "zenodo_biceps", "sEMG_data")

# Charts used to render unconditionally on every reading -- three panels under
# a one-line answer read as clutter, not help, and made the chart the first
# thing a new user had to parse instead of the answer. Now shown only on
# request (extract.wants_visual), with a plain-language caption so the panels
# don't need to be self-explanatory, and a note when one was withheld so it
# doesn't just silently disappear.
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
_VISUAL_HINT = 'ask to "show the graph" to see a chart of this'

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
    st.session_state.uploads = {}          # key -> {"seg", "fs", "baseline"}
if "confirm_clear" not in st.session_state:
    st.session_state.confirm_clear = False
if "last_turn_context" not in st.session_state:
    st.session_state.last_turn_context = None
if "last_params" not in st.session_state:
    st.session_state.last_params = None   # last resolved {subject, t_start, side}


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


def _catalogue_text() -> str:
    subs = _subjects()
    return (
        f"I have surface EMG recordings for {len(subs)} subjects "
        f"(numbered {subs[0]}-{subs[-1]}), each with a right and a left biceps "
        "recording. They're efforts held to exhaustion, so the recordings vary "
        "a lot in length -- from about 25 seconds to about 8 minutes.\n\n"
        "Things you can ask me:\n"
        "- a reading at a moment: *\"is subject 13 fatigued at 60 seconds?\"*\n"
        "- in plain terms: *\"how about subject 5 near the end?\"*\n"
        "- when it set in: *\"when did subject 13 start fatiguing?\"*\n"
        "- the whole recording: *\"summarise subject 7\"*\n"
        "- side by side: *\"compare subject 5 and 9\"*, *\"left vs right for subject 4\"*\n"
        "- the field: *\"which subject fatigued the most?\"*\n"
        "- the forecast: *\"will subject 2 get more tired over the next minute?\"*\n"
        "- definitions: *\"what does median frequency mean?\"*\n\n"
        "You can also attach your own EMG recording as a CSV with the + button.")


def _analysis_turn(user_text: str, intent, previous: dict | None) -> dict:
    """Handle every question that isn't a single-window reading."""
    kind = intent.kind

    if kind == intent_router.CATALOGUE:
        return {"content": _catalogue_text()}

    if kind == intent_router.EXPLAIN:
        text = analysis.define(intent.term)
        return {"content": text} if text else {"content": _catalogue_text()}

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
                "window": {"subject": subject, "side": side, "source": "dataset",
                           "kind": kind},
                "user_text": user_text}

    # comparisons
    if intent.both_sides:
        subject = _subject_for(intent, previous)
        if subject is None:
            return {"content": _needs_subject_msg()}
        comparison = _cached_compare_sides(subject, None)
        facts = compare_facts(comparison)
        instruction = ("Answer in 2-4 sentences. Say which arm is more "
                       "fatigued and by how much, or that they are similar.")
    elif len(intent.subjects) >= 2:
        subjects = [s for s in intent.subjects if s in _subjects()]
        if len(subjects) < 2:
            return {"content": "I need two subjects I actually have to compare. "
                               + _needs_subject_msg()}
        side = extract.side_from_text(user_text) or "R"
        t_start = extract.t_start_from_text(user_text, None)
        comparison = _cached_compare_subjects(tuple(subjects), t_start, side)
        facts = compare_facts(comparison)
        instruction = ("Answer in 2-4 sentences. Say which subject is more "
                       "fatigued and on what basis. If the subjects were read "
                       "at a fraction of their own recordings rather than the "
                       "same absolute time, say so in one clause.")
    else:
        # a superlative over the whole field ("which subject fatigued most?")
        side = extract.side_from_text(user_text) or "R"
        comparison = _cached_ranking(side)
        facts = ranking_facts(comparison)
        instruction = ("Answer in 3-5 sentences. Name the top few and the "
                       "bottom few with their median-frequency drops, and "
                       "state in one clause what the ranking is based on. "
                       "Mention any excluded subject.")

    return {"prompt": build_facts_prompt("Measured results:", facts, user_text,
                                         instruction),
            "window": None, "user_text": user_text}


def _dataset_turn(user_text: str, previous: dict | None) -> dict:
    intent = intent_router.route(user_text)
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

    chart_html, chart_caption = None, None
    if extract.wants_visual(user_text):
        try:
            chart_html = render_window(subject, t_start, side)
            chart_caption = _READING_CHART_CAPTION
        except Exception:
            pass  # chart is additive; never block the text answer
    else:
        notes.append(_VISUAL_HINT)

    seg, fs = _load_subject_segment(subject, side)
    forecast, forecast_chart_html = _forecast(seg, fs, user_text, t_start)

    return {"features": result, "chart_html": chart_html,
            "chart_caption": chart_caption, "forecast": forecast,
            "forecast_chart_html": forecast_chart_html,
            "forecast_chart_caption": _FORECAST_CHART_CAPTION if forecast_chart_html else None,
            "user_text": user_text, "window": window}


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
        cache = {"seg": seg, "fs": fs, "baseline": None}
        st.session_state.uploads[key] = cache

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

    notes = []
    chart_html, chart_caption = None, None
    # a bare upload with no question has nothing else to show back, so the
    # chart is the answer; otherwise it follows the same "only if asked" rule
    # as the dataset path.
    if extract.wants_visual(user_text) or not user_text.strip():
        mdf_t, mdf_v, _ = data_loader.mdf_trend(seg, fs=fs)
        chart_html = charts.raw_and_mdf_figure(
            seg, mdf_t, mdf_v, title=f"Uploaded: {uploaded_file.name}")
        chart_caption = _UPLOAD_CHART_CAPTION
    else:
        notes.append(_VISUAL_HINT)

    forecast, forecast_chart_html = _forecast(seg, fs, user_text, t_start)

    return {"features": result, "chart_html": chart_html,
            "chart_caption": chart_caption, "forecast": forecast,
            "forecast_chart_html": forecast_chart_html,
            "forecast_chart_caption": _FORECAST_CHART_CAPTION if forecast_chart_html else None,
            "user_text": user_text,
            "window": {"t_start": t_start, "source": "upload",
                       "name": uploaded_file.name, "notes": notes}}


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
            content = (f"[LLM phrasing unavailable: {e}]\n\n"
                       + turn["prompt"].split("Measured results:", 1)[-1]
                                       .split("User question:", 1)[0].strip())
        return {"content": content, "chart_html": turn.get("chart_html"),
                "chart_caption": turn.get("chart_caption"),
                "forecast_chart_html": None, "forecast_chart_caption": None,
                "recommendation": None,
                "user_text": turn.get("user_text"), "window": turn.get("window")}

    if "features" not in turn:
        return {"content": turn["content"], "chart_html": None,
                "chart_caption": None,
                "forecast_chart_html": None, "forecast_chart_caption": None,
                "recommendation": None}

    features, forecast, user_text = turn["features"], turn.get("forecast"), turn["user_text"]
    window = turn.get("window")
    chart_shown = turn.get("chart_html") is not None

    try:
        content = chat([{"role": "user",
                        "content": build_prompt(features, user_text, forecast,
                                                window, chart_shown)}],
                       model=st.session_state.model)
    except LLMError as e:
        content = (f"{features['fatigue_state']} "
                  f"(median frequency {features['mdf_hz']:.1f} Hz, "
                  f"confidence {features['confidence'] * 100:.1f}%). "
                  f"[LLM phrasing unavailable: {e}]")

    recommendation = None
    if wants_recommendation(user_text):
        try:
            recommendation = chat([{"role": "user", "content": build_recommendation_prompt(
                features, forecast, user_text, st.session_state.athlete_note or None)}],
                model=st.session_state.model)
        except LLMError as e:
            recommendation = f"[Recommendation unavailable: {e}]"

    return {"content": content, "chart_html": turn.get("chart_html"),
            "chart_caption": turn.get("chart_caption"),
            "forecast_chart_html": turn.get("forecast_chart_html"),
            "forecast_chart_caption": turn.get("forecast_chart_caption"),
            "recommendation": recommendation, "provenance": _provenance(window, features),
            "features": features, "forecast": forecast, "user_text": user_text,
            "window": window}


def _provenance(window: dict | None, features: dict) -> str | None:
    """One line naming exactly what was measured, shown under every answer.

    Without it the reader cannot tell which window the model actually scored,
    so a mis-resolved follow-up looks identical to a correct one.
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
    return " ".join(parts)


def _handle_turn(user_text: str, uploaded_file) -> None:
    chat_obj = st.session_state.chat
    display_text = user_text or (f"[uploaded {uploaded_file.name}]" if uploaded_file else "")
    chat_obj["messages"].append({"role": "user", "content": display_text or "(empty message)"})

    with st.spinner("Working on it..."):
        turn = (_upload_turn(user_text, uploaded_file) if uploaded_file is not None
               else _dataset_turn(user_text, st.session_state.last_params))
        final = _finalize(turn)

    chat_obj["messages"].append({"role": "assistant", **final})
    st.session_state.last_turn_context = final if "features" in final else None
    # remember the resolved window so the next turn can say "and at 90 seconds?"
    # remember the resolved window so the next turn can say "and at 90 seconds?".
    # Analysis turns (onset/overview) name a subject but no single time, so the
    # previous time is kept rather than dropped -- otherwise asking "summarise
    # subject 7" mid-conversation would strand the follow-up.
    window = final.get("window")
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
           "user_text": ctx["user_text"], "chart_html": ctx.get("chart_html"),
           "chart_caption": ctx.get("chart_caption"),
           "forecast_chart_html": ctx.get("forecast_chart_html"),
           "forecast_chart_caption": ctx.get("forecast_chart_caption"),
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

for msg in st.session_state.chat["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("provenance"):
            st.caption(msg["provenance"])
        if msg.get("chart_html"):
            if msg.get("chart_caption"):
                st.caption(msg["chart_caption"])
            st.iframe(msg["chart_html"], height=600)
        if msg.get("forecast_chart_html"):
            if msg.get("forecast_chart_caption"):
                st.caption(msg["forecast_chart_caption"])
            st.iframe(msg["forecast_chart_html"], height=420)
        if msg.get("recommendation"):
            st.info(msg["recommendation"])

messages = st.session_state.chat["messages"]

# A blank chat window with nothing but a placeholder made the assistant look
# narrower than it is -- someone has to guess a phrasing that happens to work.
# One click on any of these runs the real question through the real pipeline,
# it's not a canned reply -- so the range of examples doubles as an honest
# demo of what the intent router (frontend/intent.py) actually covers.
_SUGGESTIONS = [
    "Is subject 13 fatigued at 60 seconds?",
    "How about subject 5 near the end?",
    "When did subject 13 start fatiguing?",
    "Summarise subject 7",
    "Compare subject 5 and 9",
    "Which subject fatigued the most?",
    "What does median frequency mean?",
    "What data do you have?",
]
if not messages:
    st.caption("Try asking:")
    cols = st.columns(4)
    for i, suggestion in enumerate(_SUGGESTIONS):
        if cols[i % 4].button(suggestion, key=f"suggest_{i}", use_container_width=True):
            _handle_turn(suggestion, None)
            st.rerun()

if (messages and messages[-1]["role"] == "assistant" and st.session_state.last_turn_context):
    if st.button("🔄 Regenerate"):
        _regenerate()
        st.rerun()

chat_value = st.chat_input(
    "Ask about a subject, e.g. \"Is subject 13 fatigued at 60 seconds on the "
    "right side?\", or attach your own EMG recording (+ button).",
    accept_file=True, file_type=["csv"])

if chat_value is not None:
    _handle_turn(chat_value.text, chat_value.files[0] if chat_value.files else None)
    st.rerun()
