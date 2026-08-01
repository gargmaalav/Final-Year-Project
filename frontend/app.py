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

from classify import classify, classify_upload      # noqa: E402
from fatigue_forecast import forecast_fatigue         # noqa: E402
from render_window import render_window               # noqa: E402
import loader as data_loader                           # noqa: E402

import charts                                    # noqa: E402
import extract                                   # noqa: E402
import history                                   # noqa: E402
from llm import LLMError, chat, list_models      # noqa: E402
from prompt import build_prompt                  # noqa: E402
from recommend import build_recommendation_prompt, wants_recommendation  # noqa: E402
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
    st.session_state.sample_rate = 1000.0
if "athlete_note" not in st.session_state:
    st.session_state.athlete_note = ""
if "uploads" not in st.session_state:
    st.session_state.uploads = {}          # key -> {"seg", "fs", "baseline"}
if "confirm_clear" not in st.session_state:
    st.session_state.confirm_clear = False
if "last_turn_context" not in st.session_state:
    st.session_state.last_turn_context = None


# ---------------------------------------------------------------------------
# pipeline helpers
# ---------------------------------------------------------------------------
def _dataset_turn(user_text: str) -> dict:
    params = extract.parse_query(user_text)
    if params is None:
        return {"content": (
            "I couldn't tell which subject (1-13), time (seconds), and side "
            "(R/L) you're asking about -- could you spell that out, e.g. "
            "\"subject 13 at 60 seconds, right side\"?")}

    subject, t_start, side = params["subject"], params["t_start"], params["side"]
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

    seg = data_loader.load_biceps_segment(DATA_ROOT, subject, side,
                                          target_fs=250, bandpass=True)
    fs = int(getattr(seg, "eff_fs", 250))
    forecast, forecast_chart_html = _forecast(seg, fs, user_text)

    return {"features": result, "chart_html": chart_html, "forecast": forecast,
            "forecast_chart_html": forecast_chart_html, "user_text": user_text}


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
        fs = int(getattr(seg, "eff_fs", 250))
        cache = {"seg": seg, "fs": fs, "baseline": None}
        st.session_state.uploads[key] = cache

    seg, fs = cache["seg"], cache["fs"]
    duration = float(seg.t[-1]) if seg.t.size else 0.0
    t_start = extract.extract_t_start_seconds(user_text)
    if t_start is None:
        t_start = duration  # "how fatigued am I" with no time -> right now

    try:
        result, baseline = classify_upload(seg, fs, t_start, baseline=cache["baseline"])
    except ValueError as e:
        return {"content": f"Couldn't classify that recording: {e}"}
    cache["baseline"] = baseline
    result["fatigue_state"] = _FATIGUE_STATE.get(
        result["fatigue_label"], str(result["fatigue_label"]))

    mdf_t, mdf_v, _ = data_loader.mdf_trend(seg, fs=fs)
    chart_html = charts.raw_and_mdf_figure(
        seg, mdf_t, mdf_v, title=f"Uploaded: {uploaded_file.name}")

    forecast, forecast_chart_html = _forecast(seg, fs, user_text)

    return {"features": result, "chart_html": chart_html, "forecast": forecast,
            "forecast_chart_html": forecast_chart_html, "user_text": user_text}


def _forecast(seg, fs: int, user_text: str):
    horizon = extract.extract_horizon_seconds(user_text)
    try:
        forecast = forecast_fatigue(seg, fs, horizon_sec=horizon)
    except Exception:
        return None, None
    if not forecast.get("ok"):
        return forecast, None
    return forecast, charts.forecast_figure(forecast)


def _finalize(turn: dict) -> dict:
    if "features" not in turn:
        return {"content": turn["content"], "chart_html": None,
                "forecast_chart_html": None, "recommendation": None}

    features, forecast, user_text = turn["features"], turn.get("forecast"), turn["user_text"]

    try:
        content = chat([{"role": "user", "content": build_prompt(features, user_text, forecast)}],
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
            "forecast_chart_html": turn.get("forecast_chart_html"),
            "recommendation": recommendation,
            "features": features, "forecast": forecast, "user_text": user_text}


def _handle_turn(user_text: str, uploaded_file) -> None:
    chat_obj = st.session_state.chat
    display_text = user_text or (f"[uploaded {uploaded_file.name}]" if uploaded_file else "")
    chat_obj["messages"].append({"role": "user", "content": display_text or "(empty message)"})

    with st.spinner("Working on it..."):
        turn = (_upload_turn(user_text, uploaded_file) if uploaded_file is not None
               else _dataset_turn(user_text))
        final = _finalize(turn)

    chat_obj["messages"].append({"role": "assistant", **final})
    st.session_state.last_turn_context = final if "features" in final else None
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
           "forecast_chart_html": ctx.get("forecast_chart_html")}
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
    st.session_state.sample_rate = st.number_input(
        "Sample rate for single-column uploads (Hz)",
        min_value=1.0, value=st.session_state.sample_rate, step=1.0)
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
        if msg.get("chart_html"):
            st.iframe(msg["chart_html"], height=600)
        if msg.get("forecast_chart_html"):
            st.iframe(msg["forecast_chart_html"], height=420)
        if msg.get("recommendation"):
            st.info(msg["recommendation"])

messages = st.session_state.chat["messages"]
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
