"""
title: EMG Fatigue Classifier
author: Aryan Shah
version: 0.3.0
requirements: requests
description: Query the deep-learning EMG fatigue model for a subject at a time point, with an interactive chart embedded inline.
"""
# REFERENCE COPY -- this is not imported by the project. Paste its contents into
# Open WebUI: Workspace -> Tools -> + (Create Tool) -> paste -> Save, then enable
# the tool on your chat model. Open WebUI reads the class + method docstrings to
# build the function-calling schema the LLM uses. One method is registered:
# get_fatigue, for a named subject at a time point. (The all-subjects overview
# was dropped 2026-07-13 -- 13 overlaid lines were unreadable; the single-subject
# chart is the deliverable.)
#
# Chatbot must stay on Legacy function-calling mode (Native hallucinates instead
# of calling the tool on this Ollama build). Requires models/serve.py running on
# the host first (both /classify and /render).
#
# fastapi is already an Open WebUI dependency (Open WebUI is itself a FastAPI
# app), so HTMLResponse needs no extra `requirements:` entry above.
import requests
from fastapi.responses import HTMLResponse

# Open WebUI and models/serve.py both run natively on the same host (verified
# 2026-07-12: `.venv/bin/open-webui serve`, no Docker anywhere in this repo),
# so plain localhost reaches it. If you later containerise Open WebUI, switch
# this to http://host.docker.internal:8000 instead.
API_BASE = "http://localhost:8000"


def _embed(html: str):
    """Wrap render_window() HTML so Open WebUI renders it as an inline chart
    (open_webui/utils/middleware.py process_tool_result()'s tuple-unpack)."""
    return HTMLResponse(content=html, headers={"Content-Disposition": "inline"})


class Tools:
    def __init__(self):
        pass

    def get_fatigue(self, subject: int, t_start: float, side: str = "R"):
        """
        Get the muscle-fatigue state predicted by the EMG deep-learning model for
        a subject at a given time in their recording, with an interactive chart
        of that window. Call this whenever the user asks whether a subject is
        fatigued, or about the fatigue state / median frequency at a specific
        time point.

        :param subject: subject id (1-13 in the dataset)
        :param t_start: time in seconds into the recording (e.g. 120)
        :param side: which arm, "R" or "L" (default "R")
        :return: (chart, summary) if the chart renders, else summary alone
        """
        try:
            r = requests.get(
                f"{API_BASE}/classify",
                params={"subject": subject, "t_start": t_start, "side": side},
                timeout=30,
            )
        except requests.RequestException as e:
            return (f"ERROR: could not reach the fatigue API ({e}). "
                    f"Is models/serve.py running on the host?")

        if r.status_code != 200:
            return f"ERROR {r.status_code}: {r.text}"

        d = r.json()
        grounding = (
            f"Subject {subject}, t={t_start}s ({side} arm): "
            f"fatigue_state={d.get('fatigue_state')}, "
            f"fatigue_label={d['fatigue_label']}, "
            f"median_frequency={d['mdf_hz']} Hz, "
            f"confidence={d['confidence']:.2f}. "
            f"Use only these values in your answer; do not invent numbers."
        )

        # chart is additive -- any failure here still returns the grounding text
        try:
            rr = requests.get(
                f"{API_BASE}/render",
                params={"subject": subject, "t_start": t_start, "side": side},
                timeout=30,
            )
            if rr.status_code == 200:
                return _embed(rr.json()["html"]), grounding
        except requests.RequestException:
            pass
        return grounding
