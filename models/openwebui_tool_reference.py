"""
title: EMG Fatigue Classifier
author: Aryan Shah
version: 0.1.0
requirements: requests
description: Query the deep-learning EMG fatigue model for a subject at a time point.
"""
# REFERENCE COPY -- this is not imported by the project. Paste its contents into
# Open WebUI: Workspace -> Tools -> + (Create Tool) -> paste -> Save, then enable
# the tool on your chat model. Open WebUI reads the class + method docstrings to
# build the function-calling schema the LLM uses.
#
# Requires models/serve.py running on the host first.
import requests

# models/serve.py runs on the host; Open WebUI (in Docker) reaches it here:
API_URL = "http://host.docker.internal:8000/classify"


class Tools:
    def __init__(self):
        pass

    def get_fatigue(self, subject: int, t_start: float, side: str = "R") -> str:
        """
        Get the muscle-fatigue state predicted by the EMG deep-learning model for
        a subject at a given time in their recording. Call this whenever the user
        asks whether a subject is fatigued, or about the fatigue state / median
        frequency at a specific time point.

        :param subject: subject id (1-13 in the dataset)
        :param t_start: time in seconds into the recording (e.g. 120)
        :param side: which arm, "R" or "L" (default "R")
        :return: a short factual summary the assistant must ground its answer in
        """
        try:
            r = requests.get(
                API_URL,
                params={"subject": subject, "t_start": t_start, "side": side},
                timeout=30,
            )
        except requests.RequestException as e:
            return (f"ERROR: could not reach the fatigue API ({e}). "
                    f"Is models/serve.py running on the host?")

        if r.status_code != 200:
            return f"ERROR {r.status_code}: {r.text}"

        d = r.json()
        return (
            f"Subject {subject}, t={t_start}s ({side} arm): "
            f"fatigue_state={d.get('fatigue_state')}, "
            f"fatigue_label={d['fatigue_label']}, "
            f"median_frequency={d['mdf_hz']} Hz, "
            f"confidence={d['confidence']:.2f}. "
            f"Use only these values in your answer; do not invent numbers."
        )
