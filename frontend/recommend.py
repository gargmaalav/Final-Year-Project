"""
Sport-fit and training/diet plan suggestions, grounded in measured EMG data.
==============================================================================

There is no labeled data anywhere in this project connecting fatigue
patterns to sport suitability, so a trained classifier for "what sport
should this person play" would have no ground truth to learn from. The
defensible version of this feature: the pipeline (models/classify.py,
models/fatigue_forecast.py) supplies the objective numbers, and the LLM's own
general fitness/sports knowledge turns those into suggestions -- same
"ground the LLM in real numbers, no invented ones" principle prompt.py
already applies to plain fatigue answers, extended to a richer prompt that's
explicit about which parts are measured and which are the LLM's general
knowledge.
"""
from __future__ import annotations

import re

_KEYWORDS = re.compile(
    r"\b(sport|recommend|suggest|suggestion|plan|diet|nutrition|gym|"
    r"training|workout|exercise|good at|suitable|suited)\b", re.IGNORECASE)


def wants_recommendation(text: str) -> bool:
    return bool(_KEYWORDS.search(text))


def build_recommendation_prompt(features: dict, forecast: dict | None,
                                user_query: str,
                                athlete_note: str | None = None) -> str:
    forecast_line = (forecast.get("summary", "") if forecast and forecast.get("ok")
                     else "No fatigue-trend forecast available for this reading.")
    note_line = (f"The user says: \"{athlete_note}\"\n" if athlete_note else "")

    return (
        "You are a fitness-education assistant. A single-arm biceps EMG "
        "reading was just measured. Here is the measured data -- treat these "
        "specific numbers as fact, do not change or re-derive them:\n\n"
        f"- fatigue state: {features['fatigue_state']}\n"
        f"- median frequency: {features['mdf_hz']:.1f} Hz\n"
        f"- model confidence: {features['confidence'] * 100:.1f}%\n"
        f"- fatigue trend: {forecast_line}\n\n"
        f"{note_line}"
        f"User question: {user_query}\n\n"
        "Using your own general knowledge of sports science, fitness and "
        "nutrition (there is no dataset backing sport-suitability claims, so "
        "this part is genuinely your general knowledge, not measured data), "
        "suggest: (1) what kind of sports/activities might suit someone with "
        "this muscle's endurance/fatigue profile, and (2) a general gym "
        "training and diet direction to build on it. Keep it to a short "
        "paragraph or a few bullet points.\n\n"
        "End with exactly one line stating this is a general, educational "
        "suggestion from a single biceps EMG reading -- not medical, "
        "professional coaching, or nutrition advice."
    )
