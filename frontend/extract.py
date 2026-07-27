"""
parse_query(user_query) -> {"subject": int, "t_start": float, "side": "R"|"L"} | None
========================================================================================

Asks the LLM to pull the (subject, t_start, side) triple out of a free-text
question as a single JSON object. This is the ONLY place an LLM output is
trusted in the pipeline, and only to pick which window to look up -- the
fatigue numbers themselves always come straight from classify(), never from
the LLM. Extraction is a much smaller, more constrained task than full
tool-calling, so a 3B model handles it reliably; on top of that we validate
the result and return None (never a guess) on anything malformed, so the app
can ask the user to clarify instead of silently making something up.
"""
from __future__ import annotations

import json
import re

from llm import LLMError, chat

_SYSTEM_PROMPT = """You extract structured parameters from a question about an \
EMG muscle-fatigue dataset. The dataset has subjects numbered 1-13, a \
recording time in seconds, and a side which is either "R" (right arm) or \
"L" (left arm).

Reply with ONLY a single-line JSON object of the form:
{"subject": <int 1-13>, "t_start": <number, seconds>, "side": "R" or "L"}

If the question is missing the subject number or the time in seconds, reply \
with exactly: {"missing": true}

Do not include any other text, explanation, or markdown formatting."""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_query(user_query: str) -> dict | None:
    try:
        reply = chat([
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ])
    except LLMError:
        return None

    match = _JSON_RE.search(reply)
    if not match:
        return None

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    if data.get("missing"):
        return None

    return _validate(data)


def _validate(data: dict) -> dict | None:
    try:
        subject = int(data["subject"])
        t_start = float(data["t_start"])
        side = str(data["side"]).strip().upper()
    except (KeyError, TypeError, ValueError):
        return None

    if not (1 <= subject <= 13):
        return None
    if t_start < 0:
        return None
    if side not in ("R", "L"):
        return None

    return {"subject": subject, "t_start": t_start, "side": side}


# ---------------------------------------------------------------------------
# Lightweight regex extraction for the upload path (no dataset subject/side
# to resolve there, so a full LLM extraction call is unnecessary -- deterministic
# regex is simpler and faster for just a time offset / forecast horizon).
# ---------------------------------------------------------------------------
_HORIZON_RE = re.compile(
    r"next\s+(\d+(?:\.\d+)?)\s*(seconds?|secs?|s\b|minutes?|mins?|m\b)",
    re.IGNORECASE)
_TIME_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(seconds?|secs?|s\b)", re.IGNORECASE)


def extract_horizon_seconds(text: str, default: float = 20.0) -> float:
    match = _HORIZON_RE.search(text)
    if not match:
        return default
    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit.startswith("min") or unit == "m":
        value *= 60.0
    return value


def extract_t_start_seconds(text: str) -> float | None:
    """A plain '<N> seconds' mention, ignoring a 'next N seconds' horizon
    phrase so the two don't get confused with each other."""
    stripped = _HORIZON_RE.sub("", text)
    match = _TIME_RE.search(stripped)
    return float(match.group(1)) if match else None
