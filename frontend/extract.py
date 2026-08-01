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

GROUNDING (added after measuring the failure mode)
--------------------------------------------------
Range-validation alone is not enough. On a FOLLOW-UP turn with no subject in
it, llama3.2:3b does not answer {"missing": true} -- it invents a plausible
in-range triple:

    "what about the left arm?"  -> {"subject": 13, "t_start": 120.0, ...}
    "and at 90 seconds?"        -> {"subject": 13, "t_start": 90.0,  ...}

Nothing in either message mentions subject 13 or 120 seconds. Those pass every
range check, so the app would classify a fabricated window and report it with
full confidence -- exactly the hallucination this architecture exists to
prevent, just moved from tool-calling to parameter extraction.

So every number the LLM returns is now cross-checked against numbers the user
actually typed (`_mentioned_numbers`, which understands digits, number words,
"two minutes" -> 120, and "1:30" -> 90). Anything ungrounded is discarded and
filled from the previous turn's parameters instead, so "and at 90 seconds?"
correctly reuses the subject you were already discussing. Side never goes to
the LLM at all -- "left"/"right" is a regex.
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


_WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "ninety": 90, "hundred": 100,
}
_CLOCK_RE = re.compile(r"\b(\d{1,3}):([0-5]\d)\b")


def _mentioned_numbers(text: str) -> set[float]:
    """Every number the user could plausibly have meant, from what they typed.

    Includes digits, number words, "1:30" as 90 seconds, and minute readings
    converted to seconds -- so "two minutes in" grounds a t_start of 120.
    """
    found: set[float] = set()
    for m in _CLOCK_RE.finditer(text):
        found.add(float(m.group(1)) * 60 + float(m.group(2)))
    for tok in re.findall(r"\d+(?:\.\d+)?", text):
        found.add(float(tok))
    lowered = text.lower()
    for word, value in _WORD_NUMBERS.items():
        if re.search(rf"\b{word}\b", lowered):
            found.add(float(value))
    # a bare number followed by a minute unit also grounds its value in seconds
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:minutes?|mins?|m)\b", lowered):
        found.add(float(m.group(1)) * 60.0)
    for word, value in _WORD_NUMBERS.items():
        if re.search(rf"\b{word}\s+(?:minutes?|mins?)\b", lowered):
            found.add(float(value) * 60.0)
    return found


def _side_from_text(text: str) -> str | None:
    """Side is a two-way choice stated in plain words -- no LLM needed."""
    lowered = text.lower()
    if re.search(r"\b(left|lefthand|l\b)\b", lowered):
        return "L"
    if re.search(r"\b(right|righthand|r\b)\b", lowered):
        return "R"
    return None


def parse_query(user_query: str, previous: dict | None = None) -> dict | None:
    """Resolve a query to {subject, t_start, side}, or None if it can't be.

    `previous` is the last turn's resolved parameters. Anything the user did
    not actually state this turn is carried over from it rather than guessed,
    which is what makes follow-ups like "and at 90 seconds?" work.
    """
    try:
        reply = chat([
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ])
    except LLMError:
        reply = ""

    data = {}
    match = _JSON_RE.search(reply)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if not parsed.get("missing"):
                data = parsed
        except json.JSONDecodeError:
            data = {}

    return _resolve(data, user_query, previous)


def _grounded_subject(data: dict, mentioned: set[float]) -> int | None:
    """The LLM's subject, but only if the user actually typed that number."""
    try:
        value = int(data["subject"])
    except (KeyError, TypeError, ValueError):
        return None
    if 1 <= value <= 13 and float(value) in mentioned:
        return value
    return None


def _grounded_t_start(data: dict, mentioned: set[float]) -> float | None:
    """The LLM's t_start, but only if the user actually typed that number."""
    try:
        value = float(data["t_start"])
    except (KeyError, TypeError, ValueError):
        return None
    if value >= 0 and value in mentioned:
        return value
    return None


def _resolve(data: dict, user_query: str, previous: dict | None) -> dict | None:
    mentioned = _mentioned_numbers(user_query)
    prev = previous or {}

    subject = _grounded_subject(data, mentioned)
    t_start = _grounded_t_start(data, mentioned)
    carried = [name for name, value in (("subject", subject), ("time", t_start))
               if value is None]

    if subject is None:
        subject = prev.get("subject")
    if t_start is None:
        t_start = prev.get("t_start")
    side = _side_from_text(user_query) or prev.get("side") or "R"

    if subject is None or t_start is None:
        return None
    return {"subject": int(subject), "t_start": float(t_start), "side": side,
            "carried_over": carried}


# ---------------------------------------------------------------------------
# Lightweight regex extraction for the upload path (no dataset subject/side
# to resolve there, so a full LLM extraction call is unnecessary -- deterministic
# regex is simpler and faster for just a time offset / forecast horizon).
# ---------------------------------------------------------------------------
_HORIZON_RE = re.compile(
    r"(?:next|in|over the next|after|within)\s+(?:the\s+)?(\d+(?:\.\d+)?)\s*"
    r"(seconds?|secs?|s\b|minutes?|mins?|m\b)",
    re.IGNORECASE)
_TIME_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(seconds?|secs?|s\b)", re.IGNORECASE)

# Asking about the future without naming a horizon ("will I get more tired?").
_FUTURE_INTENT_RE = re.compile(
    r"\b(will|going to|gonna|predict|forecast|project(?:ion|ed)?|"
    r"keep going|carry on|continue|later on|from here|trend)\b", re.IGNORECASE)
DEFAULT_HORIZON_SEC = 20.0


def extract_horizon_seconds(text: str,
                            default: float | None = DEFAULT_HORIZON_SEC) -> float | None:
    """Forecast horizon in seconds, or `default` when no forecast was asked for.

    Pass default=None to distinguish "no forecast wanted" from "forecast with
    the default horizon" -- the app needs that, otherwise every single question
    silently gets a projection chart nobody requested.
    """
    match = _HORIZON_RE.search(text)
    if match:
        value = float(match.group(1))
        unit = match.group(2).lower()
        if unit.startswith("min") or unit == "m":
            value *= 60.0
        return value
    if _FUTURE_INTENT_RE.search(text):
        return DEFAULT_HORIZON_SEC
    return default


def extract_t_start_seconds(text: str) -> float | None:
    """A plain '<N> seconds' or '1:30' mention, ignoring any horizon phrase
    ('next N seconds') so the two don't get confused with each other."""
    stripped = _HORIZON_RE.sub("", text)
    clock = _CLOCK_RE.search(stripped)
    if clock:
        return float(clock.group(1)) * 60 + float(clock.group(2))
    minutes = re.search(r"(\d+(?:\.\d+)?)\s*(?:minutes?|mins?)\b", stripped,
                        re.IGNORECASE)
    if minutes:
        return float(minutes.group(1)) * 60.0
    match = _TIME_RE.search(stripped)
    return float(match.group(1)) if match else None
