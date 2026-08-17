"""
resolve_query(user_query, ...) -> Resolved(params | ask | problems)
==================================================================

Turns a free-text question into the (subject, t_start, side) triple the
pipeline needs, or into a specific thing to say back when it can't.

WHY THE LLM IS NO LONGER ASKED FIRST
------------------------------------
The original version sent every question to llama3.2:3b and range-checked the
JSON it returned. That had two problems, in opposite directions.

Too loose: on a follow-up with no subject in it, a 3B model does not answer
{"missing": true} -- it invents a plausible in-range triple.

    "what about the left arm?"  -> {"subject": 13, "t_start": 120.0}

Nothing in that message mentions subject 13 or 120 seconds, but both pass
every range check, so the app classified a fabricated window and reported it
at full confidence.

Too strict: the fix for that was to discard any number the user had not
literally typed. It worked, but it made the assistant brittle in a way users
feel immediately -- "1 minute 30" was not recognised as 90, so a *correct*
extraction got thrown away, and anything the LLM got wrong silently fell back
to the previous turn's window instead of saying so.

So the order is inverted. Cue-anchored regex reads the slots directly
("subject 13", "at 90 seconds", "two minutes in", "near the end"), which is
both more reliable than a 3B model on this task and instant -- no round-trip
on the common path, which matters in a live demo. The LLM is a fallback for
phrasings the patterns miss, and its output is still grounded against what the
user actually typed. Anything left unresolved becomes a targeted question
naming only the missing piece, never a silent guess.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

from llm import LLMError, chat

# --------------------------------------------------------------------------
# numbers
# --------------------------------------------------------------------------
_UNITS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
          "seven": 7, "eight": 8, "nine": 9}
_TEENS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
          "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
          "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}
_ZERO = {"zero": 0, "nought": 0}

_CLOCK_RE = re.compile(r"\b(\d{1,3}):([0-5]\d)\b")
# The trailing unit is optional: people write "1 minute 30" far more often
# than "1 minute 30 seconds", and without this the minute half matched alone
# and the reading came out as 60 s.
_MIN_SEC_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(?:minutes?|mins?|m)\b[\s,]*"
    r"(?:and\s+)?(\d{1,2}(?:\.\d+)?)\s*(?:seconds?|secs?|s)?(?!\w)",
    re.IGNORECASE)
_MINUTES_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:minutes?|mins?)\b", re.IGNORECASE)


def _word_numbers(text: str) -> set[float]:
    """Number words, including two-token compounds ("twenty-five" -> 25).

    Written out because a user typing "subject thirteen" or "at forty five
    seconds" is asking something perfectly ordinary, and rejecting it as
    ungrounded is exactly the brittleness this module exists to remove.
    """
    tokens = re.findall(r"[a-z]+", text.lower())
    found: set[float] = set()
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if tok in _TENS and nxt in _UNITS:
            found.add(float(_TENS[tok] + _UNITS[nxt]))
            found.add(float(_TENS[tok]))
            i += 2
            continue
        for table in (_TENS, _TEENS, _UNITS, _ZERO):
            if tok in table:
                found.add(float(table[tok]))
                break
        i += 1
    return found


def _mentioned_numbers(text: str) -> set[float]:
    """Every value the user could plausibly have meant, from what they typed.

    Covers digits, number words, "1:30" and "1 minute 30" as 90, and minute
    readings converted to seconds, so "two minutes in" grounds a t_start of
    120. Deliberately generous: this set only ever *permits* a value the LLM
    proposed, so a spurious member cannot invent a window on its own.
    """
    found: set[float] = set()
    for m in _CLOCK_RE.finditer(text):
        found.add(float(m.group(1)) * 60 + float(m.group(2)))
    for m in _MIN_SEC_RE.finditer(text):
        found.add(float(m.group(1)) * 60 + float(m.group(2)))
    for tok in re.findall(r"\d+(?:\.\d+)?", text):
        found.add(float(tok))
    found |= _word_numbers(text)

    lowered = text.lower()
    for m in _MINUTES_RE.finditer(lowered):
        found.add(float(m.group(1)) * 60.0)
    for word, value in {**_TENS, **_TEENS, **_UNITS}.items():
        if re.search(rf"\b{word}\s+(?:minutes?|mins?)\b", lowered):
            found.add(float(value) * 60.0)
    if re.search(r"\b(?:a|one)\s+minute\s+and\s+a\s+half\b", lowered):
        found.add(90.0)
    if re.search(r"\bhalf\s+a\s+minute\b", lowered):
        found.add(30.0)
    if re.search(r"\b(?:a|one)\s+minute\b", lowered):
        found.add(60.0)
    return found


# --------------------------------------------------------------------------
# cue-anchored slot extraction
# --------------------------------------------------------------------------
_NUM_WORD = "|".join(list(_TENS) + list(_TEENS) + list(_UNITS))
# Decimals must be part of the number itself. Without `(?:\.\d+)?` the pattern
# matched "60" in "60.5 seconds", failed on the "." that followed, then
# re-matched further along and read the reading as 5 seconds -- a silently
# wrong window, which is the exact failure mode this module exists to stop.
_NUM = rf"(\d+(?:\.\d+)?|(?:{_NUM_WORD})(?:[\s-]+(?:{'|'.join(_UNITS)}))?)"

# A subject number must sit next to a word that means "subject". Without this
# cue, "at 60 seconds for subject 13" can bind 60 to the subject slot: the
# number was genuinely typed, so grounding alone accepts it.
_SUBJECT_RE = re.compile(
    rf"\b(?:subject|subj|sub|participant|person|athlete|patient|volunteer)\s*"
    rf"(?:number|no\.?|#)?\s*{_NUM}\b", re.IGNORECASE)
_SUBJECT_SHORT_RE = re.compile(r"\b(?:s|p)\s*[#-]?\s*(\d{1,2})\b")

# A misspelt subject word directly in front of a number. One typo used to lose
# the subject completely -- "is subjet 5 fatigued at 60 seconds" named a
# subject and a time and was answered with "which subject did you mean?",
# which reads as though the question had not been looked at.
_SUBJECT_WORDS = ("subject", "subj", "participant", "person", "athlete",
                  "patient", "volunteer")
_WORD_NUMBER_RE = re.compile(r"\b([a-z]{4,14})\s*[#-]?\s*(\d{1,2})\b",
                             re.IGNORECASE)


def _is_subject_typo(word: str) -> bool:
    """True for a near-miss of a subject word, false for an exact one.

    0.8 is tight enough to leave the words that legitimately sit in front of a
    number alone -- "seconds", "minutes", "compare", "between" all score below
    0.6 against "subject" -- while catching a transposition or a dropped
    letter ("subjcet" 0.86, "subjet" 0.92, "particpant" 0.95).
    """
    word = word.lower()
    if word in _SUBJECT_WORDS:
        return False        # matched exactly elsewhere; nothing to add here
    return bool(difflib.get_close_matches(word, _SUBJECT_WORDS, n=1, cutoff=0.8))

_TIME_RE = re.compile(
    rf"\b(?:at|after|around|about|near|by|from|@)?\s*{_NUM}\s*"
    rf"(seconds?|secs?|s|minutes?|mins?|m)\b", re.IGNORECASE)

_SIDE_RE_L = re.compile(r"\b(left|lefthand|left-hand|l)\b", re.IGNORECASE)
_SIDE_RE_R = re.compile(r"\b(right|righthand|right-hand|r)\b", re.IGNORECASE)

# Relative positions in a recording. These need the recording's duration, which
# is why resolve_query() takes one -- without it they stay unresolved and are
# asked about rather than guessed.
_ANCHOR_START = re.compile(
    r"\b(at the start|at the beginning|from the start|initially|"
    r"the very start|the beginning|right at the start)\b", re.IGNORECASE)
_ANCHOR_END = re.compile(
    r"\b((?:at|by|near|towards?|close to|around)?\s*the\s+(?:very\s+)?end|"
    r"right now|by now|at the finish|finish(?:ing)? up|last window|"
    r"latest reading|most recent)\b", re.IGNORECASE)
_ANCHOR_MID = re.compile(
    r"\b(halfway|half way|midway|the middle|in the middle)\b", re.IGNORECASE)
_ANCHOR_LAST = re.compile(
    rf"\b(?:last|final)\s+{_NUM}\s*(seconds?|secs?|s|minutes?|mins?|m)\b",
    re.IGNORECASE)


def _number_token(token: str) -> float | None:
    """Parse '13', 'thirteen' or 'twenty-five' to a value."""
    token = token.strip().lower()
    if re.fullmatch(r"\d+(?:\.\d+)?", token):
        return float(token)
    parts = re.split(r"[\s-]+", token)
    if len(parts) == 2 and parts[0] in _TENS and parts[1] in _UNITS:
        return float(_TENS[parts[0]] + _UNITS[parts[1]])
    for table in (_TENS, _TEENS, _UNITS, _ZERO):
        if token in table:
            return float(table[token])
    return None


def _to_seconds(value: float, unit: str) -> float:
    unit = unit.lower()
    return value * 60.0 if unit.startswith("min") or unit == "m" else value


def subjects_in_text(text: str) -> list[int]:
    """Every subject the user named, in order -- "compare subject 5 and 9"
    needs both, so this is separate from the single-subject slot."""
    found: list[int] = []
    for m in _SUBJECT_RE.finditer(text):
        value = _number_token(m.group(1))
        if value is not None and value == int(value):
            found.append(int(value))
    for m in _SUBJECT_SHORT_RE.finditer(text):
        found.append(int(m.group(1)))
    for m in _WORD_NUMBER_RE.finditer(text):
        if _is_subject_typo(m.group(1)):
            found.append(int(m.group(2)))

    # "subject 5 and 9" / "subject 5 vs 9" -- the second number has no cue of
    # its own but is plainly another subject.
    for m in re.finditer(rf"\b(?:and|vs\.?|versus|against|or|to)\s+{_NUM}\b",
                         text, re.IGNORECASE):
        # only when a subject was named and no time unit follows the number
        tail = text[m.end():m.end() + 12].lower()
        if found and not re.match(r"\s*(seconds?|secs?|s\b|minutes?|mins?|m\b)", tail):
            value = _number_token(m.group(1))
            if value is not None and value == int(value):
                found.append(int(value))

    seen, ordered = set(), []
    for s in found:
        if s not in seen:
            seen.add(s)
            ordered.append(s)
    return ordered


def _subject_from_text(text: str) -> int | None:
    found = subjects_in_text(text)
    return found[0] if found else None


def _t_start_from_text(text: str, duration: float | None) -> float | None:
    """A time offset in seconds, from an explicit mention or a relative anchor."""
    stripped = _HORIZON_RE.sub("", text)     # don't read "next 30 s" as a start

    # Anchors that need a duration are skipped when there isn't one, rather
    # than returning None outright -- otherwise "at the end, say 30 seconds in"
    # resolved to nothing instead of falling through to the explicit time.
    m = _ANCHOR_LAST.search(stripped)
    if m and duration is not None:
        value = _number_token(m.group(1))
        if value is not None:
            return max(0.0, duration - _to_seconds(value, m.group(2)))
    if duration is not None and _ANCHOR_END.search(stripped):
        return duration
    if _ANCHOR_START.search(stripped):
        return 0.0
    if duration is not None and _ANCHOR_MID.search(stripped):
        return duration / 2.0

    clock = _CLOCK_RE.search(stripped)
    if clock:
        return float(clock.group(1)) * 60 + float(clock.group(2))
    min_sec = _MIN_SEC_RE.search(stripped)
    if min_sec:
        return float(min_sec.group(1)) * 60 + float(min_sec.group(2))

    # skip a number that is already bound to the subject slot
    consumed = {(m.start(1), m.end(1)) for m in _SUBJECT_RE.finditer(stripped)}
    for m in _TIME_RE.finditer(stripped):
        if (m.start(1), m.end(1)) in consumed:
            continue
        value = _number_token(m.group(1))
        if value is not None:
            # a leading minus is kept so the caller can reject it, rather than
            # silently reading "-5 seconds" as 5 seconds
            if stripped[:m.start(1)].rstrip().endswith("-"):
                value = -value
            return _to_seconds(value, m.group(2))
    if re.search(r"\b(?:a|one)\s+minute\s+and\s+a\s+half\b", stripped, re.I):
        return 90.0
    if re.search(r"\bhalf\s+a\s+minute\b", stripped, re.I):
        return 30.0
    if re.search(r"\b(?:a|one)\s+minute\s+in\b", stripped, re.I):
        return 60.0
    return None


def _side_from_text(text: str) -> str | None:
    """Side is a two-way choice stated in plain words -- no LLM needed."""
    if _SIDE_RE_L.search(text):
        return "L"
    if _SIDE_RE_R.search(text):
        return "R"
    return None


# --------------------------------------------------------------------------
# LLM fallback
# --------------------------------------------------------------------------
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


def _llm_slots(user_query: str) -> dict:
    """Ask the LLM for the slots the patterns did not find. Never trusted on
    its own -- every value it returns is grounded against the user's text by
    the caller."""
    import json
    try:
        reply = chat([{"role": "system", "content": _SYSTEM_PROMPT},
                      {"role": "user", "content": user_query}])
    except LLMError:
        return {}
    match = _JSON_RE.search(reply or "")
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return {} if parsed.get("missing") else parsed


# --------------------------------------------------------------------------
# resolution
# --------------------------------------------------------------------------
@dataclass
class Resolved:
    """The outcome of reading one question.

    Exactly one of `params` and `ask` is set. `problems` carries corrections
    worth stating either way -- an out-of-range subject used to be silently
    replaced by the previous turn's, which is indistinguishable from a
    correct answer.
    """
    params: dict | None = None
    ask: str | None = None
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.params is not None


def _ask_for(missing: list[str], subjects: list[int] | None,
             range_stated: bool = False) -> str:
    """A question naming only what is actually missing.

    `range_stated` suppresses the second mention of the subject range. Asking
    about subject 99 answered "I don't have subject 99 -- the dataset has
    subjects 1-13. Which subject did you mean? I have subjects 1-13.", which
    states the same range twice in two sentences.
    """
    known = _subject_range_phrase(subjects)
    if missing == ["subject"]:
        return ("Which subject did you mean?" if range_stated
                else f"Which subject did you mean? I have {known}.")
    if missing == ["t_start"]:
        return ("Which point in the recording -- a time in seconds, or "
                "something like \"at the start\" or \"near the end\"?")
    who = "Which subject" if range_stated else f"Which subject ({known})"
    return (f"{who} and which point in the recording "
            "(a time in seconds, or \"at the start\" / \"near the end\")?")


def _subject_range_phrase(subjects: list[int] | None) -> str:
    if not subjects:
        return "subjects 1-13"
    if subjects == list(range(subjects[0], subjects[-1] + 1)):
        return f"subjects {subjects[0]}-{subjects[-1]}"
    return "subjects " + ", ".join(str(s) for s in subjects)


def resolve_query(user_query: str, previous: dict | None = None,
                  duration: float | None = None,
                  subjects: list[int] | None = None,
                  use_llm: bool = True) -> Resolved:
    """Resolve a question to {subject, t_start, side}, or say what's missing.

    Args:
        user_query: the raw message.
        previous: last turn's resolved params, so "and at 90 seconds?" keeps
            the subject already under discussion.
        duration: length of the recording being discussed, needed to resolve
            "near the end" and to reject a time past the end of the data.
        subjects: subject ids that actually exist, used in the messages.
        use_llm: whether to fall back to the LLM for phrasings the patterns
            miss. Off makes resolution fully deterministic and instant, which
            is what the regression set uses so its results don't depend on
            which model happens to be loaded.
    """
    prev = previous or {}
    problems: list[str] = []

    subject = _subject_from_text(user_query)
    t_start = _t_start_from_text(user_query, duration)
    side = _side_from_text(user_query)

    # the LLM only sees questions the patterns could not read at all
    if use_llm and subject is None and t_start is None:
        mentioned = _mentioned_numbers(user_query)
        data = _llm_slots(user_query)
        try:
            value = int(data["subject"])
            if float(value) in mentioned:
                subject = value
        except (KeyError, TypeError, ValueError):
            pass
        try:
            value = float(data["t_start"])
            if value in mentioned:
                t_start = value
        except (KeyError, TypeError, ValueError):
            pass

    # range checks produce a message; they never fall through to the old
    # window, which looked identical to a correct answer
    range_stated = False
    if subject is not None and subjects and subject not in subjects:
        problems.append(
            f"I don't have subject {subject} -- the dataset has "
            f"{_subject_range_phrase(subjects)}.")
        subject = None
        range_stated = True
    if t_start is not None and t_start < 0:
        problems.append("A time can't be negative, so I ignored that.")
        t_start = None
    if t_start is not None and duration is not None and t_start > duration:
        problems.append(
            f"That recording is only {duration:.0f}s long, so I read the "
            f"last window instead of {t_start:.0f}s.")
        t_start = duration

    carried = []
    if subject is None and prev.get("subject") is not None and not problems:
        subject = prev["subject"]
        carried.append("subject")
    if t_start is None and prev.get("t_start") is not None and not problems:
        t_start = prev["t_start"]
        carried.append("time")
    if side is None:
        side = prev.get("side") or "R"

    missing = [name for name, value in (("subject", subject), ("t_start", t_start))
               if value is None]
    if missing:
        return Resolved(ask=_ask_for(missing, subjects, range_stated),
                        problems=problems)

    return Resolved(params={"subject": int(subject), "t_start": float(t_start),
                            "side": side, "carried_over": carried},
                    problems=problems)


# Public names for the two readers the app and the intent router need on their
# own, so neither has to reach into a private function.
def side_from_text(text: str) -> str | None:
    """"left"/"right" if the message names one, else None."""
    return _side_from_text(text)


def t_start_from_text(text: str, duration: float | None = None) -> float | None:
    """A time offset in seconds, from an explicit mention or a relative anchor
    ("near the end"). Anchors need `duration` and stay unresolved without it."""
    return _t_start_from_text(text, duration)


def parse_query(user_query: str, previous: dict | None = None) -> dict | None:
    """Back-compatible wrapper: the resolved triple, or None.

    Kept because the integration contract names this signature. New callers
    should prefer resolve_query(), which can say *why* it could not resolve.
    """
    return resolve_query(user_query, previous).params


# ---------------------------------------------------------------------------
# Forecast horizon. Shared by both paths -- the dataset and upload paths used
# to read times with different code, so the same phrasing behaved differently
# depending on whether a file was attached.
# ---------------------------------------------------------------------------
_HORIZON_RE = re.compile(
    r"(?:next|in|over the next|after|within|ahead|forward)\s+(?:the\s+)?"
    r"(\d+(?:\.\d+)?)\s*(seconds?|secs?|s\b|minutes?|mins?|m\b)",
    re.IGNORECASE)

# A horizon named as a bare unit: "over the next minute", "in the coming
# minute". Only the singular is matched -- "the next few minutes" names no
# definite span, so it is left to the default rather than guessed at.
_BARE_HORIZON_RE = re.compile(
    r"\b(?:next|following|coming)\s+(minute|min|second|sec)\b", re.IGNORECASE)

# Asking about the future without naming a horizon ("will I get more tired?").
_FUTURE_INTENT_RE = re.compile(
    r"\b(will|going to|gonna|predict|forecast|project(?:ion|ed)?|"
    r"keep going|carry on|continue|later on|from here|trend|"
    r"if i keep|how much longer|hold out|last)\b", re.IGNORECASE)
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
        return _to_seconds(float(match.group(1)), match.group(2))
    # "over the next minute" states a horizon with no digit in front of it.
    # _HORIZON_RE needs one, so this fell through to the 20 s default and a
    # question that plainly named a minute was answered with a third of it.
    bare = _BARE_HORIZON_RE.search(text)
    if bare:
        return _to_seconds(1.0, bare.group(1))
    if _FUTURE_INTENT_RE.search(text):
        return DEFAULT_HORIZON_SEC
    return default


def extract_t_start_seconds(text: str, duration: float | None = None) -> float | None:
    """Time offset for the upload path.

    Now the same reader the dataset path uses, so "near the end" and "1 minute
    30" behave identically whether or not a file is attached.
    """
    return _t_start_from_text(text, duration)
