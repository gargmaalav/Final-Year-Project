"""
route(user_query) -> Intent(kind, ...)
=====================================

Decides what *kind* of question was asked, before anything tries to answer it.

Until now there was exactly one: classify a single window. Every message was
forced through that path, so a perfectly reasonable question about the dataset
as a whole ("which subject fatigues fastest?", "when did subject 13 start
fatiguing?", "what does median frequency actually mean?") either got answered
as if it named a window, or hit the "I couldn't tell which subject and time"
message. In a live demo those are the first things anyone tries.

Routing is deterministic regex, not an LLM call, for three reasons: it is
instant (no round-trip before the real work starts, which matters when the
model is running on a laptop), it is predictable enough to write down and
test, and a 3B model is not reliable at multi-way classification. The
fallback is READING, so anything unrecognised behaves exactly as it does
today.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import extract

READING = "reading"          # classify one window -- the original behaviour
MENU = "menu"                # a subject named, but nothing specific asked
FOLLOWUP = "followup"        # "why?", "explain that" -- about the last answer
COMPARE = "compare"          # two subjects, or the two arms of one subject
ONSET = "onset"              # when does fatigue set in
OVERVIEW = "overview"        # how does one recording develop start to finish
EXPLAIN = "explain"          # what does a term mean
CATALOGUE = "catalogue"      # what data is available


@dataclass
class Intent:
    kind: str = READING
    subjects: list[int] = field(default_factory=list)
    term: str | None = None          # EXPLAIN: which term was asked about
    both_sides: bool = False         # COMPARE: left vs right of one subject


_CATALOGUE_RE = re.compile(
    r"\b(what (?:data|subjects|recordings|files)|which subjects|how many "
    r"subjects|what(?:'s| is) (?:in|available)|list the subjects|"
    r"what can (?:you|i) (?:ask|do)|what do you have|help)\b", re.IGNORECASE)

_EXPLAIN_RE = re.compile(
    r"\b(what (?:is|are|does|do)|what'?s|explain|meaning of|means?\b|"
    r"how does .{0,30}work|tell me about|define)\b", re.IGNORECASE)

# Terms worth a real definition. Order matters: the longest, most specific
# phrasings first, so "median frequency" is not matched as "frequency".
_TERMS = [
    ("median frequency", r"\b(median frequency|mdf|median freq)\b"),
    ("fatigue", r"\bfatigue[d]?\b"),
    ("confidence", r"\bconfidence\b"),
    ("EMG", r"\b(emg|electromyograph\w*|the signal)\b"),
    ("the forecast", r"\b(forecast|projection|prediction)\b"),
    ("the baseline", r"\b(baseline|calibration|calibrated)\b"),
]

_ONSET_RE = re.compile(
    r"\b(when (?:did|does|do|will)|at what (?:point|time)|how (?:long|far) "
    r"(?:in|into|before)|what (?:point|time) (?:did|does)|"
    r"start(?:ed|s)? (?:to )?(?:get |becom\w+ |being )?(?:fatigu|tir)\w*|"
    r"first (?:sign|become|get)|onset)\b", re.IGNORECASE)

_COMPARE_RE = re.compile(
    r"\b(compare|comparison|versus|vs\.?|difference between|differ|"
    r"which (?:one|subject|participant|person|of them|arm|side)\b|"
    r"who (?:is|was|has|gets?) (?:more|less|the most)|more fatigued than|"
    r"most fatigued|least fatigued|"
    r"fatigued? (?:the )?(?:most|least|fastest|slowest)|"
    r"fatigues? (?:the )?(?:fastest|slowest|quickest|soonest)|"
    r"(?:dropped|fell|declined) (?:the )?(?:most|least)|"
    r"which .{0,20}(?:better|worse))\b", re.IGNORECASE)

_BOTH_SIDES_RE = re.compile(
    r"\b(both (?:arms|sides)|left and right|right and left|"
    r"left or right|right or left|which (?:arm|side)\b|"
    r"(?:left|right) (?:arm |side )?(?:vs\.?|versus|and|or|against) "
    r"(?:the )?(?:left|right)|arms?\b.{0,20}\bcompare|each arm)\b",
    re.IGNORECASE)

_OVERVIEW_RE = re.compile(
    r"\b(overall|in general|summar\w+|overview|the whole (?:recording|"
    r"session|thing)|whole recording|entire recording|start to finish|"
    r"throughout|over time|how did .{0,25}(?:change|develop|progress|go)|"
    r"walk me through|describe the (?:recording|session))\b", re.IGNORECASE)


def _term_for(text: str) -> str | None:
    for name, pattern in _TERMS:
        if re.search(pattern, text, re.IGNORECASE):
            return name
    return None


# An open-ended request about a subject: "tell me about subject 13", "subject
# 13", "subject 13 info". These name a person but no question, and answering
# them with a single window's numbers guesses at what was wanted.
_VAGUE_RE = re.compile(
    r"^\s*(?:tell me about|about|info(?:rmation)? (?:on|about)|show me|"
    r"look at|check|analyse|analyze|how(?:'s| is| about)|what about|"
    r"data (?:on|for)|anything (?:on|about))?\s*"
    r"(?:subject|subj|participant|person|athlete|s)?\s*"
    r"[#-]?\s*(?:\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen)\s*[?.!]*\s*$", re.IGNORECASE)


def is_vague_subject_request(text: str, subjects: list[int]) -> bool:
    """True when a subject is named but nothing specific is asked about it."""
    if len(subjects) != 1:
        return False
    if extract.t_start_from_text(text, None) is not None:
        return False
    return bool(_VAGUE_RE.match(text or ""))


# Questions about the previous answer rather than about new data. Each LLM
# call is built from scratch with no conversation history, so "why?" used to be
# routed as a fresh reading -- it re-classified the same window and restated
# the same answer, never addressing what was asked. These are answered from the
# previous turn's already-measured numbers instead.
_FOLLOWUP_RE = re.compile(
    r"^\s*(why\??|why is that\??|why though\??|how come\??|"
    r"(?:can you |could you |please )?(?:explain|elaborate|expand)"
    r"(?: (?:that|this|it|more|further|again))?\??|"
    r"(?:tell me |say )?more\??|say more|go on|"
    r"what does that mean\??|what do you mean\??|"
    r"in (?:simpler|plain|simple|other) (?:terms|words|english)\??|"
    r"simpler\??|simplify(?: that| this| it)?\??|"
    r"break (?:that|it) down\??|"
    r"i don'?t (?:get|understand)(?: that| it)?\??)\s*$", re.IGNORECASE)


def is_followup(text: str) -> bool:
    """True for a question about the previous answer, not about new data."""
    return bool(_FOLLOWUP_RE.match(text or ""))


def stays_on_upload(user_query: str) -> bool:
    """True when a question with no file attached is still about the upload.

    Streamlit only hands a file over on the turn it is attached, so after
    "[uploaded run1.csv]" the obvious next question -- "when did I start
    fatiguing?" -- arrived with no file and fell through to the dataset path,
    which asked which subject was meant. The conversation should stay on the
    recording it was last about.

    Two things move it back to the dataset: naming a subject, or asking
    something the dataset owns. "why?" goes back too, because a follow-up is
    re-explained from the previous answer's own facts and needs no recording.

    The one exception is "compare me to subject 5": it names a subject but is
    still about the upload, since the upload is the other half of the
    comparison. Naming two subjects, or asking about one subject's two arms,
    is a dataset question again.
    """
    intent = route(user_query or "")
    if intent.kind in (CATALOGUE, EXPLAIN, FOLLOWUP):
        return False
    if not intent.subjects:
        return True
    return (intent.kind == COMPARE and not intent.both_sides
            and len(intent.subjects) == 1)


def route(user_query: str) -> Intent:
    """Classify one message. Precedence is deliberate and top-to-bottom."""
    text = user_query or ""
    subjects = extract.subjects_in_text(text)

    # First: these name no data at all, and any other branch would misread them
    if is_followup(text):
        return Intent(kind=FOLLOWUP)

    # "what data do you have" -- never about a specific window
    if _CATALOGUE_RE.search(text) and not subjects:
        return Intent(kind=CATALOGUE)

    # A definitional question, but only when it isn't really asking for a
    # reading: "what is subject 13's fatigue at 60s" names a window and wants
    # the number, not a glossary entry.
    if _EXPLAIN_RE.search(text):
        term = _term_for(text)
        has_window = bool(subjects) or extract.t_start_from_text(text, None) is not None
        if term and not has_window:
            return Intent(kind=EXPLAIN, term=term)

    if _ONSET_RE.search(text):
        return Intent(kind=ONSET, subjects=subjects)

    if _BOTH_SIDES_RE.search(text):
        return Intent(kind=COMPARE, subjects=subjects, both_sides=True)

    if _COMPARE_RE.search(text) or len(subjects) > 1:
        return Intent(kind=COMPARE, subjects=subjects)

    if _OVERVIEW_RE.search(text):
        return Intent(kind=OVERVIEW, subjects=subjects)

    # Checked last, so anything that names a real question wins over the menu.
    # The app suppresses this when there is a previous turn to carry a time
    # from -- mid-conversation, "what about subject 5?" is a follow-up asking
    # for the same reading, not a request to start over.
    if is_vague_subject_request(text, subjects):
        return Intent(kind=MENU, subjects=subjects)

    return Intent(kind=READING, subjects=subjects)
