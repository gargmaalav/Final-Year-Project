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

What it hands the model is the *interpreted* reading (interpret.py), not the
raw figures. Handing over "median frequency 51.2 Hz" invited exactly the
mistake the comparison answers made: 51.2 Hz means nothing on its own -- fresh
subjects in this dataset sit between 59 and 81 Hz -- so a model reasoning
about "endurance" from an absolute value is reasoning from a number that has
no scale. Percentage below that person's own fresh level does have a scale,
and that is what a training suggestion can defensibly rest on.
"""
from __future__ import annotations

import re

_KEYWORDS = re.compile(
    r"\b(sport|recommend|suggest|suggestion|plan|diet|nutrition|gym|"
    r"training|workout|exercise|good at|suitable|suited)\b", re.IGNORECASE)

# The same words in their machine-learning sense. "What training data was
# used?" and "how was the model trained?" are questions about the project, and
# answering them with a diet plan is the kind of thing that makes the whole
# tool look unserious in a demo.
_DATA_SENSE = re.compile(
    r"\b(training|test|validation|holdout)\s+(data|set|split|sample\w*|"
    r"subject\w*|window\w*)\b|"
    r"\bhow (?:was|is|did you) .{0,20}train\w*\b|\btrained on\b|"
    r"\btrain(?:ed|ing)? the (?:model|classifier|network|lstm)\b",
    re.IGNORECASE)


def wants_recommendation(text: str) -> bool:
    text = text or ""
    if _DATA_SENSE.search(text):
        return False
    return bool(_KEYWORDS.search(text))


DISCLAIMER = ("_This is a general, educational suggestion from a single "
              "biceps EMG reading — not medical, professional coaching, or "
              "nutrition advice._")


def ensure_disclaimer(text: str) -> str:
    """Append the disclaimer when the model left it out.

    The prompt asks for it and usually gets it, but "usually" is not a
    standard to hold this to: llama3.2:3b dropped it outright on one of three
    sampled answers. It is a fixed string, so there is no reason for it to
    depend on the model complying.
    """
    if not text:
        return text
    if "not medical" in text.lower():
        return text
    return f"{text.rstrip()}\n\n{DISCLAIMER}"


_MEASUREMENT = re.compile(r"\d+(?:\.\d+)?\s*(?:hz\b|%|percent\b)", re.IGNORECASE)


def strip_measurements(text: str) -> str:
    """Drop sentences quoting a hertz value or a percentage.

    The suggestion is general knowledge about training, so ordinary numbers
    are welcome in it -- sets, reps, rest days. Measurements are not: the
    model is handed the reading with its figures already removed, so any Hz
    or percentage it produces here is invented, and a training suggestion
    does not follow from those numbers anyway.
    """
    if not text:
        return text
    kept = []
    for para in re.split(r"\n\s*\n", text.strip()):
        good = [s for s in re.split(r"(?<=[.!?])\s+", para.strip())
                if s and not _MEASUREMENT.search(s)]
        if good:
            kept.append(" ".join(good))
    return "\n\n".join(kept).strip()


def _measured_block(features: dict, reading: dict | None,
                    forecast: dict | None) -> str:
    """The measured facts, stated the way a non-specialist can use them.

    The numberless prose notes are preferred over the full lines for the same
    reason build_prompt() prefers them: every figure handed to this model came
    back misattributed somewhere, and the verdict is rendered above the
    suggestion rather than phrased by it, so it needs the shape of the reading
    and not its values.
    """
    lines = []

    if reading and (reading.get("prose") or reading.get("lines")):
        lines += [f"- {line}"
                  for line in (reading.get("prose") or reading["lines"])]
    else:
        # No stored baseline for this recording, so there is no fresh level to
        # measure against and the hertz figure carries no scale. Say so rather
        # than presenting it as if it were interpretable.
        lines.append(f"- fatigue state: {features['fatigue_state']}")
        lines.append(
            f"- median frequency {features['mdf_hz']:.1f} Hz, with no fresh "
            "baseline available for this recording, so this figure cannot be "
            "read as high or low and must not be described as either")
        lines.append(
            f"- the model's own certainty in that call is "
            f"{features['confidence'] * 100:.0f}%")

    if forecast and forecast.get("ok") and forecast.get("summary"):
        lines.append(f"- fatigue trend: {forecast['summary']}")

    return "\n".join(lines)


def build_recommendation_prompt(features: dict, forecast: dict | None,
                                user_query: str,
                                athlete_note: str | None = None,
                                reading: dict | None = None) -> str:
    note_line = (f"They also say: \"{athlete_note}\"\n\n" if athlete_note else "")

    return (
        "You are a fitness-education assistant talking to someone with no "
        "background in signal processing. One biceps EMG reading from one arm "
        "has just been measured. These measured facts are already worked out "
        "-- restate them as given, do not recalculate or reinterpret them:\n\n"
        f"{_measured_block(features, reading, forecast)}\n\n"
        f"{note_line}"
        f"Their question: {user_query}\n\n"
        # Asked to open by restating the verdict "exactly as the facts above
        # state it", the model wrote "Subject 13 ... they are showing signs of
        # fatigue" directly under a rendered line saying they were NOT -- the
        # same inversion the main answer had, in the one place that had not
        # been given the same treatment. It is rendered above this text now,
        # and the model is not asked for it at all.
        "IMPORTANT: whether this muscle is showing signs of fatigue has "
        "ALREADY been written out and shown to the reader directly above your "
        "text. Do NOT restate it, do not open with it, and above all do not "
        "contradict it.\n\n"
        "Using your own general knowledge of sports science, fitness and "
        "nutrition, say what kinds of activity and what general training and "
        "diet direction would suit that profile.\n\n"
        "Rules:\n"
        "- This is your general knowledge, not measurement. Say so plainly "
        "in one short phrase; there is no data in this project linking EMG "
        "patterns to sport suitability.\n"
        "- Do not put any hertz figure or percentage anywhere in your answer. "
        "A training suggestion does not follow from those numbers, restating "
        "them implies it does, and you have not been given any -- so any you "
        "write will be wrong.\n"
        "- Do not compare this person's numbers to other people, to athletes, "
        "or to any typical or normal value. The only meaningful reference is "
        "their own fresh level, which is already accounted for above.\n"
        "- A percentage stays a percentage. Do not convert it to a fraction "
        "or a word like 'half' or 'three-quarters'.\n"
        "- Do not overturn the fatigue verdict given above.\n"
        "- A falling signal means the muscle's electrical activity has "
        "shifted towards lower frequencies. Do not describe it as less muscle "
        "activity, weaker contraction, lower energy, or reduced effort -- it "
        "is none of those.\n"
        "- Keep the whole answer under 130 words.\n\n"
        "End with exactly one line stating this is a general, educational "
        "suggestion from a single biceps EMG reading -- not medical, "
        "professional coaching, or nutrition advice."
    )
