"""
Turning a measurement into something a non-technical reader can use.
====================================================================

The pipeline's native output is "fatigue_label 1, mdf_hz 51.2, confidence
0.86". That is the right thing to compute and the wrong thing to show. A
reader who is not on this project cannot do anything with 51.2 Hz, because
median frequency has no absolute meaning -- fresh subjects in this dataset sit
anywhere from 59 to 81 Hz, so the same 51.2 Hz is a steep fall for one person
and roughly normal for another.

What is interpretable is distance from *that person's own* fresh state, which
the stored baseline already gives us: their fresh mean and their fresh spread.
So a reading becomes:

    fatigued, 60s into a 218s effort (28% of the way through)
    muscle signal 18% below their own fresh level
    3.0 standard deviations outside their normal fresh range

Every one of those is derived from numbers the pipeline already measured. This
module does no modelling and no rounding-away of uncertainty -- it restates
measured values against a measured reference, and the raw figures still travel
alongside for the technical read.
"""
from __future__ import annotations

import re

# How far outside the fresh range a reading sits, in that person's own
# standard deviations. Bands are deliberately coarse: the underlying
# classifier is ~88% accurate per window, so finer gradations would imply a
# precision the measurement does not support.
_BANDS = [
    (-3.0, "well outside their normal fresh range"),
    (-2.0, "clearly below their normal fresh range"),
    (-1.0, "a little below their normal fresh range"),
    (1.0, "within their normal fresh range"),
]
_ABOVE = "above their fresh level, which is not the fatigue direction"

# Above this, the reading is inside the person's normal fresh range -- the
# same boundary the last _BANDS entry uses. A fatigued verdict here is not
# wrong, but it is contested, and saying so is the difference between a reader
# distrusting the tool and understanding it.
CONFLICT_Z = -1.0


def _band(z: float | None) -> str | None:
    if z is None:
        return None
    for threshold, text in _BANDS:
        if z <= threshold:
            return text
    return _ABOVE


def describe_reading(features: dict, reference: dict | None,
                     t_start: float | None = None,
                     duration: float | None = None) -> dict:
    """Plain-language reading of one classified window.

    Returns the pieces rather than a finished sentence, so the prompt builder
    can order them and the LLM can phrase them, while the numbers stay exact.
    """
    fatigued = features.get("fatigue_label") in (1, 2)
    mdf = features.get("mdf_hz")

    fresh = (reference or {}).get("fresh_mdf")
    sd = (reference or {}).get("sd_mdf")

    drop_hz = drop_pct = z = None
    if fresh and mdf is not None:
        drop_hz = fresh - mdf
        drop_pct = (drop_hz / fresh) * 100.0 if fresh else None
        if sd:
            z = (mdf - fresh) / sd          # negative when fatigued

    position = None
    if t_start is not None and duration and duration > 0:
        position = {
            "t_start": t_start, "duration": duration,
            "percent": min(100.0, (t_start / duration) * 100.0),
        }

    # A fatigued verdict while the fatigue marker still sits inside the
    # person's normal fresh range. The threshold is -1.0, not 0, because -1.0
    # is where _BANDS stops saying "below their normal fresh range" and starts
    # saying "within" it: a reading at z = -0.2 produced "the muscle signal is
    # fatigued ... which is within their normal range", which reads as a
    # contradiction unless the disagreement is named.
    conflict = bool(fatigued and z is not None and z > CONFLICT_Z)

    return {
        "fatigued": fatigued,
        "conflict": conflict,
        "verdict": "showing signs of fatigue" if fatigued else "not showing signs of fatigue",
        "mdf_hz": mdf,
        "fresh_mdf": fresh,
        "drop_hz": drop_hz,
        "drop_percent": drop_pct,
        "z": z,
        "band": _band(z),
        "position": position,
        "confidence": features.get("confidence"),
    }


def plain_lines(reading: dict, who: str) -> list[str]:
    """The reading as plain statements, most useful first.

    `who` names the subject or the uploaded recording.
    """
    lines = []

    # The verdict gets its own line, and the negative case is shouted. Asked to
    # phrase "Subject 13 is not showing signs of fatigue 65 seconds into a 218
    # second effort", llama3.2:3b answered "subject 13's right arm is fatigued
    # at 65 seconds" -- it dropped the "not". A single unstressed negation
    # buried mid-sentence, surrounded by four lines about falling signals and
    # fatigue, is the easiest word in the answer to lose, and losing it inverts
    # the result.
    if reading["fatigued"]:
        lines.append(f"{who} IS showing signs of fatigue")
    else:
        lines.append(f"{who} is NOT fatigued at this point -- the correct "
                     "answer to give is that they are not showing signs of "
                     "fatigue")

    position = reading["position"]
    if position:
        lines.append(
            f"this reading is {position['t_start']:.0f} seconds into a "
            f"{position['duration']:.0f} second effort, which is "
            f"{position['percent']:.0f}% of the way through the recording")

    if reading["drop_percent"] is not None:
        fell = reading["drop_percent"] >= 0
        # The gloss follows the verdict, not just the direction. Telling the
        # reader "a falling signal is what muscle fatigue looks like" under a
        # NOT-fatigued verdict argues against the verdict it is attached to,
        # which is exactly the push that made the model invert it.
        if not fell:
            gloss = ("it has risen rather than fallen, which is not the "
                     "direction fatigue moves it")
        elif reading["fatigued"]:
            gloss = "a falling signal is what muscle fatigue looks like"
        else:
            gloss = ("a small drop like this is normal variation, not enough "
                     "to count as fatigue")
        lines.append(
            f"their muscle signal is {abs(reading['drop_percent']):.0f}% "
            f"{'below' if fell else 'above'} their own fresh level "
            f"(now {reading['mdf_hz']:.1f} Hz, against "
            f"{reading['fresh_mdf']:.1f} Hz when fresh) -- {gloss}")

    if reading["band"]:
        lines.append(f"that puts this reading {reading['band']}"
                     + (f" ({abs(reading['z']):.1f} standard deviations)"
                        if reading["z"] is not None else ""))

    # The label and the fatigue marker can disagree -- subject 7 late in their
    # effort is classified fatigued while their median frequency sits ABOVE
    # their fresh level. The classifier reads eight features, not just this
    # one, so it is not a bug; but presenting the verdict without the
    # disagreement would let a reader take a contested call as settled.
    if reading["conflict"]:
        moved = ("has not fallen at all" if (reading["z"] or 0) >= 0
                 else "has barely moved")
        lines.append(
            f"worth flagging: the classifier calls this fatigued, but the "
            f"median-frequency marker {moved} and is still inside their "
            "normal fresh range, so the two disagree here -- the model weighs "
            "eight signal features and this is only one of them, but treat "
            "this particular reading as less settled")

    if reading["confidence"] is not None:
        lines.append(
            f"the model's own certainty in the fatigued/not-fatigued call is "
            f"{reading['confidence'] * 100:.0f}% -- this is how sure the model "
            "is, not how accurate it is known to be")

    return lines


# How big a move from someone's own fresh level is, in words. The boundaries
# are the same coarse bands _BANDS uses, expressed as a percentage so they can
# be stated without a z-score.
_SIZE = [
    (1.0, "essentially unchanged from their own fresh level"),
    (5.0, "a little below their own fresh level"),
    (12.0, "clearly below their own fresh level"),
]
_FAR = "far below their own fresh level"


def _size_words(drop: float) -> str:
    if drop < 0:
        return ("slightly above their own fresh level, which is not the "
                "direction fatigue moves it")
    for threshold, text in _SIZE:
        if drop < threshold:
            return text
    return _FAR


def _when_words(position: dict | None) -> str | None:
    if not position:
        return None
    pct = position["percent"]
    if pct < 15:
        return "this reading is early in the effort"
    if pct < 50:
        return "this reading is in the first half of the effort"
    if pct < 85:
        return "this reading is well into the effort"
    return "this reading is near the end of the effort"


def prose_notes(reading: dict, who: str) -> list[str]:
    """The reading with every number removed.

    This is what the LLM gets. It is not a stylistic choice: given figures,
    llama3.2:3b has reported the current reading as the fresh baseline (twice),
    quoted "1.2 standard deviations" as proof the reading was *within* a range
    the facts described it as below, and pasted a z-score into the answer as a
    "Caveat". Each was fixed by a rule and each came back somewhere else.

    Every number the reader needs is already rendered by verdict_sentence()
    and carried on the technical line beneath the answer, both built in Python.
    So the model is given none, and the class of error disappears rather than
    being argued with.
    """
    notes = [f"{who} is "
             + ("SHOWING signs of fatigue" if reading["fatigued"]
                else "NOT showing signs of fatigue")]

    when = _when_words(reading["position"])
    if when:
        notes.append(when)

    if reading["drop_percent"] is not None:
        notes.append(f"their muscle signal is {_size_words(reading['drop_percent'])}")

    if reading["band"]:
        notes.append(f"that puts this reading {reading['band']}")

    if reading["conflict"]:
        notes.append(
            "worth flagging: the classifier calls this fatigued, but the "
            "median-frequency marker has barely moved and is still inside "
            "their normal fresh range, so the two disagree here -- the model "
            "weighs eight signal features and this is only one of them, but "
            "treat this particular reading as less settled")

    return notes


def verdict_sentence(reading: dict, who: str) -> str:
    """The finding, written here rather than by the model.

    Every prompt rule aimed at protecting this sentence has only half-held. A
    3B model asked to phrase "Subject 13 is not showing signs of fatigue"
    answered "subject 13's right arm is fatigued"; told explicitly never to
    drop the negation, it produced "showing signs of fatigue, but only
    slightly ... not yet fatigued" -- opening with the wrong verdict and
    closing with the right one. It also twice reported the current reading as
    the fresh baseline.

    So the verdict and the two numbers it rests on are rendered directly, and
    the model is left to add context around a sentence it cannot alter. This
    is the same move that fixed the ranking table and the comparisons.
    """
    # Three numbers for one idea ("60s into a 491s effort (12% of the way
    # through)") is more arithmetic than the sentence is worth. How far in
    # they are is the part that carries meaning; the total length is on the
    # chart and in the provenance line for anyone who wants it.
    where = ""
    position = reading["position"]
    if position:
        where = (f", {position['t_start']:.0f}s in — about "
                 f"{position['percent']:.0f}% of the way through")

    if reading["fatigued"]:
        head = f"**{who} is showing signs of fatigue**{where}."
    else:
        head = f"**{who} is not showing signs of fatigue**{where}."

    drop = reading["drop_percent"]
    if drop is None:
        return head

    # The raw hertz pair used to be quoted here too. It is the most technical
    # thing in the answer, it appears again in the provenance line directly
    # underneath, and the percentage already says how big the change is in a
    # form a non-specialist can act on -- so the headline keeps the percentage
    # and drops the hertz. interpret.plain_lines() still carries the labelled
    # pair for the prompt (and is tested for not swapping them).
    if drop < 0:
        detail = (f" Their muscle signal is {abs(drop):.0f}% **above** their "
                  "own fresh level, which is not the direction fatigue "
                  "moves it.")
    elif drop < 1.0:
        # "0% below" reads as a measurement failure rather than as a result
        detail = (" Their muscle signal is essentially unchanged from their "
                  "own fresh level.")
    else:
        detail = (f" Their muscle signal is {drop:.0f}% below their own "
                  "fresh level.")
    return head + detail


_ECHO = re.compile(
    # "is fatigued" alone missed "This indicates they ARE fatigued", which was
    # a paragraph of its own restating the rendered verdict and nothing else.
    r"(sign|signs) of fatigue|(is|are|was|were|isn'?t|aren'?t) (not )?fatigued|"
    r"no signs? of|"
    # "within their normal fresh range" is deliberately NOT here: a genuinely
    # useful answer opened "This reading falls within their normal fresh
    # range, indicating that ...", and the test for that caught this when it
    # was tried.
    r"showing (no|signs)|no indication of fatigue", re.IGNORECASE)


# A paragraph longer than this is assumed to carry detail beyond the verdict
# and is never dropped. Set from observation: the restatements ran 8-25 words,
# while the shortest paragraph that genuinely added something ran 33.
_ECHO_MAX_WORDS = 30


def strip_verdict_echo(prose: str, who: str | None = None) -> str:
    """Drop opening or closing paragraphs that only repeat the rendered verdict.

    The verdict is printed directly above the model's text and the prompt says
    not to restate it. It restated it anyway in every sampled reading -- and
    for a not-fatigued reading it restated it at BOTH ends, sandwiching two
    useful sentences between "Subject 13 is showing a muscle signal slightly
    below their fresh level" and "The subject is not fatigued at 65 seconds".

    `who` ("Subject 13", "This recording") catches the rest. A paragraph that
    opens by naming the subject is restating the finding -- the sentences that
    actually add something all begin "This reading...", "This small
    deviation...", "For a change to count...". That is a far better signal
    than wording, which could not be widened safely: a genuinely useful answer
    opened "This reading falls within their normal fresh range, indicating
    that ...", which any content-based pattern would have deleted.

    Only short paragraphs that are nothing but the verdict go, and never the
    last remaining one, so this cannot empty an answer or quietly delete
    reasoning.
    """
    if not prose:
        return prose
    paras = [p.strip() for p in re.split(r"\n\s*\n", prose.strip()) if p.strip()]

    names = re.compile(rf"^{re.escape(who)}\b\s*('s|is|was|has|shows|had)\b",
                       re.IGNORECASE) if who else None

    def _is_echo(p: str) -> bool:
        if len(p.split()) > _ECHO_MAX_WORDS:
            return False
        return bool(_ECHO.search(p)) or bool(names and names.match(p))

    while len(paras) > 1 and _is_echo(paras[0]):
        paras.pop(0)
    while len(paras) > 1 and _is_echo(paras[-1]):
        paras.pop()
    return "\n\n".join(paras)


# The lab-report words the model reaches for even when the prompt names them
# as forbidden. Each maps to the phrase a person would actually say. Ordered
# longest-first so "indicative of" is not half-matched by a shorter key.
#
# "indicating" is two different words. After a clause it is a connective and
# "which means" is what a person would say there; after a verb it is an
# ordinary participle, and swapping it produced "before the model would start
# which means fatigue" -- observed live, and worse English than the jargon it
# replaced. The connective sense is the one that follows a comma, so that is
# what the comma-anchored entries above the bare one are for.
_PLAIN_WORDS = [
    ("indicative of", "a sign of"),
    ("is indicative", "is a sign"),
    (", indicating that", ", which means"),
    (", indicating", ", which means"),
    ("indicating that", "which means"),
    ("indicating", "showing"),
    ("deviation from", "change from"),
    ("deviation", "change"),
]


def _word_bounded(key: str) -> str:
    """\\b only where the key actually starts or ends on a word character.

    ", indicating" opens on punctuation, and \\b there asserts a boundary
    against whatever precedes the comma rather than against the key itself.
    """
    lead = r'\b' if key[:1].isalnum() else ''
    tail = r'\b' if key[-1:].isalnum() else ''
    return lead + re.escape(key) + tail


_PLAIN_RE = [(re.compile(_word_bounded(k), re.IGNORECASE), v)
             for k, v in _PLAIN_WORDS]


def plain_words(prose: str) -> str:
    """Swap the jargon the prompt bans but the model writes anyway.

    llama3.2:3b walked through a twelve-item banned-words list and still
    produced "indicating a substantial decrease in performance capacity". The
    list in the prompt is now short, and the two or three words it still slips
    in are replaced here -- deterministic, and the same approach the verdict
    and the invented figures already take.

    Capitalisation of the original is carried over so a swap at the start of
    a sentence does not produce a lower-case opening.
    """
    if not prose:
        return prose
    for pattern, replacement in _PLAIN_RE:
        def _sub(m, r=replacement):
            return r[0].upper() + r[1:] if m.group(0)[0].isupper() else r
        prose = pattern.sub(_sub, prose)
    return prose


# Sentence end: ., ! or ? followed by whitespace, but not after a decimal
# point ("66.8 Hz") or a common abbreviation.
_SENTENCE_END = re.compile(r'(?<![0-9])([.!?])(?=\s|$)')


def trim_sentences(prose: str, limit: int = 2, total: int | None = None) -> str:
    """Keep at most `limit` sentences per paragraph of the model's prose.

    `total` additionally caps the sentences across the WHOLE prose. The
    per-paragraph limit is the right shape for the reading answers, where the
    caller appends its own paragraphs around the model's -- but it puts no
    ceiling on length at all when every paragraph is the model's own, which is
    the follow-up case: asked for 2-4 sentences, it returned three paragraphs
    of three and every one of them was under the per-paragraph limit.

    Asked for "1-3 short sentences", llama3.2:3b wrote four long ones; asked
    for "ONE or TWO", it still wrote three and padded the last with a summary
    of the first. Length is the one property of an answer that can be enforced
    exactly, so it is enforced here rather than requested -- the same move as
    rendering the verdict in code instead of trusting the model to phrase it.

    Paragraph structure is preserved: the caller appends its own paragraphs
    (the rendered verdict above, the recommendation below) and trimming across
    a blank line would splice them together.
    """
    if not prose:
        return prose

    # Split every paragraph into its complete sentences plus whatever trails
    # after the last full stop. That trailing piece is an unfinished sentence:
    # it appears when the model hits llm.NUM_PREDICT mid-thought and stops
    # dead -- "...and their muscle signal is still with".
    parsed = []
    for para in prose.split("\n\n"):
        stripped = para.strip()
        if not stripped:
            continue
        # Markdown bullets are a list, not prose -- counting "sentences"
        # across them would cut the list off mid-way.
        if stripped.startswith(("-", "*", "#", ">")):
            parsed.append((None, stripped))
            continue
        pieces, start = [], 0
        for m in _SENTENCE_END.finditer(stripped):
            pieces.append(stripped[start:m.end()].strip())
            start = m.end()
        parsed.append((pieces, stripped[start:].strip()))

    # Whether ANY complete sentence survived anywhere in the answer -- not
    # just in this paragraph. Judged across the whole prose because the model
    # writes the cut-off sentence in its own paragraph as often as not: with
    # the test applied per paragraph, "Muscles tire.\n\nAs the body's energy
    # stores are depleted, fatigue sets in, causing a gradual decline in"
    # kept the fragment, since that paragraph alone had nothing complete.
    has_complete = any(p is not None and p for p, _ in parsed)

    out, budget = [], total
    for pieces, tail in parsed:
        if pieces is None:            # a bullet list, passed through whole
            out.append(tail)
            continue
        kept = list(pieces)
        # Keep an unfinished sentence only when the alternative is showing
        # nothing at all.
        if tail and not has_complete:
            kept.append(tail)
        kept = kept[:limit]
        if budget is not None:
            if budget <= 0:
                break                 # whole paragraphs, never a part of one
            kept = kept[:budget]
            budget -= len(kept)
        if kept:
            out.append(" ".join(kept))
    return "\n\n".join(out)


# Words carrying no content, ignored when judging whether two sentences say
# the same thing. Without this, two sentences about entirely different things
# still share "the", "is", "of" and "their" and score as related.
_STOPWORDS = frozenset(
    "a an and are as at be been but by can could do does for from had has have "
    "he her him his i if in into is it its me my not of on or our she so than "
    "that the their them then there these they this to us was we were what "
    "when which who will with would you your".split())

# Above this share of the new sentence's content words already present in the
# previous answer, it is a restatement. 0.8 rather than 0.5: at 0.5 a genuinely
# new sentence about the same reading ("this sits well inside the range their
# fresh recordings covered") was scored as a repeat, because a follow-up is
# ABOUT the previous answer and is expected to reuse its subject matter.
_REPEAT_SHARE = 0.8


def _content_words(sentence: str) -> set:
    return {w for w in re.findall(r"[a-z0-9%]+", sentence.lower())
            if w not in _STOPWORDS}


def drop_repeated_sentences(prose: str, previous_answer: str) -> str:
    """Remove sentences of a follow-up that only say the previous answer again.

    A follow-up is handed the previous answer as context, and the likeliest
    continuation of "here is an answer, now write about it" is that answer
    again in different words. Asked "so what does this mean?" under a reading,
    the model returned three paragraphs of which all three restated the
    verdict -- factually perfect and worth nothing to the person who had just
    read it and wanted to know what to do with it.

    The prompt says not to (see prompt._FOLLOWUP_NEVER_RESTATE) and that only
    half-holds, which is the same position the verdict, the banned vocabulary
    and the answer length were all in before they were moved into code.

    Judged on content-word overlap rather than wording, so a paraphrase is
    caught along with a copy. Never returns empty: if every sentence is a
    repeat the whole prose is handed back for the caller to deal with, because
    showing a redundant answer beats showing none.
    """
    if not prose or not previous_answer:
        return prose
    seen = _content_words(previous_answer)
    if not seen:
        return prose

    kept_paras = []
    for para in re.split(r"\n\s*\n", prose.strip()):
        if not para.strip():
            continue
        good = []
        for sentence in re.split(r"(?<=[.!?])\s+", para.strip()):
            words = _content_words(sentence)
            # A sentence too short to carry an idea is judged on nothing and
            # would always score 1.0; leave those to trim_sentences.
            if len(words) >= 4 and len(words & seen) / len(words) >= _REPEAT_SHARE:
                continue
            good.append(sentence)
        if good:
            kept_paras.append(" ".join(good))
    return "\n\n".join(kept_paras).strip() or prose


# A hertz figure alongside a word placing it above or below something. This is
# the one shape in a follow-up that can be factually wrong while every number
# in it is real, so it is the one shape that gets removed.
#
# "rest of the sentence" has to step over a decimal point -- written as
# [^.!?]* it stopped dead at the "." in "70.0 Hz", so the very sentence this
# was built for ("would drop below 70.0 Hz") went straight through it.
_REST = r"(?:[^.!?]|(?<=\d)\.(?=\d))*"
_PLACES = (r"(?:above|below|higher|lower|greater|less|more|under|over|"
           r"exceeds?|drops? to|falls? to)")
_HZ_COMPARISON = re.compile(
    r"\bHz\b(?=" + _REST + r"\b" + _PLACES + r"\b)|"
    r"\b" + _PLACES + r"\b(?=" + _REST + r"\bHz\b)", re.IGNORECASE)


def drop_hertz_comparisons(prose: str) -> str:
    """Remove sentences that place one hertz figure above or below another.

    A follow-up is allowed to quote the figures from the answer it is
    re-explaining -- screened against the measured facts alone, every number
    was stripped and the answers came back as dangling fragments. That let
    through "their current signal is 66.8 Hz, which is still above the fresh
    level of 70.0 Hz". Both figures are real and correctly attributed; the
    relation between them is backwards. 66.8 is below 70.0.

    Nothing is lost by removing it. Which way the reading has moved is
    rendered in Python by verdict_sentence ("5% below their own fresh level")
    and the pair itself is on the provenance line directly under the answer --
    so the model has no reason to reconstruct the comparison in prose, and no
    reliable way of doing it. Percentages, times and a lone hertz figure are
    untouched.

    Never returns empty, for the same reason drop_repeated_sentences does not.
    """
    if not prose or "hz" not in prose.lower():
        return prose
    kept = []
    for para in re.split(r"\n\s*\n", prose.strip()):
        good = [s for s in re.split(r"(?<=[.!?])\s+", para.strip())
                if not _HZ_COMPARISON.search(s)]
        if good:
            kept.append(" ".join(good))
    return "\n\n".join(kept).strip() or prose


# Telling the reader to DO something. Two halves, and a sentence needs both:
# a directive construction, and a thing to do that this measurement says
# nothing about. "They should keep going" is advice; "the signal should be
# read against their own baseline" is not, and only the first has a verb from
# the second list.
_DIRECTIVE = re.compile(
    r"\b(should|shouldn'?t|ought to|need(?:s)? to|must|may want to|"
    r"might want to|may need to|consider|try to|make sure|be sure to|"
    r"it(?:'s| is) (?:important|advisable|worth) to|"
    r"(?:i|we|you) (?:recommend|suggest|advise))\b", re.IGNORECASE)
_ADVICE_TOPIC = re.compile(
    r"\b(rest|resting|break|breaks|stop|stopping|pause|recover|recovery|"
    r"hydrat\w+|drink|water|pace|pacing|slow down|ease off|back off|"
    r"grip|technique|form|posture|intensity|workload|train\w*|exercis\w*|"
    r"warm up|cool down|stretch\w*|sleep|nutrition|adjust\w*)\b",
    re.IGNORECASE)


def drop_advice(prose: str) -> str:
    """Remove sentences telling the reader what to do about the reading.

    A plain reading question got "they should take regular breaks to rest and
    recover during exercise" and "you may need to adjust your grip or
    technique to maintain control". Nothing in an EMG window measures whether
    a break, a grip change or a lighter session is warranted -- the model is
    filling a gap with generic coaching, and it reads as though the
    measurement supports it.

    This is only for answers where no recommendation was asked for. When one
    IS asked for, recommend.py builds it deliberately, with its disclaimer,
    and this must not run over that.

    Never returns empty: if advice was the entire answer the caller still has
    the rendered verdict above it, so an empty string here is safe, but the
    prose is handed back rather than silently blanked.
    """
    if not prose:
        return prose
    kept = []
    for para in re.split(r"\n\s*\n", prose.strip()):
        good = [s for s in re.split(r"(?<=[.!?])\s+", para.strip())
                if not (_DIRECTIVE.search(s) and _ADVICE_TOPIC.search(s))]
        if good:
            kept.append(" ".join(good))
    return "\n\n".join(kept).strip() or prose


def strip_invented_numbers(prose: str, who: str | None = None) -> tuple[str, list[str]]:
    """Drop sentences quoting measurements the model was never given.

    Returns (cleaned prose, the removed sentences).

    Withholding the figures stopped the model misattributing them and started
    it inventing them instead: "The current reading of 12 Hz is lower than the
    subject's fresh level of 15 Hz", where the real values were 58.2 and 62.7
    and neither was in its prompt. That is the exact failure this project's
    grounding architecture exists to prevent, so it is removed rather than
    argued with.

    ANY digit counts, not just one carrying a unit. Restricting it to hertz,
    percentages and decimals let through "The person's current reading is
    200s, which is outside their normal fresh range" -- a timestamp lifted
    from the provenance line and reported as a measurement. The model is told
    to write in words only, so a bare number is as wrong as a units one.

    `who` is exempted, because the subject's own name legitimately contains
    digits ("Subject 13"). Only whole sentences go, and a sentence that
    reasons in words survives untouched, which is all the prose is for.
    """
    if not prose:
        return prose, []

    def _has_number(sentence: str) -> bool:
        probe = sentence
        if who:
            probe = re.sub(re.escape(who), "", probe, flags=re.IGNORECASE)
        return bool(re.search(r"\d", probe))

    kept, removed = [], []
    for para in re.split(r"\n\s*\n", prose.strip()):
        sentences = re.split(r"(?<=[.!?])\s+", para.strip())
        good = [s for s in sentences if not _has_number(s)]
        removed += [s for s in sentences if _has_number(s)]
        if good:
            kept.append(" ".join(good))
    return "\n\n".join(kept).strip(), removed


def technical_line(reading: dict) -> str:
    """The raw figures, kept for the technical reader but out of the way."""
    bits = []
    if reading["mdf_hz"] is not None:
        bits.append(f"median frequency {reading['mdf_hz']:.1f} Hz")
    if reading["fresh_mdf"]:
        bits.append(f"fresh baseline {reading['fresh_mdf']:.1f} Hz")
    if reading["z"] is not None:
        bits.append(f"z = {reading['z']:+.1f}")
    if reading["confidence"] is not None:
        bits.append(f"confidence {reading['confidence'] * 100:.1f}%")
    return " · ".join(bits)
