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

    # fatigued verdict, but the fatigue marker has not actually fallen
    conflict = bool(fatigued and z is not None and z > 0)

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

    position = reading["position"]
    if position:
        lines.append(
            f"{who} is {reading['verdict']} {position['t_start']:.0f} seconds "
            f"into a {position['duration']:.0f} second effort, which is "
            f"{position['percent']:.0f}% of the way through the recording")
    else:
        lines.append(f"{who} is {reading['verdict']}")

    if reading["drop_percent"] is not None:
        fell = reading["drop_percent"] >= 0
        gloss = ("a falling signal is what muscle fatigue looks like" if fell
                 else "it has risen rather than fallen, which is not the "
                      "direction fatigue moves it")
        lines.append(
            f"their muscle signal is {abs(reading['drop_percent']):.0f}% "
            f"{'below' if fell else 'above'} their own fresh level "
            f"({reading['mdf_hz']:.1f} Hz now, {reading['fresh_mdf']:.1f} Hz "
            f"when fresh) -- {gloss}")

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
        lines.append(
            "worth flagging: the classifier calls this fatigued, but the "
            "median-frequency marker has not fallen, so the two disagree "
            "here -- the model weighs eight signal features and this is only "
            "one of them, but treat this particular reading as less settled")

    if reading["confidence"] is not None:
        lines.append(
            f"the model's own certainty in the fatigued/not-fatigued call is "
            f"{reading['confidence'] * 100:.0f}% -- this is how sure the model "
            "is, not how accurate it is known to be")

    return lines


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
