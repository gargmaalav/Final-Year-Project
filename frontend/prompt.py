"""
build_prompt(features, user_query) -> str
==========================================

The team integration contract (README.md):
  Produced by: Maalav.
  Build the LLM system prompt from pipeline features + the user's question.
  Constrain the LLM to only use the provided feature values -- no
  hallucinated numbers.

`features` is the dict returned by models/classify.py's classify():
  {"mdf_hz": float, "fatigue_label": int, "confidence": float,
   "fatigue_state": str}

`forecast` is optional: models/fatigue_forecast.py's forecast_fatigue() dict (must
have "ok": True and a "summary" string) if a trend projection is available
for this reading. Kept as an optional trailing arg so the contract's two-arg
signature (features, user_query) still works unchanged for callers that
don't have a forecast.
"""
from __future__ import annotations


def describe_window(window: dict | None) -> str:
    """Plain-English name for the window that was actually classified."""
    if not window:
        return ""
    if window.get("source") == "upload":
        return f"the uploaded recording at {window['t_start']:.0f}s"
    side = "right" if window.get("side", "R").upper() == "R" else "left"
    return (f"subject {window['subject']}, {side} arm, "
            f"at {window['t_start']:.0f}s")


def build_prompt(features: dict, user_query: str, forecast: dict | None = None,
                 window: dict | None = None) -> str:
    """Build the grounding prompt.

    `window` names the reading's provenance (which subject/time/side, or that
    it came from an upload) and is stated in the answer on purpose. Without it
    a follow-up like "and at 90 seconds?" produces an answer that never says
    which window it used, so a wrong resolution is invisible to the reader.
    """
    forecast_line = (
        f"- fatigue trend: {forecast['summary']}\n" if forecast and forecast.get("ok") else ""
    )
    where = describe_window(window)
    where_line = f"- reading taken from: {where}\n" if where else ""

    calib = features.get("calibration")
    calib_line = (
        f"- calibration caveat: {calib['note']}; this is less reliable than a "
        "reading for one of the 13 dataset subjects\n" if calib else ""
    )
    say_where = (
        f"Begin by stating that this reading is for {where}. " if where else ""
    )
    say_calib = (
        "Also mention in one short clause that the baseline was estimated from "
        "the uploaded recording itself, so the result is less reliable. "
        if calib else ""
    )

    return (
        "You are an assistant reporting the result of a lab sensor reading "
        "(muscle fatigue from an EMG sensor) -- this is a factual "
        "engineering measurement, not a medical diagnosis, so it is safe "
        "and expected to state it plainly. Ground your answer ONLY in the "
        "measured values below -- do not invent, adjust, or round "
        "differently any number that isn't listed here.\n\n"
        f"Measured result:\n"
        f"{where_line}"
        f"- fatigue state: {features['fatigue_state']}\n"
        f"- median frequency: {features['mdf_hz']:.1f} Hz\n"
        f"- model confidence: {features['confidence'] * 100:.1f}%\n"
        f"{calib_line}"
        f"{forecast_line}\n"
        f"User question: {user_query}\n\n"
        f"Answer in 1-4 sentences. {say_where}State the fatigue state and "
        f"confidence plainly. {say_calib}Mention the trend only if it's "
        "relevant to the question or a forecast is provided above. If the "
        "question also asks for sport or training suggestions, ignore that "
        "part here -- it is answered separately; just report the measured "
        "fatigue result."
    )
