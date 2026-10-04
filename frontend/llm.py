"""
Thin wrapper around Ollama's local HTTP API.
=============================================

Shared by extract.py (JSON-extraction call) and app.py (final grounded-answer
call), so the model name, endpoint, and error handling live in one place.
"""
from __future__ import annotations

import requests

OLLAMA_BASE = "http://localhost:11434"
# llama3.2:1b was tried and rejected on accuracy, not speed. It was ~2.5x
# faster to generate, but it opened all four sampled readings with "The person
# is not fatigued" -- including the two the classifier called FATIGUED. Nothing
# wrong ever reached the screen (the verdict is rendered from the label in
# Python, and strip_verdict_echo drops the inverted opener), but the answer was
# one phrasing away from contradicting the verdict printed above it, and its
# prose was measurably worse -- one answer misused the confidence figure in a
# way 3b does not. Speed is being bought elsewhere instead: see KEEP_ALIVE and
# NUM_PREDICT below, prompt.build_prompt's cacheable-prefix ordering, and
# turn.py's _cached_chart.
MODEL = "llama3.2:3b"

# Ollama drops a model from memory 5 minutes after its last request, and
# reloading llama3.2 off disk measured 6.7 s. Questions in a demo arrive
# minutes apart, so nearly every one paid that -- the same question asked
# twice in a row took 3.2 s the second time and 9.6 s after a gap. Half an
# hour covers a session; the cost is the model staying resident in RAM.
KEEP_ALIVE = "30m"

# A ceiling that should never be reached, not a length target. Length is
# controlled by asking for one or two sentences and by cutting to that in
# Python (interpret.trim_sentences); this exists only so a runaway generation
# cannot hang the answer.
#
# Sized from the longest path measured, uncapped, on llama3.2:3b:
#     reading          62 tokens
#     follow-up "why?" 106 tokens
#     recommendation  152 tokens   <- the longest, and the one that matters
# The recommendation is the case to size for: unlike the readings it is NOT
# trimmed to two sentences afterwards, so every token it generates is shown.
# At 200 it was running at 76% of the ceiling -- close enough that a slightly
# wordier one would have been cut. 320 is over twice the worst measured case.
#
# Why a high ceiling rather than no ceiling: removing it means unlimited, not
# "sensible default". Uncapped, an open-ended prompt generated 765 tokens in
# 76 s here. Anything past ~15 s of generation is a worse outcome than a
# trimmed tail, and past TIMEOUT_SEC the request fails and the prose is lost
# entirely -- the answer falls back to bullet points. A truncated sentence is
# dropped cleanly by trim_sentences; a timeout throws the whole reply away.
# 320 tokens is ~40 s at 8 tok/s, comfortably inside the timeout.
NUM_PREDICT = 320

# Warm calls to llama3.2:3b measured 12-25 s on a laptop. The risk is not the
# warm case but the first call after `ollama serve` starts, which also loads
# the model into memory and can take minutes. 60 s was tight enough to turn
# that into a visible error in the opening seconds of a demo; a slow answer is
# better than a failed one, and the timeout message now says which happened.
TIMEOUT_SEC = 120


# Every generation that hit NUM_PREDICT instead of ending on its own, as
# token counts. Empty is the expected state; anything in here means the cap is
# too low for some real question and should be raised. Exposed on /health so
# it can be checked without reading the console.
TRUNCATIONS: list = []


class LLMError(Exception):
    pass


def chat(messages: list[dict], *, model: str | None = None,
         temperature: float = 0.0, timeout: float | None = None,
         num_predict: int | None = None) -> str:
    """Send a chat completion request to Ollama, return the reply text.

    messages: list of {"role": "system"|"user"|"assistant", "content": str}.
    timeout: seconds to wait, defaulting to TIMEOUT_SEC. A cold model load
        can take most of a minute on a laptop, so callers that only want to
        know whether Ollama is answering at all should pass a short one
        rather than blocking the caller for the full default.
    num_predict: cap on generated tokens, defaulting to NUM_PREDICT. The
        warm-up call (turn.warm_up) passes 1: it is priming the prompt cache
        and throws the answer away, so generating a full one is wasted time.
    """
    waited = TIMEOUT_SEC if timeout is None else timeout
    predict = NUM_PREDICT if num_predict is None else num_predict
    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/chat",
            json={
                "model": model or MODEL,
                "messages": messages,
                "stream": False,
                "keep_alive": KEEP_ALIVE,
                "options": {"temperature": temperature,
                            "num_predict": predict},
            },
            timeout=waited,
        )
    # These three failures need three different things from the reader, and
    # they all used to arrive as the same "could not reach Ollama" wrapped
    # around a requests traceback -- which reads like the server is down even
    # when it is running fine and merely loading a model.
    except requests.Timeout as e:
        raise LLMError(
            f"the model did not answer within {waited:.0f}s. The first request "
            "after starting Ollama also loads the model, which is slow; try "
            "again and it should be quicker. The measurement above is already "
            "done either way -- only the wording of the answer is missing"
        ) from e
    except requests.ConnectionError as e:
        raise LLMError(
            f"nothing is answering at {OLLAMA_BASE}. Start it with `ollama "
            "serve`, then check the model is pulled with `ollama list`"
        ) from e
    except requests.RequestException as e:
        raise LLMError(f"could not reach Ollama at {OLLAMA_BASE} ({e})") from e

    if r.status_code != 200:
        raise LLMError(f"Ollama returned {r.status_code}: {r.text}")

    data = r.json()
    content = data.get("message", {}).get("content")
    if not content:
        raise LLMError(f"Ollama response had no content: {data}")

    # Ollama says WHY it stopped: "stop" means the model finished its own
    # sentence, "length" means num_predict cut it off. Measured across every
    # answer type this app produces, generations run 28-75 tokens against a
    # 200 cap and always report "stop" -- but "we sampled it and it was fine"
    # is not a guarantee, so the rare case announces itself instead of
    # silently handing back a half sentence. trim_sentences drops the
    # unfinished tail either way; this is how you find out it happened.
    # `num_predict is None` means the caller took the default cap, i.e. this
    # is a real answer. warm_up() passes 1 deliberately and is always cut off
    # by design, so counting it would report a truncation on every boot and
    # make the number meaningless.
    if data.get("done_reason") == "length" and num_predict is None:
        TRUNCATIONS.append(data.get("eval_count"))
        print(f"[llm] answer hit the {predict}-token cap and was cut short "
              f"(generated {data.get('eval_count')}). The unfinished sentence "
              "is dropped; raise NUM_PREDICT in frontend/llm.py if this "
              "recurs.", flush=True)
    return content


def list_models() -> list[str]:
    """Names of locally-pulled Ollama models, for the model picker. Falls
    back to just MODEL if Ollama isn't reachable."""
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        r.raise_for_status()
        names = [m["name"] for m in r.json().get("models", [])]
        return names or [MODEL]
    except requests.RequestException:
        return [MODEL]
