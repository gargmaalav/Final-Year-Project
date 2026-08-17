"""
Thin wrapper around Ollama's local HTTP API.
=============================================

Shared by extract.py (JSON-extraction call) and app.py (final grounded-answer
call), so the model name, endpoint, and error handling live in one place.
"""
from __future__ import annotations

import requests

OLLAMA_BASE = "http://localhost:11434"
# 1b generates about 2.5x faster than 3b (2.5 s vs 6.5 s for a typical answer)
# and processes the prompt faster too. The model's job here is small by design:
# every number, and the verdict itself, is rendered in Python -- it only writes
# one or two sentences of connective prose.
#
# KNOWN WEAKNESS, measured before switching: 1b opened all four sampled
# readings with "The person is not fatigued", including the two the classifier
# called FATIGUED. It survives only because it writes that as its own short
# paragraph and interpret.strip_verdict_echo drops it (tested), and because the
# verdict the reader sees is rendered from the label in Python and cannot be
# inverted. An echo written INLINE would get through -- see the two
# strip_verdict_echo tests in test_answers.py. Put this back to "llama3.2:3b"
# if answers start contradicting the verdict above them; 3b is still pulled.
MODEL = "llama3.2:1b"

# Ollama drops a model from memory 5 minutes after its last request, and
# reloading llama3.2 off disk measured 6.7 s. Questions in a demo arrive
# minutes apart, so nearly every one paid that -- the same question asked
# twice in a row took 3.2 s the second time and 9.6 s after a gap. Half an
# hour covers a session; the cost is the model staying resident in RAM.
KEEP_ALIVE = "30m"

# Answers are cut to two sentences in Python (interpret.trim_sentences), so
# tokens generated past that are paid for at ~10 tok/s and then thrown away.
# 160 leaves room for the longest answer we keep plus the analysis answers'
# three sentences, and caps a runaway generation instead of waiting it out.
NUM_PREDICT = 160

# Warm calls to llama3.2:3b measured 12-25 s on a laptop. The risk is not the
# warm case but the first call after `ollama serve` starts, which also loads
# the model into memory and can take minutes. 60 s was tight enough to turn
# that into a visible error in the opening seconds of a demo; a slow answer is
# better than a failed one, and the timeout message now says which happened.
TIMEOUT_SEC = 120


class LLMError(Exception):
    pass


def chat(messages: list[dict], *, model: str | None = None,
         temperature: float = 0.0, timeout: float | None = None) -> str:
    """Send a chat completion request to Ollama, return the reply text.

    messages: list of {"role": "system"|"user"|"assistant", "content": str}.
    timeout: seconds to wait, defaulting to TIMEOUT_SEC. A cold model load
        can take most of a minute on a laptop, so callers that only want to
        know whether Ollama is answering at all should pass a short one
        rather than blocking the caller for the full default.
    """
    waited = TIMEOUT_SEC if timeout is None else timeout
    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/chat",
            json={
                "model": model or MODEL,
                "messages": messages,
                "stream": False,
                "keep_alive": KEEP_ALIVE,
                "options": {"temperature": temperature,
                            "num_predict": NUM_PREDICT},
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
