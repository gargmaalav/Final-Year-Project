"""
Thin wrapper around Ollama's local HTTP API.
=============================================

Shared by extract.py (JSON-extraction call) and app.py (final grounded-answer
call), so the model name, endpoint, and error handling live in one place.
"""
from __future__ import annotations

import requests

OLLAMA_BASE = "http://localhost:11434"
MODEL = "llama3.2:3b"

# Warm calls to llama3.2:3b measured 12-25 s on a laptop. The risk is not the
# warm case but the first call after `ollama serve` starts, which also loads
# the model into memory and can take minutes. 60 s was tight enough to turn
# that into a visible error in the opening seconds of a demo; a slow answer is
# better than a failed one, and the timeout message now says which happened.
TIMEOUT_SEC = 120


class LLMError(Exception):
    pass


# Every prompt in this project already asks for 2-5 sentences, but nothing
# stopped the model from ignoring that and rambling -- on a CPU laptop each
# extra generated token is directly extra wall-clock time, so this caps the
# worst case rather than trusting the instruction alone. 220 is generous for
# 2-5 sentences of plain English; raise per-call for prompts that legitimately
# need more (e.g. the recommendation prompt's disclaimer).
DEFAULT_NUM_PREDICT = 220


def chat(messages: list[dict], *, model: str | None = None,
         temperature: float = 0.0, timeout: float | None = None,
         num_predict: int | None = DEFAULT_NUM_PREDICT) -> str:
    """Send a chat completion request to Ollama, return the reply text.

    messages: list of {"role": "system"|"user"|"assistant", "content": str}.
    timeout: seconds to wait, defaulting to TIMEOUT_SEC. A cold model load
        can take most of a minute on a laptop, so callers that only want to
        know whether Ollama is answering at all should pass a short one
        rather than blocking the caller for the full default.
    num_predict: max tokens to generate (Ollama's `options.num_predict`).
        None removes the cap.
    """
    waited = TIMEOUT_SEC if timeout is None else timeout
    options = {"temperature": temperature}
    if num_predict is not None:
        options["num_predict"] = num_predict
    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/chat",
            json={
                "model": model or MODEL,
                "messages": messages,
                "stream": False,
                "options": options,
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
