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
TIMEOUT_SEC = 60


class LLMError(Exception):
    pass


def chat(messages: list[dict], *, model: str | None = None,
         temperature: float = 0.0) -> str:
    """Send a chat completion request to Ollama, return the reply text.

    messages: list of {"role": "system"|"user"|"assistant", "content": str}.
    """
    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/chat",
            json={
                "model": model or MODEL,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=TIMEOUT_SEC,
        )
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
