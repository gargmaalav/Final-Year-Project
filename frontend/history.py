"""
Chat history: multiple conversations, persisted to disk as JSON.
==================================================================

Each chat is one file under .chat_history/<id>.json (gitignored -- local to
the machine, like a real ChatGPT session list).

Chart HTML is deliberately NOT persisted: one chart rendered 3.1 MB of HTML
in this session's own testing, and saving several of those per chat would
bloat the folder fast for no real benefit (charts are cheap to regenerate,
but the raw fatigue numbers/timestamps are what's worth keeping). A reloaded
past chat shows its saved text answers with no chart underneath; that's a
deliberate tradeoff, not an oversight.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone

HISTORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".chat_history")
_PERSISTED_MESSAGE_KEYS = ("role", "content", "recommendation", "timestamp")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_chat() -> dict:
    return {
        "id": str(uuid.uuid4()),
        "title": None,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "messages": [],
    }


def auto_title(first_user_message: str, max_len: int = 40) -> str:
    text = " ".join(first_user_message.split())
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."


def _path(chat_id: str) -> str:
    return os.path.join(HISTORY_DIR, f"{chat_id}.json")


def save_chat(chat: dict) -> None:
    os.makedirs(HISTORY_DIR, exist_ok=True)
    chat = dict(chat)
    chat["updated_at"] = _now_iso()
    if not chat.get("title") and chat["messages"]:
        first_user = next((m for m in chat["messages"] if m["role"] == "user"), None)
        if first_user:
            chat["title"] = auto_title(first_user["content"])

    # drop chart HTML (and anything else not in the whitelist) before writing
    # -- see module docstring
    to_write = dict(chat)
    to_write["messages"] = [
        {k: v for k, v in m.items() if k in _PERSISTED_MESSAGE_KEYS}
        for m in chat["messages"]
    ]
    with open(_path(chat["id"]), "w", encoding="utf-8") as f:
        json.dump(to_write, f, indent=2)


def list_chats() -> list[dict]:
    if not os.path.isdir(HISTORY_DIR):
        return []
    chats = []
    for fname in os.listdir(HISTORY_DIR):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(HISTORY_DIR, fname), encoding="utf-8") as f:
                chats.append(json.load(f))
        except (OSError, json.JSONDecodeError):
            continue
    chats.sort(key=lambda c: c.get("updated_at", ""), reverse=True)
    return chats


def load_chat(chat_id: str) -> dict | None:
    try:
        with open(_path(chat_id), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def delete_chat(chat_id: str) -> None:
    try:
        os.remove(_path(chat_id))
    except OSError:
        pass


def relative_time(iso_str: str) -> str:
    try:
        then = datetime.fromisoformat(iso_str)
    except ValueError:
        return ""
    now = datetime.now(timezone.utc)
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    delta = now - then
    secs = delta.total_seconds()
    if secs < 60:
        return "just now"
    if secs < 3600:
        return f"{int(secs // 60)}m ago"
    if secs < 86400:
        return f"{int(secs // 3600)}h ago"
    return f"{int(secs // 86400)}d ago"
