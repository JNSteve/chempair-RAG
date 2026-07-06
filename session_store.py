"""File-backed persistence for chat sessions.

Sessions previously lived only in process memory, so every backend restart
or redeploy silently broke ongoing conversations (the frontend's follow-up
questions lost their history). This module checkpoints the sessions dict to
<RAG_STORAGE>/sessions.json so restarts resume where they left off.

Deliberately forgiving: if the storage dir doesn't exist (unit tests, fresh
checkouts) or the file is corrupt, persistence becomes a no-op — chat must
never fail because of the checkpoint.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

SESSIONS_FILENAME = "sessions.json"


def sessions_file(rag_storage: str | Path) -> Path:
    return Path(rag_storage) / SESSIONS_FILENAME


def load_sessions(rag_storage: str | Path, ttl_seconds: float) -> dict[str, dict]:
    """Load persisted sessions, dropping expired and malformed entries."""
    path = sessions_file(rag_storage)
    try:
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
    except Exception:
        return {}

    if not isinstance(raw, dict):
        return {}

    now = time.time()
    sessions: dict[str, dict] = {}
    for session_id, session in raw.items():
        if not isinstance(session, dict):
            continue
        last_used = session.get("last_used")
        history = session.get("history")
        if not isinstance(last_used, (int, float)) or not isinstance(history, list):
            continue
        if now - last_used > ttl_seconds:
            continue
        sessions[str(session_id)] = {"history": history, "last_used": last_used}
    return sessions


def persist_sessions(rag_storage: str | Path, sessions: dict[str, dict]) -> None:
    """Atomically checkpoint sessions. No-op when the storage dir is absent;
    never raises."""
    storage = Path(rag_storage)
    if not storage.is_dir():
        return
    try:
        path = sessions_file(storage)
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(sessions), encoding="utf-8")
        tmp_path.replace(path)
    except Exception:
        return
