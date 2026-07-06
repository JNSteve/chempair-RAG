import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import session_store  # noqa: E402
import upstream_errors  # noqa: E402


def _session(age_seconds: float = 0.0, history: list | None = None) -> dict:
    return {
        "history": history
        if history is not None
        else [{"role": "user", "content": "hi"}],
        "last_used": time.time() - age_seconds,
    }


def test_persist_and_load_round_trip(tmp_path):
    sessions = {"sid-1": _session(), "sid-2": _session(history=[])}
    session_store.persist_sessions(tmp_path, sessions)

    loaded = session_store.load_sessions(tmp_path, ttl_seconds=3600)
    assert set(loaded) == {"sid-1", "sid-2"}
    assert loaded["sid-1"]["history"] == [{"role": "user", "content": "hi"}]


def test_load_drops_expired_and_malformed_entries(tmp_path):
    raw = {
        "fresh": _session(age_seconds=10),
        "stale": _session(age_seconds=7200),
        "malformed": {"history": "not-a-list", "last_used": time.time()},
        "no-timestamp": {"history": []},
        "not-a-dict": 42,
    }
    session_store.sessions_file(tmp_path).write_text(json.dumps(raw), encoding="utf-8")

    loaded = session_store.load_sessions(tmp_path, ttl_seconds=3600)
    assert set(loaded) == {"fresh"}


def test_load_missing_or_corrupt_file_returns_empty(tmp_path):
    assert session_store.load_sessions(tmp_path, 3600) == {}
    session_store.sessions_file(tmp_path).write_text("{corrupt", encoding="utf-8")
    assert session_store.load_sessions(tmp_path, 3600) == {}


def test_persist_into_missing_dir_is_a_noop(tmp_path):
    missing = tmp_path / "does-not-exist"
    session_store.persist_sessions(missing, {"sid": _session()})
    assert not missing.exists()


def test_persist_never_raises_on_unserializable_history(tmp_path):
    sessions = {"sid": {"history": [object()], "last_used": time.time()}}
    session_store.persist_sessions(tmp_path, sessions)  # must not raise
    assert session_store.load_sessions(tmp_path, 3600) == {}


# --- upstream error classification ---


def test_rate_limit_errors_map_to_429():
    status, message = upstream_errors.classify_upstream_error(
        Exception("openai.RateLimitError: insufficient_quota (429)")
    )
    assert status == 429
    assert message == upstream_errors.RATE_LIMIT_MESSAGE


def test_connection_and_retry_errors_map_to_503():
    for raw in (
        "RetryError[<Future at 0x7f2 state=finished raised APIConnectionError>]",
        "Connection error.",
        "Request timed out",
        "Incorrect API key provided",
    ):
        status, message = upstream_errors.classify_upstream_error(Exception(raw))
        assert status == 503, raw
        assert message == upstream_errors.UNAVAILABLE_MESSAGE


def test_unknown_errors_map_to_generic_500():
    status, message = upstream_errors.classify_upstream_error(
        ValueError("KeyError: 'x'")
    )
    assert status == 500
    assert message == upstream_errors.GENERIC_MESSAGE
    assert "KeyError" not in message
