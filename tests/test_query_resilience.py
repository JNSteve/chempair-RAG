"""Endpoint-level tests for session persistence and user-safe upstream error
mapping (heavy deps: runs in the full-suite CI job only)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path):
    with patch.dict(
        "os.environ",
        {"OPENAI_API_KEY": "test-key", "RAG_AUTH_REQUIRED": "false"},
    ):
        import server

        server.app.router.on_startup.clear()
        server.app.router.on_shutdown.clear()
        server.sessions.clear()

        mock_rag = MagicMock()
        mock_rag.aquery = AsyncMock(return_value="mocked rag answer")
        mock_rag.lightrag = MagicMock()
        mock_rag.lightrag.aquery_data = AsyncMock(
            return_value={"status": "success", "data": {"references": [], "chunks": []}}
        )
        mock_openai = AsyncMock()

        with patch.object(server, "RAG_STORAGE", str(tmp_path)):
            server.rag = mock_rag
            with patch.object(server, "openai_complete_if_cache", mock_openai):
                with TestClient(
                    server.app, raise_server_exceptions=False
                ) as test_client:
                    yield test_client, server, mock_rag, tmp_path

        server.sessions.clear()


def test_upstream_retry_error_returns_user_safe_503(client):
    test_client, _, mock_rag, _ = client
    mock_rag.aquery.side_effect = Exception(
        "RetryError[<Future at 0x7f25 state=finished raised APIConnectionError>]"
    )

    response = test_client.post(
        "/query", json={"question": "What are the soil guidelines?"}
    )

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "temporarily unavailable" in detail
    assert "RetryError" not in detail


def test_upstream_rate_limit_returns_429(client):
    test_client, _, mock_rag, _ = client
    mock_rag.aquery.side_effect = Exception(
        "Rate limit reached for gpt: insufficient_quota"
    )

    response = test_client.post(
        "/query", json={"question": "What are the soil guidelines?"}
    )

    assert response.status_code == 429
    assert "too many requests" in response.json()["detail"]


def test_sessions_are_checkpointed_and_survive_restart(client):
    test_client, server, _, tmp_path = client

    response = test_client.post(
        "/query", json={"question": "What are the soil guidelines?"}
    )
    assert response.status_code == 200
    session_id = response.json()["session_id"]

    persisted = json.loads((tmp_path / "sessions.json").read_text(encoding="utf-8"))
    assert session_id in persisted
    assert (
        persisted[session_id]["history"][0]["content"]
        == "What are the soil guidelines?"
    )

    # Simulate a restart: memory wiped, then reloaded from the checkpoint.
    server.sessions.clear()
    from session_store import load_sessions

    server.sessions.update(load_sessions(tmp_path, server.SESSION_TTL))
    follow_up = test_client.post(
        "/query",
        json={"question": "and for groundwater?", "session_id": session_id},
    )
    assert follow_up.status_code == 200
    assert follow_up.json()["session_id"] == session_id


def test_deleted_sessions_are_removed_from_checkpoint(client):
    test_client, _, _, tmp_path = client

    response = test_client.post(
        "/query", json={"question": "What are the soil guidelines?"}
    )
    session_id = response.json()["session_id"]

    delete = test_client.delete(f"/session/{session_id}")
    assert delete.status_code == 200
    persisted = json.loads((tmp_path / "sessions.json").read_text(encoding="utf-8"))
    assert session_id not in persisted
