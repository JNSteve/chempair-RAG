"""Optional map_image on /query: validation, and the vision call path.

A snapshot of the consultant's current map view rides alongside the
question as a data URL. It is interpretive evidence only — the unified
system prompt constrains how the model may use it.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


FAKE_JPEG = "data:image/jpeg;base64," + ("aGVsbG8=" * 4)


def _context() -> dict:
    return {
        "schemaVersion": 5,
        "projectState": {"project": {"projectName": "Ducat"}},
        "mapContext": {"selectedAnalyte": "Arsenic", "contourAreaM2": 764},
    }


@pytest.fixture()
def client():
    with patch.dict(
        "os.environ",
        {"OPENAI_API_KEY": "test-key", "RAG_AUTH_REQUIRED": "false"},
    ):
        import server

        server.app.router.on_startup.clear()
        server.app.router.on_shutdown.clear()
        server.sessions.clear()

        mock_rag = MagicMock()
        mock_rag.aquery = AsyncMock(return_value="kb answer")
        mock_rag.lightrag = MagicMock()
        mock_rag.lightrag.aquery_data = AsyncMock(
            return_value={"status": "success", "data": {"references": [], "chunks": []}}
        )
        mock_openai = AsyncMock(return_value="text-only unified answer")

        server.rag = mock_rag
        with patch.object(server, "openai_complete_if_cache", mock_openai):
            with TestClient(server.app, raise_server_exceptions=False) as test_client:
                yield test_client, server, mock_openai

        server.sessions.clear()


def _vision_client_mock(answer: str = "vision answer"):
    """A stub AsyncOpenAI whose chat.completions.create returns `answer`."""
    completion = MagicMock()
    completion.choices = [MagicMock(message=MagicMock(content=answer))]
    instance = MagicMock()
    instance.chat.completions.create = AsyncMock(return_value=completion)
    return instance


def test_malformed_map_image_is_rejected(client):
    test_client, _, _ = client
    response = test_client.post(
        "/query",
        json={
            "question": "what's on the aerial?",
            "context": _context(),
            "map_image": "https://example.com/evil.jpg",
        },
    )
    assert response.status_code == 422
    assert "data URL" in response.json()["detail"]


def test_oversize_map_image_is_rejected(client):
    test_client, server, _ = client
    huge = "data:image/jpeg;base64," + "A" * (server.MAX_MAP_IMAGE_CHARS + 1)
    response = test_client.post(
        "/query",
        json={"question": "look at the map", "context": _context(), "map_image": huge},
    )
    assert response.status_code == 422
    assert "too large" in response.json()["detail"]


def test_map_image_goes_to_vision_call_with_prompt_and_image(client):
    test_client, server, mock_openai = client
    vision = _vision_client_mock("There appears to be a stockpile near BH07.")

    with patch("openai.AsyncOpenAI", return_value=vision):
        response = test_client.post(
            "/query",
            json={
                "question": "anything on the aerial I should target?",
                "context": _context(),
                "map_image": FAKE_JPEG,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "There appears to be a stockpile near BH07."
    assert body["route_used"] == "unified"

    # The vision call carries the system prompt, the unified text prompt,
    # and the image part; the text-only client is not used.
    mock_openai.assert_not_awaited()
    call = vision.chat.completions.create.await_args
    messages = call.kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == server.UNIFIED_ANSWER_SYSTEM
    parts = messages[1]["content"]
    text_part = next(p for p in parts if p["type"] == "text")
    image_part = next(p for p in parts if p["type"] == "image_url")
    assert "=== MAP SNAPSHOT ===" in text_part["text"]
    assert "Contour area: 764 m2" in text_part["text"]
    assert image_part["image_url"]["url"] == FAKE_JPEG
    assert call.kwargs["model"] == server.LLM_MODEL


def test_without_map_image_text_client_is_used(client):
    test_client, _, mock_openai = client
    response = test_client.post(
        "/query",
        json={"question": "how big is the contour?", "context": _context()},
    )
    assert response.status_code == 200
    assert response.json()["answer"] == "text-only unified answer"
    mock_openai.assert_awaited_once()
    prompt = mock_openai.await_args.args[1]
    assert "=== MAP SNAPSHOT ===" not in prompt


def test_system_prompt_carries_map_snapshot_rules():
    import server

    assert "Map snapshot rules" in server.UNIFIED_ANSWER_SYSTEM
    assert "interpretive, never authoritative" in server.UNIFIED_ANSWER_SYSTEM
    assert "Never read measurements" in server.UNIFIED_ANSWER_SYSTEM
